"""Joint language training with exact, checkpointed physical response drafts.

Every input advances the continuing body once. Quiet response branches use the
current full physical state and live trainable parameters; they do not advance
the individual's clock or consume future tokens. CUDA uses the existing Triton
physical kernels, avoiding a detached/stale CSR matrix during plastic learning.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState, advance_fly_input_event, step_fly_physical_tick
from .fly_rtc_learning import physical_options
from .fly_streaming_learning import motor_features

ROUTE = 'fly-causal-rtc-joint-v1'


class CausalHorizonReadout(nn.Module):
    """Choose among native motor responses; all responses share one readout."""

    def __init__(self, feature_dim, *, horizon=14, key_dim=128):
        super().__init__()
        self.horizon = int(horizon)
        self.query = nn.Linear(feature_dim, key_dim, bias=False)
        self.key = nn.Linear(feature_dim, key_dim, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.horizon_bias = nn.Parameter(torch.zeros(horizon+1))

    def forward(self, responses):
        # [physical horizon+1, batch, native read feature]. No labels/tokens.
        if not 1 <= len(responses) <= self.horizon+1:
            raise ValueError('Readout exceeds declared response budget')
        query = F.normalize(self.query(responses[0]), dim=-1)
        keys = F.normalize(self.key(responses), dim=-1)
        logits = (keys*query[None]).sum(-1)*self.log_scale.exp()
        weights = F.softmax(logits+self.horizon_bias[:len(responses), None], dim=0)
        return (weights[..., None]*responses).sum(0), weights


def exact_response_features(model, origin, options, horizon, *, checkpoint_branch=True):
    """Differentiable response at fixed weights, with no no-grad teacher call."""
    if horizon == 0:
        return motor_features(model, origin)[None]
    if horizon < 0:
        raise ValueError('Nonnegative quiet response horizon required')
    if getattr(model, 'dan_plastic_lr', 0.) or model.use_read_gamma_trace:
        raise ValueError('Exact branch requires pure COBA/ALIF/STP without extra local updates or Gamma')

    def replay(h, p0, p1, p2, p3, ge, gi, b, x, u, baseline, mean):
        current = FlyPhysicalState(h, (p0,p1,p2,p3), ge, gi, b, x, u, baseline, mean)
        source = torch.zeros_like(h)
        features = [motor_features(model, current)]
        for _ in range(horizon):
            current = step_fly_physical_tick(model, current, None, source,
                baseline, options, base_rates=options['base_rates'])
            features.append(motor_features(model, current))
        return torch.stack(features)

    inputs = (origin.h, *origin.ring, origin.ge, origin.gi, origin.b,
              origin.x, origin.u, origin.baseline, origin.h_mean)
    if checkpoint_branch and torch.is_grad_enabled():
        return checkpoint(replay, *inputs, use_reentrant=False, preserve_rng_state=False)
    return replay(*inputs)


def read_causal_rtc_event(model, physical, token, *, horizon=14,
                          checkpoint_branch=True, options=None):
    options = physical_options(model) if options is None else options
    physical = advance_fly_input_event(model, physical, token, settle_ticks=0,
        writer_baseline_clock='input', **options)
    responses = exact_response_features(model, physical, options, horizon,
                                        checkpoint_branch=checkpoint_branch)
    features, weights = model.causal_read(responses)
    return features, physical, weights


def predict_causal_rtc_next(model, physical, token, *, horizon=14):
    features, physical, weights = read_causal_rtc_event(model, physical, token,
        horizon=horizon, checkpoint_branch=False)
    return model.decoder(model.read_norm(features)), physical, weights


class FlyCausalRTCLearner(FlyBPTTLearner):
    def __init__(self, model, state, *, horizon=14, checkpoint_branch=True, **options):
        if not hasattr(model, 'causal_read') or not 0 <= horizon <= model.causal_read.horizon:
            raise ValueError('Attach matching causal motor readout first')
        if model.topographic_writer is None or model.read_surface != 'output':
            raise ValueError('Sensory-only write and motor-only read required')
        if set(model.topographic_writer.injection_index.tolist()) & set(model.read_indices.tolist()):
            raise ValueError('Physical sensory and motor surfaces must be disjoint')
        if any(getattr(model, n, None) is not None for n in
               ('rtc_student', 'graph_observer', 'latent_predictor', 'streaming_observer')):
            raise ValueError('This exact-response arm has no approximate observer')
        names = list(options.pop('adam_names', None) or (
            'output_read.weight','read_norm.weight','decoder.weight','decoder.bias',
            'log_threshold','log_tau_m','log_beta','log_tau_a',
            'log_tau_s_e','log_tau_s_i','log_g_e','log_g_i',
            'topographic_writer.gate_linear.weight','topographic_writer.gate_linear.bias'))
        names.extend('causal_read.'+name for name,_ in model.causal_read.named_parameters())
        options.update(settle_ticks=0, writer_baseline_clock='input', lambda_jepa=0.)
        super().__init__(model, state, adam_names=names, **options)
        self.horizon = int(horizon)
        self.checkpoint_branch = bool(checkpoint_branch)
        self.speculative_ticks = 0
        self._read_metrics = {}

    def forward_window(self, input_ids, targets):
        if input_ids.shape[0] != 1:
            raise ValueError('One continuous individual required')
        model, physical = self.model, self.state
        options = physical_options(model)
        features, attentions = [], []
        for token in input_ids.unbind(1):
            feature, physical, attention = read_causal_rtc_event(model, physical, token,
                horizon=self.horizon, checkpoint_branch=self.checkpoint_branch, options=options)
            features.append(feature)
            attentions.append(attention.detach())
        features = torch.cat(features)
        scores = F.cross_entropy(model.decoder(model.read_norm(features)), targets.flatten(), reduction='none')
        attention = torch.cat(attentions, dim=1)
        distribution = attention.mean(1)
        self._read_metrics = dict(causal_horizon_mass=distribution.cpu().tolist(),
            causal_mean_horizon=float((distribution*torch.arange(len(distribution),device=distribution.device)).sum()),
            causal_zero_horizon_mass=float(distribution[0]),
            causal_read_entropy=float(-(distribution*distribution.clamp_min(1e-30).log()).sum()))
        self.last_jepa_loss = None
        self.last_jepa_metrics = {}
        return scores, physical, features

    def observe(self, targets):
        scores, metrics = super().observe(targets)
        self.speculative_ticks += self.horizon*len(scores)
        metrics.update(self._read_metrics, causal_speculative_ticks=self.speculative_ticks,
                       causal_horizon=self.horizon, checkpoint_branch=self.checkpoint_branch)
        return scores, metrics

    def state_dict(self):
        result = super().state_dict()
        result.update(learning_route=ROUTE, causal_horizon=self.horizon,
            checkpoint_branch=self.checkpoint_branch, speculative_ticks=self.speculative_ticks)
        return result

    def restore_learning_state(self, saved):
        if (saved.get('learning_route') != ROUTE or saved.get('causal_horizon') != self.horizon
                or saved.get('checkpoint_branch') != self.checkpoint_branch):
            raise ValueError('Exact-response continuation configuration mismatch')
        device = self.state.h.device
        self.state = FlyPhysicalState(**{name: tuple(p.to(device) for p in value)
            if name == 'ring' else value.to(device) for name,value in saved['physical'].items()})
        self.load_edge_signs(saved)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.sgd.load_state_dict(saved['sgd'])
        for name in ('events','updates','physical_ticks','previous_token','ema','speculative_ticks'):
            setattr(self, name, saved[name])
        self.latent_window.copy_(saved['latent_window'].to(device))
