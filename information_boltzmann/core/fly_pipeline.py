"""One-tick pre-observation prediction on a delayed sensory/motor substrate.

Current observations enter only after the motor feature has been sealed. Both
phases share one old-state snapshot and one delayed-arrival computation.
One tick per input is the throughput, not the maximum propagation latency:
multi-hop pathways continue through the complete persistent physical state.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .fly_reservoir import SpikeFn
from .fly_bptt_learning import FlyPhysicalState, FlyBPTTLearner


PROTOCOL = 'motor-pre-observation-one-tick-v1'
# Numerical acceptance coverage requested by the user. This is neither a
# waiting time nor a physical cutoff; recurrent paths may last longer.
VERIFIED_PATH_HORIZON = 14


@dataclass
class PendingFlyTick:
    old_state: FlyPhysicalState
    context: dict
    feature: torch.Tensor
    h_mean_decay: float
    committed: bool = False


def _check_pipeline_model(model):
    if model.synapse_model != 'coba' or model.topographic_writer is None:
        raise ValueError('Pipeline requires COBA and the functional topographic writer')
    if model.read_surface != 'output' or model.n_read == 0:
        raise ValueError('Pipeline reads the nonempty biological output surface')
    if getattr(model, 'use_read_gamma_trace', False):
        raise ValueError('This protocol uses the native motor state, without Gamma')
    if getattr(model, 'latent_predictor', None) is not None:
        raise ValueError('Motor-only pipeline excludes a token-conditioned decoder predictor')
    if getattr(model, 'dan_plastic_lr', 0.0) > 0.0:
        raise ValueError('Separate DAN learning is outside this first pipeline protocol')
    sensory = set(model.topographic_writer.injection_index.detach().cpu().tolist())
    motor = set(model.read_indices.detach().cpu().tolist())
    if sensory & motor:
        raise ValueError('Sensory injection and motor read surfaces must be disjoint')
    if len(model.splits_e) != 5 or len(model.splits_i) != 5:
        raise ValueError('Each chemical edge requires an old-ring delay 1..4; '
                         'total multi-hop latency is not limited to four ticks')


def begin_fly_prediction(model, state, *, decode=True, h_mean_decay=0.99,
                         base_rates=None, thresholds=None, conductance_gains=None,
                         alif_params=None, stp_params=None):
    """Issue a prediction without accepting the current observation.

    For windowed training, ``decode=False`` seals the feature and batches the
    pointwise decoder at unchanged weights. In generation, return actual logits
    before sampling/receiving the current token. Setup validation is performed
    by the learner; direct callers should call ``validate_pipeline_model`` once.
    """
    if not 0.0 <= h_mean_decay < 1.0:
        raise ValueError('h_mean_decay must be in [0, 1)')
    c = model.prepare_coba_tick(
        state.h, state.ring, state.ge, state.gi, state.b, state.x, state.u,
        base_rates=base_rates, thresholds=thresholds,
        conductance_gains=conductance_gains, alif_params=alif_params,
        stp_params=stp_params)
    idx = model.read_indices
    v = c['alpha'][:, idx] * state.h[:, idx] + c['beta_int'][:, idx] * c['base_current'][:, idx]
    threshold = c['eff_threshold']
    if torch.is_tensor(threshold) and threshold.numel() > 1:
        threshold = threshold[:, idx]
    motor = v * (1.0 - SpikeFn.apply(v - threshold))
    if model.read_centering:
        mean = state.h_mean[:, idx] if state.h_mean.numel() else torch.zeros_like(motor)
        mean = h_mean_decay * mean + (1.0 - h_mean_decay) * motor
        motor = motor - mean
    feature = model.output_read(motor)
    pending = PendingFlyTick(state, c, feature, h_mean_decay)
    result = model.decoder(model.read_norm(feature)) if decode else feature
    return result, pending


def commit_fly_observation(model, pending, token):
    """Observe the token and atomically complete the same physical tick once."""
    if pending.committed:
        raise RuntimeError('A pending physical tick may be committed only once')
    pending.committed = True
    old = pending.old_state
    drive, baseline = model.topographic_writer.forward_with_state(
        model.embedding(token), old.h, old.baseline)
    ret = model.finish_coba_tick(pending.context, drive)
    h, _, ring, ge, gi = ret[:5]
    offset = 5
    b, x, u = old.b, old.x, old.u
    if model.use_alif:
        b = ret[offset]
        offset += 1
    if model.use_stp:
        x, u = ret[offset:offset + 2]
    mean = old.h_mean if old.h_mean.numel() else torch.zeros_like(h)
    mean = pending.h_mean_decay * mean + (1.0 - pending.h_mean_decay) * h
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline, mean,
                            old.dan_gate, old.gamma_z1, old.gamma_z2)


validate_pipeline_model = _check_pipeline_model


class FlyPipelineLearner(FlyBPTTLearner):
    """Joint BPTT with one physical tick per scored/assimilated observation.

    Detachment occurs at optimizer windows only. Inputs and targets refer to the
    same event, with the target entering the body after its feature is sealed.
    Every recurrent state field survives that boundary. Gradient credit survives
    only within the current unrolled window, including multi-hop paths of 14
    ticks when both their source and loss lie inside that window.
    """
    def __init__(self, model, state, **kwargs):
        _check_pipeline_model(model)
        if kwargs.get('settle_ticks', 0) != 0:
            raise ValueError('The pipeline advances exactly one tick per observation')
        if kwargs.get('writer_baseline_clock', 'input') != 'input':
            raise ValueError('The writer baseline follows observed input events')
        super().__init__(model, state, **kwargs)

    def inputs_for_targets(self, tokens):
        # Every token is consumed once, including the first and window bridges.
        return tokens[None]

    def forward_window(self, input_ids, targets):
        m, state = self.model, self.state
        options = dict(base_rates=m.get_decay_rates(), thresholds=m.get_thresholds(),
                       conductance_gains=m.get_conductance_gains(),
                       alif_params=m.get_alif_params(), stp_params=m.get_stp_params())
        latents = []
        for token in input_ids.unbind(1):
            feature, pending = begin_fly_prediction(m, state, decode=False, **options)
            latents.append(feature)
            state = commit_fly_observation(m, pending, token)
        features = torch.cat(latents, dim=0)
        # This pointwise head has no sequence mixing or target-dependent stats.
        logits = m.decoder(m.read_norm(features))
        scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
        return scores, state, features

    def state_dict(self):
        result = super().state_dict()
        result.update(prediction_protocol=PROTOCOL, pipeline_version=1)
        return result
