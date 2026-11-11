"""Bind decision-time P to the existing predicted/observed next boundary.

This is an explicit, optional migration of the continuing S/P individual. P is
read with each layer's existing query projection, using the decision feature as
the common query source. The layer-specific coordinate alignment remains a
learned empirical hypothesis, not a presumed biological or BFSS identity.
The saved receipt contains only the finite action-time reads, not past S/P or
a historical computation graph.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.navigation_bfss_dynamics import MomentumEngine
from scripts.ib_local.navigation_internal_rollout import continuation, next_boundary_prediction
from scripts.ib_local.navigation_direct_read import gain_forecast


def decision_momentum_read(model, momenta, source):
    """[batch, width] -> [batch, layers*width], P^T q per layer/head."""
    if source.ndim != 2:
        raise ValueError('decision query must be a batch of body features')
    result = []
    for layer, p in zip(model.body.base.layers, momenta):
        q = layer.q_proj(layer.norm(source)).reshape(-1, layer.heads, layer.dk)
        q = F.normalize(q, dim=-1)
        if p.shape[0] != 1:
            raise ValueError('one continuous individual is required')
        values = torch.einsum('bhk,hkv->bhv', q, p[0])
        result.append(values.flatten(1))
    return torch.cat(result, -1)


class MomentumWoodEngine(MomentumEngine):
    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        self._momentum_without_offset = self._internal_offset.detach().clone()
        actions = self._internal_action
        source = h.expand_as(actions)
        p_read = decision_momentum_read(self.model, self.bfss_momenta, h)
        self._momentum_decision_read = p_read[0].detach().clone()
        delta_z = self.model.momentum_to_boundary(p_read).expand(len(actions), -1)
        original = self._internal_next_latent
        latent = original + delta_z
        current = self.model.encoder(source)
        reward_valid = self.reward_label_valid(wood=True)
        future, packet = continuation(self.model, latent, reward_valid=reward_valid)
        blind, blind_packet = continuation(self.model, current, reward_valid=reward_valid)
        wrong, _ = continuation(self.model, latent, reward_valid=reward_valid, shuffle=True)
        immediate = self._gain_forecast['mean']
        baseline = immediate + self.gamma*blind.cpu().numpy()
        delta = self.gamma*(future-blind).cpu().numpy()
        eligible = (packet['support'] & blind_packet['support']).cpu().numpy() & (delta != 0)
        forecast = gain_forecast(self.internal_gain_state, baseline, delta, eligible)
        self._internal_forecast = forecast
        self._internal_next_latent = latent.detach().clone()
        self._internal_packet.update(next_latent=latent.detach().clone(),
            future=future.cpu().tolist(), blind=blind.cpu().tolist(),
            shuffled=wrong.cpu().tolist(), support=packet['support'].cpu().tolist(),
            slots=packet['slots'].cpu().tolist(), weights=packet['weights'].cpu().tolist())
        self._internal_offset = values.new_tensor(forecast['offset'])
        self._momentum_prediction_delta = float(delta_z.norm())
        return values, inputs

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        pending = self.internal_pending
        # The pending receipt belongs to the actual earlier decision; a
        # recycled or merged episode is rejected by _capture_boundary.
        if self._internal_label is None or pending is None or 'momentum_read' not in pending:
            return loss, dict(diag, momentum_boundary_samples=0)
        old_source, old_action, target = self._internal_label
        base = next_boundary_prediction(self.model, old_source[None], old_action[None])
        p = pending['momentum_read'].detach()[None]
        prediction = base + self.model.momentum_to_boundary(p)
        error = F.mse_loss(prediction, self.model.encoder(target[None]))
        return loss + error, dict(diag, momentum_boundary_samples=1,
                                  momentum_boundary_loss=float(error.detach()))

    def step(self, directions, age_ablate=False):
        before = self.model.momentum_to_boundary.weight.detach().clone()
        result = super().step(directions, age_ablate)
        self.internal_pending['momentum_read'] = self._momentum_decision_read.detach().clone()
        logits = self._internal_delivered_offset.new_tensor(self._pending_record['choice_logits'])
        available = logits > -1e3
        without = self._momentum_without_offset.clone()
        without -= without[available].mean()
        without = without.masked_fill(~available, 0.) if self.internal_mode == 'real' else torch.zeros_like(without)
        with_offset = self._internal_delivered_offset
        without_p = logits - with_offset + without
        result['momentum_wood'] = dict(
            prediction_delta=self._momentum_prediction_delta,
            read_norm=float(self._momentum_decision_read.norm()),
            weight_norm=float(self.model.momentum_to_boundary.weight.detach().norm()),
            weight_step=float((self.model.momentum_to_boundary.weight.detach()-before).norm()),
            choice_probability_tv_without_p=float(
                (logits.softmax(-1)-without_p.softmax(-1)).abs().sum()/2))
        return result


def configure_momentum_wood(engine):
    """Add one zero-output trainable port, preserving all existing parameters."""
    if type(engine) not in (MomentumEngine, MomentumWoodEngine):
        raise TypeError('requires the S/P continuation individual')
    model = engine.model
    if not hasattr(model, 'momentum_to_boundary'):
        width = model.body.base.d
        latent = model.encoder.out_features
        layers = len(model.body.base.layers)
        weight = model.encoder.weight
        # Do not change the individual's actor/teacher random streams.
        with torch.random.fork_rng(devices=[weight.device] if weight.is_cuda else []):
            port = nn.Linear(layers*width, latent, bias=False,
                             device=weight.device, dtype=weight.dtype)
        nn.init.zeros_(port.weight)
        model.momentum_to_boundary = port
        engine.optimizer.add_param_group(dict(params=list(port.parameters()),
                                              lr=engine.optimizer.param_groups[0]['lr']))
        engine.params = [p for p in model.parameters() if p.requires_grad]
        engine.groups = (*engine.groups[:2], [p for n,p in model.named_parameters()
                                              if not n.startswith('body.')])
    engine.__class__ = MomentumWoodEngine
    engine.momentum_wood_schema = 1
    return engine
