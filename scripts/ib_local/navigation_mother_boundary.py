"""Bind the observed end of the mother's reply into the same navigation loop."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.navigation_shadow_preview import ShadowPreviewEngine
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.navigation_relation_wood import local_relation


def mother_boundary_feature(features, own, response_len, info):
    """Feature immediately after actual mother bytes, before ambient bytes."""
    count = info.get('mother_bytes', 0) if info.get('source') == 'mother' else 0
    if not isinstance(count, int) or count <= 0 or count >= response_len:
        return None
    index = own + count
    if not 0 <= index < len(features):
        return None
    return features[index].detach().clone()


class MotherBoundaryEngine(ShadowPreviewEngine):
    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        self._mother_pre_base = self._internal_next_latent.detach().clone()
        self._mother_pre_prediction = (self._mother_pre_base +
            self.model.mother_from_boundary(self._mother_pre_base)).detach().clone()
        self._mother_pre_encoder = self.model.encoder.weight.detach().clone()
        return values, inputs

    def _capture_mother_response_boundary(self, features, own, response_len, info):
        self._mother_boundary_target = mother_boundary_feature(features, own, response_len, info)

    @torch.no_grad()
    def _condition_mother_question(self, h, context):
        # Before drafting, predict what silence might elicit using the same
        # shared transition. The correction conditions the existing q port.
        source = h[None]
        p = decision_momentum_read(self.model, self.bfss_momenta, source)
        base = next_boundary_prediction(self.model, source, source)
        base = base + self.model.momentum_to_boundary(p) + context
        correction = self.model.mother_from_boundary(base)
        self._mother_question_correction = correction.detach().clone()
        return context + correction

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        target = getattr(self, '_mother_boundary_target', None)
        if target is None:
            return loss, dict(diag, mother_boundary_samples=0)
        m = self.model.memory
        valid = m.occupied & m.mother_valid & (m.count == 1) & (m.mother_stamp == m.last_write)
        slots = torch.nonzero(valid).flatten()
        if len(slots) > self.replay_batch:
            slots = slots[m.mother_stamp[slots].argsort()[-self.replay_batch:]]
        sources = torch.cat((source[None], m.source[slots]))
        actions = torch.cat((action[None], m.action[slots]))
        targets = torch.cat((target[None], m.mother_next[slots]))
        relation, _ = local_relation(self.model, sources, actions,
            ridge=self.relation_ridge, scale=self.relation_scale)
        base = next_boundary_prediction(self.model, sources, actions) + relation
        p = self.model.momentum_to_boundary(self._momentum_decision_read.detach()[None])
        base = torch.cat((base[:1]+p, base[1:]), 0)
        prediction = base + self.model.mother_from_boundary(base)
        error = F.mse_loss(prediction, self.model.encoder(targets))
        return loss + error, dict(diag, mother_boundary_samples=len(sources),
                                  mother_boundary_loss=float(error.detach()))

    def step(self, directions, age_ablate=False):
        self._mother_boundary_target = None
        before = self.model.mother_from_boundary.weight.detach().clone()
        row = super().step(directions, age_ablate)
        m, slot = self.model.memory, row['episode_slot']
        pre_score = None
        if self._mother_boundary_target is not None:
            option = self._pending_record['choice']
            actual = F.linear(self._mother_boundary_target, self._mother_pre_encoder)
            pre_score = dict(
                original_mse=float((self._mother_pre_base[option]-actual).square().mean()),
                mother_mse=float((self._mother_pre_prediction[option]-actual).square().mean()))
        m.mother_valid[slot] = False
        if self._mother_boundary_target is not None and int(m.count[slot]) == 1:
            m.mother_next[slot].copy_(self._mother_boundary_target)
            m.mother_stamp[slot] = m.last_write[slot]
            m.mother_valid[slot] = True
        row['mother_boundary'] = dict(stored=bool(m.mother_valid[slot]),
            valid_records=int((m.mother_valid & m.occupied & (m.count == 1) &
                               (m.mother_stamp == m.last_write)).sum()),
            weight_step=float((self.model.mother_from_boundary.weight.detach()-before).norm()),
            question_correction_norm=float(self._mother_question_correction.norm()),
            prequential=pre_score)
        return row


def configure_mother_boundary(engine):
    if type(engine) not in (ShadowPreviewEngine, MotherBoundaryEngine):
        raise TypeError('requires the continuing shadow-preview body')
    model, m = engine.model, engine.model.memory
    if not hasattr(model, 'mother_from_boundary'):
        w = model.encoder.weight
        with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
            port = nn.Linear(w.shape[0], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        nn.init.zeros_(port.weight)
        model.mother_from_boundary = port
        engine.optimizer.add_param_group(dict(params=list(port.parameters()),
                                              lr=engine.optimizer.param_groups[0]['lr']))
        engine.params = [p for p in model.parameters() if p.requires_grad]
        engine.groups = (*engine.groups[:2], [p for n,p in model.named_parameters()
                                              if not n.startswith('body.')])
        m.register_buffer('mother_next', torch.zeros_like(m.source))
        m.register_buffer('mother_stamp', torch.zeros_like(m.last_write))
        m.register_buffer('mother_valid', torch.zeros_like(m.occupied))
    engine.__class__ = MotherBoundaryEngine
    engine.mother_boundary_schema = 1
    return engine
