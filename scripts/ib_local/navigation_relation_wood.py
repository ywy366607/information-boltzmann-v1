"""Finite local transition relations from the individual's actual water.

Each query constructs an action-conditioned residual operator from the
currently compatible, version-valid observed boundary pairs. The operator is
not stored per event: the evidence remains in the existing bounded reservoir.
The same encoder and transition participate in the real boundary loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.ib_local.navigation_momentum_wood import MomentumWoodEngine
from scripts.ib_local.navigation_internal_rollout import continuation, next_boundary_prediction
from scripts.ib_local.navigation_direct_read import gain_forecast


def local_relation(model, source, action, *, exclude=None, receipt=None, extra=None,
                   extra_encoded=False, prior_extra=None, ridge=.1, scale=.1,
                   outcome_kind='decision'):
    """Return residual and selected versions; exclude each target's own slot.

    Source/action are raw current-body features. Optional extra is one bounded
    (source, action, observed_successor, valid) evidence tuple per query. It is
    only used in this functional solve; the caller decides whether evidence is
    hypothetical or was actually committed to the persistent reservoir.
    The local linear operator is
    solved in the small dual evidence space, keeping work independent of the
    number of events already experienced. Unsupported queries return zero.
    """
    if source.ndim != 2 or source.shape != action.shape or ridge <= 0:
        raise ValueError('expected matched source/action batches and positive ridge')
    m, enc = model.memory, model.encoder
    if outcome_kind == 'decision':
        eligible = m.occupied & m.internal_valid & (m.count == 1) & (m.internal_stamp == m.last_write)
        target_bank = m.internal_next
    elif outcome_kind == 'mother':
        eligible = m.occupied & m.mother_valid & (m.count == 1) & (m.mother_stamp == m.last_write)
        target_bank = m.mother_next
    else:
        raise ValueError('unknown relation outcome kind')
    qs, qa = enc(source), enc(action)
    bank_s = enc(m.source)
    scores = F.normalize(qs, dim=-1) @ F.normalize(bank_s, dim=-1).T
    scores = scores.masked_fill(~eligible[None], -1e4)
    if exclude is not None:
        if exclude.shape != (len(source),):
            raise ValueError('one excluded versioned slot per query')
        mask = exclude >= 0
        if bool(mask.any()):
            scores = scores.clone()
            scores[torch.arange(len(source), device=source.device)[mask], exclude[mask]] = -1e4
    k = min(m.topk, m.capacity)
    if receipt is None:
        values, slots = scores.topk(k, -1)
        valid = eligible[slots] & (values > -1e3)
    else:
        slots, stamps, observed_valid = receipt
        if slots.shape != (len(source), k) or stamps.shape != slots.shape or observed_valid.shape != slots.shape:
            raise ValueError('invalid decision-time relation receipt')
        values = scores.gather(1, slots)
        valid = observed_valid & eligible[slots] & (values > -1e3) & (m.last_write[slots] == stamps)
    old_s, old_a, old_y = m.source[slots], m.action[slots], target_bank[slots]
    zs, za, zy = enc(old_s), enc(old_a), enc(old_y)
    # A relation binds condition and action difference, in separate roles.
    train_u = torch.cat((zs, za-zs), -1)
    query_u = torch.cat((qs, qa-qs), -1)
    train_u = F.normalize(train_u, dim=-1)*valid[..., None]
    query_u = F.normalize(query_u, dim=-1)
    base_old = model.long_predictor(torch.cat((zs, za), -1))
    if outcome_kind == 'mother':
        base_old = base_old + model.mother_from_boundary(base_old)
    residual = (zy-base_old)*valid[..., None]
    for evidence, encoded in ((prior_extra, False), (extra, extra_encoded)):
        if evidence is None:
            continue
        extra_s, extra_a, extra_y, extra_valid = evidence
        y_shape = (len(source), model.encoder.out_features) if encoded else source.shape
        if (extra_s.shape != source.shape or extra_a.shape != source.shape or
                extra_y.shape != y_shape or extra_valid.shape != (len(source),)):
            raise ValueError('one correctly shaped extra evidence item per query required')
        es, ea, ey = enc(extra_s), enc(extra_a), (extra_y if encoded else enc(extra_y))
        extra_u = F.normalize(torch.cat((es, ea-es), -1), dim=-1)
        extra_residual = ey-model.long_predictor(torch.cat((es, ea), -1))
        if outcome_kind == 'mother':
            base_extra = model.long_predictor(torch.cat((es, ea), -1))
            extra_residual = ey - (base_extra + model.mother_from_boundary(base_extra))
        train_u = torch.cat((train_u, (extra_u*extra_valid[:, None])[:, None]), 1)
        residual = torch.cat((residual, (extra_residual*extra_valid[:, None])[:, None]), 1)
    kernel = train_u @ train_u.transpose(-1, -2)
    eye = torch.eye(train_u.shape[1], device=source.device, dtype=source.dtype)
    query_kernel = (train_u @ query_u[..., None])
    coefficients = torch.linalg.solve(kernel + ridge*eye, query_kernel).squeeze(-1)
    prediction = scale*(coefficients[..., None]*residual).sum(-2)
    packet = dict(slots=slots, stamps=m.last_write[slots].detach().clone(),
                  valid=valid, coefficients=coefficients[..., :k])
    if prior_extra is not None:
        packet['prior_extra_coefficient'] = coefficients[..., k:k+1]
    if extra is not None:
        packet['extra_coefficient'] = coefficients[..., k+int(prior_extra is not None):]
    return prediction, packet


class RelationWoodEngine(MomentumWoodEngine):
    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        self._relation_without_offset = self._internal_offset.detach().clone()
        actions = self._internal_action
        source = h.expand_as(actions)
        p_delta = self.model.momentum_to_boundary(self._momentum_decision_read[None])
        relation_delta, packet_r = local_relation(self.model, source, actions,
            ridge=self.relation_ridge, scale=self.relation_scale)
        latent = self._internal_next_latent + relation_delta
        current = self.model.encoder(source)
        valid = self.reward_label_valid(wood=True)
        future, packet = continuation(self.model, latent, reward_valid=valid)
        blind, blind_packet = continuation(self.model, current, reward_valid=valid)
        wrong, _ = continuation(self.model, latent, reward_valid=valid, shuffle=True)
        baseline = self._gain_forecast['mean'] + self.gamma*blind.cpu().numpy()
        delta = self.gamma*(future-blind).cpu().numpy()
        eligible = (packet['support'] & blind_packet['support']).cpu().numpy() & (delta != 0)
        self._internal_forecast = gain_forecast(self.internal_gain_state, baseline, delta, eligible)
        self._internal_next_latent = latent.detach().clone()
        self._internal_packet.update(next_latent=latent.detach().clone(), future=future.cpu().tolist(),
            blind=blind.cpu().tolist(), shuffled=wrong.cpu().tolist(),
            support=packet['support'].cpu().tolist(), slots=packet['slots'].cpu().tolist(),
            weights=packet['weights'].cpu().tolist())
        self._internal_offset = values.new_tensor(self._internal_forecast['offset'])
        # For the P ablation, hold the new relation fixed and remove only P.
        without_p, without_packet = continuation(self.model, latent-p_delta, reward_valid=valid)
        without_delta = self.gamma*(without_p-blind).cpu().numpy()
        without_eligible = (without_packet['support'] & blind_packet['support']).cpu().numpy() & (without_delta != 0)
        old = gain_forecast(self.internal_gain_state, baseline, without_delta, without_eligible)
        self._momentum_without_offset = values.new_tensor(old['offset'])
        self._relation_prediction_delta = float(relation_delta.norm())
        self._relation_decision_receipt = {k: v.detach().clone() for k,v in packet_r.items()}
        return values, inputs

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        if self._internal_label is None or self.internal_pending is None:
            return loss, dict(diag, relation_samples=0)
        s, a, target = self._internal_label
        excluded = torch.tensor([self.internal_pending['slot']], device=s.device)
        if 'relation_slots' not in self.internal_pending:
            return loss, dict(diag, relation_samples=0)
        receipt = (self.internal_pending['relation_slots'][None],
                   self.internal_pending['relation_stamps'][None],
                   self.internal_pending['relation_valid'][None])
        correction, packet = local_relation(self.model, s[None], a[None], exclude=excluded,
            receipt=receipt, ridge=self.relation_ridge, scale=self.relation_scale)
        if not bool(packet['valid'].any()):
            return loss, dict(diag, relation_samples=0)
        prediction = next_boundary_prediction(self.model, s[None], a[None])
        label = self.model.encoder(target[None])
        # Same P binding used at the actual historical decision, when available.
        if 'momentum_read' in self.internal_pending:
            prediction = prediction + self.model.momentum_to_boundary(
                self.internal_pending['momentum_read'].detach()[None])
            # Parent already added this P-conditional example. Replace that
            # term with the joint P+relation prediction, rather than charging
            # the same new observation twice. The inherited replay loss stays.
            loss = loss - F.mse_loss(prediction, label)
        error = F.mse_loss(prediction + correction, label)
        return loss + error, dict(diag, relation_samples=1,
                                  relation_boundary_loss=float(error.detach()))

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        receipt = self._relation_decision_receipt
        choice = self._pending_record['choice']
        self.internal_pending['relation_slots'] = receipt['slots'][choice].detach().clone()
        self.internal_pending['relation_stamps'] = receipt['stamps'][choice].detach().clone()
        self.internal_pending['relation_valid'] = receipt['valid'][choice].detach().clone()
        logits = self._internal_delivered_offset.new_tensor(self._pending_record['choice_logits'])
        available = logits > -1e3
        without = self._relation_without_offset.clone()
        without -= without[available].mean()
        without = without.masked_fill(~available, 0.) if self.internal_mode == 'real' else torch.zeros_like(without)
        without_relation = logits-self._internal_delivered_offset+without
        result['relation_wood'] = dict(prediction_delta=self._relation_prediction_delta,
            support=receipt['valid'].sum(-1).tolist(),
            slots=receipt['slots'].tolist(), stamps=receipt['stamps'].tolist(),
            choice_probability_tv_without_relation=float(
                (logits.softmax(-1)-without_relation.softmax(-1)).abs().sum()/2))
        return result


def configure_relation_wood(engine, *, ridge=.1, scale=.1):
    if type(engine) not in (MomentumWoodEngine, RelationWoodEngine):
        raise TypeError('requires the trainable momentum-wood individual')
    if ridge <= 0 or not 0 <= scale <= 1:
        raise ValueError('invalid relation operator configuration')
    engine.__class__ = RelationWoodEngine
    engine.relation_ridge, engine.relation_scale = float(ridge), float(scale)
    engine.relation_wood_schema = 1
    return engine
