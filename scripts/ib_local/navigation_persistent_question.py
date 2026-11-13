"""Persist versioned pre-draft relation receipts in the existing episode bank.

The first candidate keeps the *outcome* role open.  An episode address is a
provenance handle, not a claim that two similar contexts ask the same question.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.navigation_mother_boundary import MotherBoundaryEngine
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction


def retained_receipt(model, source):
    """Recall one versioned question context; never use invalidated slots."""
    m = model.memory
    eligible = (m.occupied & m.question_valid & (m.count == 1) &
                (m.question_stamp == m.last_write))
    if not bool(eligible.any()):
        return None
    read = m.read_encoded(model.encoder(source), model.encoder, eligible=eligible)
    # Select by the individual's existing read, not an external label.
    anchor = read.slots[0, read.weights[0].argmax()]
    if not bool(eligible[anchor]):
        return None
    return dict(anchor=int(anchor), stamp=int(m.last_write[anchor]),
                slots=m.question_slots[anchor:anchor+1].clone(),
                stamps=m.question_stamps[anchor:anchor+1].clone(),
                valid=m.question_support[anchor:anchor+1].clone())


def actual_boundary_evidence(engine, packet, batch=1):
    """The anchor's observed next boundary, only while its version survives."""
    if packet is None:
        return None
    m, i = engine.model.memory, packet['anchor']
    if not (bool(m.occupied[i]) and int(m.count[i]) == 1 and
            bool(m.question_valid[i]) and int(m.question_stamp[i]) == packet['stamp'] and
            bool(m.internal_valid[i]) and int(m.internal_stamp[i]) == packet['stamp'] and
            int(m.last_write[i]) == packet['stamp']):
        return None
    # Do not count the same episode both inside the original receipt and as
    # a new boundary observation.
    if bool(((packet['slots'] == i) & packet['valid'] &
             (packet['stamps'] == packet['stamp'])).any()):
        return None
    return (m.source[i:i+1].expand(batch, -1), m.action[i:i+1].expand(batch, -1),
            m.internal_next[i:i+1].expand(batch, -1),
            torch.ones(batch, dtype=torch.bool, device=m.source.device))


def mother_response_evidence(engine, packet, batch=1):
    """Return the version-matched mother-end observation, before ambient input.

    This typed evidence is not interchangeable with the decision-boundary
    target used by ``actual_boundary_evidence``. The caller must use a
    mother-boundary predictor; this function does not change the live policy.
    """
    if packet is None:
        return None
    m, i = engine.model.memory, packet['anchor']
    if not (bool(m.occupied[i]) and int(m.count[i]) == 1 and
            bool(m.question_valid[i]) and int(m.question_stamp[i]) == packet['stamp'] and
            bool(m.mother_valid[i]) and int(m.mother_stamp[i]) == packet['stamp'] and
            int(m.last_write[i]) == packet['stamp']):
        return None
    return (m.source[i:i+1].expand(batch, -1), m.action[i:i+1].expand(batch, -1),
            m.mother_next[i:i+1].expand(batch, -1),
            torch.ones(batch, dtype=torch.bool, device=m.source.device))


def carry_question_parent(engine, receipt, parent, child_slot):
    """Keep the actually reactivated mother observation in the next receipt.

    The existing fixed-size receipt remains the only allocation. A parent is
    evidence, not a semantic identity or an imagined observation. Reject a
    replaced/merged parent and never point the new record at itself.
    """
    result = {key: receipt[key].detach().clone() for key in ('slots', 'stamps', 'valid')}
    if (not getattr(engine, 'question_parent_carry', False) or
            not getattr(engine, 'joint_question_update', False) or
            not getattr(engine, 'mother_shadow_enabled', False) or
            parent is None or parent['anchor'] == child_slot or
            mother_response_evidence(engine, parent) is None):
        return result, False
    slot, stamp = parent['anchor'], parent['stamp']
    same = result['valid'] & (result['slots'] == slot) & (result['stamps'] == stamp)
    if bool(same.any()):
        return result, True
    # Invalid entries are free. Otherwise replace the lowest-ranked support;
    # local_relation returns the original nearest-neighbour order.
    free = torch.nonzero(~result['valid']).flatten()
    index = int(free[0]) if len(free) else len(result['valid']) - 1
    result['slots'][index] = slot
    result['stamps'][index] = stamp
    result['valid'][index] = True
    return result, True


def relation_from_receipt(engine, source, action, packet, *, outcome_kind='decision'):
    if packet is None:
        return source.new_zeros((len(source), engine.model.encoder.out_features)), False
    m = engine.model.memory
    anchor = packet['anchor']
    anchor_ok = (bool(m.occupied[anchor]) and int(m.count[anchor]) == 1 and
                 bool(m.question_valid[anchor]) and
                 int(m.last_write[anchor]) == packet['stamp'] and
                 int(m.question_stamp[anchor]) == packet['stamp'])
    if not anchor_ok:
        return source.new_zeros((len(source), engine.model.encoder.out_features)), False
    receipt = tuple(packet[k] for k in ('slots', 'stamps', 'valid'))
    evidence = mother_response_evidence if outcome_kind == 'mother' else actual_boundary_evidence
    actual = (evidence(engine, packet, len(source))
              if getattr(engine, 'joint_question_update', False) else None)
    relation, used = local_relation(engine.model, source, action, receipt=receipt,
                                    prior_extra=actual,
                                    ridge=engine.relation_ridge, scale=engine.relation_scale,
                                    outcome_kind=outcome_kind)
    return relation, bool(used['valid'].any()) or actual is not None


class PersistentQuestionEngine(MotherBoundaryEngine):
    def _capture_boundary(self):
        if getattr(self, '_question_boundary_captured', False):
            return
        return super()._capture_boundary()

    def _utter(self):
        if not getattr(self, 'align_question_boundary', False):
            return super()._utter()
        # The parent captures inside _utter, after the question context has
        # already been formed. Capture once before both draft and shadow.
        self._capture_boundary()
        self._question_boundary_captured = True
        try:
            return super()._utter()
        finally:
            self._question_boundary_captured = False

    def _shadow_relation_extra(self, batch):
        if not getattr(self, 'joint_question_update', False):
            return None
        return actual_boundary_evidence(self, getattr(self, '_retained_question_receipt', None), batch)

    def _shadow_relation_receipt(self, batch):
        packet = getattr(self, '_retained_question_receipt', None)
        if packet is None or not getattr(self, '_retained_question_used', False):
            return None
        return tuple(packet[k].expand(batch, -1) for k in ('slots', 'stamps', 'valid'))

    @torch.no_grad()
    def _condition_mother_question(self, h, context):
        context = super()._condition_mother_question(h, context)
        packet = retained_receipt(self.model, h[None])
        kind = 'mother' if getattr(self, 'mother_shadow_enabled', False) else 'decision'
        relation, used = relation_from_receipt(self, h[None], h[None], packet, outcome_kind=kind)
        self._retained_question_receipt = packet
        self._retained_question_used = used
        evidence = mother_response_evidence if kind == 'mother' else actual_boundary_evidence
        self._retained_actual_boundary_used = evidence(self, packet) is not None
        self._retained_question_delta = self.model.question_reactivation(relation)
        return context + self._retained_question_delta

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        target = getattr(self, '_mother_boundary_target', None)
        packet = getattr(self, '_retained_question_receipt', None)
        if target is None or packet is None:
            return loss, dict(diag, retained_question_samples=0)
        kind = 'mother' if getattr(self, 'mother_shadow_enabled', False) else 'decision'
        relation, used = relation_from_receipt(self, source[None], source[None], packet, outcome_kind=kind)
        if not used:
            return loss, dict(diag, retained_question_samples=0)
        # The same pre-draft relation affects the shared mother-boundary
        # predictor; the current actual outcome is the only training target.
        base = next_boundary_prediction(self.model, source[None], action[None])
        base = base + self.model.momentum_to_boundary(self._momentum_decision_read.detach()[None])
        prediction = base + self.model.mother_from_boundary(base) + self.model.question_reactivation(relation)
        error = F.mse_loss(prediction, self.model.encoder(target[None]))
        return loss + error, dict(diag, retained_question_samples=1,
                                  retained_question_loss=float(error.detach()))

    def step(self, directions, age_ablate=False):
        row = super().step(directions, age_ablate)
        m, slot = self.model.memory, row['episode_slot']
        receipt = self._question_turn_receipt
        receipt, parent_carried = carry_question_parent(
            self, receipt, self._retained_question_receipt, slot)
        m.question_valid[slot] = False
        if int(m.count[slot]) == 1:
            m.question_slots[slot].copy_(receipt['slots'])
            m.question_stamps[slot].copy_(receipt['stamps'])
            m.question_support[slot].copy_(receipt['valid'])
            m.question_stamp[slot] = m.last_write[slot]
            m.question_valid[slot] = True
        packet = self._retained_question_receipt
        row['retained_question'] = dict(stored=bool(m.question_valid[slot]),
            anchor=None if packet is None else packet['anchor'],
            anchor_stamp=None if packet is None else packet['stamp'],
            used=self._retained_question_used,
            parent_evidence_carried=parent_carried and bool(m.question_valid[slot]),
            correction_norm=float(self._retained_question_delta.norm()),
            actual_boundary_used=self._retained_actual_boundary_used)
        return row


def configure_persistent_question(engine):
    if type(engine) not in (MotherBoundaryEngine, PersistentQuestionEngine):
        raise TypeError('requires the continuing mother-boundary body')
    model, m = engine.model, engine.model.memory
    if not hasattr(model, 'question_reactivation'):
        w = model.encoder.weight
        with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
            port = nn.Linear(w.shape[0], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        nn.init.zeros_(port.weight)
        model.question_reactivation = port
        engine.optimizer.add_param_group(dict(params=list(port.parameters()),
                                              lr=engine.optimizer.param_groups[0]['lr']))
        engine.params = [p for p in model.parameters() if p.requires_grad]
        engine.groups = (*engine.groups[:2], [p for n,p in model.named_parameters()
                                              if not n.startswith('body.')])
        shape = (m.capacity, min(m.topk, m.capacity))
        m.register_buffer('question_slots', torch.zeros(shape, dtype=torch.long, device=w.device))
        m.register_buffer('question_stamps', torch.zeros(shape, dtype=torch.long, device=w.device))
        m.register_buffer('question_support', torch.zeros(shape, dtype=torch.bool, device=w.device))
        m.register_buffer('question_stamp', torch.zeros_like(m.last_write))
        m.register_buffer('question_valid', torch.zeros_like(m.occupied))
    engine.__class__ = PersistentQuestionEngine
    return engine
