"""Condition the existing utterance and its replay on a finite relation query.

This is the first interface stage, not a claim that the selected vector is an
interpretable question or that hypothetical evidence has been assimilated.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_bfss_dynamics import MomentumBody
from scripts.ib_local.navigation_relation_wood import RelationWoodEngine, local_relation
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction


class QuestionContextBody(MomentumBody):
    def predict_observed(self, ids, state, table_logp):
        logits, next_state, diagnostics, h, read = super().predict_observed(ids, state, table_logp)
        q = getattr(self, '_question_context', None)
        if q is not None:
            logits = logits + self.question_to_logits(q)[None, None, :]
        return logits, next_state, diagnostics, h, read


class QuestionContextEngine(RelationWoodEngine):
    @torch.no_grad()
    def _utter(self):
        # Obtain the decision feature without using any current-turn future.
        self.model._question_context = None
        self.decision.start(self.recurrent)
        _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        context, receipt = local_relation(self.model, h[None], h[None],
            ridge=self.relation_ridge, scale=self.relation_scale)
        if hasattr(self, '_condition_mother_question'):
            context = self._condition_mother_question(h, context)
        self._question_turn_context = context[0].detach().clone()
        self._question_turn_receipt = {k: v[0].detach().clone() for k,v in receipt.items()}
        self.decision.start(self.recurrent)
        before, _ = self.decision.step(self.pending, peek(self.table, self.pending))
        self.model._question_context = self._question_turn_context
        self.decision.start(self.recurrent)
        after, _ = self.decision.step(self.pending, peek(self.table, self.pending))
        self._question_initial_logit_norm = float((after-before).norm())
        self._question_initial_probability_tv = float((after.softmax(-1)-before.softmax(-1)).abs().sum()/2)
        return super()._utter()

    def _advance(self, loss, features, diagnostics, next_state, actor=None, advantage=0.):
        result = super()._advance(loss, features, diagnostics, next_state, actor, advantage)
        # The following held-out observation is not part of the utterance.
        self.model._question_context = None
        return result

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        pending = self.internal_pending
        if (not diag.get('relation_samples') or pending is None or
                'question_slots' not in pending or self._internal_label is None):
            return loss, dict(diag, question_evidence_samples=0)
        s, a, target = self._internal_label
        excluded = torch.tensor([pending['slot']], device=s.device)
        def read(prefix):
            receipt = tuple(pending[prefix + name][None] for name in ('slots', 'stamps', 'valid'))
            return local_relation(self.model, s[None], a[None], exclude=excluded, receipt=receipt,
                                  ridge=self.relation_ridge, scale=self.relation_scale)
        question, packet = read('question_')
        if not bool(packet['valid'].any()):
            return loss, dict(diag, question_evidence_samples=0)
        relation, _ = read('relation_')
        base = next_boundary_prediction(self.model, s[None], a[None])
        if 'momentum_read' in pending:
            base = base + self.model.momentum_to_boundary(pending['momentum_read'].detach()[None])
        label = self.model.encoder(target[None])
        existing = F.mse_loss(base+relation, label)
        asked = F.mse_loss(base+question, label)
        # Preserve one total real-observation term, shared equally between the
        # actual pre-draft and the actual action-time receipts.
        return loss + .5*(asked-existing), dict(diag, question_evidence_samples=1,
                                                question_evidence_loss=float(asked.detach()))

    def step(self, directions, age_ablate=False):
        before = self.model.question_to_logits.weight.detach().clone()
        result = super().step(directions, age_ablate)
        receipt = self._question_turn_receipt
        self.internal_pending['question_context'] = self._question_turn_context.detach().clone()
        self.internal_pending['question_slots'] = receipt['slots'].detach().clone()
        self.internal_pending['question_stamps'] = receipt['stamps'].detach().clone()
        self.internal_pending['question_valid'] = receipt['valid'].detach().clone()
        output = self.model.question_to_logits(self._question_turn_context)
        result['question_context'] = dict(context_norm=float(self._question_turn_context.norm()),
            support=int(receipt['valid'].sum()), slots=receipt['slots'].tolist(),
            stamps=receipt['stamps'].tolist(), post_update_logit_norm=float(output.detach().norm()),
            decision_logit_norm=self._question_initial_logit_norm,
            decision_byte_probability_tv=self._question_initial_probability_tv,
            weight_step=float((self.model.question_to_logits.weight.detach()-before).norm()))
        return result


def configure_question_context(engine):
    if type(engine) not in (RelationWoodEngine, QuestionContextEngine):
        raise TypeError('requires the existing relation-wood individual')
    model = engine.model
    if not hasattr(model, 'question_to_logits'):
        weight = model.encoder.weight
        with torch.random.fork_rng(devices=[weight.device] if weight.is_cuda else []):
            port = nn.Linear(model.encoder.out_features, model.body.base.vocab_size,
                             bias=False, device=weight.device, dtype=weight.dtype)
        nn.init.zeros_(port.weight)
        model.question_to_logits = port
        engine.optimizer.add_param_group(dict(params=list(port.parameters()),
                                              lr=engine.optimizer.param_groups[0]['lr']))
        engine.params = [p for p in model.parameters() if p.requires_grad]
        engine.groups = (*engine.groups[:2], [p for n,p in model.named_parameters()
                                              if not n.startswith('body.')])
    model.__class__ = QuestionContextBody
    engine.__class__ = QuestionContextEngine
    model._question_context = None
    engine.question_context_schema = 1
    return engine
