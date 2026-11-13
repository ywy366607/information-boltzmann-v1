"""The shared relation program inside the continuing mother-world body.

Binding, typed composition and evidence revision run on the real navigation
path. The original language/actor learning, S/P dynamics, exact byte table,
actual gains and caregiver remain in the same individual. No separate F or
question reward is introduced.
"""
from __future__ import annotations

import copy
import math

import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import (
    Evidence, Port, RelationBank, compose, imagine, log_score, mechanism,
    hypothesis_evidence, real_evidence, revise)
from scripts.ib_local.navigation_persistent_question import PersistentQuestionEngine
from scripts.ib_local.navigation_internal_rollout import InternalRolloutEngine, continuation
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.navigation_direct_read import gain_forecast, gain_state


def frozen_forecast(distribution, encoder_weight):
    return dict(mean=distribution['mean'].detach().clone(),
                scale2=distribution['scale2'].detach().clone(), df=distribution['df'],
                encoder_weight=encoder_weight.detach().clone())


def score_frozen(forecast, actual):
    y = F.linear(actual.reshape(1, -1), forecast['encoder_weight'])
    distribution = {k: forecast[k] for k in ('mean', 'scale2', 'df')}
    return dict(log_score=float(log_score(distribution, y).mean()),
                mse=float((distribution['mean']-y).square().mean()))


class CausalProgramEngine(PersistentQuestionEngine):
    @torch.no_grad()
    def _condition_mother_question(self, h, context):
        inherited = super()._condition_mother_question(h, context)
        bank, m = self.model.causal_relations, self.model.memory
        state = bank.bind(self.model, h[None])
        self._program_question_state = state
        slots = real_evidence(self.model, state, Port.MOTHER)
        # Actions are the body's own remembered action changes. The source is
        # observed now, before drafting; no world location or hidden answer.
        prior_actions = h[None] + m.action[slots] - m.source[slots]
        candidates = torch.cat((h[None], prior_actions), 0)[:m.topk]
        source = h.expand_as(candidates)
        p = decision_momentum_read(self.model, self.bfss_momenta, h[None]).expand(len(candidates), -1)
        forecast = imagine(self.model, state, source, candidates, ridge=self.relation_ridge,
            reward_valid=self.reward_label_valid(wood=True), momentum=p)
        # A question carries an open mother-outcome role and a proposed binding.
        # Existing continuation chooses a useful binding; information is NOT a
        # new reward. The actual utterance/action is chosen later by the actor.
        choice = int(forecast['expected_after_value'].argmax())
        distribution = forecast['path']['mother']
        relation = distribution['mean'][choice] - distribution['base'][choice]
        self._program_question_vector = relation.detach().clone()
        self._program_question_action = candidates[choice].detach().clone()
        self._program_question_roles = dict(open=[Port.MOTHER.value],
            bound=[Port.CONDITION.value], proposed=[Port.ACTION.value],
            subject='continuing_body', receiver='mother',
            unknown_identity=f'{state.identity}:mother',
            identity=state.identity, generation=state.generation, version=state.version)
        self._program_pre_question = dict(
            candidate_count=len(candidates), selected=choice,
            conditional_information=forecast['information'].cpu().tolist(),
            expected_after_value=forecast['expected_after_value'].cpu().tolist())
        return inherited + self.model.question_reactivation(relation[None])

    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        state = self._program_question_state
        actions = self._internal_action
        source = h.expand_as(actions)
        momentum = self._momentum_decision_read[None].expand(len(actions), -1)
        planned = imagine(self.model, state, source, actions,
            ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
            momentum=momentum)
        path = planned['path']
        blind, blind_packet = continuation(self.model, self.model.encoder(source),
                                           reward_valid=self.reward_label_valid(wood=True))
        immediate = self._gain_forecast['mean']
        baseline = immediate + self.gamma*blind.cpu().numpy()
        delta = self.gamma*(path['value']-blind).cpu().numpy()
        eligible = (path['continuation']['support'] & blind_packet['support']).cpu().numpy() & (delta != 0)
        self._internal_forecast = gain_forecast(self.internal_gain_state, baseline, delta, eligible)
        self._internal_offset = values.new_tensor(self._internal_forecast['offset'])
        self._internal_next_latent = path['decision']['mean'].detach().clone()
        self._internal_packet.update(next_latent=self._internal_next_latent,
            future=path['value'].cpu().tolist(), blind=blind.cpu().tolist(),
            support=path['continuation']['support'].cpu().tolist(),
            slots=path['continuation']['slots'].cpu().tolist(),
            weights=path['continuation']['weights'].cpu().tolist())
        # Conditional epistemic revision is judged by the same future-use
        # settlement as the inherited actor, with new calibration provenance.
        baseline = immediate + self.gamma*path['value'].cpu().numpy()
        delta = self.gamma*planned['future_delta'].cpu().numpy()
        supported = path['continuation']['support'].cpu().numpy() & (delta != 0)
        self._shadow_forecast = gain_forecast(self.shadow_gain_state, baseline, delta, supported)
        self._shadow_offset = values.new_tensor(self._shadow_forecast['offset'])
        self._mother_pre_base = path['mother']['base'].detach().clone()
        self._mother_pre_prediction = path['mother']['mean'].detach().clone()
        self._mother_pre_encoder = self.model.encoder.weight.detach().clone()
        self._program_forecasts = planned
        self._program_mother_forecast = frozen_forecast(path['mother'], self._mother_pre_encoder)
        self._program_decision_forecast = frozen_forecast(path['decision'], self._mother_pre_encoder)
        # Keep the inherited save/settlement interface, but identify these as
        # conditional operator hypotheses rather than residual neighbours.
        n, count = len(actions), planned['hypotheses'].shape[1]
        self._shadow_packet = dict(
            slots=path['continuation']['slots'][:, :count],
            stamps=self.model.memory.last_write[path['continuation']['slots'][:, :count]].clone(),
            valid=path['continuation']['support'][:, None].expand(n, count),
            weights=actions.new_full((n,count), 1/count),
            shadow_state_shift=planned['shadow_state_shift'][:, None].expand(n,count),
            future_delta=planned['future_delta'], usable=path['continuation']['support'],
            candidate_latents=planned['hypotheses'], base_latent=path['mother']['mean'],
            bound_receipt_used=torch.tensor(True, device=h.device),
            actual_evidence_used=torch.tensor(bool(len(path['mother']['slots'])), device=h.device))
        return values, inputs

    def _outcome_loss(self, source, action, long):
        # Retain the existing language-world outcome and real next-boundary
        # training. Replace the old extra mother/question/ridge losses with
        # the SAME typed program that was actually used to choose and speak.
        loss, diag = InternalRolloutEngine._outcome_loss(self, source, action, long)
        state = self._program_question_state
        target = getattr(self, '_mother_boundary_target', None)
        program_loss = loss*0
        examples = 0
        if target is not None:
            dist = mechanism(self.model, state, source[None], action[None], kind=Port.MOTHER,
                momentum=self._momentum_decision_read.detach()[None], ridge=self.relation_ridge)
            program_loss = program_loss - log_score(dist, self.model.encoder(target[None])).mean()
            # Recompute the pre-draft binding in current coordinates. The
            # current response is not evidence in this solve, only its label.
            requested = mechanism(self.model, state, source[None],
                self._program_question_action.detach()[None], ridge=self.relation_ridge)
            question = self.model.question_reactivation(requested['mean']-requested['base'])
            if self._program_question_roles['open'] == [Port.MOTHER.value]:
                program_loss = program_loss + F.mse_loss(question,
                    self.model.encoder(target[None])-dist['base'])
            examples += 1
        # Reactivate a bounded set of OLD real observations. Leave each own
        # observation out of its predictor; never self-score fitted labels.
        m = self.model.memory
        replay = real_evidence(self.model, state, Port.MOTHER)[:self.replay_batch]
        replay_losses = []
        for slot in replay.tolist():
            d = mechanism(self.model, state, m.source[slot:slot+1], m.action[slot:slot+1],
                          ridge=self.relation_ridge, exclude=slot)
            replay_losses.append(-log_score(d, self.model.encoder(m.mother_next[slot:slot+1])).mean())
        if replay_losses:
            program_loss = program_loss + torch.stack(replay_losses).mean()
        # Only the following ACTUAL decision boundary completes ambient
        # transport; it is not interchangeable with the mother's ending.
        pending = self.internal_pending
        if (self._internal_label is not None and pending is not None and
                'causal_program' in pending and pending['causal_program'].get('mother') is not None):
            previous = pending['causal_program']
            old_state = previous['state']
            i = old_state.index
            bank = self.model.causal_relations
            if int(bank.identity[i]) == old_state.identity and int(bank.generation[i]) == old_state.generation:
                s, a, next_state = self._internal_label
                mom = self.model.encoder(previous['mother'][None])
                d = mechanism(self.model, old_state, s[None], a[None], kind=Port.DECISION,
                    mother=mom, ridge=self.relation_ridge, exclude=pending['slot'])
                direct = mechanism(self.model, old_state, s[None], a[None], kind=Port.DECISION,
                    ridge=self.relation_ridge, exclude=pending['slot'],
                    momentum=pending['momentum_read'].detach()[None])
                label = self.model.encoder(next_state[None])
                # Both paths name exactly the same actual decision port.
                # World evidence, rather than forced agreement, settles them.
                program_loss = program_loss - .5*(log_score(d,label).mean()+log_score(direct,label).mean())
                examples += 1
        return loss+program_loss, dict(diag, causal_program_samples=examples,
            causal_program_replay=len(replay), causal_program_loss=float(program_loss.detach()))

    def step(self, directions, age_ablate=False):
        archive = getattr(self.model, '_causal_archive', None)
        if archive is not None:
            archive.prepare()
        previous = self.internal_pending
        old_question_weight = self.model.question_reactivation.weight.detach().clone()
        row = super().step(directions, age_ablate)
        state = self._program_question_state
        option = self._pending_record['choice']
        target = self._mother_boundary_target
        slot, m = row['episode_slot'], self.model.memory
        observed = None
        committed = False
        if target is not None:
            observed = Evidence(m.source[slot:slot+1], m.action[slot:slot+1],
                target[None], Port.MOTHER, observed=True, slot=slot, stamp=int(m.last_write[slot]))
            # The functional U is also executed on the real observation. Its
            # surviving evidence reference is then committed to the bank.
            updated = revise(state, observed, model=self.model)
            committed = self.model.causal_relations.commit(self.model, state, observed)
        else:
            updated = state
        before_score = None
        if target is not None:
            f = self._program_mother_forecast
            before_score = score_frozen({**f, 'mean':f['mean'][option:option+1],
                                         'scale2':f['scale2'][option:option+1]}, target)
        next_score = None
        if (previous is not None and 'causal_program' in previous and
                self._internal_label is not None):
            next_score = score_frozen(previous['causal_program']['decision_forecast'], self._internal_label[2])
        f = self._program_decision_forecast
        chosen_forecast = {**f, 'mean':f['mean'][option:option+1], 'scale2':f['scale2'][option:option+1]}
        self.internal_pending['causal_program'] = dict(
            state=state, roles=copy.deepcopy(self._program_question_roles),
            mother=None if target is None else target.detach().clone(),
            decision_forecast=chosen_forecast)
        row['causal_program'] = dict(
            roles=self._program_question_roles, pre_draft=self._program_pre_question,
            evidence_committed=committed, real_update_items=len(updated.extra),
            stored_version=int(self.model.causal_relations.version[state.index]),
            active_relations=int((self.model.causal_relations.identity > 0).sum()),
            evidence_count=int(self.model.causal_relations.valid[state.index].sum()),
            mother_prequential=before_score, decision_prequential=next_score,
            conditional_information=self._program_forecasts['information'].cpu().tolist(),
            expected_after_value=self._program_forecasts['expected_after_value'].cpu().tolist(),
            shared_update_future_delta=self._program_forecasts['future_delta'].cpu().tolist(),
            legal_overlap_residual_norm=self._program_forecasts['path']['overlap_residual'].norm(dim=-1).cpu().tolist(),
            question_weight_step=float((self.model.question_reactivation.weight.detach()-old_question_weight).norm()),
            execution=['bound condition/action', 'mother response', 'evidence U',
                       'ambient transport', 'next choice'],
            hypothesis_kind='conditional posterior operator draws; real predictive score recorded')
        if archive is not None:
            archive.sync()
            row['causal_archive'] = archive.stats()
        return row


def configure_causal_program(engine):
    if type(engine) not in (PersistentQuestionEngine, CausalProgramEngine):
        raise TypeError('requires the continuing joint water/momentum/mother body')
    if not (getattr(engine, 'mother_shadow_enabled', False) and
            getattr(engine, 'joint_question_update', False)):
        raise ValueError('shared program requires actual mother evidence and joint updates')
    model, m = engine.model, engine.model.memory
    model.causal_relation_ridge = engine.relation_ridge
    if not hasattr(model, 'causal_relations'):
        w = model.encoder.weight
        model.causal_relations = RelationBank(m.width, model.encoder.out_features,
            capacity=min(m.capacity, m.topk*m.topk), evidence_budget=m.topk,
            hypotheses=engine.shadow_samples, device=w.device, dtype=w.dtype)
    engine.__class__ = CausalProgramEngine
    engine.causal_program_schema = 1


def migrate_causal_program(engine):
    """An explicit new program with honest calibration and old history intact."""
    if getattr(engine, 'causal_program_schema', None) is not None:
        return engine
    engine.causal_program_gain_archive = dict(
        internal=copy.deepcopy(engine.internal_gain_state),
        shadow=copy.deepcopy(engine.shadow_gain_state))
    if engine.internal_pending is not None:
        engine.causal_program_pending_archive = copy.deepcopy(engine.internal_pending)
    configure_causal_program(engine)
    engine.internal_gain_state = gain_state(engine.wood_decay)
    engine.shadow_gain_state = gain_state(engine.wood_decay)
    # The previous observed transition and actor credit remain pending. Its
    # old forecasts settle under their old semantics; do not recalibrate them
    # as though they had been produced by the typed composed program.
    if engine.internal_pending is not None:
        for name in ('forecast', 'shadow_forecast'):
            if name in engine.internal_pending:
                engine.internal_pending['pre_program_'+name] = engine.internal_pending[name]
        # InternalRolloutEngine expects a forecast. Mark it ineligible while
        # preserving all source/action/label fields required by real learning.
        forecast = copy.deepcopy(engine.internal_pending['forecast'])
        forecast['eligible'][:] = False
        engine.internal_pending['forecast'] = forecast
        engine.internal_pending.pop('shadow_forecast', None)
    return engine
