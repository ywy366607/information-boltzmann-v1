"""The shared relation program inside the continuing mother-world body.

Binding, typed composition and evidence revision run on the real navigation
path. The original language/actor learning, S/P dynamics, exact byte table,
actual gains and caregiver remain in the same individual. No separate F or
question reward is introduced.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

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
    forecast=dict(mean=distribution['mean'].detach().clone(),
                scale2=distribution['scale2'].detach().clone(), df=distribution['df'],
                encoder_weight=encoder_weight.detach().clone())
    if 'scale_matrix' in distribution:
        forecast['scale_matrix']=distribution['scale_matrix'].detach().clone()
    return forecast


def score_frozen(forecast, actual):
    y = F.linear(actual.reshape(1, -1), forecast['encoder_weight'])
    distribution = {k: forecast[k] for k in ('mean', 'scale2', 'df')}
    if 'scale_matrix' in forecast:distribution['scale_matrix']=forecast['scale_matrix']
    return dict(log_score=float(log_score(distribution, y).mean()),
                mse=float((distribution['mean']-y).square().mean()))


@torch.no_grad()
def counterfactual_response_shadow(forecast, actual, chosen, probabilities):
    means = forecast['mean']
    observed = F.linear(actual.reshape(1, -1), forecast['encoder_weight'])[0]
    residual = observed - means[chosen]
    alternatives = [i for i, p in enumerate(probabilities) if i != chosen and p > 0]
    return dict(chosen=chosen, observed=observed.tolist(), alternatives=[
        dict(option=i, prior=means[i].tolist(), transported=(means[i]+residual).tolist())
        for i in alternatives])


class CausalProgramEngine(PersistentQuestionEngine):
    @torch.no_grad()
    def _condition_mother_question(self, h, context, *, typed_replacement=False):
        inherited = super()._condition_mother_question(h, context)
        bank, m = self.model.causal_relations, self.model.memory
        state = bank.bind(self.model, h[None])
        self._program_question_state = state
        if getattr(self.model, 'reference_use_enabled', False):
            from scripts.ib_local.reference_use import capture
            self._reference_use_boundary = capture(self.model, state)
        if typed_replacement:
            # GenerativeCycleMixin installs the actually sampled typed filler.
            # Its old path added this legacy vector only to subtract it again;
            # capture_actual also clears the legacy imagination's teacher queue.
            # Keep the real bind/reference receipt, and defer the menu receipt
            # to that actual typed choice instead of evaluating a discarded one.
            self._program_question_roles = dict(subject='continuing_body', receiver='mother',
                identity=state.identity, generation=state.generation, version=state.version)
            return inherited
        slots = real_evidence(self.model, state, Port.MOTHER)
        # Actions are the body's own remembered action changes. The source is
        # observed now, before drafting; no world location or hidden answer.
        prior_actions = h[None] + m.action[slots] - m.source[slots]
        if getattr(self.model,'joint_relation_update_revision',1)>=3:
            from scripts.ib_local.relation_evidence import mechanism_evidence
            facts=mechanism_evidence(self.model,state,Port.MOTHER)
            if getattr(self.model, 'identity_preview_revision', 0) >= 1:
                from scripts.ib_local.identity_preview import ranked_action_facts
                facts = ranked_action_facts(self.model, h, facts, m.topk - 1)
            prior_actions=(torch.cat([h[None]+e.action-e.source for e in facts])
                           if facts else h.new_empty((0,h.numel())))
        candidates = torch.cat((h[None], prior_actions), 0)[:m.topk]
        source = h.expand_as(candidates)
        p = decision_momentum_read(self.model, self.bfss_momenta, h[None]).expand(len(candidates), -1)
        forecast = imagine(self.model, state, source, candidates, ridge=self.relation_ridge,
            reward_valid=self.reward_label_valid(wood=True), momentum=p)
        # A question carries an open mother-outcome role and a proposed binding.
        # Existing continuation chooses a useful binding; information is NOT a
        # new reward. The actual utterance/action is chosen later by the actor.
        from scripts.ib_local.causal_use_value import utility
        binding_value = utility(self, state, source, candidates, forecast)
        choice = int(binding_value.argmax())
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
            selection_value=binding_value.cpu().tolist(),
            expected_after_value=forecast['expected_after_value'].cpu().tolist())
        if getattr(self.model, 'identity_preview_revision', 0) >= 1:
            from scripts.ib_local.identity_preview import fact_identity
            self._program_pre_question['candidate_identities'] = ['current'] + [fact_identity(e) for e in facts]
        return inherited + self.model.question_reactivation(relation[None])

    @torch.no_grad()
    def _value(self, h, films):
        values, inputs = super()._value(h, films)
        from scripts.ib_local.competing_action_teachers import decision_scope
        from scripts.ib_local.preview_runtime import preview_scope
        with preview_scope(self.model), decision_scope(self, h, self._internal_action):
            return self._relation_value(h, films, values, inputs)

    def _relation_value(self, h, films, values, inputs):
        state = self._program_question_state
        actions = self._internal_action
        source = h.expand_as(actions)
        momentum = self._momentum_decision_read[None].expand(len(actions), -1)
        from scripts.ib_local import phase_coupled_rhythm
        if phase_coupled_rhythm.enabled(self) and not phase_coupled_rhythm.is_theta_open(self):
            planned = phase_coupled_rhythm.gamma_relation_plan(self, state, source, actions, momentum)
        else:
            planned = imagine(self.model, state, source, actions,
                ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
                momentum=momentum)
        path = planned['path']
        if getattr(self.model,'joint_relation_update_revision',1)>=3:
            from scripts.ib_local.generative_relations import dataset
            from scripts.ib_local.relation_evidence import mechanism_evidence,fact_reference
            ox,oy,_,refs=dataset(self.model,state.organization)
            actual=path['mother']
            actual_refs=[fact_reference(e) for e in mechanism_evidence(self.model,state,Port.MOTHER)]
            equal=(refs==actual_refs and ox.shape==actual['x'].shape and oy.shape==actual['y'].shape and
                   torch.allclose(ox,actual['x'],atol=1e-6,rtol=1e-6) and
                   torch.allclose(oy,actual['y'],atol=1e-6,rtol=1e-6))
            if not equal:raise RuntimeError('prediction and organization used different evidence')
            self._evidence_contract_before=dict(same_facts=True,same_design=True,same_residual=True,
                episode_ids=refs,items=len(refs))
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
        if getattr(self, 'counterfactual_response_shadow', False):
            from scripts.ib_local.counterfactual_response_observer import capture
            self._counterfactual_response_context = capture(self, source, actions, momentum)
        if getattr(self, 'epistemic_value_schema', 0) >= 6:
            from scripts.ib_local.joint_response import distribution
            self._program_mother_joint_forecast = frozen_forecast(distribution(path['mother']), self._mother_pre_encoder)
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
            actual_evidence_used=torch.tensor(bool(len(path['mother']['slots']) or
                path['mother'].get('bound_evidence',0)), device=h.device))
        from scripts.ib_local.causal_use_value import active, utility
        if active(self):
            values = 10*utility(self, state, source, actions, planned)
        if getattr(self, 'native_amortization', {}).get('correction') == 'decision_budget':
            from scripts.ib_local.latent_action_planner import flush_teachers
            values = flush_teachers(self, decision_values=values)
        return values, inputs

    def _outcome_loss(self, source, action, long):
        # Retain the existing language-world outcome and real next-boundary
        # training. Replace the old extra mother/question/ridge losses with
        # the SAME typed program that was actually used to choose and speak.
        loss, diag = InternalRolloutEngine._outcome_loss(self, source, action, long)
        state = self._program_question_state
        target = getattr(self, '_mother_boundary_target', None)
        from scripts.ib_local.joint_response import predictive_score
        program_loss = loss*0
        examples = 0
        momentum_credit = {'current': 0, 'delayed': 0}
        coupling = getattr(self.model, 'head_coupling', None)

        def credited_read(value, eligibility, port):
            read = value.detach()[None]
            if coupling is not None and hasattr(coupling, 'bridge_momentum_read'):
                read = coupling.bridge_momentum_read(read, eligibility)
                momentum_credit[port] += int(eligibility is not None)
            return read

        if target is not None:
            dist = mechanism(self.model, state, source[None], action[None], kind=Port.MOTHER,
                momentum=credited_read(self._momentum_decision_read,
                    getattr(self, '_momentum_decision_eligibility', None), 'current'), ridge=self.relation_ridge)
            program_loss = program_loss - predictive_score(self.model, dist, self.model.encoder(target[None])).mean()
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
        if getattr(self.model,'joint_relation_update_revision',1)>=3:
            from scripts.ib_local.relation_evidence import mechanism_evidence
            old=[e for e in mechanism_evidence(self.model,state,Port.MOTHER)
                 if e.observed and not e.encoded][:self.replay_batch]
            for evidence in old:
                d=mechanism(self.model,state,evidence.source,evidence.action,
                    ridge=self.relation_ridge,exclude_fact=evidence.fact_id)
                replay_losses.append(-predictive_score(self.model,d,self.model.encoder(evidence.outcome)).mean())
        elif hasattr(self.model,'relation_basis') and state.binding is not None:
            # Cold facts may have no hot slot. Fact identity, rather than a
            # stale/absent hot index, excludes their own outcome from fitting.
            old = [e for e in state.constraints if e.kind==Port.MOTHER and e.observed
                   and not e.encoded][:self.replay_batch]
            for evidence in old:
                d = mechanism(self.model,state,evidence.source,evidence.action,
                    ridge=self.relation_ridge,exclude_fact=evidence.fact_id)
                replay_losses.append(-predictive_score(self.model,d,self.model.encoder(evidence.outcome)).mean())
        else:
            for slot in replay.tolist():
                d = mechanism(self.model, state, m.source[slot:slot+1], m.action[slot:slot+1],
                              ridge=self.relation_ridge, exclude=slot)
                replay_losses.append(-predictive_score(self.model,d,self.model.encoder(m.mother_next[slot:slot+1])).mean())
        if replay_losses:
            program_loss = program_loss + torch.stack(replay_losses).mean()
        # Only the following ACTUAL decision boundary completes ambient
        # transport; it is not interchangeable with the mother's ending.
        pending = self.internal_pending
        delayed_fact = None
        successor_learning = None
        if (self._internal_label is not None and pending is not None and
                'causal_program' in pending and pending['causal_program'].get('mother') is not None):
            previous = pending['causal_program']
            aligned = (getattr(self,'epistemic_value_schema',0) in (4,5,6) and
                       'decision_learning_state' in previous)
            old_state = previous['decision_learning_state'] if aligned else previous['state']
            i = old_state.index
            bank = self.model.causal_relations
            if int(bank.identity[i]) == old_state.identity and int(bank.generation[i]) == old_state.generation:
                s, a, next_state = self._internal_label
                mom = self.model.encoder(previous['mother'][None])
                exclusion=dict(exclude=pending['slot'])
                if getattr(self.model,'joint_relation_update_revision',1)>=4:
                    from scripts.ib_local.relation_evidence import pending_fact_identity
                    delayed_fact=pending_fact_identity(self.model,pending)
                    exclusion=dict(exclude_fact=delayed_fact)
                d = mechanism(self.model, old_state, s[None], a[None], kind=Port.DECISION,
                    mother=mom, ridge=self.relation_ridge, **exclusion)
                direct = mechanism(self.model, old_state, s[None], a[None], kind=Port.DECISION,
                    ridge=self.relation_ridge, **exclusion,
                    momentum=credited_read(pending['momentum_read'],
                        pending.get('momentum_read_eligibility'), 'delayed'))
                label = self.model.encoder(next_state[None])
                # Both paths name exactly the same actual decision port.
                # World evidence, rather than forced agreement, settles them.
                if aligned:
                    from scripts.ib_local.successor_distribution import training_distribution
                    coordinated=training_distribution(self.model,old_state,d,direct)
                    program_loss = program_loss-log_score(coordinated,label).mean()
                    from scripts.ib_local.relation_evidence import mechanism_evidence
                    successor_learning=dict(schema=1,event=delayed_fact,
                        evidence=[f.fact_id for f in mechanism_evidence(self.model,old_state,Port.MOTHER)],
                        target_excluded=all(f.fact_id!=delayed_fact for f in mechanism_evidence(
                            self.model,old_state,Port.DECISION,**exclusion)),
                        score=float(log_score(coordinated,label).mean().detach()),
                        mean_scale=float(coordinated['scale2'].mean().detach()))
                else:
                    program_loss = program_loss - .5*(log_score(d,label).mean()+log_score(direct,label).mean())
                examples += 1
        if getattr(self,'epistemic_value_schema',0) in (4,5,6):
            self.successor_learning_last=successor_learning
        return loss+program_loss, dict(diag, causal_program_samples=examples,
            causal_program_replay=len(replay_losses), causal_program_loss=float(program_loss.detach()),
            momentum_current_credit_samples=momentum_credit['current'],
            momentum_delayed_credit_samples=momentum_credit['delayed'],
            causal_program_delayed_fact=delayed_fact,successor_learning=successor_learning)

    def step(self, directions, age_ablate=False):
        archive = getattr(self.model, '_causal_archive', None)
        if archive is not None:
            archive.prepare()
        previous = self.internal_pending
        old_question_weight = self.model.question_reactivation.weight.detach().clone()
        row = super().step(directions, age_ablate)
        from scripts.ib_local.identified_event_credit import finish_labels
        finish_labels(self, row)
        state = self._program_question_state
        option = self._pending_record['choice']
        target = self._mother_boundary_target
        slot, m = row['episode_slot'], self.model.memory
        observed = None
        committed = False
        identity = None
        if target is not None:
            identity=''
            if getattr(self.model,'joint_relation_update_revision',1)>=3:
                archive.sync()
                identity=f"episode:{archive.hot[str(slot)]['id']}"
            observed = Evidence(m.source[slot:slot+1], m.action[slot:slot+1],
                target[None], Port.MOTHER, observed=True, slot=slot, stamp=int(m.last_write[slot]),fact_id=identity)
            from scripts.ib_local.identified_event_credit import current
            event = current(self)
            if event is not None:
                observed, identity = event, event.fact_id
            # The functional U is also executed on the real observation. Its
            # surviving evidence reference is then committed to the bank.
            updated = revise(state, observed, model=self.model)
            # The question owns its shared conditional constraints. Local
            # relation scope organization retains its own versioned evidence;
            # a recalled question does not silently alias different scopes.
            if getattr(self.model,'joint_relation_update_schema',None)==1:
                local_state = state.organization
                if local_state is None or updated.organization is None:
                    raise RuntimeError('joint U missing the real/shadow scope update')
                committed = self.model.causal_relations.commit(self.model,local_state,observed,
                                                              updated=updated.organization)
            else:
                local_state = replace(state,binding=None,constraints=())
                committed = self.model.causal_relations.commit(self.model, local_state, observed)
        else:
            updated = state
        before_score = None
        response_joint_score = None
        if target is not None:
            f = self._program_mother_forecast
            before_score = score_frozen({**f, 'mean':f['mean'][option:option+1],
                                         'scale2':f['scale2'][option:option+1]}, target)
            if getattr(self, 'epistemic_value_schema', 0) >= 6:
                f = self._program_mother_joint_forecast
                response_joint_score = score_frozen({**f, **{k:f[k][option:option+1]
                    for k in ('mean', 'scale2', 'scale_matrix')}}, target)
        next_score = None
        if (previous is not None and 'causal_program' in previous and
                self._internal_label is not None):
            next_score = score_frozen(previous['causal_program']['decision_forecast'], self._internal_label[2])
        f = self._program_decision_forecast
        chosen_forecast = {**f, 'mean':f['mean'][option:option+1], 'scale2':f['scale2'][option:option+1]}
        if 'scale_matrix' in f:chosen_forecast['scale_matrix']=f['scale_matrix'][option:option+1]
        self.internal_pending['causal_program'] = dict(
            state=state, roles=copy.deepcopy(self._program_question_roles),
            mother=None if target is None else target.detach().clone(),
            decision_forecast=chosen_forecast)
        if getattr(self,'epistemic_value_schema',0) in (4,5,6):
            from scripts.ib_local.successor_distribution import learning_state
            self.internal_pending['causal_program']['decision_learning_state']=learning_state(updated)
        if getattr(self.model,'joint_relation_update_revision',1)>=4 and identity:
            self.internal_pending['causal_program']['episode_identity']=identity
        row['causal_program'] = dict(
            roles=self._program_question_roles, pre_draft=self._program_pre_question,
            evidence_committed=committed, real_update_items=len(updated.extra),
            stored_version=int(self.model.causal_relations.version[state.index]),
            active_relations=int((self.model.causal_relations.identity > 0).sum()),
            evidence_count=int(self.model.causal_relations.valid[state.index].sum()),
            mother_prequential=before_score, decision_prequential=next_score,
            mother_joint_prequential=response_joint_score,
            committed_episode_identity=identity,
            conditional_information=self._program_forecasts['information'].cpu().tolist(),
            expected_after_value=self._program_forecasts['expected_after_value'].cpu().tolist(),
            shared_update_future_delta=self._program_forecasts['future_delta'].cpu().tolist(),
            legal_overlap_residual_norm=self._program_forecasts['path']['overlap_residual'].norm(dim=-1).cpu().tolist(),
            question_weight_step=float((self.model.question_reactivation.weight.detach()-old_question_weight).norm()),
            execution=['bound condition/action', 'mother response', 'evidence U',
                       'ambient transport', 'next choice'],
            hypothesis_kind='conditional posterior operator draws; real predictive score recorded')
        context = self.__dict__.pop('_counterfactual_response_context', None)
        if context is not None and observed is not None:
            row['causal_program']['counterfactual_response_shadow'] = counterfactual_response_shadow(
                self._program_mother_forecast, target, option, row['choice_probability'])
            from scripts.ib_local.counterfactual_response_observer import settle
            report = settle(self, context, state, updated, observed, option, row['choice_probability'])
            report['previous_actual_episode'] = ((previous or {}).get('causal_program') or {}).get('episode_identity')
            row['causal_program']['counterfactual_response_settlement'] = report
        if getattr(self,'epistemic_value_schema',0) in (4,5,6):
            row['causal_program']['successor_learning']=copy.deepcopy(self.successor_learning_last)
        if getattr(updated.organization,'retention_use',None) is not None:
            from dataclasses import asdict
            row['causal_program']['retention_organization']=dict(
                trace=updated.organization.retention_trace,
                after=asdict(updated.organization.retention_use))
        if archive is not None:
            archive.sync()
            row['causal_archive'] = archive.stats()
        if getattr(self.model,'joint_relation_update_schema',None)==1:
            coordinated=self._program_forecasts['path']['overlap_coordination']
            row['joint_relation_update']=dict(schema=1,
                organization_in_shared_U=True,feature_dimension=4,
                organization_operation=(updated.organization.revision_plan or {}).get('operation','retain'),
                variable_uses=[dict(identity=v.identity,port=v.port.value,version=version,seen_version=seen)
                               for v,version,seen in state.variable_uses],
                overlap_objective_before=coordinated['objective_before'].detach().cpu().tolist(),
                overlap_objective_after=coordinated['objective_after'].detach().cpu().tolist(),
                overlap_shift=(coordinated['mean']-
                    self._program_forecasts['path']['program_predictions'][0]['mean']).norm(dim=-1).detach().cpu().tolist(),
                continuation_uses_coordinated_boundary=True)
            if getattr(self.model,'joint_relation_update_revision',1)>=3:
                row['joint_relation_update']['evidence_contract']=self._evidence_contract_before
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
