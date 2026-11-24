"""One relation grammar across internal work, questions, words and real revision."""
from __future__ import annotations

import copy
from dataclasses import replace

import torch
from scripts.ib_local.navigation_gain_units import wood_choice_logits
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import (
    Port, infer_role, real_evidence, compose, imagine, log_score, release_choice)
from scripts.ib_local.generative_relations import replace_bank, PROGRAMS
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_joint_cycle import JointCycleMixin, JointCycleEngine, JointLearnedCycleEngine
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read


QUERY_ROLES = (Port.MOTHER, Port.ACTION, Port.CONDITION)


def binding_feature(model, source, action, target, role, internal=False):
    """Shared role/filler representation for every finite binding choice."""
    z, a = model.encoder(source), model.encoder(action)
    tag = z.new_zeros(len(z), 4)
    if internal:
        tag[:, 3] = 1
    else:
        tag[:, QUERY_ROLES.index(role)] = 1
    return torch.cat((z, a-z, target-z, tag), -1)


@torch.no_grad()
def question_choices(engine, state, h, *, utility=None, **candidate_options):
    """Native typed question producer, shared with prospective execution."""
    from scripts.ib_local.preview_runtime import preview_scope
    with preview_scope(engine.model):
        return _question_choices(engine, state, h, utility=utility, **candidate_options)


def _question_choices(engine, state, h, *, utility=None, **candidate_options):
    if utility is None and getattr(engine, 'native_amortization', {}).get('schema') == 1:
        from scripts.ib_local.native_amortized_decision import capture_actual
        capture_actual(engine, state, h.reshape(1, -1))
    model, m = engine.model, engine.model.memory
    slots = real_evidence(model, state, Port.MOTHER)
    # Targets and alternatives come only from real supported experience.
    actions = torch.cat((h[None], h[None]+m.action[slots]-m.source[slots]))[:m.topk]
    conditions = torch.cat((h[None], m.source[slots]))[:m.topk]
    canonical=getattr(model,'joint_relation_update_revision',1)>=3
    facts=None
    if canonical:
        from scripts.ib_local.relation_evidence import mechanism_evidence
        facts=mechanism_evidence(model,state,Port.MOTHER)
        if facts:
            actions=torch.cat((h[None],torch.cat([h[None]+e.action-e.source for e in facts])))[:m.topk]
            conditions=torch.cat((h[None],torch.cat([e.source for e in facts])))[:m.topk]
        else:actions=conditions=h[None]
    raw_target=None
    if (any(not e.encoded for e in facts) if canonical else bool(len(slots))):
        target_facts = [e for e in facts if not e.encoded] if canonical else None
        remembered = model.encoder(torch.cat([e.outcome for e in target_facts]) if canonical else m.mother_next[slots])
        from scripts.ib_local.navigation_internal_rollout import continuation
        value, _ = continuation(model, remembered, reward_valid=engine.reward_label_valid(wood=True))
        anchor=int(value.argmax())
        target = remembered[anchor][None]
        if canonical:
            fact=target_facts[anchor];target_slot,target_stamp=fact.slot,fact.stamp
            target_identity=fact.fact_id;raw_target=fact.outcome
        else:
            target_slot=int(slots[anchor]); target_stamp=int(m.last_write[target_slot])
            archive=getattr(model,'_causal_archive',None)
            saved=archive.hot.get(str(target_slot)) if archive is not None else None
            target_identity=(f'episode:{saved["id"]}' if saved is not None
                             else f'event:{target_slot}:{target_stamp}')
    else:
        target = compose(model, state, h[None], h[None], ridge=engine.relation_ridge,
            reward_valid=engine.reward_label_valid(wood=True))['mother']['mean']
        target_identity='unobserved';target_slot=target_stamp=-1
    query = infer_role(model, state, open_role=Port.ACTION, condition=h[None],
        action=None, candidates=actions, observed=target, ridge=engine.relation_ridge)
    condition = infer_role(model, state, open_role=Port.CONDITION, condition=None,
        action=h[None], candidates=conditions, observed=target, ridge=engine.relation_ridge)
    action = (query['log_weights'].exp()[:, None]*actions).sum(0, keepdim=True)
    source = (condition['log_weights'].exp()[:, None]*conditions).sum(0, keepdim=True)
    forward = infer_role(model, state, open_role=Port.MOTHER,
        condition=h[None], action=action, ridge=engine.relation_ridge)
    bindings = ((h[None], action, forward['mean']),
                (h[None], action, target), (source, h[None], target))
    proposals = None
    candidate_roles = QUERY_ROLES
    candidate_states = [state]*len(bindings)
    if getattr(engine, 'shared_unknown_schema', None) is not None:
        from scripts.ib_local.shared_unknown_bindings import candidates
        # Inverse questions need a real target in stable raw coordinates.
        # With no observed target, offer only the forward question.
        if not canonical:raw_target=m.mother_next[target_slot:target_slot+1] if target_slot>=0 else None
        proposals = candidates(engine,state,h[None],action,raw_target,target_identity, **candidate_options)
        bindings = [(p['source'],p['action'],p['target']) for p in proposals]
        candidate_roles = tuple(p['role'] for p in proposals)
        candidate_states = [p['state'] for p in proposals]
    features = torch.cat([binding_feature(model, s, a, y, role)
                          for role, (s,a,y) in zip(candidate_roles, bindings)])
    # Existing expected future utility is the shared prior. A single small
    # selector receives the SAME real delayed policy factor as word choice.
    from scripts.ib_local.causal_use_value import utility as use_utility
    utilities = []
    for qs, (s, a, _) in zip(candidate_states, bindings):
        if utility is None:
            prediction = imagine(model, qs, s, a, ridge=engine.relation_ridge,
                                 reward_valid=engine.reward_label_valid(wood=True))
            utilities.append(use_utility(engine, qs, s, a, prediction)[0])
        else:
            utilities.append(utility(qs, s, a)[0])
    prior = wood_choice_logits(engine, torch.stack(utilities))
    logits = model.binding_choice(features).flatten()+prior
    return dict(proposals=proposals, bindings=bindings, candidate_roles=candidate_roles,
                features=features, prior=prior, logits=logits, target_identity=target_identity,
                target_slot=target_slot, target_stamp=target_stamp)


class GenerativeCycleMixin:
    def _choice_offset(self, values, available):
        inherited = super()._choice_offset(values, available)
        from scripts.ib_local.causal_use_value import active
        if active(self):
            return (self.precision()*(values-values[available].mean())).masked_fill(~available, 0.)
        return inherited

    def _protect_relations(self):
        bank = self.model.causal_relations
        pending = self.internal_pending
        bank._protected_identities = set()
        if pending and 'causal_program' in pending:
            bank._protected_identities.add(pending['causal_program']['state'].identity)

    @torch.no_grad()
    def _read_source(self, recurrent=None):
        self.decision.start(self.recurrent if recurrent is None else recurrent)
        _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        return h[None]

    @torch.no_grad()
    def _utter(self):
        self._utterance_payload_cache = {}
        try:
            return self._utter_with_payload_cache()
        finally:
            del self._utterance_payload_cache

    def _utter_with_payload_cache(self):
        self._binding_choices = []
        self._protect_relations()
        self.model._question_context = None
        self._internal_admission = []
        if getattr(self, 'autonomous_events_schema', None) == 1:
            from scripts.ib_local.autonomous_cycle import take_boundary
            take_boundary(self)
            from scripts.ib_local import phase_coupled_rhythm
            if phase_coupled_rhythm.enabled(self):
                phase_coupled_rhythm.begin_cycle(self)
            return super()._utter()
        # One budget shared with the existing language search, not an assigned
        # frequency. The body can choose zero internal operations every turn.
        for _ in range(self.joint_proposal_budget):
            selected, choice, admission = self._sample_internal_choice()
            self._binding_choices.append(choice)
            self._internal_admission.append(admission)
            if selected == 0:
                break
            self.continue_internal(1)
        self._internal_budget_hit = bool(self._internal_admission and
            len(self._internal_admission) == self.joint_proposal_budget and
            self._internal_admission[-1]['selected'] == 'internal')
        from scripts.ib_local import phase_coupled_rhythm
        if phase_coupled_rhythm.enabled(self):
            phase_coupled_rhythm.begin_cycle(self)
        return super()._utter()

    @torch.no_grad()
    def _sample_internal_choice(self):
        source = self._read_source()
        state = self.model.causal_relations.bind(self.model, source)
        now = compose(self.model, state, source, source, ridge=self.relation_ridge,
            reward_valid=self.reward_label_valid(wood=True),
            momentum=decision_momentum_read(self.model, self.bfss_momenta, source))
        from scripts.quiet_force_jet import prepare_engine, finish_choice
        prepared = prepare_engine(self)
        if prepared is not None:
            next_s, next_p = prepared.result, list(prepared.next_p.unbind(0))
        elif hasattr(self.model, 'head_coupling'):
            next_s, next_p = self.model.head_coupling.quiet(self.model.body.base,
                self.recurrent, self.bfss_momenta, dt=self.bfss_dt,
                frequency=self.bfss_frequency, steps=1)
        else:
            from scripts.ib_local.bfss_gdn_capture import quiet_packed
            next_s, next_p = quiet_packed(self.model.body.base, self.recurrent,
                self.bfss_momenta, dt=self.bfss_dt, frequency=self.bfss_frequency,
                coupling=self.bfss_coupling, steps=1)
        later = self._read_source(next_s)
        preview = compose(self.model, state, later, later, ridge=self.relation_ridge,
            reward_valid=self.reward_label_valid(wood=True),
            momentum=decision_momentum_read(self.model, next_p, later))
        features = torch.cat((binding_feature(self.model, source, source,
            now['mother']['mean'], Port.MOTHER), binding_feature(self.model, later, later,
            preview['mother']['mean'], Port.MOTHER, internal=True)))
        from scripts.ib_local.causal_use_value import utility
        from scripts.ib_local import persistent_rhythm
        if persistent_rhythm.enabled(self):
            utility = persistent_rhythm.internal_utility
        prior = wood_choice_logits(self, torch.stack((
            utility(self, state, source, source, now)[0],
            utility(self, state, later, later, preview)[0])))
        logits = self.model.binding_choice(features).flatten()+prior
        selected = int(torch.multinomial(logits.softmax(0), 1))
        choice = dict(features=features.detach().clone(),
            prior=prior.detach().clone(), selected=selected,
            sampled_logp=float(logits.log_softmax(0)[selected]))
        admission = dict(selected='internal' if selected else 'external',
            probability=logits.softmax(0).cpu().tolist())
        finish_choice(self, prepared, selected)
        return selected, choice, admission

    @torch.no_grad()
    def _condition_mother_question(self, h, context):
        # Retain the same original question ports and their update receipts.
        direct = getattr(self, 'native_direct_question', False)
        inherited = super()._condition_mother_question(h, context, typed_replacement=direct)
        model, m = self.model, self.model.memory
        state = self._program_question_state
        from scripts.ib_local import phase_coupled_rhythm
        held = phase_coupled_rhythm.try_hold_theme(self, state, h) if phase_coupled_rhythm.enabled(self) else None
        if held is not None:
            state = held['state']
            self._program_question_state = state
            role, s, a, y = held['role'], held['source'], held['action'], held['target']
            from scripts.ib_local.prospective_self_execution import typed_filler
            filler = typed_filler(model, h[None], dict(role=role, source=s, action=a, target=y))
            old = 0. if direct else model.question_reactivation(self._program_question_vector[None])
            self._program_question_action = a[0].detach().clone()
            self._program_question_roles.update(open=[role.value],
                bound=[p.value for p in QUERY_ROLES if p != role],
                unknown_identity=held['unknown_id'],
                unknown_version=held['version'],
                anchor_identity=held['anchor'],
                binding_signature=held['signature'],
                resumed_unknown=True)
            self._generative_query = dict(role=role.value, alternatives=[1.0],
                source=s.detach().clone(), action=a.detach().clone(), target=y.detach().clone())
            self._program_pre_question = dict(candidate_count=1, selected=0,
                selection_value=[0.0], question_probabilities=[1.0],
                held_cycles=held['held_cycles'], unknown_identity=held['unknown_id'],
                runtime='phase_rhythm_gamma_theme_hold')
            return inherited-old+model.question_reactivation(filler)
        choices = question_choices(self, state, h)
        proposals, bindings = choices['proposals'], choices['bindings']
        candidate_roles = choices['candidate_roles']
        features, prior, logits = (choices[k] for k in ('features','prior','logits'))
        target_identity, target_slot, target_stamp = (choices[k] for k in
            ('target_identity','target_slot','target_stamp'))
        from scripts.ib_local.finite_choice_contract import choose
        selected = choose(logits.softmax(0), stage='question')
        self._binding_choices.append(dict(features=features.detach().clone(),
            prior=prior.detach().clone(), selected=selected,
            sampled_logp=float(logits.log_softmax(0)[selected])))
        role = candidate_roles[selected]
        s, a, y = bindings[selected]
        if proposals is not None:
            from scripts.ib_local.shared_unknown_bindings import select
            state = select(self,proposals[selected],state)
            self._program_question_state = state
        from scripts.ib_local.prospective_self_execution import typed_filler
        filler = typed_filler(model,h[None],dict(role=role,source=s,action=a,target=y))
        # Remove the inherited outcome-only contribution, keep every older
        # retained-question contribution, then bind the chosen typed filler.
        old = 0. if direct else model.question_reactivation(self._program_question_vector[None])
        self._program_question_action = a[0].detach().clone()
        self._program_question_roles.update(open=[role.value],
            bound=[p.value for p in QUERY_ROLES if p != role],
            unknown_identity=f'{state.identity}:{role.value}:{target_identity}',
            anchor_identity=target_identity, anchor_slot=target_slot, anchor_stamp=target_stamp)
        if proposals is not None:
            payload = proposals[selected]['payload']
            self._program_question_roles.update(unknown_identity=state.binding.identity,
                unknown_version=self._shared_question_receipt['version'],
                anchor_identity=payload['anchor'],
                binding_signature=proposals[selected]['signature'],
                resumed_unknown=proposals[selected]['recalled'])
            if phase_coupled_rhythm.enabled(self):
                raw_t = None if payload.get('target') is None else h.new_tensor(payload['target']).reshape(1, -1)
                phase_coupled_rhythm.record_theme(
                    self, state, role, s, a, y,
                    unknown_id=state.binding.identity,
                    anchor=payload['anchor'],
                    signature=proposals[selected]['signature'],
                    target_raw=raw_t)
        self._generative_query = dict(role=role.value, alternatives=logits.softmax(0).cpu().tolist(),
            source=s.detach().clone(), action=a.detach().clone(), target=y.detach().clone())
        if direct:
            self._program_pre_question = dict(candidate_count=len(bindings), selected=selected,
                selection_value=prior.cpu().tolist(),
                question_probabilities=logits.softmax(0).cpu().tolist(),
                runtime='actual typed question; superseded mother-only menu elided')
        return inherited-old+model.question_reactivation(filler)

    def _policy_grad(self, targets, own, record):
        base, replay = super()._policy_grad(targets, own, record)
        result = list(base)
        self._binding_replay_error = 0.
        for choice in self._binding_choices:
            logits = self.model.binding_choice(choice['features']).flatten()+choice['prior']
            score = logits.log_softmax(0)[choice['selected']]
            error=abs(float(score.detach())-choice['sampled_logp'])
            self._binding_replay_error=max(self._binding_replay_error,error)
            if error > 1e-5:
                raise RuntimeError(f'binding choice replay differs: {error}')
            grads = torch.autograd.grad(score, self.params, allow_unused=True)
            for i, g in enumerate(grads):
                if g is not None:
                    result[i] = g.detach() if result[i] is None else result[i]+g.detach()
        if getattr(self, 'autonomous_events_schema', None) == 1:
            from scripts.ib_local.autonomous_cycle import add_boundary_credit
            add_boundary_credit(self, result)
        return tuple(result), replay

    def _value(self, h, films):
        if getattr(self, 'native_shared_utter_preview', False):
            from scripts.ib_local.preview_runtime import preview_scope
            # The actual typed-question selection has already written its
            # shared binding. Only root comparison and its native teachers
            # share this scope; the earlier question menu has its own scope.
            with preview_scope(self.model):
                result = super()._value(h, films)
        else:
            result = super()._value(h, films)
        paths = self._program_forecasts['path']['program_predictions']
        self._generated_program_forecasts = [dict(mean=p['mean'].detach().clone(),
            scale2=p['scale2'].detach().clone(), df=p['df']) for p in paths]
        return result

    def _outcome_loss(self, source, action, long):
        loss, diag = super()._outcome_loss(source, action, long)
        target = getattr(self, '_mother_boundary_target', None)
        if target is None:
            return loss, diag
        role = Port(self._generative_query['role'])
        if role != Port.MOTHER:
            state, m = self._program_question_state, self.model.memory
            slots = real_evidence(self.model, state, Port.MOTHER)[:m.topk-1]
            actual = action if role == Port.ACTION else source
            others = m.action[slots] if role == Port.ACTION else m.source[slots]
            if getattr(self.model,'joint_relation_update_revision',1)>=3:
                from scripts.ib_local.relation_evidence import mechanism_evidence
                facts=mechanism_evidence(self.model,state,Port.MOTHER)[:m.topk-1]
                others=(torch.cat([e.action if role==Port.ACTION else e.source for e in facts])
                        if facts else actual.new_empty((0,actual.numel())))
            candidates = torch.cat((actual[None], others))
            inferred = infer_role(self.model, state, open_role=role,
                condition=source[None], action=action[None], candidates=candidates,
                observed=self.model.encoder(target[None]), ridge=self.relation_ridge)
            loss = loss-inferred['log_weights'][0]
            diag = dict(diag, inverse_binding_samples=len(candidates))
        return loss, diag

    def step(self, directions, age_ablate=False):
        previous = self.internal_pending
        row = super().step(directions, age_ablate)
        program_update = False
        if previous and 'generated_program_forecasts' in previous and self._internal_label is not None:
            receipt = previous['causal_program']
            target = F.linear(self._internal_label[2][None], receipt['decision_forecast']['encoder_weight'])
            scores = torch.stack([log_score(p,target).mean() for p in previous['generated_program_forecasts']])
            program_update = self.model.causal_relations.observe_program(receipt['state'], scores)
        selected = self._pending_record['choice']
        self.internal_pending['generated_program_forecasts'] = [
            dict(mean=p['mean'][selected:selected+1], scale2=p['scale2'][selected:selected+1],df=p['df'])
            for p in self._generated_program_forecasts]
        state = self.model.causal_relations.state(self._program_question_state.index)
        row['generative_relation'] = dict(schema=1, query_role=self._generative_query['role'],
            query_probabilities=self._generative_query['alternatives'],
            programs=[list(w) for w in PROGRAMS],program_weights=state.program_weights.cpu().tolist(),
            program_evidence_updated=program_update,
            internal_admission=copy.deepcopy(self._internal_admission),
            internal_budget_hit=self._internal_budget_hit,
            reorganizations=self.model.causal_relations.reorganizations.sum(0).cpu().tolist(),
            same_U=True, observed_response_bound=True,
            binding_policy_in_delayed_actor=True)
        row['generative_relation']['binding_replay_error']=getattr(self,'_binding_replay_error',None)
        if getattr(self, 'autonomous_events_schema', None) == 1:
            row['generative_relation']['autonomous'] = copy.deepcopy(self.autonomous_action_receipt)
        archive = getattr(self.model, '_causal_archive', None)
        if archive is not None:
            archive.sync()
        if getattr(self,'shared_unknown_schema',None) is not None:
            from scripts.ib_local.shared_unknown_bindings import commit
            row['shared_unknown'] = dict(self._shared_question_receipt,
                **{'revision':commit(self,row)},
                candidate_count=(len(self._binding_choices[-1]['features']) if self._binding_choices else 1),
                evidence_before=len({e.fact_id for e in self._program_question_state.constraints}),
                program_port=self._program_question_state.binding.port.value)
            if getattr(self, 'causal_use_arm', None) == 'C2':
                from scripts.ib_local.finite_use_value import verify_commit
                verify_commit(self, row['shared_unknown']['revision'])
                if getattr(self.model, 'reference_use_enabled', False):
                    from scripts.ib_local.reference_use import commit as commit_reference_use
                    commit_reference_use(self.model, self.causal_use_last.get('finite_U', {}).get('reference_use'))
                from scripts.ib_local.causal_use_transport import finish as finish_transport
                finish_transport(self, row)
                if not getattr(self, '_deferred_causal_use_optimizer_step', False):
                    raise RuntimeError('finite U transaction lost its value learning update')
                self.optimizer.step()
                del self._deferred_causal_use_optimizer_step
        from scripts.ib_local import phase_coupled_rhythm
        if phase_coupled_rhythm.enabled(self):
            phase_coupled_rhythm.end_cycle(self, row)
        return row


class GenerativeCycleEngine(GenerativeCycleMixin, JointCycleEngine):
    pass


class GenerativeLearnedCycleEngine(GenerativeCycleMixin, JointLearnedCycleEngine):
    pass


def configure_generative_cycle(engine):
    if not isinstance(engine, JointCycleMixin):
        raise TypeError('requires the complete continuing joint-world body')
    replace_bank(engine.model)
    if not hasattr(engine.model, 'binding_choice'):
        w = engine.model.encoder.weight
        with torch.random.fork_rng(devices=[w.device] if w.is_cuda else []):
            head = nn.Linear(3*engine.model.encoder.out_features+4, 1, bias=False,
                             device=w.device, dtype=w.dtype)
        nn.init.zeros_(head.weight)
        engine.model.binding_choice = head
        engine.optimizer.add_param_group(dict(params=list(head.parameters()),
            lr=engine.optimizer.param_groups[0]['lr']))
        engine.params = [p for p in engine.model.parameters() if p.requires_grad]
        engine.groups = (*engine.groups[:2], [p for n,p in engine.model.named_parameters()
                                             if not n.startswith('body.')])
    engine.__class__ = (GenerativeLearnedCycleEngine if hasattr(engine, 'learned_head_kind')
                        else GenerativeCycleEngine)
    engine.generative_relation_schema = 1
    return engine
