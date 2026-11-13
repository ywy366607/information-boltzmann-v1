"""One relation grammar across internal work, questions, words and real revision."""
from __future__ import annotations

import copy
from dataclasses import replace

import torch
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


class GenerativeCycleMixin:
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
        self._binding_choices = []
        self._protect_relations()
        self.model._question_context = None
        self._internal_admission = []
        # One budget shared with the existing language search, not an assigned
        # frequency. The body can choose zero internal operations every turn.
        for _ in range(self.joint_proposal_budget):
            source = self._read_source()
            state = self.model.causal_relations.bind(self.model, source)
            now = compose(self.model, state, source, source, ridge=self.relation_ridge,
                reward_valid=self.reward_label_valid(wood=True),
                momentum=decision_momentum_read(self.model, self.bfss_momenta, source))
            if hasattr(self.model, 'head_coupling'):
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
            prior = torch.stack((now['value'][0], preview['value'][0]))*self.precision()
            logits = self.model.binding_choice(features).flatten()+prior
            selected = int(torch.multinomial(logits.softmax(0), 1))
            self._binding_choices.append(dict(features=features.detach().clone(),
                prior=prior.detach().clone(), selected=selected,
                sampled_logp=float(logits.log_softmax(0)[selected])))
            self._internal_admission.append(dict(selected='internal' if selected else 'external',
                probability=logits.softmax(0).cpu().tolist()))
            if selected == 0:
                break
            self.continue_internal(1)
        self._internal_budget_hit = bool(self._internal_admission and
            len(self._internal_admission) == self.joint_proposal_budget and
            self._internal_admission[-1]['selected'] == 'internal')
        return super()._utter()

    @torch.no_grad()
    def _condition_mother_question(self, h, context):
        # Retain the same original question ports and their update receipts.
        inherited = super()._condition_mother_question(h, context)
        model, m = self.model, self.model.memory
        state = self._program_question_state
        slots = real_evidence(model, state, Port.MOTHER)
        # Targets and alternatives come only from real supported experience.
        actions = torch.cat((h[None], h[None]+m.action[slots]-m.source[slots]))[:m.topk]
        conditions = torch.cat((h[None], m.source[slots]))[:m.topk]
        if len(slots):
            remembered = model.encoder(m.mother_next[slots])
            from scripts.ib_local.navigation_internal_rollout import continuation
            value, _ = continuation(model, remembered, reward_valid=self.reward_label_valid(wood=True))
            anchor=int(value.argmax())
            target = remembered[anchor][None]
            target_slot=int(slots[anchor]); target_stamp=int(m.last_write[target_slot])
            archive=getattr(model,'_causal_archive',None)
            saved=archive.hot.get(str(target_slot)) if archive is not None else None
            target_identity=(f'episode:{saved["id"]}' if saved is not None
                             else f'event:{target_slot}:{target_stamp}')
        else:
            target = compose(model, state, h[None], h[None], ridge=self.relation_ridge,
                reward_valid=self.reward_label_valid(wood=True))['mother']['mean']
            target_identity='unobserved';target_slot=target_stamp=-1
        query = infer_role(model, state, open_role=Port.ACTION, condition=h[None],
            action=None, candidates=actions, observed=target, ridge=self.relation_ridge)
        condition = infer_role(model, state, open_role=Port.CONDITION, condition=None,
            action=h[None], candidates=conditions, observed=target, ridge=self.relation_ridge)
        action = (query['log_weights'].exp()[:, None]*actions).sum(0, keepdim=True)
        source = (condition['log_weights'].exp()[:, None]*conditions).sum(0, keepdim=True)
        forward = infer_role(model, state, open_role=Port.MOTHER,
            condition=h[None], action=action, ridge=self.relation_ridge)
        bindings = ((h[None], action, forward['mean']),
                    (h[None], action, target), (source, h[None], target))
        features = torch.cat([binding_feature(model, s, a, y, role)
                              for role, (s,a,y) in zip(QUERY_ROLES, bindings)])
        # Existing expected future utility is the shared prior. A single small
        # selector receives the SAME real delayed policy factor as word choice.
        utility = torch.stack([imagine(model, state, s, a, ridge=self.relation_ridge,
            reward_valid=self.reward_label_valid(wood=True))['expected_after_value'][0] for s,a,_ in bindings])
        prior = self.precision()*utility
        logits = model.binding_choice(features).flatten()+prior
        selected = int(torch.multinomial(logits.softmax(0), 1))
        self._binding_choices.append(dict(features=features.detach().clone(),
            prior=prior.detach().clone(), selected=selected,
            sampled_logp=float(logits.log_softmax(0)[selected])))
        role = QUERY_ROLES[selected]
        s, a, y = bindings[selected]
        filler = (y-model.encoder(h[None]) if role == Port.MOTHER else
                  model.encoder(a)-model.encoder(h[None]) if role == Port.ACTION else
                  model.encoder(s)-model.encoder(h[None]))
        # Remove the inherited outcome-only contribution, keep every older
        # retained-question contribution, then bind the chosen typed filler.
        old = model.question_reactivation(self._program_question_vector[None])
        self._program_question_action = a[0].detach().clone()
        self._program_question_roles.update(open=[role.value],
            bound=[p.value for p in QUERY_ROLES if p != role],
            unknown_identity=f'{state.identity}:{role.value}:{target_identity}',
            anchor_identity=target_identity, anchor_slot=target_slot, anchor_stamp=target_stamp)
        self._generative_query = dict(role=role.value, alternatives=logits.softmax(0).cpu().tolist(),
            source=s.detach().clone(), action=a.detach().clone(), target=y.detach().clone())
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
        return tuple(result), replay

    def _value(self, h, films):
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
        archive = getattr(self.model, '_causal_archive', None)
        if archive is not None:
            archive.sync()
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
