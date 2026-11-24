"""Joint world-learning, concrete language search and input-free continuation.

Reuses the carried body, relation identities, U, S/P, actor and experience
stores. New interfaces have explicit finite budgets and no invented objective,
question label, world answer or self-preservation set point.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from scripts.ib_local.causal_relation_program import imagine
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_causal_program import CausalProgramEngine
from scripts.ib_local.navigation_direct_read import causal_prefix, gain_state
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.table_branch import table_branch
from scripts.ib_local.navigation_gain_units import wood_choice_logits


def proposal_log_probability(choice_logp, proposals, selection_logp):
    """Score the actual joint sampling path, including unchosen proposals.

    Planner evaluations are detached, as in the inherited model-based actor.
    Every proposal was sampled before choosing whether to speak. Omitting
    unchosen proposals would give the wrong credit for this search procedure.
    Use one event's log probability, not a variable-length term average.
    """
    return choice_logp + torch.stack([p.sum() for p in proposals]).sum() + selection_logp


class JointCycleMixin:
    @torch.no_grad()
    def _film(self, words, table):
        if words is not None:
            return super()._film(words, table)
        candidates, endpoints, handovers = [], [], []
        from scripts.ib_local import phase_coupled_rhythm, root_latent_planning
        budget = phase_coupled_rhythm.proposal_budget(self, self.joint_proposal_budget)
        budget = root_latent_planning.proposal_budget(self, budget)
        # All proposed words start at the SAME real state and fast-table prefix.
        # None of the proposed observations are committed as world evidence.
        for _ in range(budget):
            with table_branch(self.table) as branch:
                result = super()._film(None, branch)
            endpoint = self._draft_actions.pop()
            # The inherited causal-address path also appends a word candidate.
            # Keep the outer silence/generate/reuse axis at exactly three.
            self._causal_words.pop()
            candidates.append(result)
            endpoints.append(endpoint)
            handovers.append(getattr(self,'_last_expression_receipt',None))
        if budget == 1 and (phase_coupled_rhythm.enabled(self) or root_latent_planning.is_latent_single_active(self)):
            # Exact singleton REINFORCE identity: log_softmax([v])[0] == 0, selected == 0.
            selected = 0
            self._joint_proposals = [dict(words=v[0], logp=v[1], probs=v[2]) for v in candidates]
            if getattr(self,'variable_expression_schema',None)==1:
                self._joint_proposals[0]['handover'] = handovers[0]
                self._selected_handover = handovers[0]
            if root_latent_planning.is_latent_single_active(self):
                self._joint_selection = root_latent_planning.build_single_selection_receipt(self)
            else:
                self._joint_selection = dict(selected=0, logp=0.0, distribution=[1.0],
                    predicted_after_value=[0.0], selection_value=[0.0],
                    relation_identity=self._program_question_state.identity,
                    relation_version=self._program_question_state.version,
                    budget=1, regime='gamma')
            self._draft_actions.append(endpoints[0])
            self._causal_words.append(candidates[0][0])
            return candidates[0]
        self.decision.start(self.recurrent)
        _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        actions = torch.stack(endpoints)
        source = h[None].expand_as(actions)
        momentum = decision_momentum_read(self.model, self.bfss_momenta, h[None]).expand(len(actions), -1)
        forecast = imagine(self.model, self._program_question_state, source, actions,
            ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
            momentum=momentum)
        from scripts.ib_local.causal_use_value import utility
        selected_value = utility(self, self._program_question_state, source, actions, forecast)
        logits = wood_choice_logits(self, selected_value)
        logp = logits.log_softmax(0)
        from scripts.ib_local.finite_choice_contract import choose
        selected = choose(logp.exp(), stage='proposal')
        self._joint_proposals = [dict(words=v[0], logp=v[1], probs=v[2]) for v in candidates]
        if getattr(self,'variable_expression_schema',None)==1:
            for proposal,handover in zip(self._joint_proposals,handovers):
                proposal['handover']=handover
            self._selected_handover=handovers[selected]
        self._joint_selection = dict(selected=selected, logp=float(logp[selected]),
            distribution=logp.exp().cpu().tolist(),
            predicted_after_value=forecast['expected_after_value'].cpu().tolist(),
            selection_value=selected_value.cpu().tolist(),
            relation_identity=self._program_question_state.identity,
            relation_version=self._program_question_state.version)
        self._draft_actions.append(endpoints[selected])
        self._causal_words.append(candidates[selected][0])
        return candidates[selected]

    def _selected_proposal_audit(self, targets, own, record):
        if getattr(self, 'variable_expression_schema', None) != 1 or not own or record['choice'] != 1:
            return None
        index = self._joint_selection['selected']
        proposal = self._joint_proposals[index]
        if (proposal.get('handover') is not self._selected_handover or
                proposal.get('handover') is None or
                proposal['words'] != bytes(targets[0, :own].tolist()) or
                not np.array_equal(proposal['probs'], record['probs'])):
            return None
        return index

    def _choice_policy_term(self, targets, own, record, *, reuse_audit=False):
        # Recompute the actual one-byte decision prefix. Do not subtract a
        # detached byte logp from the inherited mean: that would count the
        # selected proposal gradient twice.
        if getattr(self,'variable_expression_schema',None)==1 and own:
            # Choosing whether to speak precedes committing an output budget.
            # Replay that actual boundary, not the first budget-conditioned
            # draft state; byte sampling below still uses the speech prefix.
            features,_=causal_prefix(self,targets,0,record)
            # Only the boundary contributes to this categorical derivative.
            # The utterance is scored with its own proposal graph below. A
            # generated payload is replayed here just for the detached audit;
            # a reused payload has no byte audit in this term at all.
            prefix_logits = None
            if record['choice'] == 1 and not reuse_audit:
                with torch.no_grad():
                    _, prefix_logits = causal_prefix(self, targets, own, record)
        else:
            features, prefix_logits = causal_prefix(self, targets, own, record)
        habit, _, _ = self.plan(features[:1])
        combined = habit[0]+habit.new_tensor(record['offset'])
        categorical = combined.log_softmax(-1)[record['choice']]
        replay = dict(choice=combined.detach().tolist(), stops=[])
        if record['choice'] == 1 and own and not reuse_audit:
            replay['byte_logp'] = prefix_logits[0].log_softmax(-1).gather(
                1, targets[0, :own, None]).flatten().detach()
        return categorical, replay

    def _proposal_policy_terms(self, record, *, audit_index=None, replay=None, batch_size=1):
        max_error = 0.
        indexed = list(enumerate(self._joint_proposals))
        if batch_size > 1:
            from scripts.ib_local.variable_expression import proposal_score, proposal_scores_batched
            groups = {}
            others = []
            for index, proposal in indexed:
                if 'handover' in proposal:
                    has_routes = proposal['handover'].get('read_slots') is not None
                    groups.setdefault(has_routes, []).append((index, proposal))
                else:
                    others.append((index, proposal))
            for key, items in groups.items():
                for pos in range(0, len(items), batch_size):
                    chunk = items[pos:pos+batch_size]
                    if len(chunk) == 1:
                        index, proposal = chunk[0]
                        terms, error = proposal_score(self, proposal)
                        max_error = max(max_error, error)
                        if index == audit_index:
                            replay['byte_logp'] = terms[:len(proposal['words'])].detach()
                        yield terms
                        continue
                    chunk_proposals = [p for _, p in chunk]
                    valid_terms, fallbacks, batch_error = proposal_scores_batched(self, chunk_proposals)
                    max_error = max(max_error, batch_error)
                    if valid_terms:
                        for local_k, terms in valid_terms.items():
                            orig_index, proposal = chunk[local_k]
                            if orig_index == audit_index:
                                replay['byte_logp'] = terms[:len(proposal['words'])].detach()
                        yield torch.stack([t.sum() for t in valid_terms.values()]).sum()
                    for local_k in fallbacks:
                        orig_index, proposal = chunk[local_k]
                        terms, error = proposal_score(self, proposal)
                        max_error = max(max_error, error)
                        if orig_index == audit_index:
                            replay['byte_logp'] = terms[:len(proposal['words'])].detach()
                        yield terms
            indexed = others
        for index, proposal in indexed:
            if 'handover' in proposal:
                from scripts.ib_local.variable_expression import proposal_score
                terms,error=proposal_score(self,proposal)
                max_error=max(max_error,error)
                if index == audit_index:
                    replay['byte_logp'] = terms[:len(proposal['words'])].detach()
                yield terms
                continue
            words = torch.tensor(list(proposal['words']), device=self.device, dtype=torch.long)[None]
            _, predicted = causal_prefix(self, words, words.shape[1],
                dict(probs=proposal['probs'], boundary_probability=record['boundary_probability'],
                     expected_byte_logp=proposal['logp']))
            actual = predicted[0].log_softmax(-1).gather(1, words[0, :, None]).flatten()
            sampled = actual.new_tensor(proposal['logp'])
            max_error = max(max_error, float((actual.detach()-sampled).abs().max()))
            yield actual
        if max_error > 1e-3:
            raise RuntimeError(f'joint proposal replay differs: {max_error}')
        self._joint_proposal_replay_error = max_error
        self._policy_replay_fallback = dict(used=True,kind='joint exact one-byte proposal replay',
                                            exact_path_error=max_error)

    def _policy_logp(self, h, logits, targets, own, record):
        """Reference joint scalar, retained for comparisons and older callers."""
        categorical, replay = self._choice_policy_term(targets, own, record)
        proposal_terms = list(self._proposal_policy_terms(record))
        # The finite search selector has no trainable parameters of its own;
        # its actual sampling logp is retained, with no invented gradient.
        return proposal_log_probability(categorical, proposal_terms,
            categorical.new_tensor(self._joint_selection['logp'])), replay

    def _policy_grad(self, targets, own, record):
        """Accumulate exactly the same score gradient one independent graph at a time.

        Each 32-byte proposal is a separate conditional sample of the same
        pre-action state. Their score gradients add; keeping all their recurrent
        autograd graphs alive together is unnecessary. The selection logp is
        detached in both implementations and therefore contributes no gradient.
        """
        audit_index = self._selected_proposal_audit(targets, own, record)
        categorical, replay = self._choice_policy_term(targets, own, record,
                                                       reuse_audit=audit_index is not None)
        result = [None if g is None else g.detach() for g in
                  torch.autograd.grad(categorical, self.params, allow_unused=True)]
        del categorical
        proposal_batch = int(getattr(self, 'policy_proposal_batch_size', 1) or 1)
        for term in self._proposal_policy_terms(record, audit_index=audit_index, replay=replay,
                                                batch_size=proposal_batch):
            grads = torch.autograd.grad(term.sum(), self.params, allow_unused=True)
            for i, grad in enumerate(grads):
                if grad is not None:
                    if result[i] is None:
                        result[i] = grad.detach()
                    else:
                        result[i].add_(grad.detach())
            del grads, term
        return tuple(result), replay

    def step(self, directions, age_ablate=False):
        row = super().step(directions, age_ablate)
        row['joint_language_search'] = dict(self._joint_selection,
            candidates=[v['words'].hex() for v in self._joint_proposals],
            replay_max_error=getattr(self, '_joint_proposal_replay_error', None),
            executed=(row['choice'] == 'generate'),
            policy_credit='joint proposal/selection/action sampling path')
        row['internal_ticks'] = self.joint_internal_ticks
        if getattr(self,'variable_expression_schema',None)==1:
            row['expression']=dict(cap=self.max_utterance,length=row['self_bytes'],
                reason=self._pending_record.get('ending_reason','silence'),
                proposal_lengths=[len(p['words']) for p in self._joint_proposals],
                handover_policy_credit=True)
        row['runtime_replay']=dict(batched_hits=getattr(self,'_batched_replay_hits',0),
            one_byte_fallbacks=getattr(self,'_batched_replay_fallbacks',0))
        return row

    @torch.no_grad()
    def continue_internal(self, ticks=1):
        """Evolve the SAME carried S/P without hearing text or calling the world.

        This is an input-free interface, not a prescribed autonomous cadence.
        No bytes, facts, gains or optimizer steps are fabricated. Outstanding
        real action receipts remain outstanding until real consequences arrive.
        """
        if not isinstance(ticks, int) or not 1 <= ticks <= self.model.memory.topk:
            raise ValueError('internal ticks exceed the finite per-call budget')
        if self.joint_internal_ticks+ticks >= 2**63:
            raise OverflowError('internal event clock exhausted')
        before = self.recurrent
        from scripts.quiet_force_jet import commit_engine
        prepared = commit_engine(self, ticks)
        if prepared is not None:
            self.recurrent, self.bfss_momenta = prepared
        elif hasattr(self.model, 'head_coupling'):
            extra = dict(capture=True) if getattr(self,'adaptive_dynamics_schema',None)==1 else {}
            self.recurrent, self.bfss_momenta = self.model.head_coupling.quiet(
                self.model.body.base, self.recurrent, self.bfss_momenta,
                dt=self.bfss_dt, frequency=self.bfss_frequency, steps=ticks, **extra)
        else:
            from scripts.ib_local.bfss_gdn_capture import quiet_packed
            self.recurrent, self.bfss_momenta = quiet_packed(self.model.body.base,
                self.recurrent, self.bfss_momenta, dt=self.bfss_dt,
                frequency=self.bfss_frequency, coupling=self.bfss_coupling, steps=ticks)
        self.model._momentum_current = self.bfss_momenta
        self.joint_internal_ticks += ticks
        # Evaluate the existing relation program on a fresh internal read.
        # bind() changes applicability/LRU metadata, not observed evidence.
        old_context = getattr(self.model, '_question_context', None)
        self.model._question_context = None
        try:
            self.decision.start(self.recurrent)
            _, h = self.decision.step(self.pending, peek(self.table, self.pending))
            relation = self.model.causal_relations.bind(self.model, h[None])
            momentum = decision_momentum_read(self.model, self.bfss_momenta, h[None])
            from scripts.ib_local import persistent_rhythm
            if persistent_rhythm.enabled(self):
                from scripts.ib_local.causal_relation_program import compose
                preview = compose(self.model, relation, h[None], h[None],
                    ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
                    momentum=momentum)
                preview['expected_after_value'] = persistent_rhythm.internal_utility(
                    self, relation, h[None], h[None], preview)
            else:
                preview = imagine(self.model, relation, h[None], h[None],
                    ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
                    momentum=momentum)
        finally:
            self.model._question_context = old_context
        self.joint_internal_receipt = dict(ticks=self.joint_internal_ticks,
            relation=relation.identity, version=relation.version,
            state_change=float((self.recurrent-before).norm()),
            predicted_after_value=float(preview['expected_after_value'][0]),
            next_step='same continuing external interaction', new_world_evidence=False)
        return copy.deepcopy(self.joint_internal_receipt)


class JointCycleEngine(JointCycleMixin, CausalProgramEngine):
    pass


def configure_joint_cycle(engine, proposal_budget=4):
    if not isinstance(engine, CausalProgramEngine):
        raise TypeError('the existing complete shared-relation body is required')
    if not 1 <= proposal_budget <= engine.model.memory.topk:
        raise ValueError('proposal count exceeds existing finite retrieval budget')
    if hasattr(engine, 'learned_head_kind'):
        engine.__class__ = JointLearnedCycleEngine
    else:
        engine.__class__ = JointCycleEngine
    engine.joint_world_cycle_schema = 1
    engine.joint_proposal_budget = proposal_budget
    if not hasattr(engine, 'joint_internal_ticks'):
        engine.joint_internal_ticks = 0
        engine.joint_internal_receipt = None
    return engine


def migrate_joint_cycle(engine, proposal_budget=4):
    if not isinstance(engine, CausalProgramEngine):
        raise TypeError('the existing complete shared-relation body is required')
    if not 1 <= proposal_budget <= engine.model.memory.topk:
        raise ValueError('proposal count exceeds existing finite retrieval budget')
    if getattr(engine, 'joint_world_cycle_schema', None) is not None:
        return configure_joint_cycle(engine, proposal_budget)
    if engine.settlement_kind != 'continuation_joint' or engine.reward_memory_kind != 'observed':
        raise ValueError('migration requires the declared measured-gain predecessor')
    engine.joint_cycle_archive = dict(settlement=engine.settlement_kind,
        read=copy.deepcopy(engine.read_gain_state), internal=copy.deepcopy(engine.internal_gain_state),
        shadow=copy.deepcopy(engine.shadow_gain_state))
    # Historical facts, parameters, optimizer and pending actual credit stay.
    # New utility calibration does not reinterpret old labels as new scores.
    engine.settlement_kind = 'world_stream'
    for name in ('read_gain_state', 'internal_gain_state', 'shadow_gain_state'):
        setattr(engine, name, gain_state(engine.wood_decay))
    return configure_joint_cycle(engine, proposal_budget)


from scripts.ib_local.navigation_learned_coupling import LearnedCouplingEngine


class JointLearnedCycleEngine(JointCycleMixin, LearnedCouplingEngine):
    pass
