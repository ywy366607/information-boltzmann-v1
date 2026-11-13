"""Joint world-learning, concrete language search and input-free continuation.

Reuses the carried body, relation identities, U, S/P, actor and experience
stores. New interfaces have explicit finite budgets and no invented objective,
question label, world answer or self-preservation set point.
"""
from __future__ import annotations

import copy
import torch

from scripts.ib_local.causal_relation_program import imagine
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_causal_program import CausalProgramEngine
from scripts.ib_local.navigation_direct_read import causal_prefix, gain_state
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.table_branch import table_branch


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
        candidates, endpoints = [], []
        # All proposed words start at the SAME real state and fast-table prefix.
        # None of the proposed observations are committed as world evidence.
        for _ in range(self.joint_proposal_budget):
            with table_branch(self.table) as branch:
                result = super()._film(None, branch)
            endpoint = self._draft_actions.pop()
            # The inherited causal-address path also appends a word candidate.
            # Keep the outer silence/generate/reuse axis at exactly three.
            self._causal_words.pop()
            candidates.append(result)
            endpoints.append(endpoint)
        self.decision.start(self.recurrent)
        _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        actions = torch.stack(endpoints)
        source = h[None].expand_as(actions)
        momentum = decision_momentum_read(self.model, self.bfss_momenta, h[None]).expand(len(actions), -1)
        forecast = imagine(self.model, self._program_question_state, source, actions,
            ridge=self.relation_ridge, reward_valid=self.reward_label_valid(wood=True),
            momentum=momentum)
        logits = self.precision()*forecast['expected_after_value']
        logp = logits.log_softmax(0)
        selected = int(torch.multinomial(logp.exp(), 1))
        self._joint_proposals = [dict(words=v[0], logp=v[1], probs=v[2]) for v in candidates]
        self._joint_selection = dict(selected=selected, logp=float(logp[selected]),
            distribution=logp.exp().cpu().tolist(),
            predicted_after_value=forecast['expected_after_value'].cpu().tolist(),
            relation_identity=self._program_question_state.identity,
            relation_version=self._program_question_state.version)
        self._draft_actions.append(endpoints[selected])
        self._causal_words.append(candidates[selected][0])
        return candidates[selected]

    def _choice_policy_term(self, targets, own, record):
        # Recompute the actual one-byte decision prefix. Do not subtract a
        # detached byte logp from the inherited mean: that would count the
        # selected proposal gradient twice.
        features, prefix_logits = causal_prefix(self, targets, own, record)
        habit, _, _ = self.plan(features[:1])
        combined = habit[0]+habit.new_tensor(record['offset'])
        categorical = combined.log_softmax(-1)[record['choice']]
        replay = dict(choice=combined.detach().tolist(), stops=[])
        if record['choice'] == 1 and own:
            replay['byte_logp'] = prefix_logits[0].log_softmax(-1).gather(
                1, targets[0, :own, None]).flatten().detach()
        return categorical, replay

    def _proposal_policy_terms(self, record):
        max_error = 0.
        for proposal in self._joint_proposals:
            words = torch.tensor(list(proposal['words']), device=self.device, dtype=torch.long)[None]
            _, predicted = causal_prefix(self, words, words.shape[1],
                dict(probs=proposal['probs'], boundary_probability=record['boundary_probability']))
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
        categorical, replay = self._choice_policy_term(targets, own, record)
        result = [None if g is None else g.detach() for g in
                  torch.autograd.grad(categorical, self.params, allow_unused=True)]
        del categorical
        for term in self._proposal_policy_terms(record):
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
        if hasattr(self.model, 'head_coupling'):
            self.recurrent, self.bfss_momenta = self.model.head_coupling.quiet(
                self.model.body.base, self.recurrent, self.bfss_momenta,
                dt=self.bfss_dt, frequency=self.bfss_frequency, steps=ticks)
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
