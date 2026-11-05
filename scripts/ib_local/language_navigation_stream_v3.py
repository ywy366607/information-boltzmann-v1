"""Inference-only event loop connecting one GDN body to a streaming world.

There is no optimizer or policy update here. It is a connectivity scaffold for
the user's pre-training gate. Observed consequences can be stored; progress
and delayed policy credit remain unwired and must not be reported as passed.
"""
from __future__ import annotations

from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_action import _peek, peek
from scripts.ib_local.train_gdn_fast_table import FLOOR


class StreamingIndividual:
    def __init__(self, model, table, world, recurrent, pending, *, max_utterance=32, max_pending=None):
        if max_pending is None:
            # At most one action can start per external event. Matching the
            # external long horizon therefore admits all overlapping events.
            max_pending = model.memory.long
        if max_utterance <= 0 or max_pending <= 0:
            raise ValueError('bounded positive capacities required')
        self.model, self.table, self.world = model, table, world
        self.recurrent, self.pending = recurrent, int(pending)
        self.max_utterance, self.max_pending = max_utterance, max_pending
        self.open_experiences = []
        self.dropped_experiences = 0
        self.overlapping_action_events = 0
        self.events = 0
        self.external_bytes = 0
        self.self_bytes = 0
        self.utterances = deque(maxlen=256)
        self.world_actions = deque(maxlen=256)
        self.progress_receipts = deque(maxlen=256)
        self.progress_count = 0
        self.probe_candidates = 0
        self.probe_interrupted = 0
        self.probe_previously_seen = 0
        self.probe_post_scored = 0
        self.truncated = 0

    @torch.no_grad()
    def _score_probe(self, recurrent, pending, history, probe):
        """Frozen-context score, privately evaluated without changing body/world.

        The evaluator may inspect source bytes, but the score is never used by
        the agent until all probe bytes have actually arrived contiguously.
        The byte table is read-only; model recurrent context is held fixed for
        the before/after comparison. It is not a full-context likelihood.
        """
        inputs = np.concatenate((np.array([pending], np.uint8), probe[:-1]))
        probs = []
        state = int(history)
        orders = np.asarray(self.table.orders, np.int64)
        for byte in inputs:
            probs.append(_peek(byte, np.uint64(state), self.table.rows, self.table.count,
                               self.table.tag, self.table.err, orders, self.table.heads,
                               1 << self.table.slots_log2, self.table.prior_weight, self.table.err_floor))
            state = ((state << 8) | int(byte)) & ((1 << 64)-1)
        device = next(self.model.parameters()).device
        ids = torch.as_tensor(inputs.astype(np.int64), device=device)[None]
        logp = torch.as_tensor(np.log(np.maximum(np.asarray(probs), FLOOR)), device=device)[None]
        targets = torch.as_tensor(probe.astype(np.int64), device=device)
        logits, _, _ = self.model.predict(ids, recurrent, logp)
        return float(F.cross_entropy(logits.flatten(0, 1), targets))

    @torch.no_grad()
    def _frame(self):
        device = next(self.model.parameters()).device
        p = peek(self.table, self.pending)
        ids = torch.tensor([[self.pending]], device=device)
        logp = torch.as_tensor(np.log(np.maximum(p, FLOOR)), device=device)[None, None]
        logits, state, _, features, receipt = self.model.predict_observed(ids, self.recurrent, logp)
        return logits[0, 0], state, features, receipt, p

    @torch.no_grad()
    def _commit(self, frame, byte, external, source_position=None):
        _, state, features, receipt, probability = frame
        _, actual, _ = self.table.run_credit(np.array([self.pending, byte], np.uint8), max_window=1)
        if not np.array_equal(probability, actual[0]):
            raise RuntimeError('table read is not aligned with the observed target')
        self.model.commit_observed(features, receipt)
        self.recurrent = state.detach()
        self.pending = int(byte)
        self.events += 1
        if external:
            self.external_bytes += 1
            complete = []
            for episode in self.open_experiences:
                episode['age'] += 1
                span = self.model.memory.short
                if (episode['probe_valid'] and episode['age'] <= 2*span and
                        source_position != episode['response_start'] + episode['age'] - 1):
                    episode['probe_valid'] = False
                    self.probe_interrupted += 1
                if episode['age'] == span and episode['probe_valid']:
                    start = episode['response_start'] + span
                    if self.world.seen[start:start+span].any():
                        episode['probe_valid'] = False
                        self.probe_previously_seen += 1
                    else:
                        episode['post_score'] = self._score_probe(episode['probe_state'], episode['probe_pending'],
                            episode['probe_history'], episode['probe'])
                        self.probe_post_scored += 1
                if episode['age'] == 2*span and episode['probe_valid'] and episode['post_score'] is not None:
                    episode['progress'] = episode['pre_score'] - episode['post_score']
                    self.progress_count += 1
                    self.progress_receipts.append(dict(progress=episode['progress'],
                        eligibility=episode['eligibility'], bias_eligibility=episode['bias_eligibility']))
                # h describes the causal state used to score this byte; the
                # recorded horizon is therefore the pre-observation state.
                if episode['age'] == self.model.memory.short:
                    episode['short'] = features[0].detach().clone()
                if episode['age'] == self.model.memory.long:
                    self.model.memory.commit_episode(
                        episode['source'], episode['action'], episode['short'],
                        features[0].detach(), self.model.encoder,
                        source_time=episode['source_time'], progress=episode['progress'],
                        intervening=episode['intervening'], intervention_count=episode['intervention_count'])
                    complete.append(episode)
            complete_ids = {id(episode) for episode in complete}
            self.open_experiences = [episode for episode in self.open_experiences if id(episode) not in complete_ids]
        else:
            self.self_bytes += 1

    @torch.no_grad()
    def step(self):
        """One external byte plus any sampled utterance before that byte."""
        frame = self._frame()
        source = frame[2][0].detach().clone()
        source_time = int(self.model.memory.clock)
        initial_plan = self.model.plan_from_state(source[None])
        start_prob = initial_plan['decision_logits'][0, 0].sigmoid()
        start = torch.bernoulli(start_prob)
        utterance = []
        if bool(start):
            for i in range(self.max_utterance):
                stop = torch.bernoulli(self.model.plan_from_state(frame[2])['decision_logits'][0, 1].sigmoid())
                if bool(stop):
                    break
                byte = int(torch.multinomial(frame[0].softmax(-1), 1))
                self._commit(frame, byte, external=False)
                utterance.append(byte)
                frame = self._frame()
            else:
                self.truncated += 1
        if utterance:
            # The current frame consumed the last self-heard byte; it is the
            # action-state and also predicts the first response byte.
            action = frame[2][0].detach().clone()
            # Mark the observed joint history in each still-open experience.
            # The last body's action-state is a bounded compression, not a
            # proof that all intervening histories remain distinguishable.
            self.overlapping_action_events += len(self.open_experiences)
            for episode in self.open_experiences:
                episode['intervention_count'] += 1
                episode['intervening'] = action
            action_event = self.world.act(bytes(utterance))
            self.world_actions.append(action_event)
            if len(self.open_experiences) == self.max_pending:
                self.open_experiences.pop(0)
                self.dropped_experiences += 1
            span = self.model.memory.short
            probe_start = action_event['start'] + span
            probe_valid = (probe_start + span <= len(self.world.data) and
                           not self.world.seen[probe_start:probe_start+span].any())
            if probe_valid:
                self.probe_candidates += 1
            else:
                self.probe_previously_seen += 1
            probe = self.world.data[probe_start:probe_start+span].copy() if probe_valid else None
            probe_state = self.recurrent.clone() if probe_valid else None
            probe_history = int(self.table.history)
            pre_score = self._score_probe(probe_state, self.pending, probe_history, probe) if probe_valid else None
            score_factor = float(1-start_prob)
            self.open_experiences.append(dict(source=source, action=action,
                                              source_time=source_time, age=0, short=None,
                                              intervening=None, intervention_count=0,
                                              response_start=action_event['start'], probe_valid=probe_valid,
                                              probe=probe, probe_state=probe_state, probe_history=probe_history,
                                              probe_pending=self.pending, pre_score=pre_score, post_score=None,
                                              progress=None,
                                              eligibility=(score_factor*initial_plan['decision_feature'][0]).detach().clone(),
                                              bias_eligibility=score_factor))
        self.utterances.append(bytes(utterance))
        incoming, observation = self.world.observe(1)
        self._commit(frame, int(incoming[0]), external=True, source_position=observation['start'])
        return bytes(utterance)
