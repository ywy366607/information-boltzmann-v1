"""V3 turn-taking dialogue: the body decides to speak or listen, and when to stop.

Claude, 2026-11-04, continuing GPT's v3 assembly. The per-byte action
opportunity of `language_navigation_stream_v3` let an untrained body start an
utterance at ~50% of bytes, so actions overlapped, every reply was cut off and
no consequence ever arrived (see CGB_NAVIGATION_V3_ASSEMBLY_20261104.md). Here
the world answers in turns. At the end of each turn the body decides from its
own state whether to speak or keep listening (silence = the world continues
reading). While speaking, the body decides after each self-heard byte whether
to stop. The world's turn ends are world events, not a body schedule; the body
may stay silent for any number of turns, and its speech rate is not set.

One cycle = one experience, written once to the experience table after its
long successor has arrived: decision state, state after own utterance, short
and long successors, and the generalization progress when the probe positions
were previously unexposed. Policy credit covers the start decision, every
stop/continue decision and the sampled bytes, so silence is also credited.
The credit is the same stale action-time score gradient as v2 (one update
between action and outcome), not an exact derivative through learning.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_v2 import DialogueEngine
from scripts.ib_local.latent_navigation_v2 import TransitionRead
from scripts.ib_local.train_gdn_fast_table import FLOOR


class DecisionGraphV3:
    """One-byte causal rollout that also returns the injected hidden state."""
    def __init__(self, model, capture=True):
        self.model = model
        device = next(model.parameters()).device
        self.capture = capture and device.type == 'cuda'
        self.state = model.initial_state(1, device=device)
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.table = torch.full((1, 1, 256), -5.54517744448, device=device)
        if self.capture:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    model.predict_observed(self.ids, self.state, self.table)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad():
                self.logits, self.next_state, _, self.h, _ = model.predict_observed(
                    self.ids, self.state, self.table)

    @torch.no_grad()
    def start(self, state):
        self.state.copy_(state)

    @torch.no_grad()
    def step(self, byte, probability):
        self.ids[0, 0] = byte
        self.table.copy_(torch.as_tensor(np.log(np.maximum(probability, FLOOR)),
                                         device=self.table.device)[None, None])
        if self.capture:
            self.graph.replay()
            logits, state, h = self.logits, self.next_state, self.h
        else:
            logits, state, _, h, _ = self.model.predict_observed(self.ids, self.state, self.table)
        self.state.copy_(state)
        return logits[0, 0].clone(), h[0].clone()


class TurnDialogueEngine(DialogueEngine):
    """mode: 'active' (body decides), 'silence' (never speaks), 'sequential'
    (body decides and self-hears, the world ignores the words and reads on)."""
    def __init__(self, model, optimizer, recurrent, table, world, pending, rng,
                 mode='active', max_utterance=32, response_bytes=480, holdout_bytes=480,
                 baseline=0., baseline_decay=.95, capture=True):
        if mode not in ('active', 'silence', 'sequential'):
            raise ValueError('unknown v3 action mode')
        super().__init__(model, optimizer, recurrent, table, world, pending, rng, 'active',
                         max_utterance, response_bytes, holdout_bytes, baseline, baseline_decay)
        self.mode = mode
        self.max_utterance = max_utterance
        self.decision = DecisionGraphV3(model, capture)

    def decision_logits(self, h):
        return self.model.plan_from_state(h)['decision_logits']

    @torch.no_grad()
    def _age_component_ablation(self, ids, targets, start_state, logp, mask):
        """Remove experiences of one age band from the injected recall.

        The experience table injects the recalled action-state (not a
        successor), so the filtered read must return the same kind of value.
        Age is from the episode's decision time (oldest contributor if merged).
        """
        memory = self.model.memory
        base_logits, _, _ = self.model.predict(ids, start_state, logp)
        masked = mask.flatten()
        def mean_ce(logits):
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
            return (ce*masked).sum()/masked.sum().clamp_min(1)
        base = mean_ce(base_logits)
        original = memory.read
        result = {}
        edges = (('0_4k', 0, 4096), ('4k_64k', 4096, 65536), ('64k_256k', 65536, 262144),
                 ('256k_plus', 262144, float('inf')))
        for label, lo, hi in edges:
            mass = []
            def filtered(features, encoder):
                read = original(features, encoder)
                age = (memory.clock - memory.birth[read.slots]).clamp_min(0)
                selected = (age >= lo) & (age < hi)
                weights = read.weights*(~selected)
                mass.append((read.weights*selected).sum(-1))
                value = (weights[..., None]*encoder(memory.action)[read.slots]).sum(-2)
                long_value = (weights[..., None]*encoder(memory.long_successor)[read.slots]).sum(-2)
                return TransitionRead(value, long_value, read.slots, weights)
            memory.read = filtered
            try:
                logits, _, _ = self.model.predict(ids, start_state, logp)
            finally:
                del memory.read
            result[label] = dict(removed_weight=float((mass[0]*masked).sum()/masked.sum().clamp_min(1)),
                                 nll_change=float(mean_ce(logits)-base))
        return result

    @torch.no_grad()
    def _utter(self):
        """Sample start, bytes and stop decisions; return what replay must match."""
        self.decision.start(self.recurrent)
        byte = self.pending
        p = peek(self.table, byte)
        logits, h = self.decision.step(byte, p)
        start_logit = float(self.decision_logits(h[None])[0, 0])
        record = dict(start_logit=start_logit, stop_logits=[], byte_logp=[], probs=[p], truncated=False)
        if self.mode == 'silence':
            record['probs'] = []
            return b'', record
        spoke = bool(torch.rand((), device=h.device) < torch.sigmoid(torch.tensor(start_logit)))
        if not spoke:
            record['probs'] = []
            return b'', record
        utterance = []
        for i in range(self.max_utterance):
            if i:
                stop_logit = float(self.decision_logits(h[None])[0, 1])
                record['stop_logits'].append(stop_logit)
                if bool(torch.rand((), device=h.device) < torch.sigmoid(torch.tensor(stop_logit))):
                    break
            logp = logits.log_softmax(-1)
            chosen = int(torch.multinomial(logp.exp(), 1))
            record['byte_logp'].append(float(logp[chosen]))
            utterance.append(chosen)
            _, observed, _ = self.table.run_credit(np.array([byte, chosen], np.uint8), max_window=1)
            if not np.array_equal(p, observed[0]):
                raise RuntimeError('self-hearing peek differs from prequential table read')
            byte = chosen
            p = peek(self.table, byte)
            record['probs'].append(p)
            logits, h = self.decision.step(byte, p)
        else:
            record['truncated'] = True
        record['probs'] = record['probs'][:len(utterance)]
        return bytes(utterance), record

    def _policy_logp(self, h, logits, targets, own, record):
        """Differentiable log-probability of every decision actually taken.

        Positions: h[0] processes the pending byte (start decision); h[i] for
        1 <= i <= own processes the i-th own byte (stop decision before byte i,
        and after the last byte when the body stopped). Normalized per decision.
        """
        decisions = self.decision_logits(h[:own+1])
        start = decisions[0, 0]
        terms = [F.logsigmoid(start) if own else F.logsigmoid(-start)]
        replay = dict(start=float(start.detach()), stops=[])
        if own:
            # Stop decision k (1-based) reads h[k]; the body stopped iff it
            # evaluated a decision after every own byte, else it was truncated.
            n = len(record['stop_logits'])
            stopped = n == own
            for k in range(1, n+1):
                logit = decisions[k, 1]
                terms.append(F.logsigmoid(logit) if stopped and k == n else F.logsigmoid(-logit))
            replay["stops"] = [float(decisions[k, 1].detach()) for k in range(1, n+1)]
            byte_logp = logits[0, :own].log_softmax(-1).gather(1, targets[0, :own, None]).flatten()
            terms.extend(byte_logp)
            replay['byte_logp'] = byte_logp.detach()
        return torch.stack([t.reshape(()) for t in terms]).mean(), replay

    def step(self, directions, age_ablate=False):
        memory = self.model.memory
        start_state = self.recurrent.clone()
        start_pending = self.pending
        source_time = int(memory.clock)
        utterance, record = self._utter()
        own = len(utterance)
        query = b'' if self.mode == 'sequential' else utterance
        response, heldout, info = self.world.take(query, self.response_bytes, self.holdout_bytes)
        last_action = utterance[-1] if own else start_pending
        _, response_probs, _ = self.table.run_credit(
            np.concatenate((np.array([last_action], np.uint8), response)), max_window=self.response_bytes)
        raw = np.concatenate((np.array([start_pending], np.uint8), np.frombuffer(utterance, np.uint8), response))
        probs = np.concatenate((np.asarray(record['probs'], np.float32).reshape(-1, 256), response_probs), 0)
        ids, targets, logp = self._inputs(raw, probs)
        mask = torch.ones_like(targets, dtype=torch.float32)
        mask[:, :own] = 0
        component_ablation = self._age_component_ablation(ids, targets, start_state, logp, mask) if age_ablate else None

        loss, state_after_response, features, diag = self.model.objective(
            ids, targets, start_state, logp, directions, score_mask=mask)
        logits = diag['logits']
        actor = None
        policy_error = 0.
        if self.mode != 'silence':
            policy, replay = self._policy_logp(features, logits, targets, own, record)
            errors = [abs(replay['start']-record['start_logit'])]
            errors += [abs(a-b) for a, b in zip(replay['stops'], record['stop_logits'])]
            if own:
                errors.append(float((replay['byte_logp'] -
                                     torch.tensor(record['byte_logp'], device=self.device)).abs().max()))
            policy_error = max(errors)
            if policy_error > 1e-3:
                raise RuntimeError(f'sampled decisions and causal replay differ: {policy_error}')
            actor = torch.autograd.grad(policy, self.params, retain_graph=True, allow_unused=True)
            actor = tuple(None if g is None else g.detach().clone() for g in actor)
        actor_norm = 0. if actor is None else float(torch.stack(
            [g.norm() for g in actor if g is not None]).norm())

        probe_raw = np.concatenate((response[-1:], heldout))
        _, probe_probs, _ = self.table.run_credit(probe_raw, max_window=self.holdout_bytes)
        probe_ids, probe_targets, probe_logp = self._inputs(probe_raw, probe_probs)
        with torch.no_grad():
            pre_logits, _, _ = self.model.predict(probe_ids, state_after_response.detach(), probe_logp)
            pre = float(F.cross_entropy(pre_logits.flatten(0, 1), probe_targets.flatten()))
        h_cycle = [features.detach()]
        response_norm = self._advance(loss, features, diag, state_after_response)
        after_loss, after_state, after_features, after_diag = self.model.objective(
            probe_ids, probe_targets, self.recurrent, probe_logp, directions)
        post = float(after_diag['ce'])
        progress = pre-post
        fresh = bool(info['holdout_position_fresh'])
        usable = progress if fresh else 0.
        advantage = usable-self.baseline
        heldout_norm = self._advance(after_loss, after_features, after_diag, after_state,
                                     actor=actor, advantage=advantage)
        baseline_before = self.baseline
        self.baseline = self.baseline_decay*self.baseline + (1-self.baseline_decay)*usable
        self.pending = int(heldout[-1])

        # One experience: written once, after its long successor has arrived.
        h_cycle.append(after_features.detach())
        timeline = torch.cat(h_cycle, 0)
        short_at = min(own+memory.short, len(timeline)-1)
        long_at = min(memory.long, len(timeline)-1)
        slot, merged = memory.commit_episode(
            timeline[0], timeline[own], timeline[short_at], timeline[long_at], self.model.encoder,
            progress=progress if fresh else None, source_time=source_time)
        try:
            utterance.decode('utf-8'); utf8_valid = True
        except UnicodeDecodeError:
            utf8_valid = False
        return dict(mode=self.mode, utterance_hex=utterance.hex(), utf8_valid=utf8_valid, spoke=own > 0,
            start_probability=float(torch.sigmoid(torch.tensor(record['start_logit']))),
            truncated=record['truncated'],
            response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
            external_bytes=len(response)+len(heldout), self_bytes=own, world=info,
            response_nll=float(diag['ce']), heldout_pre_nll=pre, heldout_post_nll=post,
            progress=progress, fresh=fresh, usable_signal=usable,
            baseline_before=baseline_before, advantage=advantage, actor_update_used=actor is not None,
            actor_norm=actor_norm, policy_replay_max_error=policy_error,
            short_loss=float(diag['latent']), long_loss=float(diag['latent_long']),
            episode_slot=slot, episode_merged=merged, age_component_ablation=component_ablation,
            gradient_norms=[response_norm, heldout_norm], updates=2)
