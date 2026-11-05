"""V4 stage 1, 取坎填离 / 水火既济: action and experience memory exchange (Claude, 2026-11-04).

Order (user, 11-04): first make water and fire meet, then tune metal and wood.
V3 was 未济: speech went out and never came back (no consequence, 1-2 byte
babble = 离中虚), the experience table stored 2500 entries and none rose into
action (坎中阳 never taken). This stage changes only the water-fire axis:

* Fire descends. The world's passive supply is bounded (`TopicWorld`): when
  the body is silent, or speaks and is not understood (no indexed 4-gram), the
  partner keeps talking around the current topic, a 4096-byte window read
  cyclically; when understood, the partner moves to a new place. This is the
  `passiveBounded` premise of `GenerationActiveInference.active_closure`.
  Every experience is stored WITH the words said; an action-conditioned
  outcome model F(source, action) -> landing is trained on real experiences
  (current one and a replay batch from the table, re-encoded now).
* Water rises. The body may say again the words of a recalled experience that
  once brought progress from a similar state (`reuse`), next to `silence` and
  `generate`. Recall thereby changes WHAT is said, not only whether.

Metal and wood are untouched on purpose: the choice among the three is still
the weak REINFORCE of v3 (stale action-time score gradient after one update).
Imagined-progress selection is the next stage, after this axis is seen to work.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ib_local.language_action import peek
from scripts.ib_local.language_navigation_turns_v3 import TurnDialogueEngine
from scripts.ib_local.language_navigation_world_v3 import BroadLanguageWorld, random_best
from scripts.ib_local.latent_navigation_v3 import ExperienceMemory

SILENCE, GENERATE, REUSE = 0, 1, 2


class TopicWorld(BroadLanguageWorld):
    """Bounded passive supply: a topic window read cyclically until understood speech moves it."""
    def __init__(self, data, window=4096, **kw):
        super().__init__(data, **kw)
        if not 0 < window < len(data):
            raise ValueError('topic window must fit the source')
        self.window, self.anchor, self.offset = int(window), int(self.cursor), 0
        self.moves = 0

    def state_dict(self):
        state = super().state_dict()
        state.update(anchor=self.anchor, offset=self.offset, moves=self.moves)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.anchor, self.offset, self.moves = state['anchor'], state['offset'], state['moves']

    def take(self, utterance, response_bytes, holdout_bytes):
        required = response_bytes+holdout_bytes
        if required > self.window:
            raise ValueError('one turn must fit the topic window')
        score = 0
        if utterance:
            query = np.frombuffer(utterance, np.uint8)
            draw = int(self.rng.integers(0, 2**32, dtype=np.uint64))
            start, score = random_best(query, self.data, self.positions, self.window, -2, draw)
            if score > 0:
                self.anchor, self.offset = int(start), 0
                self.moves += 1
        positions = self.anchor + (self.offset + np.arange(required)) % self.window
        self.offset = (self.offset+required) % self.window
        prior_response = int(self.seen[positions[:response_bytes]].sum())
        prior_holdout = int(self.seen[positions[response_bytes:]].sum())
        self.seen[positions] = True
        self.cursor = int(positions[-1])+1
        return (self.data[positions[:response_bytes]].copy(), self.data[positions[response_bytes:]].copy(),
                dict(start=int(positions[0]), anchor=self.anchor, match=int(score),
                     response_previously_seen_bytes=prior_response,
                     holdout_previously_seen_bytes=prior_holdout,
                     holdout_position_fresh=(prior_holdout == 0),
                     source='matched' if score else 'topic'))


class UtteranceMemory(ExperienceMemory):
    """Experience table that also keeps the words of each experience."""
    def __init__(self, *args, max_utterance=32, **kw):
        super().__init__(*args, **kw)
        self.max_utterance = max_utterance
        self.register_buffer('words', torch.zeros(self.capacity, max_utterance, dtype=torch.uint8))
        self.register_buffer('word_count', torch.zeros(self.capacity, dtype=torch.long))

    @torch.no_grad()
    def commit_experience(self, source, action, short, long, encoder, utterance, progress, source_time):
        slot, merged = self.commit_episode(source, action, short, long, encoder,
                                           progress=progress, source_time=source_time)
        # A merged slot keeps the words that brought at least as much progress.
        if utterance and (not merged or progress >= float(self.progress[slot])):
            n = min(len(utterance), self.max_utterance)
            self.words[slot].zero_()
            self.words[slot, :n] = torch.as_tensor(np.frombuffer(utterance[:n], np.uint8), device=self.words.device)
            self.word_count[slot] = n
        elif not utterance and not merged:
            self.word_count[slot] = 0
        return slot, merged


class KanLiHeads(nn.Module):
    """Three-way action choice and the action-conditioned outcome model."""
    def __init__(self, latent_dim):
        super().__init__()
        self.choice = nn.Linear(4*latent_dim+1, 3)
        self.outcome = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim))


def attach(model, seed=23, max_utterance=32):
    """Replace the experience table by one that keeps words; add the heads."""
    old = model.memory
    memory = UtteranceMemory(old.width, old.capacity, old.short, old.long, old.topk, old.threshold,
                             old.temperature, old.mode, old.history_size, max_utterance=max_utterance)
    model.memory = memory.to(old.source.device)
    latent = model.encoder.out_features
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model.kanli = KanLiHeads(latent).to(old.source.device)
    return model


class KanLiEngine(TurnDialogueEngine):
    def __init__(self, *args, replay_batch=32, outcome_weight=.1, **kw):
        super().__init__(*args, **kw)
        self.replay_batch, self.outcome_weight = replay_batch, outcome_weight
        self.replay_rng = np.random.default_rng(97)

    def plan(self, h):
        """Choice logits from the decision state, its recall and the recalled words' value."""
        plan = self.model.plan_from_state(h)
        memory = self.model.memory
        reusable = ((memory.word_count[plan['slots']] > 0) & (memory.progress[plan['slots']] > 0) &
                    memory.occupied[plan['slots']])
        reuse_weight = (plan['weights']*reusable).sum(-1, keepdim=True)
        logits = self.model.kanli.choice(torch.cat((plan['decision_feature'], plan['action'], reuse_weight), -1))
        logits = torch.cat((logits[..., :2], logits[..., 2:].masked_fill(reuse_weight <= 0, -1e4)), -1)
        return logits, plan, reusable

    @torch.no_grad()
    def _speak_bytes(self, utterance_iter, record, sample):
        """Self-hear bytes (sampled or given), keeping table reads aligned with replay."""
        byte = self.pending
        p = peek(self.table, byte)
        logits, h = self.decision.step(byte, p)
        words = []
        for i in range(self.max_utterance):
            if sample:
                if i:
                    stop_logit = float(self.decision_logits(h[None])[0, 1])
                    record['stop_logits'].append(stop_logit)
                    if bool(torch.rand((), device=h.device) < torch.sigmoid(torch.tensor(stop_logit))):
                        break
                logp = logits.log_softmax(-1)
                chosen = int(torch.multinomial(logp.exp(), 1))
                record['byte_logp'].append(float(logp[chosen]))
            else:
                if i >= len(utterance_iter):
                    break
                chosen = int(utterance_iter[i])
            record['probs'].append(p)
            words.append(chosen)
            _, observed, _ = self.table.run_credit(np.array([byte, chosen], np.uint8), max_window=1)
            if not np.array_equal(p, observed[0]):
                raise RuntimeError('self-hearing peek differs from prequential table read')
            byte = chosen
            p = peek(self.table, byte)
            logits, h = self.decision.step(byte, p)
        else:
            record['truncated'] = sample
        return bytes(words)

    @torch.no_grad()
    def _utter(self):
        self.decision.start(self.recurrent)
        p = peek(self.table, self.pending)
        _, h = self.decision.step(self.pending, p)
        logits, plan, reusable = self.plan(h[None])
        record = dict(choice_logits=logits[0].tolist(), stop_logits=[], byte_logp=[], probs=[],
                      truncated=False, choice=SILENCE, reused_slot=None)
        if self.mode == 'silence':
            return b'', record
        choice = int(torch.multinomial(logits[0].softmax(-1), 1))
        record['choice'] = choice
        if choice == SILENCE:
            return b'', record
        self.decision.start(self.recurrent)
        if choice == GENERATE:
            return self._speak_bytes(None, record, True), record
        weights = (plan['weights'][0]*reusable[0])
        k = int(torch.multinomial(weights/weights.sum(), 1))
        slot = int(plan['slots'][0, k])
        record['reused_slot'] = slot
        memory = self.model.memory
        words = memory.words[slot, :int(memory.word_count[slot])].cpu().numpy()
        return self._speak_bytes(words, record, False), record

    def _policy_logp(self, h, logits, targets, own, record):
        choice_logits, _, _ = self.plan(h[:1])
        logp_choice = choice_logits[0].log_softmax(-1)
        terms = [logp_choice[record['choice']]]
        replay = dict(choice=choice_logits[0].detach().tolist(), stops=[])
        if record['choice'] == GENERATE and own:
            decisions = self.decision_logits(h[:own+1])
            n = len(record['stop_logits'])
            stopped = n == own
            for k in range(1, n+1):
                logit = decisions[k, 1]
                terms.append(F.logsigmoid(logit) if stopped and k == n else F.logsigmoid(-logit))
            replay['stops'] = [float(decisions[k, 1].detach()) for k in range(1, n+1)]
            byte_logp = logits[0, :own].log_softmax(-1).gather(1, targets[0, :own, None]).flatten()
            terms.extend(byte_logp)
            replay['byte_logp'] = byte_logp.detach()
        return torch.stack([t.reshape(()) for t in terms]).mean(), replay

    def _outcome_loss(self, source, action, long):
        """F(source, action) -> landing on the current experience plus a replay batch."""
        memory, enc, outcome = self.model.memory, self.model.encoder, self.model.kanli.outcome
        occupied = torch.nonzero(memory.occupied).flatten()
        if len(occupied):
            pick = occupied[torch.as_tensor(self.replay_rng.integers(0, len(occupied), self.replay_batch),
                                            device=occupied.device)]
            src = torch.cat((source[None], memory.source[pick]))
            act = torch.cat((action[None], memory.action[pick]))
            lng = torch.cat((long[None], memory.long_successor[pick]))
        else:
            src, act, lng = source[None], action[None], long[None]
        target = enc(lng).detach()
        zs, za = enc(src), enc(act)
        error = (outcome(torch.cat((zs, za), -1))-target).square().mean(-1)
        with torch.no_grad():
            # Action sensitivity: the same sources with actions shuffled.
            shuffled = (outcome(torch.cat((zs, za.roll(1, 0)), -1))-target).square().mean(-1)
            blind = (outcome(torch.cat((zs, zs), -1))-target).square().mean(-1)
        return error.mean(), dict(outcome_error=float(error.mean()), outcome_error_current=float(error[0]),
                                  outcome_error_shuffled=float(shuffled.mean()),
                                  outcome_error_as_silent=float(blind.mean()))

    def step(self, directions, age_ablate=False):
        memory = self.model.memory
        start_state = self.recurrent.clone()
        start_pending = self.pending
        source_time = int(memory.clock)
        utterance, record = self._utter()
        own = len(utterance)
        response, heldout, info = self.world.take(utterance, self.response_bytes, self.holdout_bytes)
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
        actor, policy_error, actor_norm = None, 0., 0.
        if self.mode != 'silence':
            policy, replay = self._policy_logp(features, logits, targets, own, record)
            errors = [max(abs(a-b) for a, b in zip(replay['choice'], record['choice_logits'])
                          if abs(b) < 1e3)]
            errors += [abs(a-b) for a, b in zip(replay['stops'], record['stop_logits'])]
            if record['choice'] == GENERATE and own:
                errors.append(float((replay['byte_logp'] -
                                     torch.tensor(record['byte_logp'], device=self.device)).abs().max()))
            policy_error = max(errors)
            if policy_error > 1e-3:
                raise RuntimeError(f'sampled decisions and causal replay differ: {policy_error}')
            actor = torch.autograd.grad(policy, self.params, retain_graph=True, allow_unused=True)
            actor = tuple(None if g is None else g.detach().clone() for g in actor)
            actor_norm = float(torch.stack([g.norm() for g in actor if g is not None]).norm())
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
        timeline = torch.cat(h_cycle+[after_features.detach()], 0)
        short_at = min(own+memory.short, len(timeline)-1)
        long_at = min(memory.long, len(timeline)-1)
        source, action, long = timeline[0], timeline[own], timeline[long_at]
        outcome_loss, outcome_diag = self._outcome_loss(source, action, long)
        post = float(after_diag['ce'])
        progress = pre-post
        fresh = bool(info['holdout_position_fresh'])
        usable = progress if fresh else 0.
        advantage = usable-self.baseline
        heldout_norm = self._advance(after_loss+self.outcome_weight*outcome_loss, after_features, after_diag,
                                     after_state, actor=actor, advantage=advantage)
        baseline_before = self.baseline
        self.baseline = self.baseline_decay*self.baseline + (1-self.baseline_decay)*usable
        self.pending = int(heldout[-1])
        # Fire descends: the experience is stored with its words and its outcome.
        # Its label is the usable progress (zero when the probe was a revisit).
        slot, merged = memory.commit_experience(source, action, timeline[short_at], long, self.model.encoder,
                                                utterance, usable, source_time)
        try:
            utterance.decode('utf-8'); utf8_valid = True
        except UnicodeDecodeError:
            utf8_valid = False
        probs3 = torch.tensor(record['choice_logits']).softmax(-1).tolist()
        return dict(mode=self.mode, choice=('silence', 'generate', 'reuse')[record['choice']],
            choice_probability=probs3, reused_slot=record['reused_slot'],
            utterance_hex=utterance.hex(), utf8_valid=utf8_valid, spoke=own > 0,
            truncated=record['truncated'], external_bytes=len(response)+len(heldout), self_bytes=own,
            world=info, response_nll=float(diag['ce']), heldout_pre_nll=pre, heldout_post_nll=post,
            progress=progress, fresh=fresh, usable_signal=usable, baseline_before=baseline_before,
            advantage=advantage, actor_norm=actor_norm, policy_replay_max_error=policy_error,
            short_loss=float(diag['latent']), long_loss=float(diag['latent_long']), **outcome_diag,
            episode_slot=slot, episode_merged=merged, age_component_ablation=component_ablation,
            gradient_norms=[response_norm, heldout_norm], updates=2)
