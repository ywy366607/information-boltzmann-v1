"""V4 stage 8: water's metal digests in quiet (Claude, 2026-11-04).

User, 11-04: meditation and sleep both digest the mess in the head and make
information more regular. This is water's metal, not the big metal, and it
is the most important metal. Literature checked the same day:
* Sleep triage: consolidate what matters for the future, forget the rest
  (Stickgold & Walker 2013; Feld & Born 2017).
* Replay priority = gain x need (Mattar & Daw 2018). Gain is how much a
  backup would improve the choice at that state. Need is how soon and how
  often that state will be visited again. The authors name a broken priority
  as a candidate for rumination. That is stage 6's noisy-TV failure, which
  prioritized by error alone.

On top of stage 5b with the reply world. Stage 7's films failed: no film
feature separated exhausted from new turns or predicted the return (see the
doc, film_feature_probe). The critic inputs are stage 5b's, whatever their width.

* Each experience keeps what wood saw at decision time: the chosen option's
  critic inputs, and the probability pi with which that option was chosen.
  Its label is the settled TD return y (stage 4).
* Gain_i = precision * pi_i (1 - pi_i) * delta_i^2, where
  delta_i = y_i - V(inputs_i) under the current critic. This is the
  first-order improvement of a softmax choice whose option value moves by
  delta.
* Need_i = the body's own recall weight of experience i from the present
  state: softmax(cos(z_now, key_i) / T), with the recall temperature T
  already used by the plan.
* Replay: one critic step on the experience of highest gain x need, then the
  priorities are recomputed. Replay continues while some stored experience
  offers more than the present one did, i.e. the experience just settled,
  scored the same way. A quiet present (well predicted, small delta) lets
  much replay through; a surprising present lets little. DIGEST_LIMIT is only
  a compute bound, and how often it binds is reported.
* Judge: a disjoint uniform sample of other settled experiences, never
  replayed in that cycle, measures whether the replays helped experiences
  that were not replayed (transfer, not rumination).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4b import SCALE
from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine

DIGEST_LIMIT = 16  # compute bound only


class DigestEngine(EqualEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        memory = self.model.memory
        memory.register_buffer('wood_inputs', None)  # width set by the first record
        memory.register_buffer('wood_pi', torch.zeros(memory.capacity, device=self.device))
        memory.register_buffer('wood_valid', torch.zeros(memory.capacity, dtype=torch.bool, device=self.device))
        self.digest_rng = torch.Generator(device=self.device).manual_seed(181)
        self.replayed = torch.zeros(memory.capacity, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _priority(self, slots, z_now):
        memory, critic, enc = self.model.memory, self.model.kanli.critic, self.model.encoder
        inputs = memory.wood_inputs[slots]
        delta = SCALE*memory.progress[slots]-(critic(inputs).squeeze(-1)+inputs[:, -2])
        pi = memory.wood_pi[slots]
        gain = self.precision()*pi*(1-pi)*delta.square()
        keys = F.normalize(enc(memory.source), dim=-1)
        sim = keys @ F.normalize(z_now, dim=-1)
        sim = sim.masked_fill(~memory.occupied, -1e4)
        need = (sim/memory.temperature).softmax(-1)[slots]
        return gain*need, delta

    def _critic_error(self, slots):
        memory, critic = self.model.memory, self.model.kanli.critic
        inputs = memory.wood_inputs[slots]
        return float((critic(inputs).squeeze(-1)+inputs[:, -2]-SCALE*memory.progress[slots]).square().mean())

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        memory, record = self.model.memory, self._pending_record
        slot, choice = result['episode_slot'], record['choice']
        prev = result['td_previous']
        info = dict(replays=0, hit_limit=False, priority_present=None)
        # The experience just settled (cycle t-1) is the present that replay must beat.
        settled_slot = self._settled_slot
        self._settled_slot = slot
        if self.mode != 'silence':
            if memory.wood_inputs is None:
                memory.wood_inputs = torch.zeros(memory.capacity, record['critic_inputs'].shape[-1], device=self.device)
            memory.wood_inputs[slot] = record['critic_inputs'][choice].detach()
            memory.wood_pi[slot] = float(torch.tensor(record['choice_logits']).softmax(-1)[choice])
            memory.wood_valid[slot] = True
        candidates = torch.nonzero(memory.occupied & memory.wood_valid).flatten()
        candidates = candidates[candidates != slot]  # its label is still provisional
        if prev is None or settled_slot is None or len(candidates) < 2 or self.precision() == 0:
            result['digest'] = info
            return result
        with torch.no_grad():
            z_now = self.model.encoder(self.recurrent_feature())
        present, _ = self._priority(torch.tensor([settled_slot], device=self.device), z_now)
        info['priority_present'] = float(present[0])
        others = candidates[candidates != settled_slot]
        judge = others[torch.randperm(len(others), generator=self.digest_rng, device=self.device)[:min(32, len(others)//2)]]
        pool = others[~torch.isin(others, judge)]
        if not len(pool):
            result['digest'] = info
            return result
        j0 = self._critic_error(judge) if len(judge) else None
        critic = self.model.kanli.critic
        picked = []
        for _ in range(DIGEST_LIMIT):
            priority, _ = self._priority(pool, z_now)
            best = int(priority.argmax())
            if float(priority[best]) <= info['priority_present']:
                break
            s = pool[best:best+1]
            inputs = memory.wood_inputs[s]
            loss = F.mse_loss(critic(inputs).squeeze(-1)+inputs[:, -2], SCALE*memory.progress[s])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            self.optimizer.step()
            picked.append(int(s))
            info['replays'] += 1
        info['hit_limit'] = info['replays'] == DIGEST_LIMIT
        if picked:
            # Advanced indexed += drops duplicate indices: count every replay,
            # including repeated selection of the same episode in one cycle.
            self.replayed.add_(torch.bincount(
                torch.tensor(picked, device=self.device), minlength=memory.capacity))
            info['distinct'] = len(set(picked))
            info['replayed_age_median'] = float((int(memory.clock)-memory.birth[torch.tensor(picked, device=self.device)]).float().median())
            if j0 is not None:
                with torch.no_grad():
                    info['judge_before'], info['judge_after'] = j0, self._critic_error(judge)
        result['digest'] = info
        return result

    _settled_slot = None

    def recurrent_feature(self):
        """Feature of the state the body now stands in (same read as the plan)."""
        from scripts.ib_local.language_action import peek
        with torch.no_grad():
            self.decision.start(self.recurrent)
            _, h = self.decision.step(self.pending, peek(self.table, self.pending))
        return h
