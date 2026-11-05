"""V4 stage 6: water overflows into wood (Claude, 2026-11-04).

User, 11-04: "枢始得其环中，以应无穷". The loop has no fixed start; only water
has stores it can take things from, so water is the endogenous drive and must
itself inject into wood. "水满则溢": overflow comes from fullness, not from a
schedule. What overflows is seeds, not words. The finished words went
straight to fire and skipped wood.

On top of stage 5b (equal-length speech):
* Water level. Each experience's unresolved part is e_i, the current outcome
  model's error on it: ||F(z_src, z_act) - z_long||^2, re-encoded now.
* Overflow (water's fire). An experience overflows when it is more unresolved
  than the present: e_i > e_present, the error on the experience just lived.
  Up to 32 seeds are sampled with probability proportional to e_i (no
  monopoly). They replace the stage-1 uniform replay batch of the outcome
  model. A stale present is well explained, so much overflows; a novel present
  is poorly explained, so little or nothing does. There is no timer.
* Into wood. A seed is (state, act, outcome, felt state), re-encoded now.
  Wood's imagination organ F learns from the seeds' REAL outcomes, and wood's
  valuation G learns from their realized TD labels with their own felt state
  (felt recall). Imagination is checked against stored truth, never trained
  on its own output.
* Digest and release (water's metal). After the update, a seed whose error
  fell below the present's typical error (running mean minus one running std
  of e_present) is digested, and water lets it go (slot freed). The
  experience committed in this same cycle is never released.

Amendment 1 (gate_stage6_1, O4 failed: released median age about 20 cycles).
Release took easy experiences after one update, not digested ones. An
experience now counts as digested only after a transformation: once
overflowed while clearly unresolved (e above the present's running mean plus
one std), and later explained (e below the mean minus one std). No count
constant is involved.

Not yet: seeds do not condition the generated words ("films"). Stored-word
reuse is kept.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.language_navigation_kanli_v4b import INTERO, SCALE
from scripts.ib_local.language_navigation_kanli_v4e import EqualEngine

SEEDS = 32


class WaterEngine(EqualEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.seed_rng = torch.Generator(device=self.device).manual_seed(173)
        self.present_mean = self.present_var = None
        self._plan_slots = None
        self.released_total = 0
        self.water = None
        self.unresolved = None  # per slot: was overflowed while clearly unresolved

    def plan(self, h):
        logits, plan, reusable = super().plan(h)
        self._plan_slots = plan['slots'][0].detach()
        return logits, plan, reusable

    @torch.no_grad()
    def _errors(self, slots):
        memory, enc, outcome = self.model.memory, self.model.encoder, self.model.kanli.outcome
        zs, za = enc(memory.source[slots]), enc(memory.action[slots])
        return (outcome(torch.cat((zs, za), -1))-enc(memory.long_successor[slots])).square().mean(-1)

    def _outcome_loss(self, source, action, long):
        memory, enc, outcome = self.model.memory, self.model.encoder, self.model.kanli.outcome
        with torch.no_grad():
            e_present = float((outcome(torch.cat((enc(source[None]), enc(action[None])), -1))
                               - enc(long[None])).square().mean())
        occupied = torch.nonzero(memory.occupied).flatten()
        seeds = occupied[:0]
        level = 0.
        if len(occupied):
            e = self._errors(occupied)
            level = float(e.mean())
            over = occupied[e > e_present]
            if len(over):
                w = e[e > e_present]
                k = min(SEEDS, len(over))
                seeds = over[torch.multinomial(w/w.sum(), k, replacement=False, generator=self.seed_rng)]
        self.water = dict(level=level, e_present=e_present, overflowing=int((self._errors(occupied) > e_present).sum()) if len(occupied) else 0,
                          seeds=seeds, e_before=self._errors(seeds) if len(seeds) else None)
        src = torch.cat((source[None], memory.source[seeds]))
        act = torch.cat((action[None], memory.action[seeds]))
        lng = torch.cat((long[None], memory.long_successor[seeds]))
        target = enc(lng).detach()
        zs, za = enc(src), enc(act)
        error = (outcome(torch.cat((zs, za), -1))-target).square().mean(-1)
        with torch.no_grad():
            shuffled = (outcome(torch.cat((zs, za.roll(1, 0)), -1))-target).square().mean(-1) if len(src) > 1 else error
            blind = (outcome(torch.cat((zs, zs), -1))-target).square().mean(-1)
        return error.mean(), dict(outcome_error=float(error.mean()), outcome_error_current=float(error[0]),
                                  outcome_error_shuffled=float(shuffled.mean()),
                                  outcome_error_as_silent=float(blind.mean()))

    @torch.no_grad()
    def _seed_recall(self, z, felt, option, k=8, temperature=.2):
        memory, enc = self.model.memory, self.model.encoder
        keys = F.normalize(enc(memory.source), dim=-1)
        similarity = keys @ F.normalize(z, dim=-1) - (memory.feeling-felt).norm(dim=-1)/INTERO**.5
        valid = memory.occupied & (memory.option == option)
        count = int(valid.sum())
        if not count:
            return torch.zeros(2, device=z.device)
        score, slot = similarity.masked_fill(~valid, -1e4).topk(min(k, count))
        w = (score/temperature).softmax(-1)
        return torch.stack(((w*memory.progress[slot]).sum()*SCALE, torch.tensor(min(k, count)/k, device=z.device)))

    def _feed_wood(self, seeds):
        """Wood's valuation learns from the seeds' realized labels, in their own felt state."""
        memory, enc, heads = self.model.memory, self.model.encoder, self.model.kanli
        seeds = seeds[memory.option[seeds] >= 0]
        if not len(seeds) or self.mode == 'silence':
            return None
        with torch.no_grad():
            z, a = enc(memory.source[seeds]), enc(memory.action[seeds])
            landing = heads.outcome(torch.cat((z, a), -1))
            onehot = torch.eye(3, device=self.device)[memory.option[seeds]]
            recalled = torch.stack([self._seed_recall(z[i], memory.feeling[seeds[i]], int(memory.option[seeds[i]]))
                                    for i in range(len(seeds))])
            inputs = torch.cat((z, landing, memory.feeling[seeds], onehot, recalled), -1)
            target = SCALE*memory.progress[seeds]
        prediction = heads.critic(inputs).squeeze(-1)+inputs[:, -2]
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.critic.parameters(), 1.)
        self.optimizer.step()
        return float(loss.detach())

    def _film(self, seeds):
        """Hook for stage 6b (v4g); stage 6 plays no films."""
        return None

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        memory, water = self.model.memory, self.water
        seeds = water['seeds']
        e_present = water['e_present']
        if self.present_mean is None:
            self.present_mean, self.present_var = e_present, 0.
        else:
            d = e_present-self.present_mean
            self.present_mean += (1-self.baseline_decay)*d
            self.present_var = self.baseline_decay*self.present_var+(1-self.baseline_decay)*d*d
        info = dict(level=water['level'], e_present=e_present, overflowing=water['overflowing'], seeds=int(len(seeds)))
        if len(seeds):
            plan_slots = self._plan_slots if self._plan_slots is not None else seeds[:0]
            info['seeds_not_recalled'] = float((~torch.isin(seeds, plan_slots)).float().mean())
            film = self._film(seeds)
            if film is not None:
                info['film'] = film
            e_after = self._errors(seeds)
            info['digestion'] = float(((water['e_before']-e_after)/water['e_before'].clamp_min(1e-12)).median())
            info['wood_loss'] = self._feed_wood(seeds)
            if self.unresolved is None:
                self.unresolved = torch.zeros(memory.capacity, dtype=torch.bool, device=self.device)
            sd = np.sqrt(self.present_var)
            high, low = self.present_mean+sd, self.present_mean-sd
            self.unresolved[seeds[water['e_before'] > high]] = True
            release = seeds[self.unresolved[seeds] & (e_after < low) & (seeds != result['episode_slot'])]
            info['marked_unresolved'] = int(self.unresolved.sum())
            if len(release):
                ages = (int(memory.clock)-memory.birth[release]).float()
                info['released'] = int(len(release))
                info['released_age'] = float(ages.median())
                memory.occupied[release] = False
                self.unresolved[release] = False
                self.released_total += int(len(release))
        info['released_total'] = self.released_total
        info['occupied_age_median'] = float((int(memory.clock)-memory.birth[memory.occupied]).float().median()) if bool(memory.occupied.any()) else 0.
        result['water'] = info
        return result
