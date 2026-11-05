"""V4 stage 6b: earth sorts water before wood; wood plays films (Claude, 2026-11-04).

gate_stage6_3 failed. The overflow saturated at the 32-seed cap, and only one
experience was released in 1024 cycles. The digestibility probe
(results/.../analysis/water_digest_*.py) found the cause.
* The live F is no better than predicting the mean outcome (0.865 vs 0.880).
* No ridge fit generalizes to held-out experiences.
* Rehearsing a copy of F memorizes the rehearsed half (0.83 -> 0.012), while
  the held-out half gets worse (0.81 -> 2.14).
So a high e_i mostly means the world's reply was unusual: noise, not
structure. Overflow in proportion to e is noisy-TV rumination, and films as
plain extra rehearsal would be rote memorization.

User, 11-04: earth "不得独主于时" rules the last 18 days of every season, i.e.
earth is the transformation at each hand-over, not a stage of its own. What
was missing at the water -> wood hand-over is earth's 运化, 升清降浊: rehearsing
a clear seed helps OTHER experiences, while a turbid one can only be
memorized. User: "这也是主动的闭环", life must not stand still when the world
has no news.

On top of stage 6 (same overflow condition e_i > e_present, same sampling,
same release rule):
* Clarity. For each seed i, c_i = max(0, cos(g_i, G_w)). g_i is the seed's
  own gradient on F's parameters; G_w is the gradient of F's mean error on
  witnesses: the present plus a uniform sample of other stored experiences,
  as many as there are seeds. An idiosyncratic gradient is nearly orthogonal
  to G_w, so c is near 0 without any threshold.
* Clear water (B). In F's loss each seed is weighted by c_i, the present by
  1. The volume wood receives is sum(c), so it follows how much clear water
  there is, not a seed count. Turbid seeds stay in water as episodic memory:
  recall and stored-word reuse are untouched.
* Films (A). After the cycle, wood replays the clear seeds through F with
  its own optimizer (F only; encoder and body untouched). It keeps replaying
  while F's error on the stop sample keeps falling (amendment 0). The first replay that does
  not lower it is undone and the film ends. FILM_LIMIT is only a compute
  bound, and how often it binds is reported. A disjoint judge sample, never
  used for any decision, measures whether the film helped experiences it
  never saw.

Amendment 0 (smoke run of 140 cycles, before any gate; recorded in the doc).
The film stopped on the same witnesses that defined clarity. Clear seeds are
by construction the ones pointing toward those witnesses, so films kept
fitting that small sample: the witness error fell (median -0.005) while the
judge error rose (+0.0004), and 39% of films ran to FILM_LIMIT. Wood's own
metal must not reuse earth's sample. The film now stops on an independent
stop sample: other stored experiences, disjoint from the seeds and the
witnesses, as many as there are seeds. The judge is disjoint from all three.
"""
from __future__ import annotations

import torch
from torch.func import functional_call, grad, vmap

from scripts.ib_local.language_navigation_kanli_v4f import SEEDS, WaterEngine

FILM_LIMIT = 16  # compute bound only; binding rate is reported


class ClearWaterEngine(WaterEngine):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        outcome = self.model.kanli.outcome
        self.film_optimizer = torch.optim.AdamW(outcome.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.clear = None

    def _sample_rest(self, exclude, n):
        memory = self.model.memory
        occupied = torch.nonzero(memory.occupied).flatten()
        rest = occupied[~torch.isin(occupied, exclude)]
        if not len(rest) or n <= 0:
            return rest[:0]
        return rest[torch.randperm(len(rest), generator=self.seed_rng, device=rest.device)[:min(n, len(rest))]]

    @torch.no_grad()
    def _xy(self, slots):
        memory, enc = self.model.memory, self.model.encoder
        return torch.cat((enc(memory.source[slots]), enc(memory.action[slots])), -1), enc(memory.long_successor[slots])

    def _err(self, x, y):
        return (self.model.kanli.outcome(x)-y).square().mean(-1)

    def _clarity(self, xs, ys, xw, yw):
        outcome = self.model.kanli.outcome
        params = {k: v.detach() for k, v in outcome.named_parameters()}
        one = lambda p, x, y: (functional_call(outcome, p, (x[None],))[0]-y).square().mean()
        per = vmap(grad(one), in_dims=(None, 0, 0))(params, xs, ys)
        gi = torch.cat([per[k].flatten(1) for k in params], 1)
        mean = lambda p: (functional_call(outcome, p, (xw,))-yw).square().mean()
        gw = grad(mean)(params)
        gw = torch.cat([gw[k].flatten() for k in params])
        return torch.nn.functional.cosine_similarity(gi, gw[None], dim=-1).clamp_min(0.)

    def _outcome_loss(self, source, action, long):
        memory, enc, outcome = self.model.memory, self.model.encoder, self.model.kanli.outcome
        with torch.no_grad():
            x0 = torch.cat((enc(source[None]), enc(action[None])), -1)
            y0 = enc(long[None])
            e_present = float(self._err(x0, y0)[0])
        occupied = torch.nonzero(memory.occupied).flatten()
        seeds = occupied[:0]
        level, overflowing = 0., 0
        if len(occupied):
            e = self._errors(occupied)
            level = float(e.mean())
            over = occupied[e > e_present]
            overflowing = int(len(over))
            if len(over):
                w = e[e > e_present]
                k = min(SEEDS, len(over))
                seeds = over[torch.multinomial(w/w.sum(), k, replacement=False, generator=self.seed_rng)]
        self.water = dict(level=level, e_present=e_present, overflowing=overflowing,
                          seeds=seeds, e_before=self._errors(seeds) if len(seeds) else None)
        witnesses = self._sample_rest(seeds, len(seeds))
        c = torch.zeros(len(seeds), device=self.device)
        if len(seeds):
            xs, ys = self._xy(seeds)
            xw, yw = self._xy(witnesses) if len(witnesses) else (x0[:0], y0[:0])
            c = self._clarity(xs, ys, torch.cat((x0, xw)), torch.cat((y0, yw)))
        self.clear = dict(c=c, seeds=seeds, witnesses=witnesses, x0=x0, y0=y0)
        src = torch.cat((source[None], memory.source[seeds]))
        act = torch.cat((action[None], memory.action[seeds]))
        lng = torch.cat((long[None], memory.long_successor[seeds]))
        target = enc(lng).detach()
        zs, za = enc(src), enc(act)
        error = (outcome(torch.cat((zs, za), -1))-target).square().mean(-1)
        weight = torch.cat((torch.ones(1, device=self.device), c))
        loss = (weight*error).sum()/len(error)
        with torch.no_grad():
            shuffled = (outcome(torch.cat((zs, za.roll(1, 0)), -1))-target).square().mean(-1) if len(src) > 1 else error
            blind = (outcome(torch.cat((zs, zs), -1))-target).square().mean(-1)
        return loss, dict(outcome_error=float(error.detach().mean()), outcome_error_current=float(error[0].detach()),
                          outcome_error_shuffled=float(shuffled.mean()),
                          outcome_error_as_silent=float(blind.mean()))

    def _film(self, seeds):
        clear = self.clear
        c = clear['c']
        keep = c > 0
        info = dict(clear_seeds=int(keep.sum()), passes=0, reverted=False, hit_limit=False)
        if not bool(keep.any()):
            return info
        outcome = self.model.kanli.outcome
        xs, ys = self._xy(clear['seeds'][keep])
        cw = c[keep]
        used = torch.cat((clear['seeds'], clear['witnesses']))
        stop = self._sample_rest(used, len(clear['seeds']))
        if not len(stop):
            return info
        xw, yw = self._xy(stop)
        judge = self._sample_rest(torch.cat((used, stop)), len(clear['seeds']))
        xj, yj = self._xy(judge) if len(judge) else (None, None)
        with torch.no_grad():
            w0 = float(self._err(xw, yw).mean())
            j0 = float(self._err(xj, yj).mean()) if xj is not None else None
        info.update(witness_before=w0, judge_before=j0, stop_sample=int(len(stop)))
        for _ in range(FILM_LIMIT):
            snapshot = {k: v.detach().clone() for k, v in outcome.state_dict().items()}
            loss = (cw*self._err(xs, ys)).sum()/cw.sum()
            self.film_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.film_optimizer.step()
            with torch.no_grad():
                w1 = float(self._err(xw, yw).mean())
            if w1 >= w0:
                outcome.load_state_dict(snapshot)
                info['reverted'] = True
                break
            w0 = w1
            info['passes'] += 1
        info['hit_limit'] = info['passes'] == FILM_LIMIT
        with torch.no_grad():
            info['witness_after'] = w0
            info['judge_after'] = float(self._err(xj, yj).mean()) if xj is not None else None
        return info

    def step(self, directions, age_ablate=False):
        result = super().step(directions, age_ablate)
        c = self.clear['c']
        result['water'].update(clear_volume=float(c.sum()), clarity=c.tolist(),
                               witnesses=int(len(self.clear['witnesses'])))
        return result
