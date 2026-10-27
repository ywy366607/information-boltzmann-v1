"""CLO5: CLO4 with a beneficial latch (soft hold) instead of the hard latch.

Derivation: GenerationContinualOptimizer.lean, "A beneficial latch, not only a healthy one".
Per element, each innovation nu = g - m- is weighed by the posterior that it is ordinary noise
rather than a jump (B2, mixture_update_bounded): small innovations (drift, plateaus) are written
almost fully (B1, frozen_hold_loses_on_drift), a single large one hardly at all. A run of
same-signed jumps is accepted as a lasting change once its posterior odds against a burst exceed
one; with the rule of succession for the change hazard, 1/(age+2), the run length needed is
k*(age) = 1 + log(pi_b (age + 2)) / log(1/s_b), nondecreasing in age (B3,
release_threshold_monotone_in_age, succession_odds_decreasing): an old value is held deep, a
newly accepted one shallow. The burst onset rate pi_b and continuation probability s_b are
pooled in the clique. Everything else as CLO4.
usage: python scripts/run_optimizer_continual_v5.py --out results/published/optimizer_continual_v5_20261027.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optimizer_continual as V3  # noqa: E402
import run_optimizer_continual_v4 as V4  # noqa: E402
import run_optimizer_shell_kalman as V1  # noqa: E402

EPS = V3.EPS
# registered starting values (then learned online in the clique or per element)
JUMP_POWER0 = 25.0      # jump variance starts at 25 x the ordinary innovation variance (5 sd)
PI_B0 = 0.01            # burst onset rate per element observation
S_B0 = 0.5              # burst continuation probability
POOL_EMA = 0.99         # clique statistics
JUMP_EMA = 0.9          # per-element jump power


class CLO5(V4.CLO4):
    def __init__(self, shape, *, hold=True, age_depth=True, **kw):
        kw.setdefault("latch", False)            # the hard latch is replaced by the hold
        super().__init__(shape, **kw)
        K, D = shape
        self.hold, self.age_depth = hold, age_depth
        z = lambda: torch.zeros(K, D)  # noqa: E731
        self.J = z()                  # jump power per element (0 = not yet seen)
        self.age = z()                # observations since the last accepted change
        self.run = z()                # length of the current same-signed jump run
        self.sh = z()                 # mean innovation of the run (the candidate new value offset)
        self.pi_b, self.s_b = PI_B0, S_B0
        self.stats.update(jumps=0, accepted=0, burst_runs=0)

    def _drift(self, r, c2):
        return (self.e2 / c2 - 2 * r).clamp(min=0)

    def _curvature(self, own, dth, later):
        if later:
            self.sxy = 0.9 * self.sxy + 0.1 * own * dth
            self.sxx = 0.9 * self.sxx + 0.1 * dth * dth
        return self.sxy / (self.sxx + EPS)

    def _step_curvature(self, h):
        return h.abs().clamp(min=EPS)

    def _bound(self, u):
        return u.clamp(-self.rho, self.rho)

    def _frac(self, P):
        S = self.S.clamp(min=0)
        return S / (S + P + EPS)

    def _hazard(self, age):
        return 1 / (age + 2)                      # rule of succession

    def _accept_var(self, Pp, J, Sn):
        return Pp + J

    def kstar(self):
        if not self.age_depth:
            return torch.ones_like(self.age)
        x = torch.log((self.pi_b / self._hazard(self.age)).clamp(min=EPS)) / math.log(1 / min(max(self.s_b, 1e-3), 0.999))
        return 1 + x.clamp(min=0)

    def step(self, W, G, X, fgrad):
        self.t += 1
        Gp = None
        if self.Wprev is not None:
            now = fgrad(W, False)[0]
            lp, Gp = fgrad(self.Wprev, True)
            self.evals += 2
            if self.scoring == "fresh":
                self._trust(lp - now, self.pred, now)
        self.Rx.mul_(0.95).add_(X @ X.T / X.shape[1], alpha=0.05)
        if self.t == 1 or self.t % 10 == 0:
            Qold = self.Q.clone()
            self._rebase(torch.linalg.eigh(self.Rx + EPS * torch.eye(self.Rx.shape[0]))[1])
            T, T2 = Qold.T @ self.Q, (Qold.T @ self.Q) ** 2
            self.S, self.J = self.S @ T2, self.J @ T2
            self.sh = self.sh @ T
        g, th = G @ self.Q, W @ self.Q
        gp = Gp @ self.Q if Gp is not None else g
        thp = self.Wprev @ self.Q if self.Wprev is not None else th
        dth, own = th - thp, g - gp
        later = self.t > 1
        h = self._curvature(own, dth, later)
        if later:
            e = gp - self.gprev
            if self.t > 2:
                self.e11 = 0.95 * self.e11 + 0.05 * e * self.eprev
            self.e2 = 0.95 * self.e2 + 0.05 * e * e
            self.eprev = e
        n = self.t - 1
        c2 = max(1 - 0.95 ** n, 0.05)
        c11 = max(1 - 0.95 ** (n - 1), 0.05)
        r = (-self.e11 / c11).clamp(min=0) + EPS
        q = self._drift(r, c2)
        if later:
            mp = self.m + h * dth
            Pp = self.P + q
            Kg = Pp / (Pp + r)
            nu = g - mp
            Sn = Pp + r
            if self.hold:
                J = torch.where(self.J > 0, self.J, JUMP_POWER0 * Sn)
                Sj = Sn + J
                # posterior that the innovation is ordinary (B2): log-likelihood ratio of the two widths
                llr = 0.5 * torch.log(Sj / Sn) - 0.5 * nu * nu * (1 / Sn - 1 / Sj)
                prior = math.log(self.pi_b / (1 - self.pi_b))
                w = torch.sigmoid(-(prior + (-llr)))     # p(ordinary)
                jump = w < 0.5
                same = jump & (torch.sign(nu) == torch.sign(self.sh)) & (self.run > 0)
                newrun = jump & ~same
                ended = ~jump & (self.run > 0)
                # clique statistics of bursts (onset rate, continuation)
                njump = int(jump.sum())
                if njump:
                    self.pi_b = POOL_EMA * self.pi_b + (1 - POOL_EMA) * float(newrun.float().mean())
                cont_events = int((same | ended).sum())
                if cont_events:
                    self.s_b = POOL_EMA * self.s_b + (1 - POOL_EMA) * float(same.sum()) / cont_events
                self.J = torch.where(jump, JUMP_EMA * J + (1 - JUMP_EMA) * (nu * nu - Sn).clamp(min=0), self.J)
                self.run = torch.where(same, self.run + 1, torch.where(newrun, torch.ones_like(self.run), torch.where(jump, self.run, torch.zeros_like(self.run))))
                self.sh = torch.where(same, self.sh + (nu - self.sh) / self.run, torch.where(newrun, nu, torch.where(jump, self.sh, torch.zeros_like(self.sh))))
                accept = jump & (self.run >= self.kstar())
                self.stats["jumps"] += njump
                self.stats["accepted"] += int(accept.sum())
                self.stats["burst_runs"] += int(ended.sum())
                # ordinary part written with weight w; an accepted run moves to the new value
                self.m = torch.where(accept, mp + self.sh, mp + w * Kg * nu)
                self.P = torch.where(accept, self._accept_var(Pp, J, Sn), Pp - w * Kg * Pp)
                self.age = torch.where(accept, torch.zeros_like(self.age), self.age + 1)
                self.run = torch.where(accept, torch.zeros_like(self.run), self.run)
                self.sh = torch.where(accept, torch.zeros_like(self.sh), self.sh)
            else:
                self.m = mp + Kg * nu
                self.P = (1 - Kg) * Pp
        else:
            self.m, self.P = g.clone(), g * g + EPS
        self.gprev = g
        self.thprev = th
        m, P = self.m, self.P
        self.ns += 1
        self.S = self.S + (m * m - P - self.S) / min(self.ns, 1 / (1 - V4.S_EMA))
        frac = self._frac(P)
        ha = self._step_curvature(h)
        ghat = m * frac
        u = ghat / ha
        binding = bool((u.abs() >= self.rho).any())
        u = self._bound(u)
        a = u * u
        act = a.sum(1, keepdim=True) > 0
        if self.alloc != "none" and act.sum() > 1:
            colT = a.sum(0, keepdim=True)
            tgt = torch.where(act, self.avail, torch.zeros_like(self.avail))
            tgt = tgt / tgt.sum() * a.sum()
            for _ in range(V3.SINKHORN_ITERS if self.alloc == "both" else 1):
                a = a * torch.where(act, tgt / (a.sum(1, keepdim=True) + EPS), torch.zeros_like(tgt))
                if self.alloc == "both":
                    a = a * torch.where(colT > 0, colT / (a.sum(0, keepdim=True) + EPS), torch.zeros_like(colT))
            u = (torch.sign(u) * a.sqrt()).clamp(-self.rho, self.rho)
        use = (u * u).sum(1, keepdim=True)
        if act.any():
            use = use / (use[act].mean() + EPS)
            self.avail += V3.RES_RATE * (1 - self.avail) - V3.RES_RATE * use * self.avail
            self.avail.clamp_(min=1e-3, max=1.0)
        pred = float((ghat * u - 0.5 * ha * u * u).sum())
        pred_sd = float(((u * u) * r).sum().sqrt())
        step = u @ self.Q.T
        if self.scoring == "same":
            before = fgrad(W, False)[0]
            after = fgrad(W - step, False)[0]
            self.evals += 2
            self.pred_sd, self.binding, self.umax = pred_sd, binding, float(u.abs().max())
            self._trust(before - after, pred, before)
        self.Wprev = W.clone()
        self.pred, self.pred_sd, self.binding, self.umax = pred, pred_sd, binding, float(u.abs().max())
        W -= step


ARMS5 = dict(clo5={}, clo5_nohold=dict(hold=False), clo5_noage=dict(age_depth=False),
             clo5_noalloc=dict(alloc="none"), clo5_rowonly=dict(alloc="row"))


def make5(name, lr, shape, T):
    if name in ARMS5:
        return CLO5(shape, **ARMS5[name])
    return V3.make(name, lr, shape, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="B_rotated_heterogeneous,B0_isotropic,C_backbone_transient,D_topic_bursts,E_two_layer")
    a = ap.parse_args()
    V3.make = make5
    res = {}
    for task in a.tasks.split(","):
        fn = V3.TASKS[task]
        res[task] = {}
        for n in ARMS5:
            runs = [fn(n, None, s) for s in range(1, 6)]
            res[task][n] = dict(lr=None, **{k: V1.summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items()}), flush=True)
            Path(a.out).write_text(json.dumps(dict(results=res), indent=1))
    Path(a.out).write_text(json.dumps(dict(
        prereg="docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md",
        registered=dict(jump_power0=JUMP_POWER0, pi_b0=PI_B0, s_b0=S_B0, pool_ema=POOL_EMA, jump_ema=JUMP_EMA,
                        inherits="run_optimizer_continual_v4.py"),
        baselines_from="results/published/optimizer_continual_20261027.json", results=res), indent=1))


if __name__ == "__main__":
    main()
