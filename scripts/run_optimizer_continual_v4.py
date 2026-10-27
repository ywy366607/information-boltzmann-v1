"""CLO4: CLO (v3) with the corrections derived after the v3 run.

* The step acts on the expected loss, so the filter estimates the expected gradient: every element
  is updated every step (fresh_decrease_mixture: the expected fresh decrease is the occurrence
  mixture, and the per-step mean of the gradient is that mixture). Occurrence detection is dropped:
  judged against a shared reference it is a rich-club rule (shared_reference_flips); judged against
  the row's own past it mistakes learning progress (a shrinking gradient) for absence.
* The predicted decrease uses the posterior mean the step acts on, ghat = m S/(S+P); predicting with
  m overstates it by (m - ghat) u (consistent_prediction_excess).
* The signal power S is the time average of the unbiased m^2 - P, clipped after averaging; the
  per-step plug-in max(m^2 - P, 0) is positive when the signal is zero (positive_part_bias).
* The efference copy uses the smoothed curvature h dtheta (second-order bias,
  estimated_curvature_residual_bias) instead of the batch's own Hessian, whose noise the filter
  cannot see; the secant and the noise residual still come from the fresh batch.
Everything else as CLO (v3).
usage: python scripts/run_optimizer_continual_v4.py --out results/published/optimizer_continual_v4_20261027.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optimizer_continual as V3  # noqa: E402  (tasks, baselines, registered constants)
import run_optimizer_shell_kalman as V1  # noqa: E402

EPS = V3.EPS
S_EMA = 0.9             # registered: signal-power average (bias-corrected EMA)


class CLO4(V3.CLO):
    def __init__(self, shape, **kw):
        super().__init__(shape, **kw)
        K, D = shape
        self.S = torch.zeros(K, D)
        self.ns = 0

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
            self.S = self.S @ ((Qold.T @ self.Q) ** 2)
        g, th = G @ self.Q, W @ self.Q
        gp = Gp @ self.Q if Gp is not None else g
        thp = self.Wprev @ self.Q if self.Wprev is not None else th
        dth, own = th - thp, g - gp
        later = self.t > 1
        if later:
            self.sxy = 0.9 * self.sxy + 0.1 * own * dth
            self.sxx = 0.9 * self.sxx + 0.1 * dth * dth
        h = self.sxy / (self.sxx + EPS)
        if later:
            e = gp - self.gprev                                   # noise at the previous parameters
            if self.t > 2:
                self.e11 = 0.95 * self.e11 + 0.05 * e * self.eprev
            self.e2 = 0.95 * self.e2 + 0.05 * e * e
            self.eprev = e
        n = self.t - 1
        c2 = max(1 - 0.95 ** n, 0.05)
        c11 = max(1 - 0.95 ** (n - 1), 0.05)
        r = (-self.e11 / c11).clamp(min=0) + EPS
        q = (self.e2 / c2 - 2 * r).clamp(min=0)
        if later:
            mp = self.m + h * dth                                 # efference copy, smoothed curvature
            Pp = self.P + q
            Kg = Pp / (Pp + r)
            self.m = mp + Kg * (g - mp)
            self.P = (1 - Kg) * Pp
        else:
            self.m, self.P = g.clone(), g * g + EPS
        self.gprev = g
        self.thprev = th
        m, P = self.m, self.P
        # signal power averaged before clipping
        self.ns += 1
        self.S = self.S + (m * m - P - self.S) / min(self.ns, 1 / (1 - S_EMA))
        S = self.S.clamp(min=0)
        frac = S / (S + P + EPS)
        signal = m * m > V3.Z2 * P
        if self.latch:
            rest_run = torch.where(~self.cons, torch.where(signal, 0.0, self.run + 1), self.run)
            hold_run = torch.where(self.cons, torch.where(signal, self.run + 1, 0.0), self.run)
            self.run = torch.where(self.cons, hold_run, rest_run)
            new_cons = ~self.cons & (self.run >= V3.WINDOW)
            release = self.cons & (self.run >= V3.WINDOW) & self.exit
            self.stats["consolidations"] += int(new_cons.sum())
            self.stats["releases"] += int(release.sum())
            self.cons = (self.cons | new_cons) & ~release
            self.run = torch.where(new_cons | release, 0.0, self.run)
        ha = h.abs().clamp(min=EPS)
        ghat = m * frac                                           # the posterior mean the step acts on
        u = ghat / ha
        live = ~self.cons
        binding = bool(((u.abs() >= self.rho) & live).any())
        u = torch.where(live, u.clamp(-self.rho, self.rho), torch.zeros_like(u))
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
        pred = float((ghat * u - 0.5 * ha * u * u).sum())       # predicted with the same belief
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


ARMS4 = {k + "4": v for k, v in V3.CLO_ARMS.items()}


def make4(name, lr, shape, T):
    if name in ARMS4:
        return CLO4(shape, **ARMS4[name])
    return V3.make(name, lr, shape, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(V3.TASKS))
    a = ap.parse_args()
    V3.make = make4                            # tasks build optimizers through V3.make
    res = {}
    for task in a.tasks.split(","):
        fn = V3.TASKS[task]
        res[task] = {}
        for n in ARMS4:
            runs = [fn(n, None, s) for s in range(1, 6)]
            res[task][n] = dict(lr=None, **{k: V1.summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items()}), flush=True)
            Path(a.out).write_text(json.dumps(dict(results=res), indent=1))
    Path(a.out).write_text(json.dumps(dict(
        prereg="docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md",
        registered=dict(s_ema=S_EMA, inherits="run_optimizer_continual.py"),
        baselines_from="results/published/optimizer_continual_20261027.json", results=res), indent=1))


if __name__ == "__main__":
    main()
