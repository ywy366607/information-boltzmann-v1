"""CLO6: experimental structured-curvature and soft-hold optimizer.

v5 on task B: q = max(E[e^2] - 2 r, 0) per element over a 20-step window is noise clipped to its
positive part (positive_part_bias): about half of the elements carry a spurious drift of 2-6% of
r although the target is fixed, so the posterior variance has a floor near sqrt(q r), the signal
power S stays below it, the signal fraction is 0 and learning stops (B: 0.25 at step 300).
Drift is a slow property of the world and jumps are handled by the hold (B2), so q comes from a
long average (factor 0.999, bias-corrected) of the unclipped per-step estimate e^2 + 2 e e_prev,
clipped after averaging, with jump-like noise differences (e^2 > 25 x 2r) left out.
Also uses a column-pooled shrinkage heuristic, age-dependent jump risk,
and a saturating proposal followed by CLO5's allocation and final clipping.
The scalar Lean bounds do not yet certify the complete update. See the v6
handoff review for unresolved statistical assumptions and revision provenance.
usage: python scripts/run_optimizer_continual_v6.py --out results/published/optimizer_continual_v6_20261027.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optimizer_continual as V3  # noqa: E402
import run_optimizer_continual_v5 as V5  # noqa: E402
import run_optimizer_shell_kalman as V1  # noqa: E402

# Capture before main installs make6 into V3; dynamic fallback would recurse.
BASE_MAKE = V3.make

EPS = V3.EPS
DRIFT_EMA = 0.999       # registered: drift is slow
JUMP_E2 = 25.0          # registered: a noise difference beyond 5 sd (e^2 > 25 x 2r) is a jump, not drift
ALPHA = 1.0             # registered: Beta(alpha, beta) prior on the change rate, one pseudo-change
CBAR0 = 0.01            # registered: starting pooled change rate
A0 = 1.0                # registered: row curvature factor before data (exact for a squared loss, an
                        # overestimate for softmax/logistic outputs, the safe side by newton_contracts_iff)
UPPER_Z = 2.0           # registered: the step uses the upper 2-sd bound of the row factor


class CLO6(V5.CLO5):
    def __init__(self, shape, *, no_shrink=False, **kw):
        super().__init__(shape, **kw)
        self.no_shrink = no_shrink
        self.tau2 = None
        K, D = shape
        self.qs = torch.zeros(K, D)
        self.nqs = torch.zeros(K, D)
        self.e_last = torch.zeros(K, D)
        self.cbar = CBAR0
        self.axy = torch.zeros(K, 1)
        self.axx = torch.zeros(K, 1)
        self.a1 = torch.zeros(K, 1)      # EMA of per-step row factors and of their squares
        self.a2 = torch.zeros(K, 1)
        self.na = torch.zeros(K, 1)
        self.h_step = None

    def _curvature(self, own, dth, later):
        # Kronecker curvature h_ij = a_i lambda_j: lambda_j from the pre-activity covariance in the
        # basis (exact input factor, eigenbasis_decouples), a_i pooled over the row
        lam = torch.diagonal(self.Q.T @ self.Rx @ self.Q).clamp(min=0)[None, :] / (1 - 0.95 ** self.t)  # bias-corrected EMA
        if later:
            num = (own * dth).sum(1, keepdim=True)
            den = (lam * dth * dth).sum(1, keepdim=True)
            ok = den > EPS
            a_now = torch.where(ok, num / den.clamp(min=EPS), torch.zeros_like(num))
            self.axy = torch.where(ok, 0.9 * self.axy + 0.1 * num, self.axy)
            self.axx = torch.where(ok, 0.9 * self.axx + 0.1 * den, self.axx)
            self.a1 = torch.where(ok, 0.9 * self.a1 + 0.1 * a_now, self.a1)
            self.a2 = torch.where(ok, 0.9 * self.a2 + 0.1 * a_now * a_now, self.a2)
            self.na = torch.where(ok, self.na + 1, self.na)
        have = self.axx > EPS
        a = torch.where(have, self.axy / self.axx.clamp(min=EPS), torch.full_like(self.axy, A0))
        neff = self.na.clamp(max=19)                            # effective samples of a 0.9 average
        c = (1 - 0.9 ** self.na).clamp(min=0.05)
        var = (self.a2 / c - (self.a1 / c) ** 2).clamp(min=0)
        se = torch.where(self.na > 1, (var / neff.clamp(min=1)).sqrt(), torch.full_like(a, A0))
        self.h_step = (a.abs() + UPPER_Z * se) * lam  # upward estimate; coverage is unproved
        return a * lam

    def _step_curvature(self, h):
        return self.h_step.clamp(min=EPS)

    def _bound(self, u):
        return self.rho * torch.tanh(u / self.rho)               # saturating_step_bounded

    def _hazard(self, age):
        # Beta(ALPHA, beta) prior with mean cbar learned in the clique: h(T) = ALPHA / (ALPHA + beta + T)
        beta = ALPHA * (1 / self.cbar - 1)
        return ALPHA / (ALPHA + beta + age)

    def _frac(self, P):
        # Heuristic signed signal-power estimate pooled across ALL rows, including
        # this element. Unbiasedness for the actual filtered m/P is unproved:
        # pooled_signal_unbiased assumes its per-element moment identity.
        if self.no_shrink:
            return torch.ones_like(P)
        tau2 = (self.m * self.m - P).mean(0, keepdim=True)
        self.tau2 = tau2 if self.tau2 is None else 0.9 * self.tau2 + 0.1 * tau2
        t2 = self.tau2.clamp(min=0)
        return t2 / (t2 + P + EPS)

    def _accept_var(self, Pp, J, Sn):
        return Sn / self.run.clamp(min=1)          # variance of the run's mean (accepted_run_variance)

    def _rebase(self, Qn):
        Qold = self.Q.clone()
        super()._rebase(Qn)
        T = Qold.T @ self.Q
        self.e_last = self.e_last @ T
        self.qs = self.qs @ (T * T)
        if self.tau2 is not None:
            # Match the existing diagonal second-moment transport convention.
            # General rotations discard off-diagonal covariance; this is not
            # an exact full-covariance change of coordinates.
            self.tau2 = self.tau2 @ (T * T)

    def _drift(self, r, c2):
        # v5 has just set self.eprev to this step's noise difference e; self.e_last holds the previous one
        if self.t > 2:
            e, ep = self.eprev, self.e_last
            ok = (e * e <= JUMP_E2 * 2 * r) & (ep * ep <= JUMP_E2 * 2 * r)
            d = e * e + 2 * e * ep                       # unbiased for q, unclipped
            self.nqs = torch.where(ok, self.nqs + 1, self.nqs)
            rate = torch.maximum(1 / self.nqs.clamp(min=1), torch.full_like(self.nqs, 1 - DRIFT_EMA))
            self.qs = torch.where(ok, self.qs + (d - self.qs) * rate, self.qs)
        # variance of the next value: drift plus the chance of a change times its size; a young value
        # (high hazard) stays responsive, an old one (low hazard) stays put (B3 in the variance)
        return self.qs.clamp(min=0) + self._hazard(self.age) * self.J

    def step(self, W, G, X, fgrad):
        self.e_last = self.eprev.clone()
        before = self.stats["accepted"]
        super().step(W, G, X, fgrad)
        if self.t > 1:
            rate = (self.stats["accepted"] - before) / self.m.numel()
            self.cbar = max(V5.POOL_EMA * self.cbar + (1 - V5.POOL_EMA) * rate, 1e-6)


ARMS6 = dict(clo6={}, clo6_nohold=dict(hold=False), clo6_noage=dict(age_depth=False),
             clo6_noalloc=dict(alloc="none"), clo6_rowonly=dict(alloc="row"), clo6_noshrink=dict(no_shrink=True))


def make6(name, lr, shape, T):
    if name in ARMS6:
        return CLO6(shape, **ARMS6[name])
    return BASE_MAKE(name, lr, shape, T)


# F: an old backbone (600 quiet steps), then bursts 100 steps apart, then one lasting change (tests B3)
STEPS_F = 1600


def task_f(name, lr, seed):
    gen = torch.Generator().manual_seed(5000 + seed)
    DD = V3.DD
    Ws = torch.randn(DD, DD, generator=gen) / DD ** 0.5
    size = 0.5 * float(Ws.norm())
    starts = [600 + 100 * i for i in range(6)]
    lengths = [V3.BURSTS[i % 4] for i in torch.randperm(6, generator=gen).tolist()]
    deltas = [V3.low_rank(gen, size) for _ in starts]
    change_at, change = 1300, V3.low_rank(gen, size)
    W = torch.zeros(DD, DD)
    opt = V3.make(name, lr, W.shape, STEPS_F)
    exc, post, adopt, lag = [], {L: [] for L in V3.BURSTS}, [], []
    for t in range(STEPS_F):
        if t == change_at:
            Ws = Ws + change
        active = [k for k, s0 in enumerate(starts) if s0 <= t < s0 + lengths[k]]
        target = Ws + deltas[active[0]] if active else Ws
        X = torch.randn(DD, V3.BATCH_D, generator=gen)
        Y = target @ X + V3.NOISE_D * torch.randn(DD, V3.BATCH_D, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / V3.BATCH_D, X, V3.lin_fg(X, Y))
        if not torch.isfinite(W).all():
            return dict(loss=float("inf"), final=float("inf"), adopt100=float("inf"),
                        **{f"post{L}": float("inf") for L in V3.BURSTS})
        e = 0.5 * float(((W - Ws) ** 2).sum())
        if t >= 600 and not active:
            exc.append(e)
        for k, s0 in enumerate(starts):
            if s0 + lengths[k] <= t < s0 + lengths[k] + 10:
                post[lengths[k]].append(e)
        if t >= STEPS_F - 100:
            adopt.append(e)
        if change_at <= t < change_at + 100:
            lag.append(e)
    return dict(loss=V3.fin(sum(exc) / len(exc)), final=V3.fin(sum(adopt) / len(adopt)), adopt100=V3.fin(sum(lag) / len(lag)),
                **{f"post{L}": V3.fin(sum(v) / len(v)) if v else float("nan") for L, v in post.items()})


V3.TASKS["F_old_backbone_bursts"] = task_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="B_rotated_heterogeneous,B0_isotropic,C_backbone_transient,D_topic_bursts,E_two_layer,F_old_backbone_bursts")
    ap.add_argument("--arms", default="", help="comma list; default all CLO6 arms (+ baselines on F)")
    a = ap.parse_args()
    V3.make = make6
    res = {}
    for task in a.tasks.split(","):
        fn = V3.TASKS[task]
        res[task] = {}
        names = list(ARMS6) + (["sgd", "adam", "muon", "soap", "adam_decay", "muon_decay"] if task == "F_old_backbone_bursts" else [])
        if a.arms:
            names = a.arms.split(",")
        for n in names:
            sweep, lr = None, None
            if n not in ARMS6:
                sweep = [(lr_, fn(n, lr_, 0)["loss"]) for lr_ in V3.LRS]
                lr = min(sweep, key=lambda x: x[1])[0]
            runs = [fn(n, lr, s) for s in range(1, 6)]
            res[task][n] = dict(lr=lr, sweep=sweep, **{k: V1.summary([r[k] for r in runs]) for k in runs[0]})
            print(task, n, json.dumps({k: (v["median"] if isinstance(v, dict) else v)
                                       for k, v in res[task][n].items() if k != "sweep"}), flush=True)
            Path(a.out).write_text(json.dumps(dict(results=res), indent=1))
    Path(a.out).write_text(json.dumps(dict(
        prereg="docs/information_boltzmann/CGB_OPTIMIZER_AXIOMATIC_DERIVATION_20261027.md",
        registered=dict(drift_ema=DRIFT_EMA, jump_e2=JUMP_E2, alpha=ALPHA, cbar0=CBAR0, a0=A0, upper_z=UPPER_Z,
                        inherits="run_optimizer_continual_v5.py"),
        baselines_from="results/published/optimizer_continual_20261027.json (F: this run)", results=res), indent=1))


if __name__ == "__main__":
    main()
