"""CLO6: CLO5 with the drift estimate averaged before clipping.

v5 on task B: q = max(E[e^2] - 2 r, 0) per element over a 20-step window is noise clipped to its
positive part (positive_part_bias): about half of the elements carry a spurious drift of 2-6% of
r although the target is fixed, so the posterior variance has a floor near sqrt(q r), the signal
power S stays below it, the signal fraction is 0 and learning stops (B: 0.25 at step 300).
Drift is a slow property of the world and jumps are handled by the hold (B2), so q comes from a
long average (factor 0.999, bias-corrected) of the unclipped per-step estimate e^2 + 2 e e_prev,
clipped after averaging, with jump-like noise differences (e^2 > 25 x 2r) left out.
Everything else as CLO5.
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

DRIFT_EMA = 0.999       # registered: drift is slow
JUMP_E2 = 25.0          # registered: a noise difference beyond 5 sd (e^2 > 25 x 2r) is a jump, not drift
ALPHA = 1.0             # registered: Beta(alpha, beta) prior on the change rate, one pseudo-change
CBAR0 = 0.01            # registered: starting pooled change rate


class CLO6(V5.CLO5):
    def __init__(self, shape, *, no_shrink=True, **kw):
        super().__init__(shape, **kw)
        self.no_shrink = no_shrink
        K, D = shape
        self.qs = torch.zeros(K, D)
        self.nqs = torch.zeros(K, D)
        self.e_last = torch.zeros(K, D)
        self.cbar = CBAR0

    def _hazard(self, age):
        # Beta(ALPHA, beta) prior with mean cbar learned in the clique: h(T) = ALPHA / (ALPHA + beta + T)
        beta = ALPHA * (1 / self.cbar - 1)
        return ALPHA / (ALPHA + beta + age)

    def _frac(self, P):
        # the step acts on the posterior mean; the prior is in the filter's start, not a second shrink
        # (posterior_mean_is_bayes_action)
        return torch.ones_like(P) if self.no_shrink else super()._frac(P)

    def _accept_var(self, Pp, J, Sn):
        return Sn / self.run.clamp(min=1)          # variance of the run's mean (accepted_run_variance)

    def _rebase(self, Qn):
        Qold = self.Q.clone()
        super()._rebase(Qn)
        T = Qold.T @ self.Q
        self.e_last = self.e_last @ T
        self.qs = self.qs @ (T * T)

    def _drift(self, r, c2):
        # v5 has just set self.eprev to this step's noise difference e; self.e_last holds the previous one
        if self.t > 2:
            e, ep = self.eprev, self.e_last
            ok = (e * e <= JUMP_E2 * 2 * r) & (ep * ep <= JUMP_E2 * 2 * r)
            d = e * e + 2 * e * ep                       # unbiased for q, unclipped
            self.nqs = torch.where(ok, self.nqs + 1, self.nqs)
            rate = torch.maximum(1 / self.nqs.clamp(min=1), torch.full_like(self.nqs, 1 - DRIFT_EMA))
            self.qs = torch.where(ok, self.qs + (d - self.qs) * rate, self.qs)
        return self.qs.clamp(min=0)

    def step(self, W, G, X, fgrad):
        self.e_last = self.eprev.clone()
        before = self.stats["accepted"]
        super().step(W, G, X, fgrad)
        if self.t > 1:
            rate = (self.stats["accepted"] - before) / self.m.numel()
            self.cbar = max(V5.POOL_EMA * self.cbar + (1 - V5.POOL_EMA) * rate, 1e-6)


ARMS6 = dict(clo6={}, clo6_nohold=dict(hold=False), clo6_noalloc=dict(alloc="none"))


def make6(name, lr, shape, T):
    if name in ARMS6:
        return CLO6(shape, **ARMS6[name])
    return V3.make(name, lr, shape, T)
