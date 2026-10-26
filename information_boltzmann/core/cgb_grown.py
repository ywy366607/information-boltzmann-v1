"""Grown interior couplings for LG-1 (diagnostic, not admitted).

The interior coupling R is not drawn at birth; it grows from an empty, symmetric
seed under the generation chain's constraints:
  - structure is content with a finite budget (P1a, G16): a shared pool of slots;
  - building reads only the two ends of a coupling (G7): the rate is set by their
    slow co-activity c_ij (Hebb), the sign by the sign of c_ij (reciprocal, P2b);
  - removal is selective (G17b): a deep (occupied) connection is removed more slowly,
    k_off * exp(-beta * n), so retained connections have removal rates near zero and
    occupancy splits into strong and empty; with a constant removal rate (beta = 0)
    occupancy is only proportional to use (diagnostic 2026-10-26: no strong
    connections, 5.5 bits/byte).

`use_driven=False` makes building blind to use (d = 1): the G17a control.
Registered choices: co-activity averaging rate, k_on, k_off, budget, coupling scale g.
"""

from __future__ import annotations

import numpy as np

from cgb_lg1 import LG1


class GrownLG1(LG1):
    def __init__(self, hidden=512, *, budget=0.1, g=None, k_on=0.02, k_off=0.002,
                 alpha=0.005, beta=4.0, use_driven=True, every=4, seed=0, **kw):
        kw.pop("recur", None)
        super().__init__(hidden, seed=seed, recur=0.0, **kw)
        H = hidden
        self.c = np.zeros((H, H), np.float32)
        self.n = np.zeros((H, H), np.float32)
        self.budget, self.k_on, self.k_off, self.alpha, self.beta = budget, k_on, k_off, alpha, beta
        self.g = 2.0 / np.sqrt(budget * H) if g is None else g
        self.use_driven, self.every, self.t = use_driven, every, 0
        self.R = np.zeros((H, H), np.float32)
        self.growing = True

    def occupancy(self) -> float:
        return float(self.n.sum()) / (self.n.shape[0] * (self.n.shape[0] - 1))

    def observe(self, token: int) -> float:
        nll = super().observe(token)
        if self.growing:
            self.t += 1
            if self.t % self.every == 0:
                self._grow()
        return nll

    def _grow(self):
        x = self.p1 - 0.5
        h = float(self.every)
        self.c += self.alpha * h * (np.outer(x, x) - self.c)
        np.fill_diagonal(self.c, 0.0)
        free = max(0.0, 1.0 - self.occupancy() / self.budget)
        if self.use_driven:
            a = np.abs(self.c)
            d = a / (a.mean() + 1e-12)
        else:
            d = 1.0
        self.n += h * (self.k_on * d * free * (1.0 - self.n) - self.k_off * np.exp(-self.beta * self.n) * self.n)
        np.clip(self.n, 0.0, 1.0, out=self.n)
        np.fill_diagonal(self.n, 0.0)
        self.R = (self.g * np.sign(self.c) * self.n).astype(np.float32)


def random_sparse_R(hidden, density, g, seed):
    rng = np.random.default_rng(seed + 303)
    m = np.triu(rng.random((hidden, hidden)) < density, 1)
    s = np.triu(np.where(rng.random((hidden, hidden)) < 0.5, -1.0, 1.0), 1)
    R = g * m * s
    return (R + R.T).astype(np.float32)
