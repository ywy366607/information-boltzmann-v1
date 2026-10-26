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


class ForcedGrownLG1(LG1):
    """Interior couplings grown by the rule the chain forces (G18, G13, G16/G17, G3).

    - The only term coupling a connection unit's two ends is kappa * a * b (G18a), so
      growth reads the uncentered co-activity c_ij = <p_i p_j> (slow average).
    - One slot's equilibrium occupancy is occupancy(kappa * c_ij - theta) (G13, G18c):
      strictly increasing in use, i.e. state-selective export (G17).
    - theta is the content potential of the shared pool (conservation, G2/G3, G16),
      moved so that the mean occupancy stays at the budget.
    - Reciprocity (G18b): the same kappa couples the slot back to activity; a connection
      unit acts on one unit of a compartment of size n_c, so the compartment field is
      R = kappa * n / n_c.
    - `local_pool=True`: a connection unit belongs to two compartments and draws content
      from both; each compartment's content is finite (P1a), so each compartment has
      its own potential and h_ij = kappa * c_ij - theta_i - theta_j (G3 with one
      invariant per compartment). With one global pool (False) the uncentered
      co-activity is ranked by mean rates and grows a rich club (diagnostic 10-26).
    Constants only: kappa, n_c, budget, rates. Mean-field: c uses p (compartment
    charge fractions) for <a b>, a registered approximation.
    """

    def __init__(self, hidden=512, *, budget=0.1, kappa=20.0, n_c=None, rate=0.01,
                 alpha=0.005, theta_rate=0.05, every=4, local_pool=False, seed=0, **kw):
        kw.pop("recur", None)
        super().__init__(hidden, seed=seed, recur=0.0, **kw)
        H = hidden
        self.local_pool = local_pool
        self.c = np.full((H, H), 0.25, np.float32)
        self.n = np.full((H, H), budget, np.float32)
        np.fill_diagonal(self.n, 0.0)
        self.budget, self.kappa, self.rate, self.alpha = budget, kappa, rate, alpha
        self.n_c = kappa * budget * H / 2.0 if n_c is None else n_c  # row field ~2 at budget
        self.theta = np.full(H, kappa * 0.125) if local_pool else kappa * 0.25
        self.theta_rate = theta_rate
        self.every, self.t, self.growing = every, 0, True
        self.R = (kappa / self.n_c * self.n).astype(np.float32)

    def occupancy(self) -> float:
        H = self.n.shape[0]
        return float(self.n.sum()) / (H * (H - 1))

    def observe(self, token: int) -> float:
        nll = super().observe(token)
        if self.growing:
            self.t += 1
            if self.t % self.every == 0:
                self._grow()
        return nll

    def _grow(self):
        h = float(self.every)
        p = self.p1
        self.c += self.alpha * h * (np.outer(p, p) - self.c)
        th = self.theta[:, None] + self.theta[None, :] if self.local_pool else self.theta
        target = 1.0 / (1.0 + np.exp(-np.clip(self.kappa * self.c - th, -60, 60)))
        self.n += self.rate * h * (target - self.n)
        np.fill_diagonal(self.n, 0.0)
        H = self.n.shape[0]
        occ = self.n.sum(1) / (H - 1) if self.local_pool else self.occupancy()
        self.theta += self.theta_rate * h * (occ - self.budget) * self.kappa
        self.R = (self.kappa / self.n_c * self.n).astype(np.float32)


class ForcedGrowth:
    """The G18 growth law with one content pool per compartment, as a component
    (same rule as ForcedGrownLG1 with local_pool=True)."""

    def __init__(self, hidden, *, budget=0.1, kappa=100.0, n_c=None, rate=0.01, alpha=0.005,
                 theta_rate=0.05):
        H = hidden
        self.c = np.full((H, H), 0.25)
        self.n = np.full((H, H), budget)
        np.fill_diagonal(self.n, 0.0)
        self.budget, self.kappa, self.rate, self.alpha, self.theta_rate = budget, kappa, rate, alpha, theta_rate
        self.n_c = kappa * budget * H / 2.0 if n_c is None else n_c
        self.theta = np.full(H, kappa * 0.125)

    def target(self):
        field = self.kappa * self.c - (self.theta[:, None] + self.theta[None, :])
        return 1.0 / (1.0 + np.exp(-np.clip(field, -60, 60)))

    def coupling(self):
        return (self.kappa / self.n_c * self.n).astype(np.float32)

    def step(self, p, h=1.0):
        self.c += self.alpha * h * (np.outer(p, p) - self.c)   # two-state trace of a*b (G20a)
        self.n += self.rate * h * (self.target() - self.n)      # toward occupancy (G13, G18c)
        np.fill_diagonal(self.n, 0.0)
        H = self.n.shape[0]
        self.theta += self.theta_rate * h * (self.n.sum(1) / (H - 1) - self.budget) * self.kappa
        return self.coupling()
