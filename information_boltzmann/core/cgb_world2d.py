"""A two-dimensional food world: a random, driven medium (prereg CGB_WORLD2D_PREREG_20261026).

G x G torus, each cell holds n in 0..V quanta of food. Every tick:
  * injection (the world's outer boundary): a Poisson number of pellets (mean PELLET_RATE) land on
    random cells, PELLET quanta each (capped at V);
  * diffusion (local): every quantum hops to one of the four neighbours with probability DIFF;
  * dissipation: every quantum vanishes with probability DECAY.
An organism couples only through its boundary: it reads the cell potential
mu = log(n / (V - n + 1)) (T-N4) and takes quanta out of / puts quanta into its cell (conserved).
"""

from __future__ import annotations

import math

import numpy as np

SCALE = 50   # quanta per food unit: keeps cells in the reservoir limit (T-N4)
G, V = 48, 30 * SCALE
PELLET_RATE, PELLET = 0.5, 20 * SCALE
DIFF, DECAY = 0.1, 0.002


class World2D:
    def __init__(self, seed, g=G, v=V, pellet_rate=PELLET_RATE, pellet=PELLET, diff=DIFF, decay=DECAY,
                 warm=3000, rate_sigma=0.0, rate_tau=2000.0, size_sigma=0.0):
        """rate_sigma > 0: the pellet rate is pellet_rate * exp(X), X an Ornstein-Uhlenbeck
        process (std rate_sigma, correlation time rate_tau): random lean and rich times (W2).
        size_sigma > 0: lognormal pellet sizes with the same mean."""
        self.rng = np.random.default_rng(seed)
        self.g, self.v = g, v
        self.pellet_rate, self.pellet, self.diff, self.decay = pellet_rate, pellet, diff, decay
        self.rate_sigma, self.rate_tau, self.size_sigma = rate_sigma, rate_tau, size_sigma
        self.x = self.rng.normal(0.0, rate_sigma) if rate_sigma > 0 else 0.0
        self.food = np.zeros((g, g), dtype=np.int64)
        self.supplied = 0
        for _ in range(warm):          # start from the world's own steady state
            self.step()
        self.supplied = 0

    def rate(self):
        return self.pellet_rate * math.exp(self.x)

    def step(self):
        rng, f = self.rng, self.food
        if self.rate_sigma > 0:
            a = math.exp(-1.0 / self.rate_tau)
            self.x = a * self.x + self.rate_sigma * math.sqrt(1 - a * a) * rng.normal()
        for _ in range(rng.poisson(self.rate())):
            i, j = rng.integers(self.g, size=2)
            size = self.pellet
            if self.size_sigma > 0:
                size = int(round(self.pellet * math.exp(rng.normal(-self.size_sigma ** 2 / 2, self.size_sigma))))
            add = min(size, self.v - f[i, j])
            f[i, j] += add
            self.supplied += add
        f -= rng.binomial(f, self.decay)
        out = rng.binomial(f, self.diff)
        f -= out
        # split the movers over the four neighbours (multinomial by successive binomials)
        up = rng.binomial(out, 0.25)
        rest = out - up
        down = rng.binomial(rest, 1 / 3)
        rest -= down
        left = rng.binomial(rest, 0.5)
        right = rest - left
        f += np.roll(up, -1, 0) + np.roll(down, 1, 0) + np.roll(left, -1, 1) + np.roll(right, 1, 1)
        over = np.maximum(f - self.v, 0)   # capacity: excess stays out (dissipated)
        f -= over

    def cell(self, x, y):
        return int(x) % self.g, int(y) % self.g

    def mu(self, x, y):
        n = self.food[self.cell(x, y)]
        return math.log((n + 0.5) / (self.v - n + 1))

    def take(self, x, y, k):
        c = self.cell(x, y)
        k = min(k, int(self.food[c]))
        self.food[c] -= k
        return k

    def give(self, x, y, k):
        c = self.cell(x, y)
        room = self.v - int(self.food[c])
        put = min(k, room)
        self.food[c] += put
        return put


def policy_life(seed, policy, ticks=10000, cost=None, theta=0.0, speed=0.5, eat=0.02, reserve0=50.0,
                cap=150.0, base=0.02, full=None, world=None, margin=0.02, wkw=None, tau_c=300.0,
                move_cost=None):
    """An ideal (non-organism) forager on the exact field. Returns (survival, mean intake), intake
    and reserve in food units (quanta / SCALE). The comparator compares a fast average of mu
    (time constant 2 ticks) with a slow one (10 ticks): the innovation against a filtered
    prediction (G10a, G13)."""
    w = world if world is not None else World2D(seed, **(wkw or {}))
    rng = np.random.default_rng(seed + 77)
    x, y = rng.uniform(0, w.g, 2)
    ang = rng.uniform(0, 2 * math.pi)
    reserve, intake = reserve0, 0.0
    prev = w.mu(x, y)
    fast = slow = prev
    chat = None     # adaptive policy: slow average of the own intake rate (estimate of c)
    for t in range(ticks):
        mu = w.mu(x, y)
        fast += (mu - fast) / 2.0
        slow += (mu - slow) / 10.0
        if policy == "cruise":
            tumble = False
        elif policy == "threshold":
            tumble = mu >= theta
        elif policy == "comparator":
            tumble = fast < slow - margin or rng.random() < base
        elif policy == "graze":    # comparator, but stays to eat while the cell is rich
            tumble = fast < slow - margin or rng.random() < base or (full is not None and mu >= full)
        elif policy == "adaptive":  # marginal value rule with a remembered average (C-1)
            q = eat * w.food[w.cell(x, y)] / SCALE
            if chat is None:
                chat = q
            stay = q >= chat
            tumble = stay or fast < slow - margin or rng.random() < base
        else:
            raise ValueError(policy)
        prev = mu
        moved = not tumble
        if tumble:
            ang = rng.uniform(0, 2 * math.pi)
        else:
            x, y = (x + speed * math.cos(ang)) % w.g, (y + speed * math.sin(ang)) % w.g
        got = w.take(x, y, int(rng.binomial(w.food[w.cell(x, y)], eat))) / SCALE
        intake += got
        if chat is not None:
            chat += (got - chat) / tau_c
        reserve = min(cap, reserve + got) - (cost or 0.0) - (move_cost if (move_cost and moved) else 0.0)
        w.step()
        if cost is not None and reserve <= 0:
            return t + 1, intake / (t + 1)
    return ticks, intake / ticks
