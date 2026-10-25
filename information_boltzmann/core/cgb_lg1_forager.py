"""LG-1 forager: LG-1 plus intake, travel and its own action as sensory input.

Inputs written into the hidden layer through fixed random W_in columns: the heard
byte (if any), the intake bit, the travel bit and the efference copy of its own
last action (the minimal self-model rung: "I just moved"). Readouts from the
same hidden state: the byte head of LG-1 and an intake head pi(intake | P),
both learned by the boundary-modulated delta rule (layer 3.3). The intake head
is trained only on ticks where the organism stayed (it predicts "intake if I
stay").

Policy (registered choice, not derived): switch when the predicted intake if
staying falls below the value of going elsewhere minus `margin`. That value is
the organism's running estimate of intake in the first `fresh` ticks after
arriving at a source (what a new place gives before anything is known). An
earlier comparator, the organism's own running mean, was wrong: doing well
raised it and drove the organism away from good sources.
"""

from __future__ import annotations

import math

import numpy as np

from cgb_lg1 import LG1

N_EXTRA = 6  # intake 0/1, travelled 0/1, own action 0/1


class LG1Forager(LG1):
    def __init__(self, hidden=256, *, eta_head=0.05, margin=0.05, baseline_rate=0.02,
                 fresh=5, seed=0, **kw):
        super().__init__(hidden, seed=seed, **kw)
        rng = np.random.default_rng(seed + 101)
        self.W_extra = rng.normal(0, 0.5, (hidden, N_EXTRA)).astype(np.float32)
        self.V1 = np.zeros(hidden, np.float32)
        self.V2 = np.zeros(hidden, np.float32)
        self.vb = 0.0
        self.eta_head, self.margin, self.baseline_rate = eta_head, margin, baseline_rate
        self.elsewhere, self.fresh, self.since = 0.5, fresh, 0
        self.last_action = 0
        self.byte_nll = []

    def predict_intake(self) -> float:
        z = self.vb + float((self.p1 - 0.5) @ self.V1 + (self.p2 - 0.5) @ self.V2)
        return 1.0 / (1.0 + math.exp(-z))

    def act(self) -> int:
        stay_intake = self.predict_intake()
        a = int(stay_intake < self.elsewhere - self.margin and self.last_action == 0
                and self.since >= self.fresh)
        self.last_action = a
        return a

    def sense(self, byte, intake: int, travelled: bool):
        if not travelled and self.last_action == 0 and self.learning:
            err = intake - self.predict_intake()
            self.V1 += self.eta_head * err * (self.p1 - 0.5)
            self.V2 += self.eta_head * err * (self.p2 - 0.5)
            self.vb += self.eta_head * err
        if travelled:
            self.since = 0
        else:
            if self.since < self.fresh:
                self.elsewhere += self.baseline_rate * (intake - self.elsewhere)
            self.since += 1
        extra = self.W_extra[:, intake] + self.W_extra[:, 2 + int(travelled)] \
            + self.W_extra[:, 4 + self.last_action]
        if byte is not None:
            self.byte_nll.append(self.observe_with(byte, extra))
        else:
            self._relax(extra)

    def observe_with(self, token: int, extra) -> float:
        h = self.h
        self.h = h + extra
        try:
            return self.observe(token)
        finally:
            self.h = h

    def _relax(self, extra):
        drive = extra + self.h + self.R @ (self.p1 - 0.5)
        target = 1.0 / (1.0 + np.exp(-drive))
        p1 = (1 - self.relax) * self.p1 + self.relax * target
        d = self.diffuse * (p1 - self.p2)
        self.p1, self.p2 = p1 - d, self.p2 + d


class Reflex:
    """E. coli without memory: switch after a tick without intake."""
    def __init__(self):
        self.last, self.moved = 1, 0
    def act(self):
        self.moved = int(self.last == 0 and self.moved == 0)
        return self.moved
    def sense(self, byte, intake, travelled):
        if not travelled:
            self.last = intake


class Integrator:
    """E. coli chemotaxis: one leaky intake average compared with the value of a
    fresh place (same comparator as the forager)."""
    def __init__(self, fast=0.2, rate=0.02, margin=0.05, fresh=5):
        self.f, self.e, self.fr, self.er, self.m, self.k = 0.5, 0.5, fast, rate, margin, fresh
        self.moved, self.since = 0, 0
    def act(self):
        self.moved = int(self.f < self.e - self.m and self.moved == 0 and self.since >= self.k)
        if self.moved:
            self.f = self.e
        return self.moved
    def sense(self, byte, intake, travelled):
        if travelled:
            self.since = 0
            return
        if self.since < self.k:
            self.e += self.er * (intake - self.e)
        self.since += 1
        self.f += self.fr * (intake - self.f)


class RandomPolicy:
    def __init__(self, rate=0.02, seed=0):
        self.rng, self.rate = np.random.default_rng(seed), rate
    def act(self):
        return int(self.rng.random() < self.rate)
    def sense(self, *a):
        pass


class Oracle:
    """Upper bound: reads the world's hidden state."""
    def __init__(self, world):
        self.w, self.moved = world, 0
    def act(self):
        w = self.w
        self.moved = int(not w.rich[w.at] and any(w.rich) and self.moved == 0)
        return self.moved
    def sense(self, *a):
        pass
