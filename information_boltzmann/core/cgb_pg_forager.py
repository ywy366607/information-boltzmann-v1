"""G15 forager: the decision rule derived in Generation.lean, run as a diagnostic.

State: the intake history the forager must keep (G11/G12): whether it has eaten
since arriving, and the consecutive misses since the last meal or arrival
(capped). Action: one motor particle over {stay, switch}, read out with the
Gibbs/softmax law (G13, R1). Learning: the reward-modulated two-phase rule (G15a):
drive phase +score of the action taken, free phase -softmax, equal weights (G15d),
both scaled by intake minus a baseline (any baseline, G15a; outcome-blind
modulation learns nothing, G15c).

Registered choices (not derived): the exponential trace that carries the score
to the delayed intake (only the existence of some carrier is proven,
`LanguageLifeKernel.delayed_credit_requires_history_carrier`), the step size,
the trace decay, and the baseline's averaging rate.
"""

from __future__ import annotations

import numpy as np


class PGForager:
    def __init__(self, nmax=16, eta=0.05, trace=0.95, base_rate=0.01, relax=0.0, state_baseline=False,
                 couple=0.0, slow_c=10.0, slow_leak=0.0, seed=0):
        self.nmax = nmax
        self.w = np.zeros((2 * (nmax + 1), 2))  # symmetric birth (P3): pi(switch) = 1/2
        self.e = np.zeros_like(self.w)
        # deep compartment (G17 diagnostic): conserved diffusive transfer w <-> C * ws
        self.ws = np.zeros_like(self.w)
        self.g, self.C, self.sl = couple, slow_c, slow_leak
        self.eta, self.lam, self.br, self.relax = eta, trace, base_rate, relax
        self.rng = np.random.default_rng(seed)
        self.bl, self.ate, self.n = 0.5, 0, 0
        self.state_bl = state_baseline  # baseline = predicted intake of the state acted in (G11)
        self.bls = np.full(len(self.w), 0.5)
        self.s_act = 0
        self.learning = True

    def state(self) -> int:
        return self.ate * (self.nmax + 1) + self.n

    def policy(self, s=None):
        z = self.w[self.state() if s is None else s]
        p = np.exp(z - z.max())
        return p / p.sum()

    def act(self) -> int:
        s = self.state()
        self.s_act = s
        p = self.policy(s)
        a = int(self.rng.random() < p[1])
        score = -p
        score[a] += 1.0  # drive phase (clamped at a) minus free phase (softmax)
        self.e *= self.lam
        self.e[s] += score
        return a

    def sense(self, byte, intake: int, travelled: bool):
        if self.learning:
            self.w *= 1.0 - self.relax  # calm relaxation toward the symmetric (birth) state
            bl = self.bls[self.s_act] if self.state_bl else self.bl
            self.w += self.eta * (intake - bl) * self.e
            if self.g:
                flow = self.g * (self.w - self.ws)
                self.w -= flow
                self.ws += flow / self.C
                self.ws *= 1.0 - self.sl
            self.bl += self.br * (intake - self.bl)
            self.bls[self.s_act] += self.br * (intake - self.bls[self.s_act])
        if travelled:
            self.ate, self.n = 0, 0
        elif intake:
            self.ate, self.n = 1, 0
        else:
            self.n = min(self.n + 1, self.nmax)


class LangPGForager:
    """G15 forager that also listens: the motor readout reads the intake-history state
    (G11/G12) and a slow trace of every heard byte (the cue memory that predicting
    intake from text needs, by the G11 argument). Same G15 rule on linear features.

    Registered choices: leaky traces per byte value at time constants `taus` (set to
    their long-run mean, i.e. "no evidence yet", on arrival at a new source), online
    per-feature standardization scaled by 1/sqrt(group size), trace carrier,
    step size, baseline rate.
    """

    def __init__(self, nmax=16, eta=0.1, trace=0.9, base_rate=0.01, taus=(16.0, 300.0),
                 lang=True, seed=0):
        self.nmax, self.lang = nmax, lang
        self.ns = 2 * (nmax + 1)
        self.L = 256 * len(taus) if lang else 0
        self.F = self.ns + self.L
        self.W = np.zeros((2, self.F))
        self.e = np.zeros_like(self.W)
        self.eta, self.lam, self.br = eta, trace, base_rate
        self.k = np.repeat(1.0 / np.asarray(taus), 256)
        self.tr = np.zeros(self.L)
        self.mu, self.var = np.zeros(self.L), np.full(self.L, 1e-4)
        self.rng = np.random.default_rng(seed)
        self.bl, self.ate, self.n = 0.5, 0, 0
        self.learning = True

    def features(self):
        x = np.zeros(self.F)
        x[self.ate * (self.nmax + 1) + self.n] = 1.0
        if self.lang:
            # group-normalized: the whole language group moves a logit O(eta) per tick
            x[self.ns:] = (self.tr - self.mu) / np.sqrt(self.var * self.L)
        return x

    def act(self) -> int:
        x = self.features()
        z = self.W @ x
        p = np.exp(z - z.max())
        p /= p.sum()
        a = int(self.rng.random() < p[1])
        score = -p
        score[a] += 1.0
        self.e *= self.lam
        self.e += np.outer(score, x)
        return a

    def sense(self, byte, intake: int, travelled: bool):
        if self.learning:
            self.W += self.eta * (intake - self.bl) * self.e
            self.bl += self.br * (intake - self.bl)
        if travelled:
            self.ate, self.n = 0, 0
            self.tr[:] = self.mu
        else:
            self.ate, self.n = (1, 0) if intake else (self.ate, min(self.n + 1, self.nmax))
            if self.lang and byte is not None:
                self.tr *= 1.0 - self.k
                self.tr[byte::256] += self.k[byte::256]
                self.mu += 1e-3 * (self.tr - self.mu)
                self.var += 1e-3 * ((self.tr - self.mu) ** 2 - self.var)
