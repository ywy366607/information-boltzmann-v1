"""Smallest CGB organism: one unit of charge filtered over N cells, integer masses.

The living spec's joint kernel W(b,h'|s), conditioned on the arrived byte, with
one hidden variable of N cells:

    q_j   = sum_i alpha_i  A_ij / R_i          (transport, R_i = row mass)
    p_b   = sum_j q_j      B_jb / S_j          (prediction before b arrives)
    alpha'_j = q_j B_jb / S_j / p_b            (condition on the arrived b)

A and B hold nonnegative integer masses (exact in float32 below 2**24).
Learning deposits `deposit` units on K edge pairs (i, j) sampled from the joint
posterior alpha_i A_ij/R_i B_jb/S_j / p_b: a local three-factor product (source
charge, edge, target evidence) over one broadcast scalar p_b. Credit across
ticks is carried only by the filtered charge alpha (one-lag online EM); no
backward pass, no stored trajectory. With `lag=L > 0` the deposit for tick s
waits L ticks and uses the posterior smoothed by the L bytes that have since
arrived (fixed-lag online EM); the extra carrier is the last L charges and
bytes, L*N numbers, and each tick costs L more transport products. A row whose mass exceeds `row_cap` is
halved with floor 1 (finite capacity; old evidence is forgotten).
"""

from __future__ import annotations

import math

import torch


class ChargeFilter:
    def __init__(self, cells=256, *, alphabet=256, deposit=16, samples=4,
                 row_cap=1 << 20, init_spread=16, lag=0, seed=0, device="cpu"):
        if row_cap >= 1 << 23:
            raise ValueError("row_cap must keep integer masses exact in float32")
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.cells, self.alphabet = cells, alphabet
        self.deposit, self.samples, self.row_cap = deposit, samples, row_cap
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.A = (1 + torch.randint(0, init_spread, (cells, cells), generator=g)).float().to(self.device)
        self.B = (1 + torch.randint(0, init_spread, (cells, alphabet), generator=g)).float().to(self.device)
        self.R, self.S = self.A.sum(1), self.B.sum(1)
        self.alpha = torch.full((cells,), 1.0 / cells, device=self.device)
        self.learning = True
        self.lag, self._recent = lag, []
        self._q = None

    def predict(self, **_):
        self._q = (self.alpha / self.R) @ self.A
        return (self._q / self.S) @ self.B

    def observe(self, token: int, *, external=True, **_) -> float:
        if self._q is None:
            self.predict()
        q, self._q = self._q, None
        evidence = q * self.B[:, token] / self.S
        pb = evidence.sum()
        if self.learning:
            if self.lag == 0:
                self._learn(token, evidence, self.alpha)
            else:
                self._recent.append((self.alpha, token))
                if len(self._recent) > self.lag:
                    self._learn_smoothed()
        self.alpha = evidence / pb
        return -math.log(float(pb))

    def _learn_smoothed(self):
        alpha_prev, token = self._recent.pop(0)
        beta = torch.ones(self.cells, device=self.device)
        for _, later in reversed(self._recent):
            beta = (self.A @ (self.B[:, later] / self.S * beta)) / self.R
            beta = beta / beta.sum()
        q = (alpha_prev / self.R) @ self.A
        self._learn(token, q * self.B[:, token] / self.S * beta, alpha_prev)

    def _learn(self, token, evidence, alpha_prev):
        js = torch.multinomial(evidence, self.samples, replacement=True, generator=self.gen)
        w = (alpha_prev / self.R)[:, None] * self.A[:, js]
        is_ = torch.multinomial(w.T, 1, generator=self.gen).squeeze(1)
        self.A.index_put_((is_, js), torch.full_like(js, self.deposit, dtype=torch.float32), accumulate=True)
        self.B.index_put_((js, torch.full_like(js, token)),
                          torch.full_like(js, self.deposit, dtype=torch.float32), accumulate=True)
        self.R.index_add_(0, is_, torch.full_like(is_, self.deposit, dtype=torch.float32))
        self.S.index_add_(0, js, torch.full_like(js, self.deposit, dtype=torch.float32))
        for M, rows, sums in ((self.A, is_, self.R), (self.B, js, self.S)):
            over = rows[sums[rows] > self.row_cap].unique()
            if len(over):
                M[over] = torch.clamp(torch.floor(M[over] / 2), min=1)
                sums[over] = M[over].sum(1)

    def filter_only(self, context, start=None):
        """Charge after reading `context` from `start` (default uniform), no learning."""
        alpha = torch.full((self.cells,), 1.0 / self.cells, device=self.device) if start is None else start
        for b in context:
            e = ((alpha / self.R) @ self.A) * self.B[:, int(b)] / self.S
            alpha = e / e.sum()
        return alpha

    def nll_from(self, alpha, token: int) -> float:
        p = (((alpha / self.R) @ self.A) / self.S) @ self.B
        return -math.log(float(p[token]))

    def state_dict(self):
        return dict(A=self.A.cpu(), B=self.B.cpu(), alpha=self.alpha.cpu(),
                    learning=self.learning, gen=self.gen.get_state(), lag=self.lag,
                    recent=[(a.cpu(), b) for a, b in self._recent])

    def resident_bytes(self):
        tensors = [self.A, self.B, self.R, self.S, self.alpha] + [a for a, _ in self._recent]
        return sum(t.numel() * t.element_size() for t in tensors) + len(self._recent)


class OnlineNgram:
    """Prequential order-k byte model, KT (1/2) smoothing; the same stream exposure."""

    def __init__(self, order=1, alphabet=256):
        self.order, self.V, self.counts, self.hist = order, alphabet, {}, ()

    def observe(self, token: int) -> float:
        row = self.counts.get(self.hist)
        c, total = (0, 0) if row is None else (row.get(token, 0), row["_n"])
        nll = -math.log((c + 0.5) / (total + 0.5 * self.V))
        if row is None:
            row = self.counts[self.hist] = {"_n": 0}
        row[token] = c + 1
        row["_n"] += 1
        if self.order:
            self.hist = (self.hist + (token,))[-self.order:]
        return nll
