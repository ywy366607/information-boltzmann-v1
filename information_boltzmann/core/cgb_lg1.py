"""LG-1 mean-field: the predictor and learning rule derived in generation layer 3.

Body: sensory compartments Q_in (boundary, driven by the arrived byte for the
whole tick), prediction compartments Q_out (fast boundary, read in the free
phase), hidden compartments P1 (slow; written mainly through W_in) and P2 (fed
from P1 by fixed symmetric diffusion, hence lagged). Per world tick:

  free phase   predict  pi(b | P) = softmax(bias + W1^T p1 + W2^T p2)   (3.1)
  drive phase  Q_in and, briefly, Q_out clamped to the arrived byte b
               W_k += eta * p_k (onehot(b) - pi)   (boundary-modulated, 3.3)
               p1 relaxes toward sigmoid(W_in[:, b] + feedback * W1[:, b] + h);
               p1, p2 diffuse   (G6DB)

W1, W2 are the Q_out readout couplings. Reciprocity makes them also push on P
when Q_out is clamped; a slow P answers a brief clamp weakly (`feedback`). A
single shared Q (feedback = 1, no W_in) collapses the representation:
diagnostic 2026-10-26, 4.49 vs 3.63 bits/byte. W_in is fixed (genome-given) in
this version; interior learning may only be equilibrium-following (3.2).

Approximations, registered: mean-field charges replace sampled unit states;
the free-phase sample is replaced by its exact conditional; the phase average
of 3.3 is taken as exact. Readout and learning use each hidden compartment's
deviation from its resting charge 1/2 (the excess that can flow). Not
admitted: no Lean Medium instance yet.
"""

from __future__ import annotations

import math

import numpy as np


class LG1:
    def __init__(self, hidden=512, *, alphabet=256, eta=0.02, relax=0.5, diffuse=0.3,
                 init_scale=0.5, feedback=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W_in = rng.normal(0, init_scale, (hidden, alphabet)).astype(np.float32)
        self.W1 = np.zeros((hidden, alphabet), np.float32)
        self.feedback = feedback
        self.W2 = np.zeros((hidden, alphabet), np.float32)
        self.h = np.zeros(hidden, np.float32)
        self.bias = np.zeros(alphabet, np.float32)
        self.p1 = np.full(hidden, 0.5, np.float32)
        self.p2 = np.full(hidden, 0.5, np.float32)
        self.eta, self.relax, self.diffuse = eta, relax, diffuse
        self.learning = True

    def predict(self):
        z = self.bias + (self.p1 - 0.5) @ self.W1 + (self.p2 - 0.5) @ self.W2
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def observe(self, token: int) -> float:
        pi = self.predict()
        nll = -math.log(max(float(pi[token]), 1e-30))
        if self.learning:
            err = -pi
            err[token] += 1.0
            self.W1 += self.eta * np.outer(self.p1 - 0.5, err)
            self.W2 += self.eta * np.outer(self.p2 - 0.5, err)
            self.bias += self.eta * err
        drive = self.W_in[:, token] + self.feedback * self.W1[:, token] + self.h
        target = 1.0 / (1.0 + np.exp(-drive))
        p1 = (1 - self.relax) * self.p1 + self.relax * target
        d = self.diffuse * (p1 - self.p2)
        self.p1, self.p2 = p1 - d, self.p2 + d
        return nll
