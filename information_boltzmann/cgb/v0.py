"""CGB V0: the smallest closed-loop living brain (docs/information_boltzmann/CGB_V0_MINIMAL_LIFE_20261019.md).

Cell classes, all generated from GENOME (no connectome, no pretrained embedding):
  R receptors   fixed random sparse code per GPT-2 token (transducer, never learns)
  C traces      leaky pools of a few R / X cells, time constants log-uniform in [tau_min, tau_max] ticks (memory)
  X expansion   sparse fan-in from R and C; global inhibition lets the top `expansion_active` fraction fire (k-WTA);
                per-cell gain/threshold homeostasis (physiology only)
  O prediction  one per receptor; word hypotheses compete (p(word) ~ exp(sum of O at its code)); the error is computed
                at the receptor (arrived code - expected code under the competing hypotheses); local delta rule on
                X->O (the exact gradient of the word log-likelihood of a linear readout)
  G mouth gate  in its own turn, speaks with probability g a word sampled from its own prediction; three-factor:
                eligibility trace x (information the world delivered beyond its usual rate)

No BPTT, no Adam, every update is local.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

VOCAB = 50257

GENOME = {
    'n_receptor': 1024, 'k_receptor': 48,
    'n_trace': 2048, 'trace_fan_in': 8, 'trace_from_expansion': 0.5, 'tau_min': 1.0, 'tau_max': 512.0,
    'n_expansion': 8192, 'expansion_receptor_fan_in': 6, 'expansion_trace_fan_in': 6, 'expansion_active': 0.03,
    'receptor_gain': 1.0, 'trace_gain': 1.0, 'homeostasis_tau': 2000.0,
    'ticks_per_word': 4,
    'readout_rate': 4.0, 'bias_rate': 0.2,
    'gate_rate': 2e-4, 'gate_trace_tau': 48.0, 'gate_bias': -1.0, 'info_tau': 4000.0,
    'seed': 0,
}


def fan_in_matrix(rng, rows, cols, fan_in, weight=1.0):
    indices = np.stack([rng.choice(cols, fan_in, replace=False) for _ in range(rows)]).ravel()
    indptr = np.arange(0, rows * fan_in + 1, fan_in)
    return sp.csr_matrix((np.full(rows * fan_in, weight, np.float32), indices, indptr), shape=(rows, cols))


def receptor_codes(n, k, seed, vocab=VOCAB):
    rng = np.random.default_rng(seed + 7919)
    codes = np.empty((vocab, k), np.int32)
    draws = rng.integers(0, n, (vocab, 3 * k))
    for v in range(vocab):
        _, first = np.unique(draws[v], return_index=True)
        codes[v] = draws[v][np.sort(first)[:k]]
    return codes


class Brain:
    def __init__(self, genome=None):
        g = dict(GENOME, **(genome or {}))
        self.g = g
        rng = np.random.default_rng(g['seed'])
        nr, nc, nx = g['n_receptor'], g['n_trace'], g['n_expansion']
        self.codes = receptor_codes(nr, g['k_receptor'], g['seed'])
        vocab, k = self.codes.shape
        ones = np.ones(vocab * k, np.float32)
        self.code_matrix = sp.csr_matrix((ones, self.codes.ravel(), np.arange(0, vocab * k + 1, k)), shape=(vocab, nr))
        self.code_matrix_t = self.code_matrix.T.tocsr()
        # traces: each pools trace_fan_in cells of R or of X
        from_x = rng.random(nc) < g['trace_from_expansion']
        rows_r, rows_x = np.flatnonzero(~from_x), np.flatnonzero(from_x)
        pr = fan_in_matrix(rng, len(rows_r), nr, g['trace_fan_in'], 1.0 / g['trace_fan_in'])
        px = fan_in_matrix(rng, len(rows_x), nx, g['trace_fan_in'], 1.0 / g['trace_fan_in'])
        self.trace_r = sp.csr_matrix((pr.data, pr.indices, pr.indptr), shape=pr.shape)
        self.trace_x = px
        self.rows_r, self.rows_x = rows_r, rows_x
        tau = np.exp(rng.uniform(math.log(g['tau_min']), math.log(g['tau_max']), nc))
        self.decay = np.exp(-1.0 / tau).astype(np.float32)
        self.tau = tau.astype(np.float32)
        # expansion fan-in
        self.x_from_r = fan_in_matrix(rng, nx, nr, g['expansion_receptor_fan_in'])
        self.x_from_c = fan_in_matrix(rng, nx, nc, g['expansion_trace_fan_in'])
        self.k_active = max(1, int(round(g['expansion_active'] * nx)))
        # physiology homeostasis: running mean / variance of the trace drive, excitability threshold
        self.drive_mean = np.zeros(nx, np.float32)
        self.drive_var = np.full(nx, 1e-2, np.float32)
        self.threshold = np.zeros(nx, np.float32)
        self.rate = np.full(nx, g['expansion_active'], np.float32)
        # prediction readout X -> O (one O per receptor) and its bias
        self.w_out = np.zeros((nx, nr), np.float32)    # row = one X cell's synapses onto the O cells
        self.b_out = np.zeros(nr, np.float32)
        # mouth gate
        self.w_gate = np.zeros(nx, np.float32)
        self.b_gate = np.float32(g['gate_bias'])
        self.e_gate = np.zeros(nx, np.float32)
        self.e_bias = 0.0
        self.e_decay = math.exp(-1.0 / g['gate_trace_tau'])
        self.info_mean = 0.0
        # state
        self.r = np.zeros(nr, np.float32)
        self.c = np.zeros(nc, np.float32)
        self.active = np.zeros(0, np.int64)
        self.ticks = 0
        self.rng = np.random.default_rng(g['seed'] + 1)

    # -- one tick of the medium --------------------------------------------------------------------------------------
    def tick(self, receptors=None):
        g = self.g
        r = np.zeros(g['n_receptor'], np.float32)
        if receptors is not None:
            r[receptors] = 1.0
        trace_drive = self.x_from_c @ self.c
        a = 1.0 / g['homeostasis_tau']
        self.drive_mean += a * (trace_drive - self.drive_mean)
        self.drive_var += a * ((trace_drive - self.drive_mean) ** 2 - self.drive_var)
        score = (g['receptor_gain'] * (self.x_from_r @ r)
                 + g['trace_gain'] * (trace_drive - self.drive_mean) / np.sqrt(self.drive_var + 1e-6) - self.threshold)
        active = np.argpartition(-score, self.k_active)[:self.k_active]
        x = np.zeros(g['n_expansion'], np.float32)
        x[active] = 1.0
        self.rate += a * (x - self.rate)
        self.threshold += a * (self.rate - g['expansion_active']) / g['expansion_active']
        u = np.empty_like(self.c)
        u[self.rows_r] = self.trace_r @ r
        u[self.rows_x] = self.trace_x @ x
        self.c = self.decay * self.c + (1 - self.decay) * u
        self.r, self.active = r, active
        self.ticks += 1
        self.e_gate *= self.e_decay
        self.e_bias *= self.e_decay
        return active

    # -- prediction and its local learning --------------------------------------------------------------------------
    def prediction(self):
        return self.b_out + self.w_out[self.active].sum(0)

    def word_distribution(self, o):
        """Word hypotheses compete (divisive normalization over the vocabulary): p(word) ~ exp(sum of O at its code)."""
        logits = self.code_matrix @ o
        z = np.exp(logits - logits.max())
        return z / z.sum()

    def arrive(self, token, learn=True):
        """A word arrives: the surprisal of the standing prediction (measurement), then local learning from the
        receptor error: arrived code minus the expected code under the competing word hypotheses. On the linear
        readout this is the exact gradient of the word's log-likelihood."""
        o = self.prediction()
        p = self.word_distribution(o)
        s = -math.log(max(float(p[token]), 1e-300))
        if learn:
            expected = (self.code_matrix_t @ p).astype(np.float32)
            err = -expected
            err[self.codes[token]] += 1.0
            if len(self.active):
                # rates are per word logit: a word's logit sums k receptors, so each receptor takes 1/k of the step
                self.w_out[self.active] += (self.g['readout_rate'] / (len(self.active) * self.g['k_receptor'])) * err[None, :]
            self.b_out += (self.g['bias_rate'] / self.g['k_receptor']) * err
        return s

    def hear(self, token, learn=True):
        """Its own word heard back: time passes, the world delivers no information on these ticks."""
        s = self.arrive(token, learn)
        for _ in range(self.g['ticks_per_word']):
            self.tick(self.codes[token])
            self.reward(0.0)
        return s

    def silence(self, ticks):
        for _ in range(int(ticks)):
            self.tick(None)
            self.reward(0.0)

    # -- the mouth ---------------------------------------------------------------------------------------------------
    def gate(self):
        return 1.0 / (1.0 + math.exp(-(self.b_gate + self.w_gate[self.active].sum())))

    def decide(self):
        p = self.gate()
        speak = bool(self.rng.random() < p)
        self.e_gate[self.active] += (float(speak) - p)
        self.e_bias += (float(speak) - p)
        return speak, p

    def babble(self, temperature=1.0):
        logits = (self.code_matrix @ self.prediction()) / temperature
        z = np.exp(logits - logits.max())
        return int(self.rng.choice(len(z), p=z / z.sum()))

    def reward(self, info):
        """Three-factor update of the gate: eligibility x (information the world delivered this tick minus its usual
        rate). Information = surprisal of the world's words; the brain's own words never count."""
        m = info - self.info_mean
        self.info_mean += (info - self.info_mean) / self.g['info_tau']
        if self.e_bias != 0.0:
            eta = self.g['gate_rate']
            self.w_gate += eta * m * self.e_gate
            self.b_gate = np.float32(self.b_gate + eta * m * self.e_bias)

    def heard_from_world(self, token, learn=True):
        """A world word over its ticks: the information arrives on the first tick."""
        s = self.arrive(token, learn)
        for t in range(self.g['ticks_per_word']):
            self.tick(self.codes[token])
            self.reward(s if t == 0 else 0.0)
        return s

    def stats(self):
        return {'x_rate_min': float(self.rate.min()), 'x_rate_max': float(self.rate.max()),
                'x_rate_dead': float((self.rate < 1e-4).mean()), 'trace_mean': float(self.c.mean()),
                'gate_bias': float(self.b_gate), 'w_out_abs': float(np.abs(self.w_out).mean())}

    def state_dict(self):
        keys = ('drive_mean', 'drive_var', 'threshold', 'rate', 'w_out', 'b_out', 'w_gate', 'e_gate', 'r', 'c',
                'active')
        out = {k: getattr(self, k) for k in keys}
        out.update(b_gate=np.float32(self.b_gate), e_bias=np.float64(self.e_bias),
                   info_mean=np.float64(self.info_mean), ticks=np.int64(self.ticks))
        return out

    def load_state_dict(self, saved):
        for k, v in saved.items():
            v = np.asarray(v)
            setattr(self, k, v.item() if v.ndim == 0 else v.copy())
