"""Chemotaxis world for generated rule-table organisms (prereg CGB_GENERATED_CHEMOTAXIS_PREREG_20261026).

A 1-D ring of length L with a concentration field of 6 Gaussian peaks whose heights span
two decades. Interface: page 0 F (boundary; reservoir potential mu_F = ln c(x), the form of a
chemical potential), page 1 M (boundary, one unit, potential 0; state 1 = tumble: no move this
tick and a new random direction). Pages >= 2 are interior. Intake per tick = c(x) / C_REF.
The stochastic kernel is the same compiled `_run` as in `cgb_table_fast`.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from cgb_table_fast import _run

F, M = 0, 1
L, V, SIGMA, BASE, C_REF = 400.0, 1.0, 25.0, 0.01, 20.0


@njit(cache=True)
def conc(x, centers, heights):
    c = BASE
    for k in range(centers.shape[0]):
        d = abs(x - centers[k])
        d = min(d, L - d)
        c += heights[k] * math.exp(-d * d / (2 * SIGMA * SIGMA))
    return c


@njit(cache=True)
def _live_ct(counts, phi, bnd, ri, rj, a0, a1, b0, b1, kap, centers, heights, x0, uT, ticks, cost,
             reserve0, cap, blind, mode, seed, trace):
    """mode 0: organism; 1: never tumble; 2: tumble with fixed probability 0.1."""
    np.random.seed(seed)
    P = counts.shape[0]
    n = np.zeros((P, 2), dtype=np.int64)
    for p in range(P):
        n[p, 0] = counts[p]
    mu = np.zeros(P)
    flux = np.zeros(P)
    x, direction = x0, 1.0
    mu[F] = 0.0 if blind else math.log(conc(x, centers, heights))
    _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 5.0, flux)
    reserve, survival, intake_sum, tumbles = reserve0, ticks, 0.0, 0
    for t in range(ticks):
        if mode == 0:
            tumble = n[M, 1] > 0
        elif mode == 1:
            tumble = False
        else:
            tumble = uT[t, 1] < 0.1
        if tumble:
            tumbles += 1
            direction = 1.0 if uT[t, 0] < 0.5 else -1.0
        else:
            x = (x + V * direction) % L
        c = conc(x, centers, heights)
        intake = c / C_REF
        intake_sum += intake
        reserve = min(cap, reserve + intake) - cost
        if trace.shape[0] > 0:
            trace[t, 0] = 1.0 if tumble else 0.0
            trace[t, 1] = math.log(c)
            for p in range(2, P):
                trace[t, p] = n[p, 1]
        if reserve <= 0:
            survival = t + 1
            break
        mu[F] = 0.0 if blind else math.log(c)
        mu[M] = 0.0
        _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 1.0, flux)
    return survival, intake_sum / survival, tumbles / survival


@njit(cache=True)
def _respond(counts, phi, bnd, ri, rj, a0, a1, b0, b1, kap, mus, reps, seed):
    """Tumble frequency per tick for a fixed sequence of mu_F, averaged over replicate cells."""
    np.random.seed(seed)
    P = counts.shape[0]
    out = np.zeros(mus.shape[0])
    for r in range(reps):
        n = np.zeros((P, 2), dtype=np.int64)
        for p in range(P):
            n[p, 0] = counts[p]
        mu = np.zeros(P)
        flux = np.zeros(P)
        mu[F] = mus[0]
        _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 20.0, flux)
        for t in range(mus.shape[0]):
            out[t] += 1.0 if n[M, 1] > 0 else 0.0
            mu[F] = mus[t]
            _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 1.0, flux)
    return out / reps


RELATIVE_COST = 2.0  # amendment: cost = 2 x the ring-average intake (passive strategies must die)


def make_world(seed, ticks):
    rng = np.random.default_rng([seed, 11])
    centers = rng.uniform(0, L, 6)
    heights = np.exp(rng.uniform(0, math.log(100), 6))
    xs = np.linspace(0, L, 2000, endpoint=False)
    mean_intake = float(np.mean([conc(x, centers, heights) for x in xs])) / C_REF
    return centers, heights, float(rng.uniform(0, L)), rng.random((ticks, 2)), RELATIVE_COST * mean_intake


def arrays(g):
    P = 2 + g["interior"]
    counts = np.array(g["counts"], dtype=np.int64)
    phi = np.array(g["phi"], dtype=np.float64).reshape(P, 2)
    bnd = np.zeros(P, dtype=np.bool_)
    bnd[:2] = True
    r = g["rules"]
    cols = [np.array([x[c] for x in r], dtype=np.int64) for c in range(6)]
    kap = np.array([math.exp(x[6]) for x in r], dtype=np.float64)
    return (counts, phi, bnd, *cols, kap)


def live(g, seed, ticks=20000, blind=False, mode=0, trace=False, world=None, cost=None):
    """`cost` is ignored: the world sets it (2 x ring-average intake of that seed)."""
    w = world if world is not None else make_world(seed, ticks)
    tr = np.zeros((ticks, 2 + g["interior"])) if trace else np.zeros((0, 0))
    out = _live_ct(*arrays(g), w[0], w[1], w[2], w[3], ticks, w[4], 150.0, 300.0, blind, mode,
                   seed * 7919 + 3, tr)
    return (out, tr) if trace else out


def respond(g, mus, reps=300, seed=5):
    return _respond(*arrays(g), np.asarray(mus, dtype=np.float64), reps, seed)


def to_table(g, name):
    P = 2 + g["interior"]
    names = ["F", "M"] + [f"I{k}" for k in range(g["interior"])]
    return {
        "name": name,
        "status": "GENERATED by selection over admitted rule tables in the chemotaxis world",
        "pages": {nm: p for p, nm in enumerate(names)},
        "boundary_pages": [0, 1],
        "units": {nm: int(g["counts"][p]) for p, nm in enumerate(names)},
        "phi": {str(p): [float(g["phi"][2 * p]), float(g["phi"][2 * p + 1])] for p in range(P)},
        "birth": {str(p): 0 for p in range(P)},
        "rules": [{"name": f"r{k}", "i": x[0], "j": x[1], "a": [x[2], x[3]], "b": [x[4], x[5]],
                   "kappa": float(round(math.exp(x[6]), 6))} for k, x in enumerate(g["rules"])],
        "stimulus": {"calm": 0, "food": "ln c(x)", "motor": 0},
        "tick_time": 1.0,
    }
