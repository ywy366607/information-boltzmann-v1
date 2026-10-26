"""Fast exact simulation and selection of rule-table organisms (M3) in the forage world.

The organism is only its rule table. The world interface is fixed (prereg 2026-10-26):
page 0 F food (boundary; reservoir potential +MU when intake this tick, else -MU),
page 1 A arrival (boundary; +MU on a travel tick, else -MU), page 2 M motor (boundary,
one unit, potential 0; the world switches source when it is in state 1). Pages >= 3 are
interior. Same lumped Gillespie as `cgb_table_medium.LumpedSim` (exact for well-mixed
tables), compiled with numba. World histories are drawn per seed in advance, so every
genome lives through the same hidden states and intake draws.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

F, A, M = 0, 1, 2
MU = 6.0


@njit(cache=True)
def _rates(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, w):
    """Total rate of each rule direction (2k: a -> b, 2k+1: b -> a); returns the sum."""
    tot = 0.0
    for d in range(2 * ri.shape[0]):
        k = d // 2
        if d % 2 == 0:
            s0, s1, d0, d1 = a0[k], a1[k], b0[k], b1[k]
        else:
            s0, s1, d0, d1 = b0[k], b1[k], a0[k], a1[k]
        i, j = ri[k], rj[k]
        m = n[i, s0] * (n[j, s1] - (1 if (i == j and s0 == s1) else 0))
        if m <= 0:
            w[d] = 0.0
            continue
        dc = d0 + d1 - s0 - s1
        res = mu[i] if bnd[i] else (mu[j] if bnd[j] else 0.0)
        dE = phi[i, d0] + phi[j, d1] - phi[i, s0] - phi[j, s1] - res * dc
        w[d] = kap[k] * math.exp(-dE / 2.0) * m
        tot += w[d]
    return tot


@njit(cache=True)
def _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, T, flux):
    R = ri.shape[0]
    w = np.empty(2 * R)
    t = 0.0
    while True:
        tot = _rates(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, w)
        if tot <= 0.0:
            return
        t += -math.log(1.0 - np.random.random()) / tot
        if t > T:
            return
        u = np.random.random() * tot
        d = 0
        acc = w[0]
        while acc < u and d < 2 * R - 1:
            d += 1
            acc += w[d]
        k = d // 2
        if d % 2 == 0:
            s0, s1, d0, d1 = a0[k], a1[k], b0[k], b1[k]
        else:
            s0, s1, d0, d1 = b0[k], b1[k], a0[k], a1[k]
        i, j = ri[k], rj[k]
        n[i, s0] -= 1
        n[i, d0] += 1
        n[j, s1] -= 1
        n[j, d1] += 1
        # content taken in through each boundary page (flux bookkeeping)
        dc = d0 + d1 - s0 - s1
        if bnd[i]:
            flux[i] += dc
        elif bnd[j]:
            flux[j] += dc


@njit(cache=True)
def _live(counts, phi, bnd, ri, rj, a0, a1, b0, b1, kap, rich, uI, uG, ticks, cost, reserve0,
          cap, p_rich, p_poor, cut, blind, seed, trace):
    np.random.seed(seed)
    P = counts.shape[0]
    n = np.zeros((P, 2), dtype=np.int64)
    for p in range(P):
        n[p, 0] = counts[p]
    mu = np.zeros(P)
    flux = np.zeros(P)
    mu[F] = -MU
    mu[A] = -MU
    _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 5.0, flux)
    flux[:] = 0.0
    at, travelling, reserve = 0, False, reserve0
    requested, intake_sum, survival = 0, 0, ticks
    for t in range(ticks):
        action = 1 if n[M, 1] > 0 else 0
        requested += action
        if cut:
            action = 1 if uG[t] < requested / (t + 1) else 0
        if action == 1 and not travelling:
            at = (at + 1) % rich.shape[1]
            travelling = True
        travelled = travelling
        intake = 0
        if travelling:
            travelling = False
        else:
            p = p_rich if rich[t, at] else p_poor
            intake = 1 if uI[t] < p else 0
        intake_sum += intake
        reserve = min(cap, reserve + intake) - cost
        if trace.shape[0] > 0:
            trace[t, 0] = action
            trace[t, 1] = intake
            trace[t, 2] = 1 if travelled else 0
            for p in range(3, min(P, trace.shape[1] - 3 + 3)):
                trace[t, p] = n[p, 1]
        if reserve <= 0:
            survival = t + 1
            break
        mu[F] = MU if (intake == 1 and not blind) else -MU
        mu[A] = MU if travelled else -MU
        mu[M] = 0.0
        _run(n, phi, bnd, mu, ri, rj, a0, a1, b0, b1, kap, 1.0, flux)
    return survival, intake_sum / survival, requested / survival, flux / survival


def make_world(seed, ticks, sources=3, hazard=0.005):
    rng = np.random.default_rng([seed, 7])
    rich = np.zeros((ticks, sources), dtype=np.bool_)
    state = rng.random(sources) < 0.5
    for t in range(ticks):
        flip = rng.random(sources) < hazard
        state = np.where(flip, ~state, state)
        rich[t] = state
    return rich, rng.random(ticks), rng.random(ticks)


def arrays(g):
    P = 3 + g["interior"]
    counts = np.array(g["counts"], dtype=np.int64)
    phi = np.array(g["phi"], dtype=np.float64).reshape(P, 2)
    bnd = np.zeros(P, dtype=np.bool_)
    bnd[:3] = True
    r = g["rules"]
    ri = np.array([x[0] for x in r], dtype=np.int64)
    rj = np.array([x[1] for x in r], dtype=np.int64)
    a0 = np.array([x[2] for x in r], dtype=np.int64)
    a1 = np.array([x[3] for x in r], dtype=np.int64)
    b0 = np.array([x[4] for x in r], dtype=np.int64)
    b1 = np.array([x[5] for x in r], dtype=np.int64)
    kap = np.array([math.exp(x[6]) for x in r], dtype=np.float64)
    return counts, phi, bnd, ri, rj, a0, a1, b0, b1, kap


def live(g, seed, ticks=3000, cut=False, blind=False, trace=False, world=None, cost=0.6):
    w = world if world is not None else make_world(seed, ticks)
    tr = np.zeros((ticks, 3 + max(g["interior"], 1)), dtype=np.int64) if trace else np.zeros((0, 0), dtype=np.int64)
    out = _live(*arrays(g), w[0], w[1], w[2], ticks, cost, 150.0, 300.0, 0.9, 0.1, cut, blind, seed * 7919 + 1, tr)
    return (out, tr) if trace else out


def to_table(g, name):
    """The genome as a rule table in the cgb_table_medium JSON format (for Lean)."""
    P = 3 + g["interior"]
    names = ["F", "A", "M"] + [f"I{k}" for k in range(g["interior"])]
    return {
        "name": name,
        "status": "GENERATED by selection over admitted rule tables (cgb_table_fast)",
        "pages": {nm: p for p, nm in enumerate(names)},
        "boundary_pages": [0, 1, 2],
        "units": {nm: int(g["counts"][p]) for p, nm in enumerate(names)},
        "phi": {str(p): [float(g["phi"][2 * p]), float(g["phi"][2 * p + 1])] for p in range(P)},
        "birth": {str(p): 0 for p in range(P)},
        "rules": [{"name": f"r{k}", "i": x[0], "j": x[1], "a": [x[2], x[3]], "b": [x[4], x[5]],
                   "kappa": float(round(math.exp(x[6]), 6))} for k, x in enumerate(g["rules"])],
        "stimulus": {"calm": 0, "food_on": MU, "food_off": -MU, "arrive_on": MU, "arrive_off": -MU,
                     "motor": 0},
        "tick_time": 1.0,
    }


def table_genome(t):
    """A cgb_table_medium JSON table (generated layout: pages F, A, M, then interior) as a genome."""
    P = len(t["pages"])
    return {"interior": P - 3, "counts": [t["units"][nm] for nm in sorted(t["pages"], key=t["pages"].get)],
            "phi": [v for p in range(P) for v in t["phi"][str(p)]],
            "rules": [[r["i"], r["j"], r["a"][0], r["a"][1], r["b"][0], r["b"][1], math.log(r["kappa"])]
                      for r in t["rules"]]}
