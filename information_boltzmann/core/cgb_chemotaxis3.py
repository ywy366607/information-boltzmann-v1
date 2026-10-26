"""Chemotaxis in a world that is itself a medium (G26, nesting).

The ring of `cgb_chemotaxis` (L cells) now holds conserved food. Transport inside the world is
local (P1b holds for the world too): food enters only at source cells (the world's own
boundary, one per old peak, rate proportional to the peak height), spreads to neighbouring
cells by diffusion DIFF and decays at DECAY everywhere (the world's outer reservoir). The
steady profile is BASE + sum_k h_k exp(-|x - x_k| / LAM), LAM = sqrt(DIFF / DECAY): gradients
point to the sources. The organism at x eats a fraction BETA of the cell per tick (the eaten
food leaves the world). The receptor reservoir is the cell's potential, mu_F = log n_x (G26a,
dilute limit). Staying put yields at most what flows into the site (G26b, `stay_starves`).

A first version regenerated every cell from its own reservoir (every cell on the world's
boundary): there the best policy was to cruise blind, senses were useless (recorded in
CGB_NESTED_ECOLI_PREREG).

Reserve gain is eaten / (BETA * C_REF), so a fresh cell pays exactly what the old world paid.
World mean-field (deterministic food), organism exact (Gillespie). Same organism arrays and
page layout as `cgb_chemotaxis2` (F, M, E, D, interior).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from cgb_chemotaxis import BASE, C_REF, L, V, make_world as _make_world
from cgb_chemotaxis2 import D, E, F, FUEL, M, MU0, NB, arrays  # noqa: F401
from cgb_table3 import _run

BETA, NCELL, DIFF, LAM = 0.2, 400, 0.45, 25.0
DECAY = DIFF / LAM ** 2
COST_FACTOR = 1.0   # cost = COST_FACTOR x ring-average fresh intake (calibrated with ideal policies)


def make_world(seed, ticks, cost_factor=None):
    """The same peak positions and heights, start and tumble randomness as
    `cgb_chemotaxis.make_world`; peaks become sources. Returns (food0, src, x0, uT, cost) with
    src[c] the inflow into cell c."""
    centers, heights, x0, uT, _ = _make_world(seed, ticks)
    xs = (np.arange(NCELL) + 0.5) * (L / NCELL)
    food0 = np.full(NCELL, BASE)
    src = np.zeros(NCELL)
    for c0, h in zip(centers, heights):
        d = np.abs(xs - c0)
        d = np.minimum(d, L - d)
        food0 += h * np.exp(-d / LAM)
        src[int(c0 / L * NCELL) % NCELL] += h * 2.0 * math.sqrt(DIFF * DECAY)
    src += BASE * DECAY
    cost = (COST_FACTOR if cost_factor is None else cost_factor) * float(food0.mean()) / C_REF
    return food0, src, x0, uT, cost


@njit(cache=True)
def _world_step(food, src, tmp):
    n = food.shape[0]
    for c in range(n):
        tmp[c] = food[c] + DIFF * (food[(c - 1) % n] + food[(c + 1) % n] - 2.0 * food[c])             - DECAY * food[c] + src[c]
    for c in range(n):
        food[c] = max(tmp[c], 0.0)


@njit(cache=True)
def _live(counts, phi, bnd, R, kap, Sx, food0, src, x0, uT, ticks, cost, blind, mode, calm, seed, trace, xs):
    np.random.seed(seed)
    P = counts.shape[0]
    n = np.zeros((P, Sx), dtype=np.int64)
    for p in range(P):
        n[p, 0] = counts[p]
    food = food0.copy()
    tmp = np.zeros_like(food)
    mu = np.zeros(P)
    fl = np.zeros(P)
    x, direction, reserve = x0, 1.0, 150.0
    cell = int(x / L * NCELL) % NCELL
    mu[F] = 0.0 if blind else math.log(food[cell] + 1e-9)
    mu[E], mu[D] = (0.0, 0.0) if calm else (MU0, -MU0)
    _run(n, phi, bnd, mu, R, kap, 20.0, fl)
    tumbles, intake_sum, survival = 0, 0.0, ticks
    for t in range(ticks):
        if mode == 0:
            tumble = n[M, 0] == 0
        elif mode == 1:
            tumble = False
        else:
            tumble = uT[t, 1] < 0.1
        if tumble:
            tumbles += 1
            direction = 1.0 if uT[t, 0] < 0.5 else -1.0
        else:
            x = (x + V * direction) % L
        cell = int(x / L * NCELL) % NCELL
        eaten = BETA * food[cell]
        food[cell] -= eaten
        _world_step(food, src, tmp)
        intake = eaten / (BETA * C_REF)
        intake_sum += intake
        fuel = max(fl[E], 0.0)
        reserve = min(300.0, reserve + intake) - cost - FUEL * fuel
        if xs.shape[0] > 0:
            xs[t] = x
            trace[t, 0] = 1.0 if tumble else 0.0
            trace[t, 1] = reserve
            trace[t, 2] = food[cell]
            for p in range(NB, P):
                occ = 0.0
                for s in range(1, Sx):
                    occ += n[p, s] * s
                trace[t, 3 + p - NB] = occ / max(1, counts[p])
        if reserve <= 0:
            survival = t + 1
            break
        mu[F] = 0.0 if blind else math.log(food[cell] + 1e-9)
        mu[E] = 0.0 if calm else MU0 + math.log(max(reserve, 1e-3) / 150.0)
        fl[:] = 0.0
        _run(n, phi, bnd, mu, R, kap, 1.0, fl)
    return survival, intake_sum / survival, tumbles / survival


def live(g, seed, ticks=10000, blind=False, mode=0, calm=False, trace=False, world=None, **_):
    w = world if world is not None else make_world(seed, ticks)
    P = NB + g["interior"]
    tr = np.zeros((ticks, 3 + P - NB)) if trace else np.zeros((0, 0))
    xs = np.zeros(ticks) if trace else np.zeros(0)
    out = _live(*arrays(g), w[0], w[1], w[2], w[3], ticks, w[4], blind, mode, calm, seed * 7919 + 3, tr, xs)
    return (out, tr, xs) if trace else out


def policy_survival(w, policy, ticks=10000, theta=0.0, base=0.02):
    """Ideal policies on the exact field (no organism): 'never', 'threshold' (stop while
    log food >= theta), 'comparator' (tumble when log food fell over the last tick, or at the
    base rate)."""
    food0, src, x0, uT, cost = w
    food = food0.copy()
    tmp = np.zeros_like(food)
    x, d, reserve = x0, 1.0, 150.0
    prev = math.log(food[int(x / L * NCELL) % NCELL] + 1e-9)
    for t in range(ticks):
        cell = int(x / L * NCELL) % NCELL
        lc = math.log(food[cell] + 1e-9)
        if policy == "never":
            tumble = False
        elif policy == "threshold":
            tumble = lc >= theta
        else:
            tumble = lc < prev or uT[t, 1] < base
        prev = lc
        if tumble:
            d = 1.0 if uT[t, 0] < 0.5 else -1.0
        else:
            x = (x + V * d) % L
        cell = int(x / L * NCELL) % NCELL
        eaten = BETA * food[cell]
        food[cell] -= eaten
        _world_step(food, src, tmp)
        reserve = min(300.0, reserve + eaten / (BETA * C_REF)) - cost
        if reserve <= 0:
            return t + 1
    return ticks
