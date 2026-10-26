"""An organism (rule table) living in the 2D food world, coupled only through its boundary
(prereg CGB_WORLD2D_PREREG_20261026 §3).

Pages (fixed interface, everything else is the organism's own table):
  F 0  receptors: reservoir = the cell's food (binding takes a quantum out of the cell, unbinding
       returns it; otherwise content could enter through F for free and be passed to E)
  M 1  motor (1 unit): state 1 = run (the organism moves this tick), state 0 = tumble (new heading)
  E 2  intake / reserve: reservoir = the cell's food; every quantum that enters E leaves the cell,
       every quantum that leaves E returns to it (exact, event by event; no food, no intake)
  D 3  waste: reservoir = the outer sink (constant low potential); what leaves through D is gone
  S 4  body structure: the reversible rule S1 + D0 <-> S0 + D1 has a physical rate and energy
       (world temperature, not selectable): structure decays into waste, spontaneous repair from
       waste is suppressed by the energy; repair S0 + E1 -> S1 + E0 is the organism's rule
  5..  interior pages
Fixed interface rules (rates selectable except decay):
  uptake        E0 + E0 -> E1 + E0   (a quantum from the cell into the reserve)
  binding       F0 + F0 -> F1 + F0   (a receptor binds a quantum of the cell)
  excretion     D1 + D0 -> D0 + D0   (waste leaves to the sink)
  motor stroke  M0 + E1 -> M1 + E0   (fuel into the motor)
  motor release D0 + M1 -> D0 + M0   (the stroke's quantum leaves to the sink)
  repair        S0 + E1 -> S1 + E0
  decay         S1 + D0 -> S0 + D1   (physics: fixed rate and energy)
Reserve = content of E. Death = no intact structure (S1 = 0): the body's content returns to its
cell as food. Content unit = one food unit = SCALE quanta of the world.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

F, M, E, D, S = 0, 1, 2, 3, 4
NB = 5          # pages 0..4 are the interface
MU_SINK = -6.0
LOG_DECAY = math.log(0.002)   # structure decay prefactor (physics constant, prereg §3)
PHI_S1 = 4.0                   # energy of intact structure (physics constant)
CELL = (F, E)                  # pages whose reservoir is the cell's food
FOOD_MU0 = 8.0   # chemical energy of food (world physics): the cell's potential is
                 # FOOD_MU0 + log(n / (V - n + 1)); injection brings free energy, not only matter.
                 # Food must outrank fuel (E1) and fuel must outrank structure (S1) for life to run.


@njit(cache=True)
def _mult(n, i, s0, j, s1):
    m = n[i, s0]
    if m <= 0:
        return 0
    c1 = n[j, s1] - (1 if (j == i and s1 == s0) else 0)
    if c1 <= 0:
        return 0
    return m * c1


@njit(cache=True)
def _rates(n, phi, mu, R, kap, w, food):
    """Rates of all rule directions. mu[p]: reservoir potential of boundary page p (F, E: the
    cell's food potential; D: the sink; others 0). An event whose content change goes through E
    needs that many food quanta in the cell (food = quanta available, in content units)."""
    tot = 0.0
    for d in range(2 * R.shape[0]):
        r = d // 2
        i, j = R[r, 0], R[r, 1]
        if d % 2 == 0:
            s0, s1, d0, d1 = R[r, 2], R[r, 3], R[r, 4], R[r, 5]
        else:
            s0, s1, d0, d1 = R[r, 4], R[r, 5], R[r, 2], R[r, 3]
        m = _mult(n, i, s0, j, s1)
        if m <= 0:
            w[d] = 0.0
            continue
        dc = d0 + d1 - s0 - s1
        dE = phi[i, d0] + phi[j, d1] - phi[i, s0] - phi[j, s1]
        res = -1
        if i < NB and i != M and i != S:
            res = i
        elif j < NB and j != M and j != S:
            res = j
        if res >= 0:
            dE -= mu[res] * dc
            if (res == E or res == F) and dc > food:   # no food in the cell: nothing to take
                w[d] = 0.0
                continue
        w[d] = kap[r] * math.exp(-dE / 2.0) * m
        tot += w[d]
    return tot


@njit(cache=True)
def run_tick(n, phi, mu, R, kap, food):
    """One tick of the organism's own events (exact Gillespie). `food`: content units available
    in the cell. Returns (food taken from the cell, content sent to the sink); negative taken
    means food returned to the cell."""
    w = np.empty(2 * R.shape[0])
    t, taken, sunk = 0.0, 0, 0
    while True:
        tot = _rates(n, phi, mu, R, kap, w, food - taken)
        if tot <= 0.0:
            break
        t += -math.log(1.0 - np.random.random()) / tot
        if t > 1.0:
            break
        u = np.random.random() * tot
        d = 0
        acc = w[0]
        while acc < u and d < w.shape[0] - 1:
            d += 1
            acc += w[d]
        r = d // 2
        i, j = R[r, 0], R[r, 1]
        if d % 2 == 0:
            s0, s1, d0, d1 = R[r, 2], R[r, 3], R[r, 4], R[r, 5]
        else:
            s0, s1, d0, d1 = R[r, 4], R[r, 5], R[r, 2], R[r, 3]
        n[i, s0] -= 1
        n[i, d0] += 1
        n[j, s1] -= 1
        n[j, d1] += 1
        dc = d0 + d1 - s0 - s1
        res = -1
        if i < NB and i != M and i != S:
            res = i
        elif j < NB and j != M and j != S:
            res = j
        if res == E or res == F:
            taken += dc
        elif res == D:
            sunk -= dc
    return taken, sunk


def content(n):
    return int(sum(n[p, s] * s for p in range(n.shape[0]) for s in range(n.shape[1])))


# ---------- genome and a life in the world ----------

IFACE = ("uptake", "binding", "excretion", "stroke", "release", "repair")


def interface_rules(g):
    """Fixed interface rules [i, j, a0, a1, b0, b1, logk]; rates from the genome except decay."""
    k = g["iface_logk"]
    return [[E, E, 0, 0, 1, 0, k["uptake"]],
            [F, F, 0, 0, 1, 0, k["binding"]],
            [D, D, 1, 0, 0, 0, k["excretion"]],
            [M, E, 0, 1, 1, 0, k["stroke"]],
            [D, M, 0, 1, 0, 0, k["release"]],
            [S, E, 0, 1, 1, 0, k["repair"]],
            [S, D, 1, 0, 0, 1, LOG_DECAY]]


def arrays(g):
    """counts, phi, R, kap, Smax. Pages 0..4 interface (2 states each), 5.. interior."""
    states = [2, 2, 2, 2, 2] + list(g["states"])
    P, Sx = len(states), max(states)
    counts = np.array([g["nF"], 1, g["nE"], g["nD"], g["nS"]] + list(g["counts"]), dtype=np.int64)
    phi = np.zeros((P, Sx))
    phi[F, 1], phi[M, 1], phi[E, 1], phi[D, 1], phi[S, 1] = g["phiF"], g["phiM"], g["phiE"], g["phiD"], PHI_S1
    for p, e in enumerate(g["phi"]):
        phi[NB + p, :len(e)] = e
    rules = interface_rules(g) + [list(r) for r in g["rules"]]
    R = np.array([r[:6] for r in rules], dtype=np.int64)
    N = float(counts.sum())
    kap = np.array([math.exp(r[6]) / N for r in rules])   # per-pair prefactor (per concentration)
    return counts, phi, R, kap, Sx


def birth_state(g, counts, Sx):
    n = np.zeros((counts.shape[0], Sx), dtype=np.int64)
    n[:, 0] = counts
    n[S, 0], n[S, 1] = 0, counts[S]                       # born with an intact body
    r0 = min(int(g.get("reserve0", counts[E] // 2)), int(counts[E]))
    n[E, 0], n[E, 1] = counts[E] - r0, r0                 # and a birth reserve
    return n


def live(g, seed, ticks=10000, wkw=None, blind=False, trace=False, world=None):
    """One life in the 2D world. Returns dict(survival, intake, ...) and optionally a trace."""
    import cgb_world2d as WD
    w = world if world is not None else WD.World2D(seed, **(wkw or {}))
    counts, phi, R, kap, Sx = arrays(g)
    np.random.seed(seed * 7919 + 13)
    rng = np.random.default_rng(seed + 991)
    n = birth_state(g, counts, Sx)
    x, y = rng.uniform(0, w.g, 2)
    ang = rng.uniform(0, 2 * math.pi)
    mu = np.zeros(counts.shape[0])
    mu[D] = MU_SINK
    mu_blind = FOOD_MU0 + math.log((2.0 * WD.SCALE + 0.5) / (w.v - 2.0 * WD.SCALE + 1))   # a typical cell
    intake, survival, tr = 0, ticks, []
    for t in range(ticks):
        c = w.cell(x, y)
        q = int(w.food[c])
        m = FOOD_MU0 + w.mu(x, y)
        mu[F] = mu_blind if blind else m
        mu[E] = m
        taken, _ = run_tick(n, phi, mu, R, kap, q // WD.SCALE)
        if taken > 0:
            w.food[c] -= taken * WD.SCALE
            intake += taken
        elif taken < 0:
            w.give(x, y, -taken * WD.SCALE)
        moved = n[M, 1] == 1
        if moved:
            x, y = (x + 0.5 * math.cos(ang)) % w.g, (y + 0.5 * math.sin(ang)) % w.g
        else:
            ang = rng.uniform(0, 2 * math.pi)
        if trace:
            tr.append((x, y, int(moved), int(n[E, 1]), int(n[S, 1]), q))
        if n[S, 1] == 0:                                   # dead: the body returns as food
            w.give(x, y, content(n) * WD.SCALE)
            survival = t + 1
            break
        w.step()
    out = dict(survival=survival, intake=intake / max(1, survival))
    return (out, tr) if trace else out
