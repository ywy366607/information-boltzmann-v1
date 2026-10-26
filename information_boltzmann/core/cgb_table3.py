"""Rule tables with pair AND three-body rules (generator M4; Lean: TripleCRN, pending).

A three-body rule acts on units of pages (i, j, k) in states (a0, a1, a2) and moves them to
(b0, b1, b2). A unit whose state does not change is a catalyst or modulator: it sets the rate
(mass action counts it) but cannot shift equilibrium. Content of a state is its index. The
energy change is the change of single-unit energies minus reservoir work mu * dc, taken at the
FIRST boundary participant (i, then j, then k). Rates kappa * exp(-dE / 2) * multiplicity.
A pair rule is a three-body rule with k = -1. Exact lumped Gillespie, compiled with numba.

Rule array R: one row per rule, columns (i, j, k, a0, a1, a2, b0, b1, b2); kap: prefactors.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _mult(n, i, s0, j, s1, k, s2):
    m = n[i, s0]
    if m <= 0:
        return 0
    c1 = n[j, s1] - (1 if (j == i and s1 == s0) else 0)
    if c1 <= 0:
        return 0
    m *= c1
    if k >= 0:
        c2 = n[k, s2] - (1 if (k == i and s2 == s0) else 0) - (1 if (k == j and s2 == s1) else 0)
        if c2 <= 0:
            return 0
        m *= c2
    return m


@njit(cache=True)
def _rates(n, phi, bnd, mu, R, kap, w):
    tot = 0.0
    for d in range(2 * R.shape[0]):
        r = d // 2
        i, j, k = R[r, 0], R[r, 1], R[r, 2]
        if d % 2 == 0:
            s0, s1, s2, d0, d1, d2 = R[r, 3], R[r, 4], R[r, 5], R[r, 6], R[r, 7], R[r, 8]
        else:
            s0, s1, s2, d0, d1, d2 = R[r, 6], R[r, 7], R[r, 8], R[r, 3], R[r, 4], R[r, 5]
        m = _mult(n, i, s0, j, s1, k, s2)
        if m <= 0:
            w[d] = 0.0
            continue
        dc = d0 + d1 - s0 - s1
        dE = phi[i, d0] + phi[j, d1] - phi[i, s0] - phi[j, s1]
        if k >= 0:
            dc += d2 - s2
            dE += phi[k, d2] - phi[k, s2]
        if bnd[i]:
            res = mu[i]
        elif bnd[j]:
            res = mu[j]
        elif k >= 0 and bnd[k]:
            res = mu[k]
        else:
            res = 0.0
        dE -= res * dc
        w[d] = kap[r] * math.exp(-dE / 2.0) * m
        tot += w[d]
    return tot


@njit(cache=True)
def _run(n, phi, bnd, mu, R, kap, T, flux):
    w = np.empty(2 * R.shape[0])
    t = 0.0
    while True:
        tot = _rates(n, phi, bnd, mu, R, kap, w)
        if tot <= 0.0:
            return
        t += -math.log(1.0 - np.random.random()) / tot
        if t > T:
            return
        u = np.random.random() * tot
        d = 0
        acc = w[0]
        while acc < u and d < w.shape[0] - 1:
            d += 1
            acc += w[d]
        r = d // 2
        i, j, k = R[r, 0], R[r, 1], R[r, 2]
        if d % 2 == 0:
            s0, s1, s2, d0, d1, d2 = R[r, 3], R[r, 4], R[r, 5], R[r, 6], R[r, 7], R[r, 8]
        else:
            s0, s1, s2, d0, d1, d2 = R[r, 6], R[r, 7], R[r, 8], R[r, 3], R[r, 4], R[r, 5]
        n[i, s0] -= 1
        n[i, d0] += 1
        n[j, s1] -= 1
        n[j, d1] += 1
        dc = d0 + d1 - s0 - s1
        if k >= 0:
            n[k, s2] -= 1
            n[k, d2] += 1
            dc += d2 - s2
        if bnd[i]:
            flux[i] += dc
        elif bnd[j]:
            flux[j] += dc
        elif k >= 0 and bnd[k]:
            flux[k] += dc


def arrays(table):
    """table: dict with 'counts' (per page), 'phi' (per page, list of state energies),
    'boundary' (list of pages), 'rules' (list of (i, j, k, a, b, kappa) with a, b tuples)."""
    P = len(table["counts"])
    S = max(len(p) for p in table["phi"])
    counts = np.array(table["counts"], dtype=np.int64)
    phi = np.zeros((P, S))
    for p, e in enumerate(table["phi"]):
        phi[p, :len(e)] = e
    bnd = np.zeros(P, dtype=np.bool_)
    bnd[list(table["boundary"])] = True
    rows, kap = [], []
    for (i, j, k, a, b, kappa) in table["rules"]:
        a = list(a) + [0] * (3 - len(a))
        b = list(b) + [0] * (3 - len(b))
        rows.append([i, j, k, a[0], a[1], a[2], b[0], b[1], b[2]])
        kap.append(kappa)
    return counts, phi, bnd, np.array(rows, dtype=np.int64), np.array(kap, dtype=np.float64), S


def fresh_state(counts, S):
    n = np.zeros((counts.shape[0], S), dtype=np.int64)
    n[:, 0] = counts
    return n
