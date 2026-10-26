"""Reachability witness (NOT an organism): chemotaxis from an M4 table (pair + three-body rules).

Interface (unified, essence-derived): F receptors (boundary, mu = ln c), M motor (boundary, one
unit, mu = 0; state 1 = tumble), E store (boundary, mu = MU0 + ln(reserve/150)), D waste
(boundary, mu = -MU0). Interior: Y memory (slow copy of receptor occupancy), K kinase (few),
S substrate (many), P phosphatase (few).

  F0 + D0 <-> F1 + D0          receptor binds ligand (reservoir work at F: mu_F)
  F1 + Y0 <-> F0 + Y1          slow transfer: Y tracks receptor occupancy (memory, "methylation")
  E0 + F0 + K0 <-> E0 + F0 + K1  kinase charged from the store, catalysed by FREE receptors
  E0 + Y1 + K0 <-> E0 + Y1 + K1  ... and by memory: rate ~ (1 - f) + y, constant when y = f
  K1 + S0 <-> K0 + S1          downhill transfer to substrate (energy staircase)
  S1 + P0 <-> S0 + P1          downhill transfer to phosphatase
  P1 + D0 <-> P0 + D0          drain to waste (reservoir work at D)
  M0 + S1 <-> M1 + S0          motor follows the substrate (tumble when s is high)
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_table3 as T  # noqa: E402
from cgb_chemotaxis import conc, make_world, L, V, C_REF  # noqa: E402

F, M, E, D, Y, K, S, P = range(8)
STEP = 4.0
MU0 = 3 * STEP


def table(nF=512, nS=100, nK=4, nP=4, ky=0.002, kc=1e-4, kd=0.1, km=1.0, phim=0.0):
    rules = [
        (F, D, -1, (0, 0), (1, 0), 1.0),
        (F, Y, -1, (1, 0), (0, 1), ky),
        (E, F, K, (0, 0, 0), (0, 0, 1), kc),
        (E, Y, K, (0, 1, 0), (0, 1, 1), kc),
        (K, S, -1, (1, 0), (0, 1), 1.0),
        (S, P, -1, (1, 0), (0, 1), 1.0),
        (D, P, -1, (0, 1), (0, 0), kd),
        (M, S, -1, (0, 1), (1, 0), km),
    ]
    phi = [[0, 0], [0, phim], [0, 0], [0, 0], [0, 0], [0, 2 * STEP], [0, 0], [0, -STEP]]
    return {"counts": [nF, 1, 4, 4, nF, nK, nS, nP], "phi": phi, "boundary": [F, M, E, D], "rules": rules}


@njit(cache=True)
def _respond(counts, phi, bnd, R, kap, Sx, mus, reps, seed):
    np.random.seed(seed)
    P_ = counts.shape[0]
    out = np.zeros(mus.shape[0])
    for _ in range(reps):
        n = np.zeros((P_, Sx), dtype=np.int64)
        for p in range(P_):
            n[p, 0] = counts[p]
        mu = np.zeros(P_)
        fl = np.zeros(P_)
        mu[E], mu[D], mu[F] = MU0, -MU0, mus[0]
        T._run(n, phi, bnd, mu, R, kap, 300.0, fl)
        for t in range(mus.shape[0]):
            out[t] += 1.0 if n[M, 1] > 0 else 0.0
            mu[F] = mus[t]
            T._run(n, phi, bnd, mu, R, kap, 1.0, fl)
    return out / reps


@njit(cache=True)
def _live(counts, phi, bnd, R, kap, Sx, centers, heights, x0, uT, ticks, cost, seed):
    np.random.seed(seed)
    P_ = counts.shape[0]
    n = np.zeros((P_, Sx), dtype=np.int64)
    for p in range(P_):
        n[p, 0] = counts[p]
    mu = np.zeros(P_)
    fl = np.zeros(P_)
    x, direction, reserve = x0, 1.0, 150.0
    mu[F], mu[E], mu[D] = math.log(conc(x, centers, heights)), MU0, -MU0
    T._run(n, phi, bnd, mu, R, kap, 300.0, fl)
    tumbles = 0
    for t in range(ticks):
        if n[M, 1] > 0:
            tumbles += 1
            direction = 1.0 if uT[t, 0] < 0.5 else -1.0
        else:
            x = (x + V * direction) % L
        c = conc(x, centers, heights)
        reserve = min(300.0, reserve + c / C_REF) - cost
        if reserve <= 0:
            return t + 1, tumbles / (t + 1)
        mu[F] = math.log(c)
        mu[E] = MU0 + math.log(reserve / 150.0)
        T._run(n, phi, bnd, mu, R, kap, 1.0, fl)
    return ticks, tumbles / ticks


def respond(tab, mus, reps=200, seed=3):
    c, phi, bnd, R, kap, Sx = T.arrays(tab)
    return _respond(c, phi, bnd, R, kap, Sx, np.asarray(mus, dtype=np.float64), reps, seed)


def live(tab, seed, ticks=20000):
    c, phi, bnd, R, kap, Sx = T.arrays(tab)
    w = make_world(seed, ticks)
    return _live(c, phi, bnd, R, kap, Sx, w[0], w[1], w[2], w[3], ticks, w[4], seed * 7919 + 3)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "step":
        for kc, kd in itertools.product((3e-5, 1e-4, 3e-4), (0.03, 0.1, 0.3)):
            tab = table(kc=kc, kd=kd)
            line = []
            for mu0 in (-1.0, 1.0, 3.0):
                r = respond(tab, [mu0] * 100 + [mu0 + 1.0] * 100 + [mu0] * 100, reps=100)
                line.append(f"mu0 {mu0:+}: base {r[80:100].mean():.2f} up {r[100:105].mean():.2f} "
                            f"late {r[180:200].mean():.2f} down {r[200:205].mean():.2f}")
            print(f"kc {kc} kd {kd} | " + " | ".join(line), flush=True)
    else:
        kc, kd = float(sys.argv[2]), float(sys.argv[3])
        tab = table(kc=kc, kd=kd)
        s = [live(tab, sd) for sd in range(101, 109)]
        sv = [x[0] for x in s]
        print("survival", sv, "median", np.median(sv), "mean", np.mean(sv), "tumble", [round(x[1], 3) for x in s])
