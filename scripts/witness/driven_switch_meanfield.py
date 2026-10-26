"""Driven two-step enzyme cycle from pair rules with an energy staircase.
State-1 energies: kinase phiK, substrate 0, phosphatase phiP. Fuel E at muE, drain D at muD.
  r1: E0 + K0 <-> E0 + K1   dE = phiK - muE
  r2: K1 + S0 <-> K0 + S1   dE = 0 - phiK         (downhill if phiK > 0)
  r3: S1 + P0 <-> S0 + P1   dE = phiP - 0         (downhill if phiP < 0)
  r4: D0 + P1 <-> D0 + P0   dE = -phiP + muD      (downhill if muD < phiP)
Input: number of kinase units nK (Vmax only). Output s = S1/nS.
"""
import itertools
import numpy as np
from numba import njit


@njit(cache=True)
def steady(nK, nS, nP, muE, muD, phiK, phiP, k1, k2, k3, k4):
    f1, b1 = k1 * np.exp(-(phiK - muE) / 2), k1 * np.exp((phiK - muE) / 2)
    f2, b2 = k2 * np.exp(phiK / 2), k2 * np.exp(-phiK / 2)
    f3, b3 = k3 * np.exp(-phiP / 2), k3 * np.exp(phiP / 2)
    f4, b4 = k4 * np.exp(-(muD - phiP) / 2), k4 * np.exp((muD - phiP) / 2)
    K1, S1, P1 = 0.5 * nK, 0.5 * nS, 0.5 * nP
    rate = f1 + b1 + (f2 + b2) * (nS + nK) + (f3 + b3) * (nS + nP) + f4 + b4
    dt = 0.05 / rate
    for _ in range(50000000):
        K0, S0, P0 = nK - K1, nS - S1, nP - P1
        J1 = f1 * K0 - b1 * K1
        J2 = f2 * K1 * S0 - b2 * K0 * S1
        J3 = f3 * S1 * P0 - b3 * S0 * P1
        J4 = f4 * P1 - b4 * P0
        dK, dS, dP = J1 - J2, J2 - J3, J3 - J4
        K1 += dt * dK
        S1 += dt * dS
        P1 += dt * dP
        if abs(dK) + abs(dS) + abs(dP) < 1e-11 * nS:
            break
    return S1 / nS


def hill(xs, s):
    if not (s.min() < 0.1 and s.max() > 0.9):
        return None
    a, b = np.interp(0.1, s, xs), np.interp(0.9, s, xs)
    return float(np.log(81) / np.log(b / a))


nKs = np.exp(np.linspace(np.log(0.5), np.log(32), 61))
rows = []
for nS, step, k1 in itertools.product((100.0, 1000.0), (4.0, 8.0), (1e-3, 1e-2, 1e-1)):
    muE, phiK, phiP, muD = 3 * step, 2 * step, -step, -2 * step
    s = np.array([steady(nK, nS, 4.0, muE, muD, phiK, phiP, k1, 1.0, 1.0, k1) for nK in nKs])
    h = hill(nKs, s)
    rows.append((h if h else 0.0, nS, step, k1))
    print(f"nS {nS:6} step {step} k1=k4 {k1}: n_H {h}  s at nK=3,4,5: {np.round(np.interp([3, 4, 5], nKs, s), 3)}"
          f"  range [{s.min():.3f}, {s.max():.3f}]", flush=True)
print("best", max(rows))
