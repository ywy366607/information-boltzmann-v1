"""Diagnostic (not an organism): is temporal comparison + multi-quanta amplification reachable
inside the ms generator's space (4 boundary pages, <= 4 interior pages, <= 4 states, <= 16 pair
rules)? Built by hand from two existing results, run in the same worlds as the search:

* comparator (CGB_GENERATED_CHEMOTAXIS_PREREG s2, reachability witness): a fast page X tracks the
  present, a slow page Y the past; Y raises the motor, X lowers it (`X+M1 -> X0+M0` lowers content,
  paid at the motor reservoir, which is what flips the sign);
* amplifier (G25d `quanta_hill`): X and Y are ladders 0..3 bound from the receptor reservoir
  (`F0 + X_s <-> F0 + X_{s+1}`, so X_s ~ exp(s mu - phi_s)); the motor rules use the 3-quanta
  states, so P(tumble) ~ sigma(3 (mu_past - mu_now) + b).

Pages: F 0, M 1, E 2, D 3, X 4, Y 5. Genome format of `cgb_chemotaxis2`.
If it survives where s0 dies, failure of the search is reachability or selection pressure; if
not, the cause is the world or scale. A diagnostic is never admitted as an organism.

usage: python scripts/witness/ms_comparator_witness.py [--quick]
"""

from __future__ import annotations

import itertools
import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import cgb_chemotaxis2 as W  # noqa: E402

HELD = tuple(range(101, 109))
VAL = tuple(range(500, 516))


def genome(theta, k_fast, k_slow, bias, nX=512, nY=512, nF=1024):
    """theta: per-quantum energy of X and Y (phi_s = s * theta); k_*: log rate prefactors;
    bias: log of motor-on over motor-off prefactor."""
    F, M, E, D, X, Y = 0, 1, 2, 3, 4, 5
    phi_ladder = [0.0, theta, 2 * theta, 3 * theta]
    rules = []
    for p, k in ((X, k_fast), (Y, k_slow)):
        for s in range(3):
            rules.append([F, p, 0, s, 0, s + 1, k])          # F0 catalyses binding from mu_F
    # motor: past high -> tumble; present high -> run. Strongly downhill (dE = -3 theta).
    rules.append([Y, M, 3, 0, 0, 1, 1.0 + bias / 2])
    rules.append([X, M, 3, 1, 0, 0, 1.0 - bias / 2])
    # metabolism / export through the store and waste pages (life gate), weak
    rules.append([E, D, 1, 0, 0, 1, -6.0])
    return {"interior": 2, "states": [2, 2, 2, 2, 4, 4], "counts": [nF, 1, 4, 4, nX, nY],
            "phi": [[0.0, 8.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], phi_ladder, list(phi_ladder)],
            "rules": rules}


def survival(args):
    g, seeds, kw = args
    return [int(W.live(g, s, 10000, **kw)[0]) for s in seeds]


def step_response(g, mu0=1.0, dmu=0.5, reps=200):
    """Tumble fraction in the 20 ticks after mu steps up vs down, adapted at mu0 for 400 ticks."""
    from cgb_table3 import _run
    counts, phi, bnd, R, kap, S = W.arrays(g)
    out = {}
    for sign in (1, -1):
        tum = 0
        for r in range(reps):
            np.random.seed(1000 * (sign + 2) + r)
            n = np.zeros((counts.shape[0], S), dtype=np.int64)
            n[:, 0] = counts
            mu = np.zeros(counts.shape[0])
            mu[W.F], mu[W.E], mu[W.D] = mu0, W.MU0, -W.MU0
            fl = np.zeros(counts.shape[0])
            _run(n, phi, bnd, mu, R, kap, 400.0, fl)
            mu[W.F] = mu0 + sign * dmu
            for _ in range(20):
                _run(n, phi, bnd, mu, R, kap, 1.0, fl)
                tum += int(n[W.M, 0] == 0)
        out["up" if sign > 0 else "down"] = tum / (20 * reps)
    return out


def main():
    quick = "--quick" in sys.argv
    grid = list(itertools.product((1.0, 2.0, 3.0), (2.0,), (-5.0, -3.5), (-4.0, 0.0, 4.0)))
    if quick:
        grid = grid[:6]
    gs = [genome(*p) for p in grid]
    with Pool(3) as pool:
        val = pool.map(survival, [(g, VAL, {}) for g in gs])
    scored = sorted(zip(grid, gs, val), key=lambda t: -statistics.mean(t[2]))
    for p, _, v in scored[:6]:
        print("val", p, "median", statistics.median(v), "mean", round(statistics.mean(v)))
    p, g, _ = scored[0]
    with Pool(3) as pool:
        org, blind, never = pool.map(survival, [(g, HELD, {}), (g, HELD, {"blind": True}),
                                                (g, HELD, {"mode": 1})])
    s0 = json.loads((ROOT / "results/published/cgb_generated/ecoli_ms_s0_20261026.json")
                    .read_text(encoding="utf-8"))["held_out"]["organism"]
    sr = step_response(g)
    res = dict(best_params=dict(zip(("theta", "k_fast", "k_slow", "bias"), p)), held_out=org, blind=blind,
               never_tumble=never, s0_held_out=[x["survival"] for x in s0], step_response=sr,
               grid=[dict(params=q, val_median=statistics.median(v), val_mean=statistics.mean(v))
                     for q, _, v in scored])
    out = ROOT / "results/published/cgb_generated/ms_comparator_witness_20261026.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in ("held_out", "blind", "never_tumble", "s0_held_out"):
        v = res[k]
        print(k, v, "median", statistics.median(v), "mean", round(statistics.mean(v)))
    print("step response", sr)


if __name__ == "__main__":
    main()
