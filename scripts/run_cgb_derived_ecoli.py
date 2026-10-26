"""E. coli with structure derived from the theorems and numbers found by selection.

Structure (prereg docs/information_boltzmann/CGB_DERIVED_ECOLI_PREREG_20261026.md): one interior
page A of 6-state receptor complexes, state = 3 * bound + methylation level m. Binding takes 3
quanta from the receptor reservoir (G25d), the threshold moves with m (G25e/f), bound units are
methylated through E and unbound ones demethylated through D (driven integral feedback, G25c,
G13), active methylated units hand a quantum to the motor, which drains to D.
Only the numbers (energies, log rates, unit counts) are selected.

usage: python scripts/run_cgb_derived_ecoli.py --out results/published/cgb_generated/ecoli_derived_s0_20261026.json
"""

from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import cgb_chemotaxis2 as W  # noqa: E402
import run_cgb_generate_ms as G  # noqa: E402
from cgb_table3 import _run  # noqa: E402

F, M, E, D, A = 0, 1, 2, 3, 4
HELD, VAL, TICKS = tuple(range(101, 109)), tuple(range(500, 516)), 10000
# rule order (fixed structure); the numbers are params["k"][i]
RULES = ([("bind", m) for m in range(3)] + [("meth", m) for m in range(2)] +
         [("demeth", m) for m in range(2)] + [("motor_on", m) for m in (1, 2)] + [("motor_off", 0)])


def genome(p):
    rules = []
    for (kind, m), k in zip(RULES, p["k"]):
        if kind == "bind":
            rules.append([F, A, 0, m, 0, m + 3, k])
        elif kind == "meth":
            rules.append([E, A, 0, 3 + m, 0, 4 + m, k])
        elif kind == "demeth":
            rules.append([D, A, 0, m + 1, 0, m, k])
        elif kind == "motor_on":
            rules.append([A, M, m, 0, m - 1, 1, k])
        else:
            rules.append([D, M, 0, 1, 0, 0, k])
    return {"interior": 1, "states": [2, 2, 2, 2, 6],
            "counts": [int(p["nF"]), 1, int(p["nE"]), int(p["nD"]), int(p["nA"])],
            "phi": [[0.0, 8.0], [0.0, float(p["phiM"])], [0.0, 8.0], [0.0, 8.0], [0.0] + list(map(float, p["phiA"]))],
            "rules": rules}


def initial(rng):
    th0, a, dl = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 5)
    phiA = [a, 2 * a, 3 * th0, 3 * th0 + a + dl, 3 * th0 + 2 * a + 2 * dl]
    return {"phiA": phiA, "phiM": float(rng.uniform(-2, 4)), "k": [float(rng.uniform(-4, 1)) for _ in RULES],
            "nF": int(rng.integers(16, 257)), "nA": int(rng.integers(128, 1025)),
            "nE": int(rng.integers(1, 9)), "nD": int(rng.integers(1, 9))}


def mutate(p, rng):
    p = copy.deepcopy(p)
    for _ in range(int(rng.integers(1, 4))):
        what = rng.integers(5)
        if what == 0:
            i = int(rng.integers(5))
            p["phiA"][i] = float(np.clip(p["phiA"][i] + rng.normal(0, 0.7), -15, 20))
        elif what == 1:
            i = int(rng.integers(len(RULES)))
            p["k"][i] = float(np.clip(p["k"][i] + rng.normal(0, 0.7), -10, 4))
        elif what == 2:
            p["phiM"] = float(np.clip(p["phiM"] + rng.normal(0, 0.7), -8, 8))
        elif what == 3:
            key, lo, hi = [("nF", 1, 1024), ("nA", 16, 1024)][int(rng.integers(2))]
            p[key] = int(np.clip(round(p[key] * np.exp(rng.normal(0, 0.3))), lo, hi))
        else:
            key = ["nE", "nD"][int(rng.integers(2))]
            p[key] = int(np.clip(p[key] + rng.integers(-2, 3), 1, 8))
    return p


def fitness(args):
    p, seeds = args
    return G.fitness((genome(p), seeds))


def admitted(p):
    return G.life_ok(genome(p))


def step_response(g, mu0, dmu=0.5, reps=120, settle=600.0):
    counts, phi, bnd, R, kap, S = W.arrays(g)
    out = {}
    for sign in (1, -1):
        tum = 0
        for r in range(reps):
            np.random.seed(7000 + 100 * (sign + 2) + r)
            n = np.zeros((counts.shape[0], S), dtype=np.int64)
            n[:, 0] = counts
            mu = np.zeros(counts.shape[0])
            mu[F], mu[E], mu[D] = mu0, W.MU0, -W.MU0
            fl = np.zeros(counts.shape[0])
            _run(n, phi, bnd, mu, R, kap, settle, fl)
            mu[F] = mu0 + sign * dmu
            for _ in range(20):
                _run(n, phi, bnd, mu, R, kap, 1.0, fl)
                tum += int(n[M, 0] == 0)
        out["up" if sign > 0 else "down"] = tum / (20 * reps)
    out["contrast"] = (out["down"] - out["up"]) / max(out["down"], 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    history = []
    with Pool(a.workers) as pool:
        pop, tries = [], 0
        while len(pop) < a.pop:
            cand = [initial(rng) for _ in range(4 * a.workers)]
            tries += len(cand)
            pop += [c for c, ok in zip(cand, pool.map(admitted, cand)) if ok]
        pop = pop[:a.pop]
        print(f"life-admitted initial population: {len(pop)} of {tries}", flush=True)
        for gen in range(a.gens):
            seeds = tuple(range(1000 + gen * a.train, 1000 + (gen + 1) * a.train))
            fit = pool.map(fitness, [(p, seeds) for p in pop])
            order = np.argsort(fit)[::-1]
            elite = [pop[i] for i in order[:a.elite]]
            history.append(dict(gen=gen, best=float(fit[order[0]]), mean=float(np.mean(fit))))
            print(gen, round(float(fit[order[0]]), 1), round(float(np.mean(fit)), 1), flush=True)
            kids = []
            while len(kids) < a.pop - a.elite:
                cand = [mutate(elite[int(rng.integers(a.elite))], rng) for _ in range(2 * a.workers)]
                kids += [c for c, ok in zip(cand, pool.map(admitted, cand)) if ok]
            pop = elite + kids[:a.pop - a.elite]
        val = pool.map(fitness, [(p, VAL) for p in pop])
        best = pop[int(np.argmax(val))]
        g = genome(best)
        arms = {"organism": {}, "blind": {"blind": True}, "never_tumble": {"mode": 1},
                "fixed_tumble_0p1": {"mode": 2}, "calm_fuel": {"calm": True}}
        held = {k: G.evaluate(g, HELD, **kw) for k, kw in arms.items()}
    steps = {mu0: step_response(g, mu0) for mu0 in (-2, -1, 0, 1, 2, 3)}
    out = dict(prereg="docs/information_boltzmann/CGB_DERIVED_ECOLI_PREREG_20261026.md", args=vars(a),
               history=history, validation=dict(scores=[float(v) for v in val]), params=best, best=g,
               table=W.to_table(g, "ecoli_derived_s0"), held_out=held, step_response=steps)
    out["table"]["status"] = ("DERIVED structure (theorems G13, G18, G24, G25c-f) with numbers found by selection "
                              "in the chemotaxis ring world")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in held.items():
        s = [x["survival"] for x in v]
        print(k, s, "[min|med|mean|max]", min(s), statistics.median(s), round(statistics.mean(s)), max(s))
    for mu0, r in steps.items():
        print("step", mu0, {k: round(v, 3) for k, v in r.items()})


if __name__ == "__main__":
    main()
