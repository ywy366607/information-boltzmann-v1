"""Generate the numbers of the nested organism by selection on survival in the rain torus
(prereg docs/information_boltzmann/CGB_NESTED_ORGANISM_PREREG_20261027.md).

Structure (pages, rules, exchanges, leaks, membrane, physics constants) is fixed by the prereg and
the Lean theorems; selection only sets the energies and prefactors in the prereg ranges, with the
time-scale premise enforced. Every generation the whole population, elites included, lives in
fresh worlds (no winner's curse), each genome in `ntrain` worlds; fitness = median survival
(mean breaks ties). Elites are validated every `vevery` generations on seeds 500-515.

Usage: python scripts/run_cgb_generate_nested.py --seed 0 --gens 40 --out results/published/cgb_generated/nested_s0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_organism_triton as NX  # noqa: E402

KEYS = ("phi", "logk_rule", "logk_x", "log_flip", "log_move")
RANGES = {"phi": NO.phi_range(), "logk_rule": (-9.0, 0.0), "logk_x": (-9.0, 0.0),
          "log_flip": (-9.0, 0.0), "log_move": (-9.0, 0.0)}
VAL = tuple(range(500, 516))


def rows(g, idx):
    return {k: np.asarray(g[k])[idx] for k in KEYS}


def mutate(g, rng, sigma):
    out = {}
    for k in KEYS:
        lo, hi = RANGES[k]
        v = np.asarray(g[k], dtype=float)
        out[k] = np.clip(v + rng.normal(0.0, sigma, size=v.shape), lo, hi)
    return NO.premise(out)


def crossover(a, b, rng):
    return {k: np.where(rng.random(np.shape(a[k])) < 0.5, a[k], b[k]) for k in KEYS}


def evaluate(tbl, pop, worlds_per, seed_base, ticks):
    P = len(pop["phi"])
    idx = np.repeat(np.arange(P), worlds_per)
    seeds = seed_base + np.arange(len(idx))
    r = NX.run_exact(tbl, rows(pop, idx), seeds, ticks=ticks)
    surv = r["survival"].reshape(P, worlds_per)
    return surv, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--ntrain", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--vevery", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tbl = NW.load("rain_torus_w0")
    rng = np.random.default_rng(a.seed)
    pop = NO.random_genomes(a.pop, rng)
    hist, vals = [], []
    best = None
    out = Path(a.out)
    for gen in range(a.gens):
        t0 = time.time()
        surv, r = evaluate(tbl, pop, a.ntrain, 100000 * (a.seed + 1) + 1000 * gen, a.ticks)
        med, mean = np.median(surv, 1), surv.mean(1)
        order = np.lexsort((-mean, -med))
        over = int((r["overflow"] > 0).sum())
        hist.append(dict(gen=gen, median=[float(x) for x in np.percentile(med, [0, 50, 100])],
                         top_median=float(med[order[0]]), top_mean=float(mean[order[0]]),
                         overflow_lanes=over, seconds=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        elites = rows(pop, order[:a.elite])
        if gen % a.vevery == a.vevery - 1 or gen == a.gens - 1:
            vs, _ = evaluate(tbl, elites, len(VAL), 0, a.ticks)
            vs_med = np.median(vs, 1)
            k = int(np.argmax(vs_med + 1e-6 * vs.mean(1)))
            row = dict(gen=gen, val_median=[float(x) for x in vs_med], picked=k,
                       val=[float(x) for x in np.percentile(vs[k], [0, 50, 100])], val_mean=float(vs[k].mean()))
            vals.append(row)
            print("VAL", json.dumps(row), flush=True)
            cand = {kk: np.asarray(v)[k].tolist() for kk, v in elites.items()}
            if best is None or (row["val"][1], row["val_mean"]) > (best["val"][1], best["val_mean"]):
                best = dict(gen=gen, genome=cand, val=row["val"], val_mean=row["val_mean"])
        sigma = 0.6 * (1.0 - gen / a.gens) + 0.15
        kids = []
        for _ in range(a.pop - a.elite):
            i, j = rng.integers(a.elite, size=2)
            child = crossover(rows(elites, i), rows(elites, j), rng) if rng.random() < 0.5 else rows(elites, i)
            kids.append(mutate(child, rng, sigma))
        pop = {k: np.concatenate([np.asarray(elites[k]), np.stack([c[k] for c in kids])]) for k in KEYS}
        out.write_text(json.dumps(dict(prereg="docs/information_boltzmann/CGB_NESTED_ORGANISM_PREREG_20261027.md",
                                       world="rain_torus_w0", seed=a.seed, args=vars(a), history=hist,
                                       validation=vals, best=best), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
