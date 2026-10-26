"""Generate an organism from multi-state rule tables (Λ = Fin 2..4), inside the life gate.

Only the world interface is fixed (receptors F, motor M, store E, waste D). Interior pages, their
state counts, unit counts, state energies and pair rules are found by selection on survival.
Interior-interior rules conserve content (content = state index), so every candidate is a
`tableCRN` medium (M3) and admitted by construction; only tables that also pass the life layer
(sense + metabolism witness, export, selective export, one-tick memory) enter the population.
Training worlds are drawn fresh every generation and the elites are re-evaluated, so a lucky
history cannot carry a genome (winner's curse, seen in chemo s0-s2 and ignition s0-s1).

Prereg: docs/information_boltzmann/CGB_GENERATED_ECOLI_MS_PREREG_20261026.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_chemotaxis2 as W  # noqa: E402
from cgb_table3 import _run  # noqa: E402

NB = W.NB
CAPS = {W.F: 1024, W.M: 1, W.E: 8, W.D: 8}
MAX_INTERIOR, MAX_UNITS, MAX_STATES, MAX_RULES = 4, 512, 4, 16
TICKS = 10000
HELD = tuple(range(101, 109))
G = {"ntrain": 8}


def n_pages(g):
    return NB + g["interior"]


def content_ok(g, i, j, a, b):
    if i >= NB and j >= NB:
        return a[0] + a[1] == b[0] + b[1]
    return True


def random_rule(rng, g):
    P = n_pages(g)
    for _ in range(200):
        i, j = int(rng.integers(P)), int(rng.integers(P))
        a = (int(rng.integers(g["states"][i])), int(rng.integers(g["states"][j])))
        b = (int(rng.integers(g["states"][i])), int(rng.integers(g["states"][j])))
        if a == b or not content_ok(g, i, j, a, b):
            continue
        if i == W.M and j == W.M:
            continue
        return [i, j, a[0], a[1], b[0], b[1], float(rng.uniform(-6, 2))]
    return None


def random_genome(rng):
    k = int(rng.integers(1, 4))
    g = {"interior": k, "states": [2, 2, 2, 2] + [int(rng.integers(2, MAX_STATES + 1)) for _ in range(k)]}
    g["counts"] = [int(rng.integers(16, CAPS[W.F] + 1)), 1, int(rng.integers(1, 9)), int(rng.integers(1, 9))]
    g["counts"] += [int(rng.integers(1, MAX_UNITS + 1)) for _ in range(k)]
    g["phi"] = [[0.0] + [float(rng.uniform(-6, 6)) for _ in range(s - 1)] for s in g["states"]]
    g["rules"] = []
    for _ in range(int(rng.integers(3, 9))):
        r = random_rule(rng, g)
        if r:
            g["rules"].append(r)
    return g


def mutate(g, rng):
    g = copy.deepcopy(g)
    op = int(rng.integers(8))
    P = n_pages(g)
    if op == 0 and g["rules"]:
        r = g["rules"][int(rng.integers(len(g["rules"])))]
        r[6] = float(np.clip(r[6] + rng.normal(0, 0.8), -10, 4))
    elif op == 1 and g["rules"]:
        g["rules"][int(rng.integers(len(g["rules"])))] = random_rule(rng, g) or g["rules"][0]
    elif op == 2 and len(g["rules"]) < MAX_RULES:
        r = random_rule(rng, g)
        if r:
            g["rules"].append(r)
    elif op == 3 and len(g["rules"]) > 1:
        g["rules"].pop(int(rng.integers(len(g["rules"]))))
    elif op == 4:
        p = int(rng.integers(P))
        if g["states"][p] > 1:
            s = int(rng.integers(1, g["states"][p]))
            g["phi"][p][s] = float(np.clip(g["phi"][p][s] + rng.normal(0, 1.0), -12, 12))
    elif op == 5:
        p = int(rng.integers(P))
        if p != W.M:
            hi = CAPS.get(p, MAX_UNITS)
            g["counts"][p] = int(np.clip(round(g["counts"][p] * math.exp(rng.normal(0, 0.4))), 1, hi))
    elif op == 6 and g["interior"] < MAX_INTERIOR:
        g["interior"] += 1
        s = int(rng.integers(2, MAX_STATES + 1))
        g["states"].append(s)
        g["counts"].append(int(rng.integers(1, MAX_UNITS + 1)))
        g["phi"].append([0.0] + [float(rng.uniform(-6, 6)) for _ in range(s - 1)])
        r = random_rule(rng, g)
        if r:
            g["rules"].append(r)
    elif op == 7:
        p = int(rng.integers(NB, P)) if g["interior"] else None
        if p is not None and g["states"][p] < MAX_STATES and rng.random() < 0.5:
            g["states"][p] += 1
            g["phi"][p].append(float(rng.uniform(-6, 6)))
        elif p is not None and g["states"][p] > 2:
            s = g["states"][p] - 1
            g["states"][p] = s
            g["phi"][p] = g["phi"][p][:s]
            g["rules"] = [r for r in g["rules"] if not ((r[0] == p and (r[2] >= s or r[4] >= s)) or
                                                       (r[1] == p and (r[3] >= s or r[5] >= s)))]
    if not g["rules"]:
        r = random_rule(rng, g)
        g["rules"] = [r] if r else []
    return g


# ---------- life layer for multi-state tables ----------

def witness(g):
    """A rule with a boundary participant that changes content (sense + metabolism witness)."""
    for r in g["rules"]:
        if (r[0] < NB or r[1] < NB) and r[2] + r[3] != r[4] + r[5]:
            if r[0] != r[1] or g["counts"][r[0]] >= 2:
                return True
    return False


def export_selective(g):
    props = {}
    for r in g["rules"]:
        for src, dst in (((r[2], r[3]), (r[4], r[5])), ((r[4], r[5]), (r[2], r[3]))):
            if not (r[0] < NB or r[1] < NB) or sum(dst) >= sum(src):
                continue
            dE = (g["phi"][r[0]][dst[0]] + g["phi"][r[1]][dst[1]] - g["phi"][r[0]][src[0]] - g["phi"][r[1]][src[1]])
            key = (r[0], src[0], r[1], src[1])
            props[key] = props.get(key, 0.0) + math.exp(r[6]) * math.exp(-dE / 2)
    v = sorted(props.values())
    return bool(v) and len(v) > 1 and v[-1] > v[0] * (1 + 1e-9)


def memory_one_tick(g, reps=160, seed=11, mu_pulse=2.0):
    counts, phi, bnd, R, kap, S = W.arrays(g)
    P = counts.shape[0]
    np.random.seed(seed)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=reps)
    X = np.zeros((reps, P * S))
    fl = np.zeros(P)
    for k in range(reps):
        n = np.zeros((P, S), dtype=np.int64)
        n[:, 0] = counts
        mu = np.zeros(P)
        mu[W.E], mu[W.D] = W.MU0, -W.MU0
        _run(n, phi, bnd, mu, R, kap, 20.0, fl)
        mu[W.F] = signs[k] * mu_pulse
        _run(n, phi, bnd, mu, R, kap, 1.0, fl)
        mu[W.F] = 0.0
        _run(n, phi, bnd, mu, R, kap, 1.0, fl)
        X[k] = n.reshape(-1)
    half = reps // 2
    w = X[:half][signs[:half] > 0].mean(0) - X[:half][signs[:half] < 0].mean(0)
    thr = np.median(X[:half] @ w)
    acc = float(np.mean(np.where(X[half:] @ w > thr, 1.0, -1.0) == signs[half:]))
    return (acc - 0.5) / (0.25 / (reps - half)) ** 0.5 > 3.0


def life_ok(g):
    return witness(g) and export_selective(g) and memory_one_tick(g)


# ---------- selection ----------

WORLDS = {}


def world(s):
    # training worlds are fresh every generation: caching them grows memory without bound
    # (s2 died of it at generation 198), so only the fixed validation / held-out seeds are kept
    if s >= 1000:
        return W.make_world(s, TICKS)
    if s not in WORLDS:
        WORLDS[s] = W.make_world(s, TICKS)
    return WORLDS[s]


def fitness(args):
    g, seeds = args
    vals = []
    for s in seeds:
        surv, intake, _ = W.live(g, s, TICKS, world=world(s))
        vals.append(surv + 100 * intake)
    return statistics.mean(vals)


def evaluate(g, seeds, **kw):
    out = []
    for s in seeds:
        surv, intake, tum = W.live(g, s, TICKS, world=world(s), **kw)
        out.append(dict(survival=int(surv), intake=float(intake), tumble=float(tum)))
    return out


def fill(pool, make, n):
    out, tries = [], 0
    while len(out) < n:
        cand = [make() for _ in range(2 * (n - len(out)))]
        tries += len(cand)
        out += [c for c, ok in zip(cand, pool.map(life_ok, cand)) if ok][: n - len(out)]
    return out, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    history = []
    with Pool(a.workers) as pool:
        pop, tries = fill(pool, lambda: random_genome(rng), a.pop)
        print("life-admitted initial population:", a.pop, "of", tries, flush=True)
        for gen in range(a.gens):
            seeds = tuple(range(1000 + gen * a.train, 1000 + (gen + 1) * a.train))   # fresh every generation
            fit = pool.map(fitness, [(g, seeds) for g in pop])
            order = np.argsort(fit)[::-1]
            history.append(dict(gen=gen, best=float(fit[order[0]]), median=float(np.median(fit))))
            print(gen, round(fit[order[0]], 1), round(float(np.median(fit)), 1), flush=True)
            elite = [pop[i] for i in order[:8]]

            def make():
                i, j = rng.integers(len(pop), size=2)
                parent = pop[i] if fit[i] >= fit[j] else pop[j]
                child = mutate(parent, rng)
                for _ in range(int(rng.integers(0, 3))):
                    child = mutate(child, rng)
                return child

            children, _ = fill(pool, make, a.pop - len(elite))
            pop = elite + children
            last_fit, last_order = fit, order
        # final choice: best mean over a fixed validation set (seeds 500..515), not the training luck
        val = tuple(range(500, 516))
        vfit = pool.map(fitness, [(g, val) for g in pop[:8]])
        best = pop[int(np.argmax(vfit))]
    report = dict(prereg="docs/information_boltzmann/CGB_GENERATED_ECOLI_MS_PREREG_20261026.md",
                  args=vars(a), history=history, validation=[float(v) for v in vfit], best=best,
                  table=W.to_table(best, f"ecoli_ms_s{a.seed}"),
                  held_out=dict(organism=evaluate(best, HELD), blind=evaluate(best, HELD, blind=True),
                                never_tumble=evaluate(best, HELD, mode=1),
                                fixed_tumble_0p1=evaluate(best, HELD, mode=2),
                                calm_fuel=evaluate(best, HELD, calm=True)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in report["held_out"].items():
        s = [x["survival"] for x in v]
        print(k, "survival [min|med|mean|max]", min(s), statistics.median(s), round(statistics.mean(s)), max(s))


if __name__ == "__main__":
    main()
