"""Generate an organism: selection over admitted rule tables (prereg CGB_GENERATED_ORGANISM_PREREG_20261026).

Only the world interface is fixed (food page, arrival page, one motor unit). Interior pages,
unit counts, energies and rules are found by selection on survival. Interior-interior rules
must conserve content (M3), so every candidate is admitted by construction.
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
import cgb_table_fast  # noqa: E402
import cgb_chemotaxis  # noqa: E402

# World selection: forage (pages F, A, M; motor index 2) or chemo (pages F, M; motor index 1).
WORLD = {"name": "forage", "mod": cgb_table_fast, "nb": 3, "motor": 2, "maxu": 16}


def set_world(name):
    if name == "chemo":
        WORLD.update(name="chemo", mod=cgb_chemotaxis, nb=2, motor=1, maxu=64)
    else:
        WORLD.update(name="forage", mod=cgb_table_fast, nb=3, motor=2, maxu=16)


def live(*a, **k):
    return WORLD["mod"].live(*a, **k)


def make_world(seed, ticks):
    return WORLD["mod"].make_world(seed, ticks)


def to_table(g, name):
    return WORLD["mod"].to_table(g, name)

TICKS = 20000  # 3000 was too easy: random tables survived and ignored food (smoke 2026-10-26)
TRAIN = (1, 2, 3, 4)
HELD = tuple(range(101, 109))
WORLDS = {}
COST = [0.6]


def world(seed):
    if seed not in WORLDS:
        WORLDS[seed] = make_world(seed, TICKS)
    return WORLDS[seed]


def random_rule(rng, P):
    while True:
        i, j = int(rng.integers(P)), int(rng.integers(P))
        a = (int(rng.integers(2)), int(rng.integers(2)))
        b = (int(rng.integers(2)), int(rng.integers(2)))
        if a == b:
            continue
        if i >= WORLD["nb"] and j >= WORLD["nb"] and sum(a) != sum(b):
            continue  # interior-interior rules conserve content (M3)
        return [i, j, a[0], a[1], b[0], b[1], float(rng.uniform(-3, 3))]


def random_genome(rng):
    k = int(rng.integers(0, 4))
    nb = WORLD["nb"]
    P = nb + k
    counts = [int(rng.integers(1, 5)) for _ in range(nb)]
    counts[WORLD["motor"]] = 1
    counts += [int(rng.integers(1, WORLD["maxu"] + 1)) for _ in range(k)]
    phi = []
    for _ in range(P):
        phi += [0.0, float(rng.uniform(-4, 4))]
    rules = [random_rule(rng, P) for _ in range(int(rng.integers(2, 7)))]
    return {"interior": k, "counts": counts, "phi": phi, "rules": rules}


def mutate(g, rng):
    g = copy.deepcopy(g)
    nb = WORLD["nb"]
    P = nb + g["interior"]
    op = rng.integers(7)
    if op == 0 and g["rules"]:
        k = int(rng.integers(len(g["rules"])))
        r = g["rules"][k]
        f = int(rng.integers(7))
        if f == 6:
            r[6] = float(np.clip(r[6] + rng.normal(0, 0.7), -4, 4))
        else:
            g["rules"][k] = random_rule(rng, P) if f < 2 else r
            if 2 <= f < 6:
                for _ in range(20):
                    c = r.copy()
                    c[f] = 1 - c[f]
                    if (c[2], c[3]) != (c[4], c[5]) and not (c[0] >= nb and c[1] >= nb and c[2] + c[3] != c[4] + c[5]):
                        g["rules"][k] = c
                        break
    elif op == 1 and len(g["rules"]) < 12:
        g["rules"].append(random_rule(rng, P))
    elif op == 2 and len(g["rules"]) > 1:
        g["rules"].pop(int(rng.integers(len(g["rules"]))))
    elif op == 3:
        p = int(rng.integers(P))
        g["phi"][2 * p + 1] = float(np.clip(g["phi"][2 * p + 1] + rng.normal(0, 0.7), -4, 4))
    elif op == 4:
        p = int(rng.integers(P))
        if p != WORLD["motor"]:
            hi = 4 if p < nb else WORLD["maxu"]
            g["counts"][p] = int(np.clip(g["counts"][p] + rng.choice([-2, -1, 1, 2]), 1, hi))
    elif op == 5 and g["interior"] < 3:
        g["interior"] += 1
        g["counts"].append(int(rng.integers(1, WORLD["maxu"] + 1)))
        g["phi"] += [0.0, float(rng.uniform(-4, 4))]
        g["rules"].append(random_rule(rng, P + 1))
    elif op == 6 and g["interior"] > 0:
        last = P - 1
        g["interior"] -= 1
        g["counts"].pop()
        g["phi"] = g["phi"][:-2]
        g["rules"] = [r for r in g["rules"] if r[0] != last and r[1] != last] or [random_rule(rng, P - 1)]
    return g


def fitness(g):
    vals = []
    for s in TRAIN:
        surv, intake, sw = live(g, s, TICKS, world=world(s), cost=COST[0])[:3]
        vals.append(surv + 100 * intake)
    return statistics.mean(vals)


def evaluate(g, seeds, **kw):
    out = []
    for s in seeds:
        surv, intake, sw = live(g, s, TICKS, world=world(s), cost=COST[0], **kw)[:3]
        out.append(dict(survival=int(surv), intake=float(intake), switch=float(sw)))
    return out


def _init(cost, world_name, ntrain=4):
    global TRAIN
    COST[0] = cost
    TRAIN = tuple(range(1, ntrain + 1))
    set_world(world_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.6)
    ap.add_argument("--world", default="forage", choices=["forage", "chemo"])
    ap.add_argument("--train", type=int, default=4, help="number of training worlds (seeds 1..N)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _init(a.cost, a.world, a.train)
    rng = np.random.default_rng(a.seed)
    pop = [random_genome(rng) for _ in range(a.pop)]
    history = []
    with Pool(a.workers, initializer=_init, initargs=(a.cost, a.world, a.train)) as pool:
        fit = pool.map(fitness, pop)
        for gen in range(a.gens):
            order = np.argsort(fit)[::-1]
            elite = [pop[i] for i in order[:8]]
            history.append(dict(gen=gen, best=float(fit[order[0]]), median=float(np.median(fit))))
            print(gen, round(fit[order[0]], 1), round(float(np.median(fit)), 1), flush=True)
            children = []
            while len(children) < a.pop - len(elite):
                i, j = rng.integers(len(pop), size=2)
                parent = pop[i] if fit[i] >= fit[j] else pop[j]
                child = mutate(parent, rng)
                for _ in range(int(rng.integers(0, 3))):
                    child = mutate(child, rng)
                children.append(child)
            cfit = pool.map(fitness, children)
            pop = elite + children
            fit = [fit[i] for i in order[:8]] + cfit
    best = pop[int(np.argmax(fit))]
    report = dict(prereg=("docs/information_boltzmann/CGB_GENERATED_ORGANISM_PREREG_20261026.md" if a.world == "forage"
                          else "docs/information_boltzmann/CGB_GENERATED_CHEMOTAXIS_PREREG_20261026.md"),
                  args=vars(a), history=history, best=best, table=to_table(best, f"generated_s{a.seed}"),
                  world=a.world,
                  held_out=(dict(organism=evaluate(best, HELD), loop_cut=evaluate(best, HELD, cut=True),
                                 blind=evaluate(best, HELD, blind=True)) if a.world == "forage" else
                            dict(organism=evaluate(best, HELD), blind=evaluate(best, HELD, blind=True),
                                 never_tumble=evaluate(best, HELD, mode=1),
                                 fixed_tumble_0p1=evaluate(best, HELD, mode=2))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in report["held_out"].items():
        s = [x["survival"] for x in v]
        print(k, "survival [min|med|mean|max]", min(s), statistics.median(s), round(statistics.mean(s)), max(s),
              "intake", round(statistics.mean(x["intake"] for x in v), 3))


if __name__ == "__main__":
    main()
