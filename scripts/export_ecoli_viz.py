"""Export a generated chemotaxis organism's lives for the showcase page.

A colony of replicate organisms (same table, different internal noise) lives in one ring world,
next to a blind colony (receptors cut) for contrast. Positions, tumbles, reserves and the interior
page occupancies are downsampled to frames.

usage: python scripts/export_ecoli_viz.py <run json> <world seed> <out json> [--n 12] [--every 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_chemotaxis2 as W  # noqa: E402
from cgb_chemotaxis import conc  # noqa: E402


def colony(g, w, n, ticks, blind, every):
    lives = []
    P = W.NB + g["interior"]
    for k in range(n):
        tr = np.zeros((ticks, 2 + P - W.NB))
        xs = np.zeros(ticks)
        surv, intake, tum = W._live(*W.arrays(g), w[0], w[1], w[2], w[3], ticks, w[4], blind, 0, False,
                                    9973 * (k + 1) + (1 if blind else 0), tr, xs)
        surv = int(surv)
        idx = np.arange(0, surv, every)
        lives.append(dict(survival=surv, intake=float(intake), tumble=float(tum),
                          x=[round(float(v), 1) for v in xs[idx]],
                          tumbling=[int(v) for v in tr[idx, 0]],
                          reserve=[round(float(v), 1) for v in tr[idx, 1]],
                          interior=[[round(float(v), 3) for v in tr[idx, 2 + p]] for p in range(P - W.NB)]))
    return lives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=10000)
    a = ap.parse_args()
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    g = run["best"]
    w = W.make_world(a.seed, a.ticks)
    xs = np.linspace(0, W.L, 400, endpoint=False)
    field = [float(conc(x, w[0], w[1])) for x in xs]
    data = dict(name=run["table"]["name"], world_seed=a.seed, ring=W.L, every=a.every, ticks=a.ticks,
                cost=float(w[4]), field=[round(v, 4) for v in field],
                pages={k: v for k, v in run["table"]["pages"].items()},
                units=run["table"]["units"], states=run["table"].get("states", {}),
                rules=run["table"]["rules"],
                colony=colony(g, w, a.n, a.ticks, False, a.every),
                blind=colony(g, w, max(4, a.n // 3), a.ticks, True, a.every))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data), encoding="utf-8")
    s = [l["survival"] for l in data["colony"]]
    b = [l["survival"] for l in data["blind"]]
    print(a.out, "colony survival", s, "blind", b)


if __name__ == "__main__":
    main()
