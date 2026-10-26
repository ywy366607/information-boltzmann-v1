"""Prereg W1 (CGB_WORLD2D_PREREG): which ideal policies live in the 2D food world?
Cost = 1.5 x the blind cruiser's mean intake (training seeds 1000-1015, no cost). Threshold and
graze levels tuned on the training seeds; held-out worlds 101-116."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_world2d as W  # noqa: E402

TRAIN, HELD = range(1000, 1016), range(101, 117)


def run(args):
    seed, policy, cost, kw = args
    return W.policy_life(seed, policy, cost=cost, **kw)[0]


def surv(pool, seeds, policy, cost, **kw):
    return pool.map(run, [(s, policy, cost, kw) for s in seeds])


def main():
    with Pool(4) as pool:
        cruise_intake = [x[1] for x in pool.starmap(W.policy_life, [(s, "cruise") for s in TRAIN])]
        cost = 1.5 * statistics.mean(cruise_intake)
        print("cruise intake (no cost)", round(statistics.mean(cruise_intake), 3), "-> cost", round(cost, 3), flush=True)
        thetas = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
        th = max(thetas, key=lambda t: statistics.mean(surv(pool, TRAIN, "threshold", cost, theta=t)))
        fulls = [-2.0, -1.0, 0.0, 1.0, None]
        fu = max(fulls, key=lambda t: statistics.mean(surv(pool, TRAIN, "graze", cost, full=t)))
        print("tuned theta", th, "graze full", fu, flush=True)
        out = {"cost": cost, "theta": th, "full": fu}
        for name, pol, kw in (("cruise", "cruise", {}), ("threshold", "threshold", {"theta": th}),
                              ("comparator", "comparator", {}), ("graze", "graze", {"full": fu})):
            v = surv(pool, HELD, pol, cost, **kw)
            out[name] = v
            print(name, "[min|med|mean|max]", min(v), statistics.median(v), round(statistics.mean(v)), max(v),
                  "alive", sum(x == 10000 for x in v), flush=True)
    (ROOT / "results/published/cgb_generated/world2d_policies_20261026.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
