"""Prereg W2 (CGB_WORLD2D_PREREG): random lean and rich times. Cost = blind cruiser's mean intake
on training seeds 1000-1015; fixed threshold tuned on them; held-out worlds 101-116."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_world2d as W  # noqa: E402

TRAIN, HELD = range(1000, 1016), range(101, 117)
WKW = dict(rate_sigma=1.15, rate_tau=2000.0, size_sigma=0.5)


def run(args):
    seed, policy, cost, kw = args
    return W.policy_life(seed, policy, cost=cost, move_cost=cost, wkw=WKW, **kw)


def main():
    with Pool(4) as pool:
        cruise = pool.map(run, [(s, "cruise", None, {}) for s in TRAIN])
        cost = statistics.mean(x[1] for x in cruise) / 2
        print("b = half the cruise intake", round(cost, 4), "(basal b, moving costs another b)", flush=True)
        thetas = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
        scores = {t: statistics.mean(x[0] for x in pool.map(run, [(s, "threshold", cost, {"theta": t}) for s in TRAIN]))
                  for t in thetas}
        th = max(scores, key=scores.get)
        print("threshold training means", {k: round(v) for k, v in scores.items()}, "best", th, flush=True)
        out = {"cost": cost, "theta": th, "world": WKW}
        for name, pol, kw in (("cruise", "cruise", {}), ("threshold", "threshold", {"theta": th}),
                              ("adaptive", "adaptive", {})):
            v = [x[0] for x in pool.map(run, [(s, pol, cost, kw) for s in HELD])]
            out[name] = v
            print(name, "[min|med|mean|max]", min(v), statistics.median(v), round(statistics.mean(v)), max(v),
                  "alive", sum(x == 10000 for x in v), flush=True)
    (ROOT / "results/published/cgb_generated/world2d_w2b_20261026.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
