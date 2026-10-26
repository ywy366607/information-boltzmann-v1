"""Which strategies can live in the nested (depleting) world? Ideal policies only, no organism:
never tumble, the best absolute threshold (tuned per cost on the training seeds 1000-1063, then
fixed), and the noise-free comparator. Survival on held-out worlds 101-108 and validation 500-515.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_chemotaxis3 as W  # noqa: E402

TRAIN, HELD, VAL = range(1000, 1032), range(101, 109), range(500, 516)


def surv(seeds, cf, policy, theta=0.0):
    return [W.policy_survival(W.make_world(s, 10000, cf), policy, theta=theta) for s in seeds]


def main():
    out = {}
    for cf in (0.5, 1.0, 1.5, 2.0):
        thetas = np.linspace(-2, 5, 15)
        best = max(thetas, key=lambda th: statistics.mean(surv(TRAIN, cf, "threshold", th)))
        row = {}
        for name, pol, th in (("never", "never", 0), ("threshold", "threshold", float(best)),
                              ("comparator", "comparator", 0)):
            h, v = surv(HELD, cf, pol, th), surv(VAL, cf, pol, th)
            row[name] = dict(held=h, val_median=statistics.median(v), val_mean=statistics.mean(v),
                             val_alive=sum(x == 10000 for x in v))
        row["threshold"]["theta"] = float(best)
        out[cf] = row
        print("cost x", cf, {k: (r["held"], r["val_median"], round(r["val_mean"]), r["val_alive"]) for k, r in row.items()},
              "theta", round(float(best), 2), flush=True)
    (ROOT / "results/published/cgb_generated/ecoli_nested_world_bounds_20261026.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
