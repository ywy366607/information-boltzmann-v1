"""Fairness check for W2: the best fixed threshold sat on the lower grid edge (theta = -3);
extend the grid downward and re-test the fixed threshold only (can only help the competitor)."""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "witness"))
import world2d_w2 as S  # noqa: E402

OUT = ROOT / "results/published/cgb_generated/world2d_w2b_20261026.json"


def main():
    r = json.loads(OUT.read_text())
    cost = r["cost"]
    with Pool(4) as pool:
        th = {t: statistics.mean(x[0] for x in pool.map(S.run, [(s, "threshold", cost, {"theta": t}) for s in S.TRAIN]))
              for t in (-6.0, -5.0, -4.5, -4.0, -3.5, -3.0)}
        best = max(th, key=th.get)
        print("training", {k: round(v) for k, v in th.items()}, "best", best, flush=True)
        v = [x[0] for x in pool.map(S.run, [(s, "threshold", cost, {"theta": best}) for s in S.HELD])]
    print("threshold (lower grid)", min(v), statistics.median(v), round(statistics.mean(v)), max(v),
          "alive", sum(x == 10000 for x in v), flush=True)
    r["threshold_low_grid"] = dict(training=th, theta=best, held=v)
    OUT.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
