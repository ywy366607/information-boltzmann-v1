"""Upper bounds per world (diagnostic): which held-out worlds can any organism survive?

* oracle: knows the field; swims straight (left or right) to a chosen spot and stays there,
  best over all spots and both directions;
* comparator: noise-free run-and-tumble on the exact ln c; tumbles when ln c fell over the last
  tick (and with a small base rate); stays put while tumbling, like the organisms.
Same world, start, cost, intake and reserve rules as `cgb_chemotaxis2._live`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_chemotaxis2 as W  # noqa: E402
from cgb_chemotaxis import conc, C_REF, L, V  # noqa: E402

TICKS = 10000


def oracle(w):
    centers, heights, x0, _, cost = w
    best = 0
    for d in (1.0, -1.0):
        for steps in range(0, int(L / V)):
            x, reserve, t = x0, 150.0, 0
            alive = True
            while t < TICKS:
                if t < steps:
                    x = (x + V * d) % L
                reserve = min(300.0, reserve + conc(x, centers, heights) / C_REF) - cost
                t += 1
                if reserve <= 0:
                    alive = False
                    break
                if t > steps and reserve >= 299.0:   # settled and full: stays alive forever
                    t = TICKS
            best = max(best, t if not alive else TICKS)
            if best == TICKS:
                return best
    return best


def comparator(w, base=0.02, seed=0):
    centers, heights, x0, uT, cost = w
    x, d, reserve = x0, 1.0, 150.0
    prev = math.log(conc(x, centers, heights))
    for t in range(TICKS):
        lc = math.log(conc(x, centers, heights))
        tumble = lc < prev or uT[t, 1] < base
        prev = lc
        if tumble:
            d = 1.0 if uT[t, 0] < 0.5 else -1.0
        else:
            x = (x + V * d) % L
        reserve = min(300.0, reserve + conc(x, centers, heights) / C_REF) - cost
        if reserve <= 0:
            return t + 1
    return TICKS


def main():
    out = {}
    for s in range(101, 109):
        w = W.make_world(s, TICKS)
        out[s] = dict(oracle=oracle(w), comparator=comparator(w), cost=round(w[4], 3))
        print(s, out[s])
    (ROOT / "results/published/cgb_generated/ecoli_world_bounds_20261026.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
