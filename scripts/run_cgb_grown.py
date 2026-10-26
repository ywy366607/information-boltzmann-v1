"""Grown vs drawn interior couplings in LG-1 on real OWT bytes (diagnostic).

Arms share seed, W_in and stream: none (R=0), random_dense (birth draw, recur=2),
grown (use-selective, G17), grown_useblind (G17a control), random_sparse (same
occupancy and scale as the grown arm's final R, random signs). Phase A: life,
prequential. Phase B: validation future, learning twin and frozen twin.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_observation_stream import load_owt_sources  # noqa: E402
from cgb_lg1 import LG1  # noqa: E402
from cgb_grown import GrownLG1, ForcedGrownLG1, random_sparse_R  # noqa: E402
import numpy as np  # noqa: E402

LN2 = math.log(2)


def stats(v):
    return dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))


def live(m, life, future):
    for b in life:
        m.observe(b)
    frozen = copy.deepcopy(m)
    frozen.learning = False
    if hasattr(frozen, "growing"):
        frozen.growing = False
    a = f = 0.0
    for b in future:
        a += m.observe(b)
        f += frozen.observe(b)
    return dict(future_learning=a / len(future) / LN2, future_frozen=f / len(future) / LN2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--life", type=int, default=200_000)
    ap.add_argument("--future", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--arms", default="none,random_dense,grown,grown_useblind,random_sparse")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = ROOT / "data" / "ib_owt_10000"
    train, val, _, prov = load_owt_sources(d / "train.npy", d / "validation.npy", "byte")
    start = (args.seed * 7_000_003) % (len(train) - args.life)
    life = [int(b) for b in train[start:start + args.life]]
    future = [int(b) for b in val[0:args.future]]
    kw = dict(seed=args.seed, eta=0.05, relax=1.0, diffuse=0.3, feedback=0.1)
    res, grown_occ = {}, None
    for arm in args.arms.split(","):
        if arm == "none":
            m = LG1(args.hidden, recur=0.0, **kw)
        elif arm == "random_dense":
            m = LG1(args.hidden, recur=2.0, **kw)
        elif arm in ("grown", "grown_useblind"):
            m = GrownLG1(args.hidden, budget=args.budget, beta=args.beta, use_driven=(arm == "grown"), **kw)
        elif arm in ("forced", "forced_local"):
            m = ForcedGrownLG1(args.hidden, budget=args.budget, kappa=args.kappa,
                               local_pool=(arm == "forced_local"), **kw)
        elif arm == "random_sparse":
            m = LG1(args.hidden, recur=0.0, **kw)
            occ = grown_occ if grown_occ is not None else args.budget
            m.R = random_sparse_R(args.hidden, occ, 2.0 / np.sqrt(args.budget * args.hidden), args.seed)
        r = live(m, life, future)
        if isinstance(m, ForcedGrownLG1):
            r["occupancy"] = m.occupancy()
            r["strong_fraction"] = float((m.n > 0.5).mean())
            r["theta"] = float(np.mean(m.theta))
            r["degree_strong"] = stats([int(x) for x in (m.n > 0.5).sum(1)])
            r["n_quantiles"] = [float(q) for q in np.quantile(m.n, [0.5, 0.9, 0.99])]
        if isinstance(m, GrownLG1):
            r["occupancy"] = m.occupancy()
            nz = m.n > 0.5
            r["strong_fraction"] = float(nz.mean())
            deg = nz.sum(1)
            r["degree"] = stats([int(x) for x in deg])
            if arm == "grown":
                grown_occ = r["strong_fraction"]
        res[arm] = r
        print(arm, json.dumps(r), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), provenance=prov, results=res, admitted=False), indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
