"""Held-out tests O1-O3 of a selected nested organism (prereg
docs/information_boltzmann/CGB_NESTED_ORGANISM_PREREG_20261027.md).

O1: median survival above the blind control (receptor exchanges at zero conductance) and the
    still control (movement at zero conductance).
O2: adaptation: after a step of the cell's food the motor's run fraction relaxes back, ending
    within half of its peak deviation from the pre-step level, at two different food levels.
O3: with rich and lean times (the world's drifting rain), freezing the memory page Y (its rules
    at zero conductance) lowers survival.
Survival is reported as [min, median, mean, max] on the same held-out world seeds for every arm.

Usage: python scripts/eval_nested_organism.py <selection record> <out.json> [--worlds 64]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_organism as NO  # noqa: E402
import cgb_nested_organism_triton as NX  # noqa: E402

HELD = 9000


def dist(x):
    x = np.asarray(x, dtype=float)
    return dict(min=float(x.min()), median=float(np.median(x)), mean=float(x.mean()), max=float(x.max()),
                full=int((x >= x.max()).sum()))


def step_response(tbl, g, lo, hi, lanes=2048, pre=300, post=600, seed=3):
    class Stub:
        pass
    st = Stub()
    st.B, st.G, st.V, st.dt = lanes, 3, tbl["table"]["V"], 0.05
    st.foodE = tbl["table"]["foodE"]
    st.f = torch.full((lanes, 3, 3), lo, dtype=torch.int32, device="cuda")
    st.w = torch.zeros((lanes, 3, 3), dtype=torch.int32, device="cuda")
    gg = {k: np.repeat(np.asarray(v)[None], lanes, axis=0) for k, v in g.items()}
    o = NX.ExactOrganisms(st, gg, seed=seed)
    run = []
    for t in range(pre + post):
        level = lo if t < pre else hi
        for _ in range(20):
            st.f.fill_(level)              # the cell is held at the level (a reservoir)
            st.w.zero_()
            o.leap()
        run.append(float((o.n32[:, NO.M, NO.RUN] > 0).float().mean()))
    run = np.array(run)
    r0 = run[pre - 100:pre].mean()
    peak = np.abs(run[pre:pre + 100] - r0).max()
    final = abs(run[-200:].mean() - r0)
    return dict(lo=lo, hi=hi, pre=float(r0), peak_dev=float(peak), final_dev=float(final),
                adapted=bool(peak > 0.02 and final < 0.5 * peak), trace=run[::10].round(4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("out")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=8000)
    a = ap.parse_args()
    rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
    tbl = NW.load(rec.get("world", "rain_torus_w0"))
    g = {k: np.asarray(v, dtype=float) for k, v in rec["best"]["genome"].items()}
    seeds = HELD + np.arange(a.worlds)
    rows = {k: np.repeat(np.asarray(v)[None], a.worlds, axis=0) for k, v in g.items()}
    arms = {}
    for arm, kw in (("organism", {}), ("blind", dict(blind=True)), ("still", dict(still=True)),
                    ("freeze_y", dict(freeze_y=True))):
        r = NX.run_exact(tbl, rows, seeds, ticks=a.ticks, **kw)
        arms[arm] = dict(dist(r["survival"]), intake_median=float(np.median(r["intake"])),
                         moves_median=float(np.median(r["moves"])), run_frac_median=float(np.median(r["run_frac"])),
                         overflow_lanes=int((r["overflow"] > 0).sum()), survival=r["survival"].tolist())
        print(arm, {k: v for k, v in arms[arm].items() if k != "survival"}, flush=True)
    o2 = [step_response(tbl, g, 2, 20), step_response(tbl, g, 20, 80)]
    for s in o2:
        print("O2", {k: v for k, v in s.items() if k != "trace"}, flush=True)
    org = arms["organism"]["median"]
    res = dict(record=a.record, world=rec.get("world", "rain_torus_w0"), held_out_seeds=[int(seeds[0]), int(seeds[-1])],
               ticks=a.ticks, arms=arms, O2=o2,
               predictions=dict(O1=bool(org > arms["blind"]["median"] and org > arms["still"]["median"]),
                                O2=bool(all(s["adapted"] for s in o2)),
                                O3=bool(org > arms["freeze_y"]["median"])))
    print("predictions", res["predictions"])
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
