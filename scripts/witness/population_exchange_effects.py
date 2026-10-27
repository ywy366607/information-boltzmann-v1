"""Effects of exchange moves in the GPU population kernel (`cgb_nested_population`, Lean
`NestedWorld.move_exchanges`): run populations of a species and check, every tick,
  1. occupancy: every live lane's cell holds exactly that lane, and no other cell holds a body;
  2. bodies: exchanges keep the number of bodies (births and dissolutions are counted separately);
and report exchanges made and lost (fuel returned), and the share of dissolved bodies that never
moved (the JS core on the same species gave 78/174 = 0.45 with exchanges, 154/211 = 0.73 before).

Usage: python scripts/witness/population_exchange_effects.py <espop record> [--worlds 8] [--ticks 3000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
import cgb_nested_world as NW  # noqa: E402
import cgb_nested_world_triton as NT  # noqa: E402
import cgb_nested_population as NP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--founders", type=int, default=12)
    a = ap.parse_args()
    rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
    tbl = NW.load(rec.get("world", "rain_torus_w0"))
    theta = np.asarray(rec["best"]["theta"], dtype=float)
    W = NT.FusedWorld(tbl, B=a.worlds, seed=3)
    wea = NW.Weather(tbl, B=a.worlds, device="cuda", seed=3)
    buf, replay = W.captured()
    for _ in range(400):
        buf.copy_(wea.tick())
        replay()
    P = NP.Population(W, tbl, SL=128, seed=5, pmut=0.0)
    P.seed_population(theta, a.founders)
    nleap = max(1, int(round(1.0 / W.dt)))
    bad_occ, bad_bodies, gone_moves = 0, 0, []
    t0 = time.time()
    for t in range(a.ticks):
        W.mu32.copy_(wea.tick())
        for _ in range(nleap):
            before = int(P.alive.sum())
            W.leap()
            P.leap()
            gone_now = int(P.gone.sum())
            if int(P.alive.sum()) + gone_now - before != 0 and gone_now == 0:
                bad_bodies += 1
        g = P.gone.cpu().numpy().astype(bool)
        if g.any():
            gone_moves += P.moves.cpu().numpy()[g].tolist()
        P.end_tick()
        alive = P.alive.cpu().numpy().astype(bool)
        occ = P.occ.cpu().numpy()
        x, y = P.x.cpu().numpy(), P.y.cpu().numpy()
        lanes = np.nonzero(alive)[0]
        w = lanes // P.SL
        cells = x[lanes] * P.G + y[lanes]
        bad_occ += int((occ[w, cells] != lanes + 1).sum()) + int(((occ > 0).sum() != len(lanes)))
    gm = np.array(gone_moves)
    out = dict(record=a.record, worlds=a.worlds, ticks=a.ticks, founders=a.founders,
               occupancy_errors=bad_occ, body_count_errors=bad_bodies,
               exchanges=int(P.exchanges_total.sum()), exchanges_lost=int(P.exchanges_lost.sum()),
               dissolved=int(len(gm)), dissolved_never_moved=int((gm == 0).sum()),
               dissolved_never_moved_share=float((gm == 0).mean()) if len(gm) else None,
               living_end=P.counts()[1].tolist(), seconds=round(time.time() - t0, 1),
               passed=bad_occ == 0 and bad_bodies == 0)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
