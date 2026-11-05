"""E3: one body grown in the real byte stream. The score reads only where the boundary particle stayed."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from information_boltzmann.cgb.voxel_body import Body, Genome


def stream(start, total, shuffle_seed=None):
    values = np.asarray(np.load("D:/information-boltzmann-v1/data/ib_owt_10000/train.npy", mmap_mode="r")
                        [start:start + int(total * 1.02)])
    text = (values[values >= 4].astype(np.int64) - 4)[:total]
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(text)
    return text


def seed_numba(seed):
    from numba import njit

    @njit
    def apply(value):
        np.random.seed(value)

    apply(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=100_000)
    parser.add_argument("--start", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--every", type=int, default=2_500)
    parser.add_argument("--match_si", type=float, default=.35)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed_numba(args.seed)
    text = stream(args.start, args.bytes, args.seed if args.shuffle else None).tolist()
    body = Body(seed=args.seed, genome=Genome(match_si=args.match_si))
    state = body.state
    counts = np.full(256, .05)
    rows, nll, base, hit, t0 = [], 0., 0., 0, time.time()
    for position, y in enumerate(text, 1):
        occupancy = body.free()
        nll -= math.log((occupancy[y] + .5) / (occupancy.sum() + 128.))
        base -= math.log(counts[y] / counts.sum())
        hit += int(occupancy.argmax() == y)
        counts[y] += 1.
        body.arrive(y)
        if position % args.every == 0:
            n, m = int(state.counters[0]), int(state.counters[1])
            live = state.seg_cell[:n] >= 0
            row = dict(bytes=position, nll=round(nll / args.every, 4), order0=round(base / args.every, 4),
                       hit=round(hit / args.every, 4), cells=int(state.cell_n[0]), segments=int(live.sum()),
                       synapses=m - int(state.life[1]), z=[int(state.site_z.min()), int(state.site_z.max())],
                       spiked=int(state.cell_spiked.sum()), material=round(float(state.fld[:, 0].mean()), 2),
                       signal=[int(state.fld[:, 1].sum()), int(state.fld[:, 2].sum())], tips=body._tip_count,
                       recycled=[int(state.life[0]), int(state.life[1])], slots=int(state.syn[:m, :3].sum()),
                       tags=int(state.syn[:m, 3:].sum()), seconds=round(time.time() - t0))
            state.cell_spiked[:] = 0
            rows.append(row)
            print(row, flush=True)
            nll, base, hit = 0., 0., 0
            args.out.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()}, rows=rows)), encoding="utf-8")


if __name__ == "__main__":
    main()
