"""Life of one ChargeFilter on real OWT bytes, then an independent future.

Phase A: train split from a seed-dependent offset, prequential (score, then learn).
Phase B: validation split; the living organism keeps learning, a frozen twin forks
from the same state; online n-grams continue on the same exposure.
Memory profile on phase B: next-byte NLL when the charge is rebuilt from only the
last D bytes (starting from the life-average charge) versus the true charge.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_observation_stream import load_owt_sources  # noqa: E402
from cgb_charge_filter import ChargeFilter, OnlineNgram  # noqa: E402

LN2 = math.log(2)


def stats(v):
    return dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--life", type=int, default=1_000_000)
    ap.add_argument("--future", type=int, default=100_000)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--deposit", type=int, default=64)
    ap.add_argument("--init-spread", type=int, default=2)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--probe-points", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = ROOT / "data" / "ib_owt_10000"
    train, val, _, prov = load_owt_sources(d / "train.npy", d / "validation.npy", "byte")
    start = (args.seed * 7_000_003) % (len(train) - args.life)
    life = [int(b) for s in range(start, start + args.life, 1_000_000)
            for b in train[s:min(s + 1_000_000, start + args.life)]]
    future = [int(b) for b in val[0:args.future]]
    assert (sum(b == 0 for b in life[:100_000]) < 10), "zero bytes in stream: decoding error"

    m = ChargeFilter(args.cells, seed=args.seed, deposit=args.deposit, init_spread=args.init_spread,
                    lag=args.lag)
    ngrams = {f"online_order{k}": OnlineNgram(k) for k in (0, 1, 2, 3)}
    window, curve, occ = 50_000, [], torch.zeros(args.cells)
    acc = {k: 0.0 for k in ["organism", *ngrams]}
    t0 = time.time()
    for t, b in enumerate(life):
        acc["organism"] += m.observe(b)
        occ += m.alpha
        for k, g in ngrams.items():
            acc[k] += g.observe(b)
        if (t + 1) % window == 0:
            curve.append(dict(bytes=t + 1, **{k: v / window / LN2 for k, v in acc.items()}))
            acc = {k: 0.0 for k in acc}
            print(json.dumps(curve[-1]), flush=True)
    life_seconds = time.time() - t0
    stationary = occ / occ.sum()

    frozen = copy.deepcopy(m)
    frozen.learning = False
    seg_len = args.future // args.segments
    seg = {k: [0.0] * args.segments for k in ["organism_learning", "organism_frozen", *ngrams]}
    probe_every = max(1, args.future // args.probe_points)
    depths = [0, 1, 2, 4, 8, 16, 32, 64]
    probe = {D: [] for D in depths}
    for t, b in enumerate(future):
        i = min(t // seg_len, args.segments - 1)
        if t % probe_every == 0 and t >= 64:
            m.learning = False
            true_nll = m.nll_from(m.alpha, b)
            for D in depths:
                alpha_d = m.filter_only(future[t - D:t], stationary.clone()) if D else stationary
                probe[D].append((m.nll_from(alpha_d, b) - true_nll) / LN2)
            m.learning = True
        seg["organism_learning"][i] += m.observe(b)
        seg["organism_frozen"][i] += frozen.observe(b)
        for k, g in ngrams.items():
            seg[k][i] += g.observe(b)
    future_bits = {k: [x / seg_len / LN2 for x in v] for k, v in seg.items()}

    report = dict(
        organism="ChargeFilter", cells=args.cells, seed=args.seed, deposit=args.deposit,
        init_spread=args.init_spread, lag=args.lag, life_bytes=args.life, life_start=start,
        future_bytes=args.future, segments=args.segments, provenance=prov,
        resident_bytes=m.resident_bytes(), life_seconds=life_seconds,
        life_curve_bits_per_byte=curve,
        future_bits_per_byte={k: dict(segments=v, **stats(v)) for k, v in future_bits.items()},
        memory_profile_extra_bits_if_only_last_D_bytes={str(D): stats(v) for D, v in probe.items()},
        effective_cells=float(2 ** (-(stationary[stationary > 0] * stationary[stationary > 0].log2()).sum())),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    ck = ROOT / "checkpoints" / "cgb_charge_filter" / (out.stem + ".pt")
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=m.state_dict(), stationary=stationary, args=vars(args)), ck)
    print("future", {k: round(v["mean"], 4) for k, v in report["future_bits_per_byte"].items()})


if __name__ == "__main__":
    main()
