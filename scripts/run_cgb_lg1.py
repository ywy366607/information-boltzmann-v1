"""Life of one LG-1 (mean-field) on real OWT bytes, then an independent future.

Same protocol as run_cgb_charge_filter.py (shared helpers are not factored out
because that script's file is frozen with published results).

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "information_boltzmann" / "core"))
from cgb_observation_stream import load_owt_sources  # noqa: E402
from cgb_charge_filter import OnlineNgram  # noqa: E402
from cgb_lg1 import LG1  # noqa: E402
import numpy as np  # noqa: E402

LN2 = math.log(2)


def stats(v):
    return dict(min=min(v), median=statistics.median(v), mean=statistics.mean(v), max=max(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--relax", type=float, default=1.0)
    ap.add_argument("--diffuse", type=float, default=0.3)
    ap.add_argument("--feedback", type=float, default=0.1)
    ap.add_argument("--recur", type=float, default=0.0)
    ap.add_argument("--anneal", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--life", type=int, default=1_000_000)
    ap.add_argument("--future", type=int, default=100_000)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--probe-points", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = ROOT / "data" / "ib_owt_10000"
    train, val, _, prov = load_owt_sources(d / "train.npy", d / "validation.npy", "byte")
    start = (args.seed * 7_000_003) % (len(train) - args.life)
    life = [int(b) for s in range(start, start + args.life, 1_000_000)
            for b in train[s:min(s + 1_000_000, start + args.life)]]
    future = [int(b) for b in val[0:args.future]]
    assert (sum(b == 0 for b in life[:100_000]) < 10), "zero bytes in stream: decoding error"

    m = LG1(args.hidden, seed=args.seed, eta=args.eta, relax=args.relax, diffuse=args.diffuse,
            feedback=args.feedback, recur=args.recur, anneal=args.anneal)
    ngrams = {f"online_order{k}": OnlineNgram(k) for k in (0, 1, 2, 3)}
    window, curve = 50_000, []
    acc = {k: 0.0 for k in ["organism", *ngrams]}
    t0 = time.time()
    for t, b in enumerate(life):
        acc["organism"] += m.observe(b)
        for k, g in ngrams.items():
            acc[k] += g.observe(b)
        if (t + 1) % window == 0:
            curve.append(dict(bytes=t + 1, **{k: v / window / LN2 for k, v in acc.items()}))
            acc = {k: 0.0 for k in acc}
            print(json.dumps(curve[-1]), flush=True)
    life_seconds = time.time() - t0

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
            saved = (m.p1.copy(), m.p2.copy())
            true_nll = -math.log(float(m.predict()[b]))
            for D in depths:
                m.p1[:] = 0.5
                m.p2[:] = 0.5
                m.learning = False
                for c in future[t - D:t]:
                    m.observe(c)
                probe[D].append((-math.log(float(m.predict()[b])) - true_nll) / LN2)
                m.p1, m.p2 = saved[0].copy(), saved[1].copy()
            m.learning = True
        seg["organism_learning"][i] += m.observe(b)
        seg["organism_frozen"][i] += frozen.observe(b)
        for k, g in ngrams.items():
            seg[k][i] += g.observe(b)
    future_bits = {k: [x / seg_len / LN2 for x in v] for k, v in seg.items()}

    report = dict(
        organism="LG1-meanfield", admitted=False, hidden=args.hidden, seed=args.seed, eta=args.eta,
        relax=args.relax, diffuse=args.diffuse, feedback=args.feedback, recur=args.recur, anneal=args.anneal,
        life_bytes=args.life, life_start=start,
        future_bytes=args.future, segments=args.segments, provenance=prov,
        resident_bytes=sum(a.nbytes for a in (m.W_in, m.W1, m.W2, m.bias, m.h, m.p1, m.p2)),
        life_seconds=life_seconds,
        life_curve_bits_per_byte=curve,
        future_bits_per_byte={k: dict(segments=v, **stats(v)) for k, v in future_bits.items()},
        memory_profile_extra_bits_if_only_last_D_bytes={str(D): stats(v) for D, v in probe.items()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    ck = ROOT / "checkpoints" / "cgb_lg1" / (out.stem + ".npz")
    ck.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ck, W_in=m.W_in, W1=m.W1, W2=m.W2, bias=m.bias, h=m.h, p1=m.p1, p2=m.p2)
    print("future", {k: round(v["mean"], 4) for k, v in report["future_bits_per_byte"].items()})


if __name__ == "__main__":
    main()
