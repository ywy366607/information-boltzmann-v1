"""Frozen-weight, live-state evaluation for runs of train_gdn_fast_table.py --stream mode.

The run's final trained state reads a fresh mode stream (unseen text, 32768-byte segments,
so distances far beyond training's 4096).  At each query, scored over the two mode letters:
  accuracy  = share of queries where p(correct letter) > p(other letter),
  margin    = log p(correct) - log p(other),
  flip_effect = log p(correct | true marker) - log p(correct | marker letter flipped),
            same text and queries, only the letter in the marker changed (does the marker
            reach the query?),
binned by distance from the marker.  Reported per bin as [min | median | mean | max] over
segments for accuracy, and the mean over queries for margin and flip_effect.
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)  # ib_local/types.py would shadow the standard library
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.mode_task import EVAL_SEGMENT, eval_stream, marker  # noqa: E402

CHUNK = 512
BINS = (0, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
A, B = ord("A"), ord("B")


@torch.no_grad()
def margins(model, state, stream, queries):
    """log p(correct) - log p(other) at every query."""
    n = len(stream) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n].astype(np.int64)).cuda()[None]
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    pair = np.empty((n, 2), dtype=np.float32)
    for at in range(0, n, CHUNK):
        logits, state, _ = model.predict(data[:, at:at + CHUNK], state, uniform)
        pair[at:at + CHUNK] = torch.log_softmax(logits[0], -1)[:, [A, B]].cpu().numpy()
    out = []
    for position, mode, _ in queries:
        if position < n:
            right, wrong = pair[position, mode], pair[position, 1 - mode]
            out.append((right - wrong, right))
    return np.array(out)


def flipped(stream: np.ndarray) -> np.ndarray:
    out = stream.copy()
    raw = out.tobytes()
    for mode in (0, 1):
        tag = marker(mode)
        at = raw.find(tag)
        while at >= 0:
            out[at + len(b"\n<mode:")] = ord("B") if mode == 0 else ord("A")
            at = raw.find(tag, at + 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--segments", type=int, default=48)
    parser.add_argument("--source-start", type=int, default=12_000_000)
    parser.add_argument("--segment", type=int, default=EVAL_SEGMENT)
    parser.add_argument("--output-name", default="mode_eval.json")
    args = parser.parse_args()
    gdn_fused.enable()
    stream, _, queries = eval_stream(args.segments * args.segment, source_start=args.source_start,
                                     segment=args.segment)
    flip = flipped(stream)
    distance = np.array([q[2] for q in queries])
    segment = np.array([q[0] // args.segment for q in queries])
    for run in args.runs:
        model, state0, _ = load(run)
        true = margins(model, state0.clone(), stream, queries)
        false = margins(model, state0.clone(), flip, queries)
        k = len(true)
        rows = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            mask = (distance[:k] >= lo) & (distance[:k] < hi)
            if not mask.any():
                continue
            per_segment = [float((true[mask & (segment[:k] == s), 0] > 0).mean())
                           for s in np.unique(segment[:k][mask])]
            rows[f"{lo}-{hi}"] = dict(
                queries=int(mask.sum()),
                accuracy=[float(np.min(per_segment)), float(np.median(per_segment)),
                          float(np.mean(per_segment)), float(np.max(per_segment))],
                margin=float(true[mask, 0].mean()),
                flip_effect=float((true[mask, 1] - false[mask, 1]).mean()))
        (run / args.output_name).write_text(json.dumps(rows, indent=1))
        print(run.name)
        for name, r in rows.items():
            print(f"  {name:>11} n={r['queries']:4d} acc[min|med|mean|max]="
                  + "|".join(f"{v:.2f}" for v in r["accuracy"])
                  + f" margin={r['margin']:+.3f} flip={r['flip_effect']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
