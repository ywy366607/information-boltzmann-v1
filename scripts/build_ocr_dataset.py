#!/usr/bin/env python3
"""Materialize large synthetic 1px OCR digit dataset.

Usage:
  python scripts/build_ocr_dataset.py --out data/ocr1px --n_train 50000 --n_val 5000 --res 32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.ocr_1px import OCR_DIGITS, make_ocr_1px  # noqa: E402


def _save_png(path: str, chw: torch.Tensor) -> None:
    arr = (chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)


def build_split(out_split: str, n: int, res: int, seed: int, hard_frac: float) -> dict:
    os.makedirs(os.path.join(out_split, "images"), exist_ok=True)
    rng = np.random.default_rng(seed)
    # balanced digits
    per = n // 10
    extra = n - per * 10
    labs = []
    for d in range(10):
        labs.extend([d] * (per + (1 if d < extra else 0)))
    rng.shuffle(labs)
    labs = np.array(labs[:n], dtype=np.int64)

    rows = []
    t0 = time.time()
    # generate in chunks to limit RAM
    chunk = 256
    idx = 0
    while idx < n:
        b = min(chunk, n - idx)
        img, lab, msk = make_ocr_1px(
            rng, labs[idx : idx + b], res=res, hard_frac=hard_frac,
        )
        for j in range(b):
            i = idx + j
            path = os.path.join(out_split, "images", f"{i:06d}.png")
            _save_png(path, img[j])
            ink = int(msk[j].sum().item()) if hasattr(msk[j], "sum") else int(np.asarray(msk[j]).sum())
            rows.append({
                "id": f"{i:06d}",
                "file": f"images/{i:06d}.png",
                "digit": OCR_DIGITS[int(lab[j].item())],
                "label": int(lab[j].item()),
                "ink_pixels": ink,
            })
        idx += b
        if idx % 2000 == 0 or idx >= n:
            print(f"  {out_split}: {idx}/{n}", flush=True)

    csv_path = os.path.join(out_split, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n": n,
        "res": res,
        "seed": seed,
        "hard_frac": hard_frac,
        "seconds": time.time() - t0,
        "digits": list(OCR_DIGITS),
        "ink_mean": float(np.mean([r["ink_pixels"] for r in rows])),
    }
    with open(os.path.join(out_split, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ocr1px")
    ap.add_argument("--n_train", type=int, default=50_000)
    ap.add_argument("--n_val", type=int, default=5_000)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--hard_frac", type=float, default=0.35)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=90_001)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"building train n={args.n_train} …", flush=True)
    tr = build_split(
        os.path.join(args.out, "train"), args.n_train, args.res,
        args.train_seed, args.hard_frac,
    )
    print(f"building val n={args.n_val} …", flush=True)
    va = build_split(
        os.path.join(args.out, "val"), args.n_val, args.res,
        args.val_seed, args.hard_frac,
    )
    meta = {
        "task": "ocr_1px_digits",
        "train": tr,
        "val": va,
        "question": "What digit is drawn with the thin stroke?",
        "alphabet": list(OCR_DIGITS),
        "stroke": "1px Bresenham stick digits on noisy canvas",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"done → {args.out}", flush=True)


if __name__ == "__main__":
    main()
