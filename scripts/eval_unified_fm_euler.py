#!/usr/bin/env python3
"""Euler-step sweep on a unified FM checkpoint. No retraining."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import eval_ports, render_gallery

CKPT = ROOT / "checkpoints" / "omni_d256_unified_fm_t2i_best.pt"
OUT = ROOT / "results" / "published" / "omni_unified_fm_t2i_euler_sweep.json"
FIG = ROOT / "present" / "figs"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE).to(dev)
    raw = torch.load(args.ckpt, map_location="cpu")
    miss = model.load_state_dict(raw, strict=False)
    print(f"loaded {args.ckpt} miss={len(miss.missing_keys)} extra={len(miss.unexpected_keys)}", flush=True)
    recs = {}
    for n in (1, 2, 4, 8, 16):
        ev = eval_ports(
            model, np.random.default_rng(9001), 32, dev, n=48, kinds=["t2i"], flow_steps=n,
        )["t2i"]
        recs[str(n)] = ev
        print(
            f"  Euler {n:2d}  psnr={ev['psnr']:.2f}  bg={ev['bg_psnr']:.2f}  "
            f"str={ev['stroke_psnr']:.2f}  flood={ev['flood']*100:.1f}  ink={ev['ink']*100:.1f}",
            flush=True,
        )
    OUT.write_text(json.dumps({"ckpt": args.ckpt, "steps": recs}, indent=2), encoding="utf-8")
    best_n = max(recs, key=lambda k: recs[k]["psnr"])
    render_gallery(
        model, np.random.default_rng(7), 32, dev,
        FIG / "omni_unified_fm_t2i_euler_best_gallery.png",
        kinds=["t2i"] * 5, flow_steps=int(best_n),
    )
    print(f"saved {OUT} best_steps={best_n}", flush=True)


if __name__ == "__main__":
    main()
