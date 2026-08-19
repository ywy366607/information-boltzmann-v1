#!/usr/bin/env python3
"""Same xyt ckpt, same color, digits 0-9 from noise. Do they look different?"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import fm_generate, to_unit

CKPT = ROOT / "checkpoints" / "omni_d256_unified_jit_xyt_best.pt"
OUT = ROOT / "present" / "figs" / "omni_xyt_digit_grid.png"
TAB = ROOT / "results" / "published" / "omni_xyt_digit_grid.json"


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x", fm_signed=True).to(dev)
    missing = m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    print(f"loaded {CKPT.name} missing={len(missing.missing_keys)}", flush=True)
    m.eval()

    seeds = (0, 1)
    color = "yellow"
    imgs = []
    prompts = []
    torch.manual_seed(0)
    for seed in seeds:
        g = torch.Generator(device=dev).manual_seed(seed)
        for d in range(10):
            prompt = f"Draw digit {d} with a thin {color} stroke blank image"
            z = torch.randn(1, 3, 32, 32, device=dev, generator=g)
            rgb = to_unit(fm_generate(
                m, z, [prompt], [True], n_steps=8, method="heun",
                clamp=False, cfg=2.0,
            )).clamp(0, 1)
            imgs.append(rgb[0].cpu())
            prompts.append(prompt)

    # Pairwise MSE of per-digit mean images (seed-averaged).
    stack = torch.stack(imgs).view(len(seeds), 10, 3, 32, 32)
    mean = stack.mean(0)
    mse = torch.zeros(10, 10)
    for i in range(10):
        for j in range(10):
            mse[i, j] = (mean[i] - mean[j]).pow(2).mean()
    off = mse[~torch.eye(10, dtype=torch.bool)]
    rec = {
        "ckpt": str(CKPT),
        "color": color,
        "mean_offdiag_mse": float(off.mean()),
        "min_offdiag_mse": float(off.min()),
        "max_offdiag_mse": float(off.max()),
        "diag_zero": float(mse.diag().mean()),
    }
    TAB.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(
        f"off-diag MSE mean={rec['mean_offdiag_mse']:.5f} "
        f"min={rec['min_offdiag_mse']:.5f} max={rec['max_offdiag_mse']:.5f}",
        flush=True,
    )

    fig, axes = plt.subplots(len(seeds), 10, figsize=(14, 3.0), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for s in range(len(seeds)):
        for d in range(10):
            ax = axes[s, d]
            ax.imshow(imgs[s * 10 + d].permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis("off")
            ax.set_facecolor("#0f172a")
            if s == 0:
                ax.set_title(str(d), color="#e2e8f0", fontsize=11)
        axes[s, 0].set_ylabel(f"seed {seeds[s]}", color="#fbbf24", fontsize=9)
    fig.suptitle("xyt  ·  yellow  ·  from noise  ·  do digits differ?", color="white")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
