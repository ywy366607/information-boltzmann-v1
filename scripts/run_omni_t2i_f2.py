#!/usr/bin/env python3
"""F2 DualStreamOmni on the *fixed* t2i port.

Previous broken run: Champion B DualStreamOmni (no vfe_coef),
d=256 M=64 h=8, ckpt checkpoints/omni_d256_t2i_best.pt.
Headline PSNR ~7 was background: target canvas ≠ input paper.
Stroke PSNR was already ~74 / ink ~91%.

This script:
  1. re-evals that old ckpt on the fixed same-paper task
  2. trains a fresh F2 model (λ=0.1 gap, write A, g(U)) on t2i only
  3. does not overwrite the old ckpt

  python scripts/run_omni_t2i_f2.py --device cuda
"""
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

from fine_grain.omni_model import DualStreamOmni
from scripts.train_omni_probe import eval_ports, render_gallery, main as train_main

OLD_CKPT = ROOT / "checkpoints" / "omni_d256_t2i_best.pt"
OLD_TABLE = ROOT / "results" / "published" / "omni_t2i_2000step_table.json"
OUT = ROOT / "results" / "published" / "omni_t2i_f2_2000step_table.json"
CMP = ROOT / "results" / "published" / "omni_t2i_f2_vs_old.json"
FIG = ROOT / "present" / "figs"


def _make_omni(device, vfe_coef: float = 0.0) -> DualStreamOmni:
    return DualStreamOmni(
        d_model=256, n_slices=64, n_layers=4, res=32, n_heads=8,
        surprise_mode="v1_bayes", s_update="rms_dir",
        prior_loss_coef=0.1, use_stiefel=True, deslice_topk=2,
        gate_on="u", deslice_write="absolute", gate_h_local=False,
        vfe_coef=vfe_coef,
    ).to(device)


def _eval_old(device: str) -> dict:
    if not OLD_CKPT.exists():
        print(f"  no old ckpt at {OLD_CKPT}", flush=True)
        return {}
    dev = torch.device(device)
    model = _make_omni(dev, vfe_coef=0.0)
    raw = torch.load(OLD_CKPT, map_location="cpu")
    miss = model.load_state_dict(raw, strict=False)
    print(
        f"  loaded {OLD_CKPT} miss={len(miss.missing_keys)} extra={len(miss.unexpected_keys)}",
        flush=True,
    )
    ev = eval_ports(model, np.random.default_rng(9001), 32, dev, n=64, kinds=["t2i"])
    print(
        f"  OLD ckpt on FIXED t2i  psnr={ev['t2i']['psnr']:.2f}  "
        f"stroke={ev['t2i']['stroke_psnr']:.2f}  bg={ev['t2i']['bg_psnr']:.2f}  "
        f"ink={ev['t2i']['ink']*100:.1f}",
        flush=True,
    )
    return ev


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-old-only", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    print("=== t2i  old Champion-B DualStreamOmni on fixed same-paper task ===", flush=True)
    old_fixed = _eval_old(args.device)
    old_orig = {}
    if OLD_TABLE.exists():
        old_orig = json.loads(OLD_TABLE.read_text(encoding="utf-8")).get("final", {})
    if args.eval_old_only:
        CMP.write_text(json.dumps({
            "old_model": "DualStreamOmni Champion B (v1_bayes+rms, no F2)",
            "old_ckpt": str(OLD_CKPT),
            "old_on_mismatched_canvas": old_orig.get("t2i"),
            "old_on_fixed_same_paper": old_fixed.get("t2i"),
        }, indent=2), encoding="utf-8")
        return

    print("=== t2i  F2 DualStreamOmni  d=256 M=64 h=8  λ=0.1  write A ===", flush=True)
    sys.argv = [
        "train_omni_probe.py",
        "--device", args.device,
        "--steps", str(args.steps),
        "--d-model", "256",
        "--n-slices", "64",
        "--n-heads", "8",
        "--mix", "t2i",
        "--vfe-coef", "0.1",
        "--tag", "t2i_f2",
        "--out", str(OUT),
    ]
    train_main()

    new_table = json.loads(OUT.read_text(encoding="utf-8"))
    new = new_table["final"]["t2i"]
    old_m = old_orig.get("t2i") or {}
    cmp = {
        "old_model": "DualStreamOmni Champion B (v1_bayes + rms_dir, vfe_coef=0)",
        "old_ckpt": str(OLD_CKPT),
        "old_size": "d=256 M=64 h=8  ~8.1M",
        "old_on_mismatched_canvas": old_m,
        "old_on_fixed_same_paper": old_fixed.get("t2i"),
        "f2_on_fixed_same_paper": new,
        "f2_ckpt": new_table.get("ckpt"),
        "note": (
            "Old headline PSNR~7 was a second random canvas as target. "
            "Stroke/ink were already high. F2 trained on same-paper t2i."
        ),
    }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    labs = ["PSNR", "Stroke", "BG", "Ink×20"]
    old_bar = [
        float(old_m.get("psnr") or 0),
        float(old_m.get("stroke_psnr") or 0),
        float(old_m.get("bg_psnr") or 0),
        float(old_m.get("ink") or 0) * 20.0,
    ]
    new_bar = [
        float(new.get("psnr") or 0),
        float(new.get("stroke_psnr") or 0),
        float(new.get("bg_psnr") or 0),
        float(new.get("ink") or 0) * 20.0,
    ]
    x = np.arange(4)
    ax.bar(x - 0.18, old_bar, 0.36, label="Old ChampB (mismatch tgt)", color="#fbbf24")
    ax.bar(x + 0.18, new_bar, 0.36, label="F2 same-paper t2i", color="#2dd4bf")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("t2i  ·  old Champion B vs F2", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "omni_t2i_f2_vs_old_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {CMP}", flush=True)


if __name__ == "__main__":
    main()
