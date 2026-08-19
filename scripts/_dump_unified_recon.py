#!/usr/bin/env python3
"""Recover recon table/gallery from the ckpt saved before the gallery crash."""
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


def main() -> None:
    ckpt = ROOT / "checkpoints" / "omni_d256_unified_recon_best.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    ev = eval_ports(model, np.random.default_rng(9001), 32, dev, n=64, kinds=["recon"])
    gal = ROOT / "present" / "figs" / "omni_unified_recon_gallery.png"
    render_gallery(model, np.random.default_rng(7), 32, dev, gal, kinds=["recon"] * 5)
    rec = ev["recon"]
    table = {
        "steps": 2000,
        "res": 32,
        "d_model": 256,
        "n_slices": 64,
        "n_heads": 8,
        "final": ev,
        "ckpt": str(ckpt),
        "gallery": str(gal),
        "vfe_coef": 0.1,
        "deslice_write": "increment",
        "s_lang_topk": 0,
        "mix": ["recon"],
        "note": "Unified residual. Recovered after gallery crash; ckpt was already saved.",
    }
    out = ROOT / "results" / "published" / "omni_unified_recon_2000step_table.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(
        f"RECON psnr={rec['psnr']:.2f} bg={rec['bg_psnr']:.2f} "
        f"str={rec['stroke_psnr']:.2f} flood={rec['flood']*100:.1f}",
        flush=True,
    )
    sum_p = ROOT / "results" / "published" / "omni_unified_ports.json"
    summary = json.loads(sum_p.read_text(encoding="utf-8")) if sum_p.exists() else {"ports": {}}
    summary.setdefault("ports", {})
    summary["ports"]["recon"] = {
        "kind": "recon",
        "acc": rec.get("acc", 0.0),
        "psnr": rec["psnr"],
        "stroke_psnr": rec["stroke_psnr"],
        "bg_psnr": rec["bg_psnr"],
        "ink": rec["ink"],
        "flood": rec["flood"],
        "checkpoint": str(ckpt),
        "deslice_write": "increment",
        "vfe_coef": 0.1,
    }
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
