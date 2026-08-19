#!/usr/bin/env python3
"""Re-score published JiT ckpts with paired + free metrics, both starts."""
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
from scripts.train_omni_probe import eval_ports

CKPTS = {
    "paper_xpred": ROOT / "checkpoints" / "omni_d256_unified_jit_t2i_best.pt",
    "noise_xpred": ROOT / "checkpoints" / "omni_d256_unified_jit_noise_best.pt",
}
OUT = ROOT / "results" / "published" / "jit_fair_metrics.json"


def _fmt(ev: dict) -> str:
    return (
        f"psnr={ev['psnr']:.2f} str={ev['stroke_psnr']:.2f} bg={ev['bg_psnr']:.2f} "
        f"col={ev['color_acc']*100:.0f} dig={ev['digit_top1']*100:.0f} "
        f"iou={ev['digit_iou']:.3f} inkf={ev['ink_frac']*100:.1f} "
        f"fld={ev['flood']*100:.0f}"
    )


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = {}
    setups = (
        ("product_paper", "pair", "paper"),
        ("gen_noise_paper_tgt", "noise", "paper"),
        ("gen_noise_black", "noise", "black"),
    )
    for name, ckpt in CKPTS.items():
        if not ckpt.exists():
            table[name] = {"error": str(ckpt)}
            continue
        m = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x").to(dev)
        missing = m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        print(f"=== {name} missing={len(missing.missing_keys)} ===", flush=True)
        table[name] = {"missing_time_mod": sum("time_mod" in k for k in missing.missing_keys)}
        for tag, x0, canvas in setups:
            ev = eval_ports(
                m, np.random.default_rng(9001), 32, dev, n=32, kinds=["t2i"],
                flow_steps=8, flow_method="heun", fm_x0=x0, t2i_canvas=canvas,
            )["t2i"]
            table[name][tag] = ev
            print(f"  {tag:22s} {_fmt(ev)}", flush=True)
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
