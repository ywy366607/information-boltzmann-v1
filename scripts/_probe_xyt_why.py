#!/usr/bin/env python3
"""Quantify why 1px xyt digits are unreadable.

H1 class not bound: between-class MSE ≈ 0, reader acc ≈ 10%.
H2 position chaos: COM std large vs canvas.
H3 too thin / smear: ink_frac far from a 1px template.
"""
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

from fine_grain.gen_metrics import digit_shift_scores, free_ink_mask
from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import fm_generate, to_unit

GEN = ROOT / "checkpoints" / "omni_d256_unified_jit_xyt_best.pt"
READ = ROOT / "checkpoints" / "omni_d256_unified_i2t_best.pt"
OUT = ROOT / "results" / "published" / "omni_xyt_why.json"
N_PER = 8
COLOR = "yellow"


def _com(mask: torch.Tensor):
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() < 1:
        return None
    return float(ys.float().mean()), float(xs.float().mean())


@torch.no_grad()
def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x", fm_signed=True).to(dev)
    gen.load_state_dict(torch.load(GEN, map_location="cpu"), strict=False)
    gen.eval()

    reader = None
    if READ.exists():
        reader = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE).to(dev)
        miss = reader.load_state_dict(torch.load(READ, map_location="cpu"), strict=False)
        reader.eval()
        print(f"reader i2t missing={len(miss.missing_keys)}", flush=True)

    by_d = {d: [] for d in range(10)}
    template_hit = 0
    reader_hit = 0
    n = 0
    conf = torch.zeros(10, 10)
    ink_fracs = []
    coms = {d: [] for d in range(10)}

    torch.manual_seed(0)
    for d in range(10):
        prompt = f"Draw digit {d} with a thin {COLOR} stroke blank image"
        for i in range(N_PER):
            z = torch.randn(1, 3, 32, 32, device=dev)
            rgb = to_unit(fm_generate(
                gen, z, [prompt], [True], n_steps=8, method="heun",
                clamp=False, cfg=2.0,
            )).clamp(0, 1)
            by_d[d].append(rgb[0].cpu())
            ink = free_ink_mask(rgb, COLOR)[0]
            ink_fracs.append(float(ink.float().mean()))
            c = _com(ink)
            if c is not None:
                coms[d].append(c)
            rec = digit_shift_scores(rgb, str(d), COLOR)
            pred_d = int(rec["digit_best"])
            template_hit += int(pred_d == d)
            conf[d, pred_d] += 1
            if reader is not None:
                out = reader(rgb, ["What is this"], need_pix=[False])
                rid = int(out["logits"][0].argmax().item())
                # answers are colors + kinks + digits; map back
                ans = reader.answers[rid] if rid < len(reader.answers) else "?"
                reader_hit += int(str(ans) == str(d))
            n += 1
        print(f"  digit {d} done", flush=True)

    # within / between (pixel MSE on [0,1] images)
    means = torch.stack([torch.stack(by_d[d]).mean(0) for d in range(10)])
    within = []
    for d in range(10):
        s = torch.stack(by_d[d])
        within.append(float((s - means[d]).pow(2).mean()))
    within_m = float(np.mean(within))
    between = float((means - means.mean(0)).pow(2).mean())
    eta = between / max(between + within_m, 1e-8)

    com_std = []
    for d, pts in coms.items():
        if len(pts) < 2:
            continue
        arr = np.array(pts)
        com_std.append(float(arr.std(0).mean()))

    rec = {
        "n_per_digit": N_PER,
        "color": COLOR,
        "template_acc": template_hit / max(n, 1),
        "i2t_reader_acc": (reader_hit / max(n, 1)) if reader is not None else None,
        "chance": 0.1,
        "within_mse": within_m,
        "between_mse": between,
        "eta_squared": eta,
        "ink_frac_mean": float(np.mean(ink_fracs)),
        "ink_frac_std": float(np.std(ink_fracs)),
        "com_std_px": float(np.mean(com_std)) if com_std else None,
        "confusion": conf.tolist(),
        "note": (
            "eta≈1: digits are distinct blobs. eta≈0: one shared scribble. "
            "High com_std: same digit jumps around. Reader is unified i2t (paper OCR)."
        ),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k != "confusion"}, indent=2), flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
