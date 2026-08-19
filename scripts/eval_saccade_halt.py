#!/usr/bin/env python3
"""S1: residual halt on the trained 4×2 saccade ckpt (no retrain).

  python scripts/eval_saccade_halt.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model

CKPT = ROOT / "checkpoints" / "v1_bayes_2000step_saccade_l4i2_run1_best.pt"
OUT = ROOT / "results" / "published" / "saccade_s1_halt.json"


def _load(device, halt: bool, eps: float) -> DualStreamVQAModel:
    m = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms_dir", use_stiefel=True, deslice_topk=2, n_heads=4,
        gate_on="u", deslice_write="absolute", gate_h_local=False,
        share_layers=False, saccade=True, saccade_inner=2, saccade_gain=1.0,
        saccade_halt=halt, saccade_halt_eps=eps, saccade_halt_train=False,
    ).to(device)
    raw = torch.load(CKPT, map_location="cpu")
    miss = m.load_state_dict(raw, strict=False)
    print(f"  load miss={len(miss.missing_keys)} extra={len(miss.unexpected_keys)}", flush=True)
    m.eval()
    return m


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, default=35)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    device = torch.device(args.device)
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    rows = []
    # halt extra look if residual focus (max/mean pixel mass) <= eps
    for halt, eps in [(False, 0.0)] + [(True, e) for e in (2.4, 2.2, 2.0, 1.8, 1.6, 1.4)]:
        print(f"=== halt={halt} eps={eps} ===", flush=True)
        m = _load(device, halt, eps)
        rng = np.random.default_rng(42 + 9999)
        ev = eval_model(m, rng, val_batches=args.batches, batch_size=32, res=32, device=device)
        row = {
            "halt": halt,
            "eps": eps,
            "acc": ev["acc"],
            "loss": ev["loss"],
            "tasks": ev["task_accs"],
            "mean_looks": ev.get("mean_looks", 8.0 if not halt else None),
            "looks_task": ev.get("looks_task", {}),
        }
        rows.append(row)
        print(
            f"  acc={ev['acc']*100:.2f} ocr={ev['task_accs'].get('ocr',0)*100:.1f} "
            f"kinks={ev['task_accs'].get('kinks',0)*100:.1f} color={ev['task_accs'].get('color',0)*100:.1f} "
            f"looks={row['mean_looks']:.2f}  task_looks={row['looks_task']}",
            flush=True,
        )
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"checkpoint": str(CKPT), "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
