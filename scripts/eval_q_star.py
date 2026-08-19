#!/usr/bin/env python3
"""Inference-only: snap q ← q* on all / residual layers. No retraining.

  python scripts/eval_q_star.py --device cuda
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

from scripts.probe_deslice_memory import load_model
from scripts.run_v0_surprise_eval import eval_model

CKPTS = {
    "f2_vfe": ROOT / "checkpoints" / "v1_bayes_2000step_f2_vfe_run1_best.pt",
    "champ_b": ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run1_best.pt",
}
MODES = ("amortized", "residual", "star")
OUT = ROOT / "results" / "published" / "q_star_infer.json"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=35)
    ap.add_argument("--thresh", type=float, default=0.05)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")
    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    all_rows = {}

    for name, path in CKPTS.items():
        if not path.exists():
            print(f"SKIP {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        model = load_model(path, device)
        model.mot_stack.set_write_knobs("absolute", gate_h_local=False)
        all_rows[name] = {}
        for mode in MODES:
            model.mot_stack.set_q_infer(mode, thresh=args.thresh)
            ev = eval_model(model, rng, val_batches=args.batches, batch_size=32, res=32, device=device)
            rec = {
                "acc": ev["acc"],
                "ocr": ev["task_accs"].get("ocr", 0.0),
                "kinks": ev["task_accs"].get("kinks", 0.0),
                "color": ev["task_accs"].get("color", 0.0),
                "layer_u": ev.get("layer_surprise"),
                "layer_gap": ev.get("layer_gap"),
            }
            all_rows[name][mode] = rec
            print(
                f"  {mode:10s} acc={rec['acc']*100:5.1f} ocr={rec['ocr']*100:5.1f} "
                f"kinks={rec['kinks']*100:5.1f}  U={ [round(x,3) for x in (rec['layer_u'] or [])] }  "
                f"gap={ [round(x,3) for x in (rec['layer_gap'] or [])] }",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
