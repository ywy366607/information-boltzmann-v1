#!/usr/bin/env python3
"""Kalman-as-S-update on top of F2. Update only (no predict A).

  S ← μ* = μp + K(S − μp), then write A.
  Everything else = F2: g(U), vfe_coef=0.1, 4 unshared, rms_dir unused on S.

  python scripts/run_kalman_s_2000.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_triad_2000steps import train_single_run

PUB_F2 = ROOT / "results" / "published" / "f2_vfe_2000step_table.json"
PUB_B = ROOT / "results" / "published" / "upd_rms_bayes_2000step_table.json"
FIG = ROOT / "present" / "figs"
OUT = ROOT / "results" / "published" / "kalman_s_2000step_table.json"
CMP = ROOT / "results" / "published" / "kalman_s_vs_f2.json"


def _seed42(table: dict) -> dict:
    arm = table.get("v1_bayes", table)
    for r in arm.get("runs", [arm]):
        if int(r.get("seed", -1)) == 42:
            return r
    runs = arm.get("runs")
    return runs[0] if runs else arm


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vfe-coef", type=float, default=0.1)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    print(
        "=== Kalman-S  S←μ*  write=A  g(U)  F2-λ=0.1  no predict  seed=42 ===",
        flush=True,
    )
    res = train_single_run(
        arm="v1_bayes",
        run_idx=0,
        seed=42,
        steps=args.steps,
        device=args.device,
        s_update="rms_dir",
        use_stiefel=True,
        deslice_topk=2,
        gate_on="u",
        deslice_write="absolute",
        gate_h_local=False,
        vfe_coef=args.vfe_coef,
        s_kalman_update=True,
        ckpt_tag="kalman_s",
    )
    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "note": "F2 + Kalman S update: S←μ*, write A, g(U), no predict A.",
            "vfe_coef": args.vfe_coef,
            "s_kalman_update": True,
            "gate_on": "u",
            "mean_acc": res["final_acc"] * 100,
            "mean_ocr": res["final_tasks"].get("ocr", 0) * 100,
            "mean_kinks": res["final_tasks"].get("kinks", 0) * 100,
            "mean_color": res["final_tasks"].get("color", 0) * 100,
            "mean_loss": res["final_loss"],
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
            "layer_K": res.get("layer_K"),
            "runs": [res],
        }
    }
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")

    f2 = _seed42(json.loads(PUB_F2.read_text(encoding="utf-8")))
    b = _seed42(json.loads(PUB_B.read_text(encoding="utf-8")))
    cmp = {
        "champion_b_seed42": {"acc": b["final_acc"], "tasks": b["final_tasks"]},
        "f2_vfe_seed42": {"acc": f2["final_acc"], "tasks": f2["final_tasks"]},
        "kalman_s_seed42": {
            "acc": res["final_acc"],
            "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
            "layer_K": res.get("layer_K"),
        },
        "delta_pp_vs_f2": {
            "acc": (res["final_acc"] - f2["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - f2["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - f2["final_tasks"]["kinks"]) * 100,
        },
    }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(
        f"Kalman-S FINAL acc={res['final_acc']*100:.2f} "
        f"ocr={res['final_tasks'].get('ocr',0)*100:.1f} "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f} "
        f"K={res.get('layer_K')}",
        flush=True,
    )
    print(
        f"  DELTA vs F2  acc={cmp['delta_pp_vs_f2']['acc']:+.2f}pp "
        f"ocr={cmp['delta_pp_vs_f2']['ocr']:+.2f} "
        f"kinks={cmp['delta_pp_vs_f2']['kinks']:+.2f}",
        flush=True,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    labs = ["Overall", "OCR", "Kinks", "Color"]
    old = [
        f2["final_acc"] * 100,
        f2["final_tasks"]["ocr"] * 100,
        f2["final_tasks"]["kinks"] * 100,
        f2["final_tasks"]["color"] * 100,
    ]
    new = [
        res["final_acc"] * 100,
        res["final_tasks"]["ocr"] * 100,
        res["final_tasks"]["kinks"] * 100,
        res["final_tasks"]["color"] * 100,
    ]
    x = np.arange(4)
    ax.bar(x - 0.18, old, 0.36, label="F2 (default)", color="#2dd4bf")
    ax.bar(x + 0.18, new, 0.36, label="F2 + Kalman S", color="#c084fc")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("Kalman-as-S-update vs F2  ·  seed 42  ·  write A, no predict",
                 color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "kalman_s_vs_f2_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {OUT} and {FIG / 'kalman_s_vs_f2_bars.png'}", flush=True)


if __name__ == "__main__":
    main()
