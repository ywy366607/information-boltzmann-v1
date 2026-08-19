#!/usr/bin/env python3
"""F2: train amortized q on E[gap], keep g(U) and write A.

  python scripts/run_f2_vfe_2000.py --device cuda
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

PUB_B = ROOT / "results" / "published" / "upd_rms_bayes_2000step_table.json"
FIG = ROOT / "present" / "figs"
OUT = ROOT / "results" / "published" / "f2_vfe_2000step_table.json"
CMP = ROOT / "results" / "published" / "f2_vfe_vs_champb.json"


def _seed42(table: dict) -> dict:
    arm = table.get("v1_bayes", table)
    for r in arm["runs"]:
        if int(r["seed"]) == 42:
            return r
    return arm["runs"][0]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vfe-coef", type=float, default=0.1)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    print("=== F2  train q on gap  gate=g(U)  write=A  seed=42 ===", flush=True)
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
        ckpt_tag="f2_vfe",
    )
    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "note": "F2: CE+pred_loss+λ E[KL(q||q*)]. Gate still g(U). Write A.",
            "vfe_coef": args.vfe_coef,
            "gate_on": "u",
            "mean_acc": res["final_acc"] * 100,
            "mean_ocr": res["final_tasks"].get("ocr", 0) * 100,
            "mean_kinks": res["final_tasks"].get("kinks", 0) * 100,
            "mean_color": res["final_tasks"].get("color", 0) * 100,
            "mean_loss": res["final_loss"],
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
            "runs": [res],
        }
    }
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")
    pub = json.loads(PUB_B.read_text(encoding="utf-8"))
    b = _seed42(pub)
    cmp = {
        "champion_b_seed42": {"acc": b["final_acc"], "tasks": b["final_tasks"]},
        "f2_vfe_seed42": {
            "acc": res["final_acc"], "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
        },
        "delta_pp": {
            "acc": (res["final_acc"] - b["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - b["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - b["final_tasks"]["kinks"]) * 100,
        },
    }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(
        f"F2 FINAL acc={res['final_acc']*100:.2f} ocr={res['final_tasks'].get('ocr',0)*100:.1f} "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f}",
        flush=True,
    )
    print(
        f"  DELTA vs B  acc={cmp['delta_pp']['acc']:+.2f}pp "
        f"ocr={cmp['delta_pp']['ocr']:+.2f} kinks={cmp['delta_pp']['kinks']:+.2f}",
        flush=True,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    labs = ["Overall", "OCR", "Kinks", "Color"]
    old = [b["final_acc"] * 100, b["final_tasks"]["ocr"] * 100,
           b["final_tasks"]["kinks"] * 100, b["final_tasks"]["color"] * 100]
    new = [res["final_acc"] * 100, res["final_tasks"]["ocr"] * 100,
           res["final_tasks"]["kinks"] * 100, res["final_tasks"]["color"] * 100]
    x = np.arange(4)
    ax.bar(x - 0.18, old, 0.36, label="Champion B", color="#fbbf24")
    ax.bar(x + 0.18, new, 0.36, label="F2 train q on gap", color="#2dd4bf")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("F2 vs Champion B  ·  seed 42  ·  g(U)+write A", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "f2_vfe_vs_champb_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {OUT} and {FIG / 'f2_vfe_vs_champb_bars.png'}", flush=True)


if __name__ == "__main__":
    main()
