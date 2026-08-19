#!/usr/bin/env python3
"""4 unique layers (same params as F2), 2 residual looks per layer.

  python scripts/run_saccade_k4_2000.py --device cuda
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
OUT = ROOT / "results" / "published" / "saccade_l4i2_2000step_table.json"
CMP = ROOT / "results" / "published" / "saccade_l4i2_vs_f2.json"


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
    ap.add_argument("--n-loops", type=int, default=2, help="Residual looks per unique layer.")
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\\ml_cache")

    print(
        "=== S0  4 unique layers × %d looks  residual mass  =F2 params  seed=42 ==="
        % args.n_loops,
        flush=True,
    )
    res = train_single_run(
        arm="v1_bayes",
        run_idx=0,
        seed=42,
        steps=args.steps,
        device=args.device,
        n_layers=4,
        s_update="rms_dir",
        use_stiefel=True,
        deslice_topk=2,
        gate_on="u",
        deslice_write="absolute",
        gate_h_local=False,
        vfe_coef=args.vfe_coef,
        share_layers=False,
        saccade=True,
        saccade_gain=1.0,
        saccade_inner=args.n_loops,
        ckpt_tag="saccade_l4i2",
    )
    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "note": (
                "4 unique layers (iso-param F2). Each layer: look, residual mass, look again. "
                "Q per layer unchanged. Time decoupled from depth."
            ),
            "vfe_coef": args.vfe_coef,
            "share_layers": False,
            "saccade": True,
            "n_layers": 4,
            "saccade_inner": args.n_loops,
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")

    f2 = _seed42(json.loads(PUB_F2.read_text(encoding="utf-8"))) if PUB_F2.exists() else None
    b = _seed42(json.loads(PUB_B.read_text(encoding="utf-8"))) if PUB_B.exists() else None
    cmp = {
        "saccade_l4i2_seed42": {
            "acc": res["final_acc"],
            "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
        }
    }
    if f2 is not None:
        cmp["f2_vfe_seed42"] = {"acc": f2["final_acc"], "tasks": f2["final_tasks"]}
        cmp["delta_pp_vs_f2"] = {
            "acc": (res["final_acc"] - f2["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - f2["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - f2["final_tasks"]["kinks"]) * 100,
        }
    if b is not None:
        cmp["champion_b_seed42"] = {"acc": b["final_acc"], "tasks": b["final_tasks"]}
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(
        f"S0 FINAL acc={res['final_acc']*100:.2f} ocr={res['final_tasks'].get('ocr',0)*100:.1f} "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f}",
        flush=True,
    )
    if "delta_pp_vs_f2" in cmp:
        d = cmp["delta_pp_vs_f2"]
        print(
            f"  DELTA vs F2  acc={d['acc']:+.2f}pp ocr={d['ocr']:+.2f} kinks={d['kinks']:+.2f}",
            flush=True,
        )

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    labs = ["Overall", "OCR", "Kinks", "Color"]
    new = [
        res["final_acc"] * 100, res["final_tasks"]["ocr"] * 100,
        res["final_tasks"]["kinks"] * 100, res["final_tasks"]["color"] * 100,
    ]
    x = np.arange(4)
    width = 0.28
    if f2 is not None:
        old = [
            f2["final_acc"] * 100, f2["final_tasks"]["ocr"] * 100,
            f2["final_tasks"]["kinks"] * 100, f2["final_tasks"]["color"] * 100,
        ]
        ax.bar(x - width / 2, old, width, label="F2 4-unshared", color="#38bdf8")
        ax.bar(x + width / 2, new, width, label="S0 saccade Φ×4", color="#c084fc")
    else:
        ax.bar(x, new, width, label="S0 saccade Φ×4", color="#c084fc")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("S0 IG-gaze vs F2  ·  seed 42", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "saccade_k4_vs_f2_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {OUT} and {FIG / 'saccade_k4_vs_f2_bars.png'}", flush=True)


if __name__ == "__main__":
    main()
