#!/usr/bin/env python3
"""R0 shared-Φ loop: K=4, LTI assemble Δ_Φ, deep-sup CE, write A, F2 vfe.

Does not overwrite F2 / Champion B / the failed mix-then-Φ recur_k4 ckpts.

  python scripts/run_recur_k4_2000.py --device cuda
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
OUT = ROOT / "results" / "published" / "recur_k4_delta_2000step_table.json"
CMP = ROOT / "results" / "published" / "recur_k4_delta_vs_f2.json"


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
    ap.add_argument("--n-loops", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\\ml_cache")

    print(
        "=== R0 recur  share Φ × K=%d  LTI assemble Δ  deep CE  write=A  F2  seed=42 ==="
        % args.n_loops,
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
        share_layers=True,
        n_loops=args.n_loops,
        ckpt_tag="recur_k4_delta",
    )
    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "note": (
                "R0: shared Φ × K. After each cell: X ← ĀX + B̄ e + (X_Φ−X). "
                "Mean CE over loops. Gate g(U). Write A. F2. No loop-index."
            ),
            "vfe_coef": args.vfe_coef,
            "gate_on": "u",
            "share_layers": True,
            "n_loops": args.n_loops,
            "mean_acc": res["final_acc"] * 100,
            "mean_ocr": res["final_tasks"].get("ocr", 0) * 100,
            "mean_kinks": res["final_tasks"].get("kinks", 0) * 100,
            "mean_color": res["final_tasks"].get("color", 0) * 100,
            "mean_loss": res["final_loss"],
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
            "layer_ce": res.get("layer_ce"),
            "rho_x": res.get("rho_x"),
            "rho_h": res.get("rho_h"),
            "runs": [res],
        }
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")

    f2 = _seed42(json.loads(PUB_F2.read_text(encoding="utf-8"))) if PUB_F2.exists() else None
    b = _seed42(json.loads(PUB_B.read_text(encoding="utf-8"))) if PUB_B.exists() else None
    cmp = {
        "recur_k4_seed42": {
            "acc": res["final_acc"],
            "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "layer_u": res.get("layer_surprise"),
            "layer_gap": res.get("layer_gap"),
            "layer_ce": res.get("layer_ce"),
            "rho_x": res.get("rho_x"),
            "rho_h": res.get("rho_h"),
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
        cmp["delta_pp_vs_b"] = {
            "acc": (res["final_acc"] - b["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - b["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - b["final_tasks"]["kinks"]) * 100,
        }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    print(
        f"R0 FINAL acc={res['final_acc']*100:.2f} ocr={res['final_tasks'].get('ocr',0)*100:.1f} "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f} "
        f"rho_x={res.get('rho_x', 0):.4f} rho_h={res.get('rho_h', 0):.4f}",
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
        res["final_acc"] * 100,
        res["final_tasks"]["ocr"] * 100,
        res["final_tasks"]["kinks"] * 100,
        res["final_tasks"]["color"] * 100,
    ]
    x = np.arange(4)
    width = 0.28
    offset = 0
    if b is not None:
        old_b = [
            b["final_acc"] * 100, b["final_tasks"]["ocr"] * 100,
            b["final_tasks"]["kinks"] * 100, b["final_tasks"]["color"] * 100,
        ]
        ax.bar(x - width, old_b, width, label="Champion B", color="#fbbf24")
        offset = 1
    if f2 is not None:
        old_f = [
            f2["final_acc"] * 100, f2["final_tasks"]["ocr"] * 100,
            f2["final_tasks"]["kinks"] * 100, f2["final_tasks"]["color"] * 100,
        ]
        ax.bar(x, old_f, width, label="F2 train q on gap", color="#38bdf8")
        offset = 1
    ax.bar(x + width * offset, new, width, label="R0 Φ×4  assemble Δ", color="#a78bfa")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("R0 assemble-Δ K=4 vs F2 / Champion B  ·  seed 42", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "recur_k4_delta_vs_f2_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {OUT} and {FIG / 'recur_k4_delta_vs_f2_bars.png'}", flush=True)


if __name__ == "__main__":
    main()
