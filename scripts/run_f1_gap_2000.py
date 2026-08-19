#!/usr/bin/env python3
"""F1: one-seed 2000-step Champion-B recipe with g = g(gap), write A.

Does not overwrite published RMS / Champion B ckpts.

  python scripts/run_f1_gap_2000.py --device cuda
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
OUT = ROOT / "results" / "published" / "f1_gap_2000step_table.json"
CMP = ROOT / "results" / "published" / "f1_gap_vs_champb.json"


def _seed42(table: dict) -> dict:
    arm = table.get("v1_bayes", table)
    for r in arm["runs"]:
        if int(r["seed"]) == 42:
            return r
    return arm["runs"][0]


def render(p0: dict, pub: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    b = _seed42(pub)
    labels = ["Overall", "1px OCR", "Kinks", "Color"]
    old = [
        b["final_acc"] * 100,
        b["final_tasks"]["ocr"] * 100,
        b["final_tasks"]["kinks"] * 100,
        b["final_tasks"]["color"] * 100,
    ]
    new = [
        p0["final_acc"] * 100,
        p0["final_tasks"]["ocr"] * 100,
        p0["final_tasks"]["kinks"] * 100,
        p0["final_tasks"]["color"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, old, w, label="Champion B  g(U)", color="#fbbf24")
    ax.bar(x + w / 2, new, w, label="F1  g(gap)", color="#2dd4bf")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, axis="y", ls=":", alpha=0.3)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    ax.set_title("F1 gap gate vs Champion B   ·   seed 42, write A", color="white", fontweight="bold")
    fig.tight_layout()
    p = FIG / "f1_gap_vs_champb_bars.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, ls=":", alpha=0.3)
    sb = [pt["step"] for pt in b["trajectory"]]
    sp = [pt["step"] for pt in p0["trajectory"]]
    ax1.plot(sb, [pt["val_acc"] * 100 for pt in b["trajectory"]], color="#fbbf24", label="B acc")
    ax1.plot(sp, [pt["val_acc"] * 100 for pt in p0["trajectory"]], color="#2dd4bf", label="F1 acc")
    ax1.plot(sb, [pt["val_kinks"] * 100 for pt in b["trajectory"]], color="#fbbf24", ls="--", alpha=0.55)
    ax1.plot(sp, [pt["val_kinks"] * 100 for pt in p0["trajectory"]], color="#2dd4bf", ls="--", alpha=0.55)
    ax1.set_title("Val acc / kinks", color="white")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    # gap vs U if present
    if p0["trajectory"] and p0["trajectory"][-1].get("layer_gap"):
        steps = [pt["step"] for pt in p0["trajectory"]]
        u_l3 = [pt.get("layer_u", [0, 0, 0, 0])[-1] if pt.get("layer_u") else 0 for pt in p0["trajectory"]]
        g_l3 = [pt.get("layer_gap", [0, 0, 0, 0])[-1] if pt.get("layer_gap") else 0 for pt in p0["trajectory"]]
        ax2.plot(steps, u_l3, color="#fb7185", label="L3 U")
        ax2.plot(steps, g_l3, color="#38bdf8", label="L3 gap")
        ax2.set_title("U vs gap (should split)", color="white")
        ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    p2 = FIG / "f1_gap_vs_champb_trajectories.png"
    fig.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  saved {p2}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

    print("=== F1  g(gap)  write=A  seed=42  v1_bayes+RMS ===", flush=True)
    res = train_single_run(
        arm="v1_bayes",
        run_idx=0,
        seed=42,
        steps=args.steps,
        device=args.device,
        s_update="rms_dir",
        use_stiefel=True,
        deslice_topk=2,
        gate_on="gap",
        deslice_write="absolute",
        gate_h_local=False,
        ckpt_tag="f1_gap",
    )
    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "steps": args.steps,
            "n_runs": 1,
            "gate_on": "gap",
            "deslice_write": "absolute",
            "s_update": "rms_dir",
            "ckpt_tag": "f1_gap",
            "mean_acc": res["final_acc"] * 100,
            "std_acc": 0.0,
            "mean_ocr": res["final_tasks"].get("ocr", 0) * 100,
            "mean_kinks": res["final_tasks"].get("kinks", 0) * 100,
            "mean_color": res["final_tasks"].get("color", 0) * 100,
            "mean_loss": res["final_loss"],
            "runs": [res],
        }
    }
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)
    print(
        f"F1 FINAL  acc={res['final_acc']*100:.2f}%  ocr={res['final_tasks'].get('ocr',0)*100:.1f}  "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f}  gap_L={res.get('layer_surprise')}",
        flush=True,
    )
    pub = json.loads(PUB_B.read_text(encoding="utf-8"))
    b = _seed42(pub)
    cmp = {
        "champion_b_seed42": {
            "acc": b["final_acc"], "tasks": b["final_tasks"],
        },
        "f1_gap_seed42": {
            "acc": res["final_acc"], "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "layer_u": res.get("layer_surprise"),
        },
        "delta_pp": {
            "acc": (res["final_acc"] - b["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - b["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - b["final_tasks"]["kinks"]) * 100,
        },
    }
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(
        f"  DELTA vs B  acc={cmp['delta_pp']['acc']:+.2f}pp  "
        f"ocr={cmp['delta_pp']['ocr']:+.2f}  kinks={cmp['delta_pp']['kinks']:+.2f}",
        flush=True,
    )
    render(res, pub)


if __name__ == "__main__":
    main()
