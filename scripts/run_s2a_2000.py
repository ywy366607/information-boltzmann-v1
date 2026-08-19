#!/usr/bin/env python3
"""S2a: F2 backbone + answer-IG halt head. Init from F2, see if G helps kinks.

  python scripts/run_s2a_2000.py --device cuda
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_triad_2000steps import train_single_run
from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model

F2_CKPT = ROOT / "checkpoints" / "v1_bayes_2000step_f2_vfe_run1_best.pt"
PUB_F2 = ROOT / "results" / "published" / "f2_vfe_2000step_table.json"
OUT = ROOT / "results" / "published" / "s2a_l4i2_2000step_table.json"
CMP = ROOT / "results" / "published" / "s2a_vs_f2.json"
FIG = ROOT / "present" / "figs"


def _seed42(table: dict) -> dict:
    arm = table.get("v1_bayes", table)
    for r in arm.get("runs", [arm]):
        if int(r.get("seed", -1)) == 42:
            return r
    runs = arm.get("runs")
    return runs[0] if runs else arm


def _eval_modes(model, device, batches: int = 35) -> dict:
    rng = np.random.default_rng(42 + 9999)
    model.mot_stack.s2a_eval_halt = True
    ev_ad = eval_model(model, rng, val_batches=batches, batch_size=32, res=32, device=device)
    rng = np.random.default_rng(42 + 9999)
    model.mot_stack.s2a_eval_halt = False
    ev_full = eval_model(model, rng, val_batches=batches, batch_size=32, res=32, device=device)
    return {"adaptive": ev_ad, "force_2look": ev_full}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no-init-f2", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("ML_CACHE_ROOT", r"D:\\ml_cache")

    init = None if args.no_init_f2 else (str(F2_CKPT) if F2_CKPT.exists() else None)
    print(
        f"=== S2a  4 layers × 2 looks  Ĝ~KL(p2||p1)  init_f2={init is not None} ===",
        flush=True,
    )
    res = train_single_run(
        arm="v1_bayes",
        run_idx=0,
        seed=42,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        n_layers=4,
        s_update="rms_dir",
        use_stiefel=True,
        deslice_topk=2,
        gate_on="u",
        deslice_write="absolute",
        gate_h_local=False,
        vfe_coef=0.1,
        share_layers=False,
        saccade=True,
        saccade_inner=2,
        s2a=True,
        init_ckpt=init,
        ckpt_tag="s2a_l4i2",
    )

    # Re-eval best ckpt: adaptive vs force 2-look
    device = torch.device(args.device)
    best = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms_dir", use_stiefel=True, deslice_topk=2, n_heads=4,
        gate_on="u", deslice_write="absolute", vfe_coef=0.1,
        saccade=True, saccade_inner=2, s2a=True,
    ).to(device)
    ckpt_path = Path(res["checkpoint"])
    if ckpt_path.exists():
        best.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    modes = _eval_modes(best, device)
    print(
        f"S2a train-best  acc={res['final_acc']*100:.2f} "
        f"ocr={res['final_tasks'].get('ocr',0)*100:.1f} "
        f"kinks={res['final_tasks'].get('kinks',0)*100:.1f}",
        flush=True,
    )
    print(
        f"  adaptive  acc={modes['adaptive']['acc']*100:.2f} "
        f"looks={modes['adaptive'].get('mean_looks',0):.2f} "
        f"task_looks={modes['adaptive'].get('looks_task')}",
        flush=True,
    )
    print(
        f"  force-2   acc={modes['force_2look']['acc']*100:.2f} "
        f"looks={modes['force_2look'].get('mean_looks',0):.2f}",
        flush=True,
    )

    table = {
        "v1_bayes": {
            "arm": "v1_bayes",
            "note": "S2a: F2 init + per-layer 2nd look. Ĝ trained on KL(p2||p1). CE every look.",
            "init_f2": init is not None,
            "mean_acc": res["final_acc"] * 100,
            "mean_ocr": res["final_tasks"].get("ocr", 0) * 100,
            "mean_kinks": res["final_tasks"].get("kinks", 0) * 100,
            "mean_color": res["final_tasks"].get("color", 0) * 100,
            "adaptive": {
                "acc": modes["adaptive"]["acc"],
                "tasks": modes["adaptive"]["task_accs"],
                "mean_looks": modes["adaptive"].get("mean_looks"),
                "looks_task": modes["adaptive"].get("looks_task"),
            },
            "force_2look": {
                "acc": modes["force_2look"]["acc"],
                "tasks": modes["force_2look"]["task_accs"],
            },
            "runs": [res],
        }
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")

    f2 = _seed42(json.loads(PUB_F2.read_text(encoding="utf-8"))) if PUB_F2.exists() else None
    cmp = {
        "s2a_seed42": {
            "acc": res["final_acc"],
            "tasks": res["final_tasks"],
            "checkpoint": res.get("checkpoint"),
            "adaptive": table["v1_bayes"]["adaptive"],
            "force_2look": table["v1_bayes"]["force_2look"],
        }
    }
    if f2 is not None:
        cmp["f2_vfe_seed42"] = {"acc": f2["final_acc"], "tasks": f2["final_tasks"]}
        cmp["delta_pp_vs_f2"] = {
            "acc": (res["final_acc"] - f2["final_acc"]) * 100,
            "ocr": (res["final_tasks"]["ocr"] - f2["final_tasks"]["ocr"]) * 100,
            "kinks": (res["final_tasks"]["kinks"] - f2["final_tasks"]["kinks"]) * 100,
        }
        print(
            f"  DELTA vs F2  acc={cmp['delta_pp_vs_f2']['acc']:+.2f} "
            f"ocr={cmp['delta_pp_vs_f2']['ocr']:+.2f} "
            f"kinks={cmp['delta_pp_vs_f2']['kinks']:+.2f}",
            flush=True,
        )
    CMP.write_text(json.dumps(cmp, indent=2), encoding="utf-8")

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
    w = 0.28
    if f2 is not None:
        old = [
            f2["final_acc"] * 100, f2["final_tasks"]["ocr"] * 100,
            f2["final_tasks"]["kinks"] * 100, f2["final_tasks"]["color"] * 100,
        ]
        ax.bar(x - w / 2, old, w, label="F2 1 look", color="#38bdf8")
        ax.bar(x + w / 2, new, w, label="S2a + Ĝ", color="#a78bfa")
    else:
        ax.bar(x, new, w, color="#a78bfa")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color="#e2e8f0")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_title("S2a vs F2  ·  seed 42", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "s2a_vs_f2_bars.png", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
