#!/usr/bin/env python3
"""Ablation Study: What happens if surprise gate gradients are NOT detached?

Compares:
  1. v0_jepa_detached   (detach_gate = True)
  2. v0_jepa_leaking    (detach_gate = False)
  3. v1_bayes_detached  (detach_gate = True)
  4. v1_bayes_leaking   (detach_gate = False)

Outputs:
  - results/published/detach_ablation_table.json
  - present/figs/detach_ablation_trajectories.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.native_mot import NativeMoTStack
from fine_grain.vlm_data import COLORS, KINK_KS, OCR_DIGITS, make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel, eval_model


def run_single_detach_arm(
    surprise_mode: str,
    surprise_detach: bool,
    steps: int = 300,
    batch_size: int = 32,
    res: int = 32,
    d_model: int = 128,
    n_layers: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    eval_interval: int = 20,
) -> Dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 999)

    model = DualStreamVQAModel(
        d_model=d_model,
        n_slices=32,
        n_layers=n_layers,
        res=res,
        surprise_mode=surprise_mode,
        surprise_beta=1.5,
    ).to(dev)

    # Apply detach flag to all layers
    for layer in model.mot_stack.layers:
        layer.surprise_gate.detach_gate = surprise_detach

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_cached = 100
    cached_batches = []
    for _ in range(n_cached):
        b = make_vqa_batch(rng_train, batch=batch_size, res=res, mix=["ocr", "kinks", "color"])
        cached_batches.append({
            "imgs": b["image"].to(dev),
            "prompts": b["prompt"],
            "targets": torch.tensor([model.ans_to_idx[a] for a in b["answer"]], device=dev),
        })

    trajectory = []
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        b = cached_batches[(step - 1) % n_cached]
        imgs = b["imgs"]
        prompts = b["prompts"]
        targets = b["targets"]

        optimizer.zero_grad()
        out = model(imgs, prompts)
        loss = model.task_loss(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == steps:
            preds = out["logits"].argmax(dim=-1)
            train_acc = float((preds == targets).float().mean().item())
            quick_val = eval_model(model, rng_val, val_batches=4, batch_size=batch_size, res=res, device=dev)
            trajectory.append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": quick_val["loss"],
                "val_acc": quick_val["acc"],
                "val_ocr": quick_val["task_accs"].get("ocr", 0.0),
            })

    elapsed = time.time() - t0
    val_res = eval_model(model, rng_val, val_batches=25, batch_size=batch_size, res=res, device=dev)

    return {
        "arm": f"{surprise_mode}_{'detached' if surprise_detach else 'leaking'}",
        "mode": surprise_mode,
        "detach": surprise_detach,
        "seed": seed,
        "steps": steps,
        "val_acc": val_res["acc"],
        "val_loss": val_res["loss"],
        "task_accs": val_res["task_accs"],
        "layer_surprise": val_res["layer_surprise"],
        "trajectory": trajectory,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation: Detached vs Leaking Gate Gradients.")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per arm")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/detach_ablation_table.json")
    args = parser.parse_args()

    configurations = [
        ("v0_jepa", True, "v0_jepa_detached (Detached / Normal)"),
        ("v0_jepa", False, "v0_jepa_leaking (No Detach / Gradient Leaks)"),
        ("v1_bayes", True, "v1_bayes_detached (Detached / Normal)"),
        ("v1_bayes", False, "v1_bayes_leaking (No Detach / Gradient Leaks)"),
    ]

    seed_list = [42 + s * 101 for s in range(args.seeds)]
    print(f"[Detach Ablation] Running {len(configurations)} arms with {args.seeds} seeds on {args.device.upper()}...", flush=True)

    results = {}
    for mode, detach_flag, label in configurations:
        arm_key = f"{mode}_{'detached' if detach_flag else 'leaking'}"
        results[arm_key] = []
        for seed in seed_list:
            res = run_single_detach_arm(
                surprise_mode=mode,
                surprise_detach=detach_flag,
                steps=args.steps,
                seed=seed,
                device=args.device,
            )
            results[arm_key].append(res)
            print(f"  {label:<48} Seed {seed}: Val Acc={res['val_acc']*100:.2f}%, Loss={res['val_loss']:.4f}, OCR={res['task_accs'].get('ocr', 0)*100:.1f}%, Surprise={res['layer_surprise']}", flush=True)

    # Summary
    summary = {}
    for arm_key, runs in results.items():
        accs = [r["val_acc"] * 100 for r in runs]
        losses = [r["val_loss"] for r in runs]
        ocr_accs = [r["task_accs"].get("ocr", 0.0) * 100 for r in runs]
        summary[arm_key] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)),
            "ocr_mean": float(np.mean(ocr_accs)),
            "runs": runs,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"{'Arm':<28} {'Mean Acc (%)':<18} {'OCR Acc (%)':<16} {'Mean Loss':<12}", flush=True)
    print("=" * 80, flush=True)
    for arm_key, s in summary.items():
        print(f"{arm_key:<28} {s['mean_acc']:.2f} ± {s['std_acc']:.2f}%     {s['ocr_mean']:.1f}%             {s['mean_loss']:.4f}", flush=True)
    print("=" * 80, flush=True)

    # Render trajectory plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    colors = {
        "v0_jepa_detached": "#38bdf8",
        "v0_jepa_leaking": "#f87171",
        "v1_bayes_detached": "#2dd4bf",
        "v1_bayes_leaking": "#fbbf24",
    }

    for arm_key, s in summary.items():
        col = colors.get(arm_key, "#ffffff")
        runs = s["runs"]
        all_trajs = [r["trajectory"] for r in runs if "trajectory" in r and r["trajectory"]]
        if not all_trajs:
            continue
        steps = [pt["step"] for pt in all_trajs[0]]
        val_losses = np.mean([[pt["val_loss"] for pt in t] for t in all_trajs], axis=0)
        val_accs = np.mean([[pt["val_acc"] for pt in t] for t in all_trajs], axis=0)

        ls = "-" if "detached" in arm_key else "--"
        ax1.plot(steps, val_losses, label=arm_key, color=col, linewidth=2.5, linestyle=ls)
        ax2.plot(steps, val_accs * 100, label=arm_key, color=col, linewidth=2.5, linestyle=ls)

    ax1.set_title("Validation Loss (Detached vs Leaking Gradients)", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Steps", color="#cbd5e1")
    ax1.set_ylabel("Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    ax2.set_title("Validation Accuracy % (Detached vs Leaking Gradients)", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Steps", color="#cbd5e1")
    ax2.set_ylabel("Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    plt.tight_layout()
    fig_path = ROOT / "present" / "figs" / "detach_ablation_trajectories.png"
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved trajectory plot to {fig_path}", flush=True)


if __name__ == "__main__":
    main()
