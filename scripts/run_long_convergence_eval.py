#!/usr/bin/env python3
"""Long-Horizon Convergence Study (500 Steps) on 4-Layer Dual-Stream MoT.

Investigates asymptotic convergence, steady-state accuracy, and probabilistic loss calibration
across key arms:
  1. baseline       - Unmodulated write-back (g=1.0)
  2. constant       - Fixed 0.5 residual damping
  3. v0_jepa        - Deterministic JEPA prediction error gate
  4. v0_shuffled    - Negative control: scrambled slice correspondence
  5. v0_reverse     - Negative control: reversed surprise gate
  6. v0_global_only - Global adaptive step size
  7. v1_bayes       - Full Gaussian Bayesian surprise gate with learned uncertainty

Outputs:
  - results/published/long_convergence_table.json
  - results/published/long_convergence_conclusion.md
  - present/figs/long_convergence_trajectories.png
  - present/figs/long_convergence_task_breakdown.png
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


def train_single_arm_long(
    arm: str,
    steps: int = 500,
    batch_size: int = 32,
    res: int = 32,
    d_model: int = 128,
    n_layers: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    eval_interval: int = 25,
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
        surprise_mode=arm,
        surprise_beta=1.5,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-generate 100 training batches in memory for maximum throughput
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
            trajectory.append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
            })

    elapsed = time.time() - t0

    # Final evaluation on 25 held-out batches = 800 test samples
    val_res = eval_model(model, rng_val, val_batches=25, batch_size=batch_size, res=res, device=dev)

    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "n_layers": n_layers,
        "val_acc": val_res["acc"],
        "val_loss": val_res["loss"],
        "task_accs": val_res["task_accs"],
        "layer_surprise": val_res["layer_surprise"],
        "trajectory": trajectory,
        "elapsed_sec": round(elapsed, 2),
    }


def render_long_plots(summary: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Long-Horizon Trajectory Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    arm_colors = {
        "baseline": "#94a3b8",
        "constant": "#475569",
        "v0_jepa": "#38bdf8",
        "v0_shuffled": "#f59e0b",
        "v0_reverse": "#ef4444",
        "v0_global_only": "#a855f7",
        "v1_bayes": "#2dd4bf",
    }

    for arm, s in summary.items():
        col = arm_colors.get(arm, "#ffffff")
        runs = s["runs"]
        all_trajs = [r["trajectory"] for r in runs if "trajectory" in r and r["trajectory"]]
        if not all_trajs:
            continue
        steps = [pt["step"] for pt in all_trajs[0]]
        losses = np.mean([[pt["train_loss"] for pt in t] for t in all_trajs], axis=0)
        accs = np.mean([[pt["train_acc"] for pt in t] for t in all_trajs], axis=0)

        lw = 2.8 if arm in ("v0_jepa", "v1_bayes", "baseline") else 1.8
        ls = "--" if "shuffled" in arm or "reverse" in arm else "-"
        ax1.plot(steps, losses, label=arm, color=col, linewidth=lw, linestyle=ls)
        ax2.plot(steps, accs * 100, label=arm, color=col, linewidth=lw, linestyle=ls)

    ax1.set_title("Long-Horizon Convergence: Loss(t) (500 Steps)", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", color="#cbd5e1")
    ax1.set_ylabel("Cross Entropy Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    ax2.set_title("Long-Horizon Convergence: Accuracy Acc(t) % (500 Steps)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", color="#cbd5e1")
    ax2.set_ylabel("Batch Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    traj_path = out_dir / "long_convergence_trajectories.png"
    plt.savefig(traj_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved long convergence trajectory plot to {traj_path}")

    # 2. Per-Task Accuracy Breakdown (500 Steps)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    arms = list(summary.keys())
    tasks = ["ocr", "kinks", "color"]
    task_labels = ["1px OCR (Digit)", "Kinks (Polyline Corners)", "Needle (Color Square)"]
    x = np.arange(len(arms))
    width = 0.26
    task_colors = ["#38bdf8", "#fb7185", "#fbbf24"]

    for i, (t_key, t_label, t_col) in enumerate(zip(tasks, task_labels, task_colors)):
        means = []
        for arm in arms:
            t_accs = [r["task_accs"].get(t_key, 0.0) * 100 for r in summary[arm]["runs"]]
            means.append(np.mean(t_accs))
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, label=t_label, color=t_col, alpha=0.85, edgecolor="#0f172a")

    ax.set_title("Per-Task Steady-State Accuracy Breakdown (500 Steps, 4 Layers)", color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=25, ha="right", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("Validation Accuracy (%)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)

    plt.tight_layout()
    task_path = out_dir / "long_convergence_task_breakdown.png"
    plt.savefig(task_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved long task breakdown plot to {task_path}")


def main():
    parser = argparse.ArgumentParser(description="Long-Horizon 500-Step Convergence Study.")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per arm")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--res", type=int, default=32, help="Resolution")
    parser.add_argument("--dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of MoT layers")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/long_convergence_table.json")
    args = parser.parse_args()

    arms = [
        "baseline",
        "constant",
        "v0_jepa",
        "v0_shuffled",
        "v0_reverse",
        "v0_global_only",
        "v1_bayes",
    ]
    print(f"[Long Convergence] Starting 500-step study across {len(arms)} arms with {args.seeds} seeds on {args.device.upper()}...")
    seed_list = [42 + s * 101 for s in range(args.seeds)]

    results = {}
    for arm in arms:
        results[arm] = []
        for seed in seed_list:
            res = train_single_arm_long(
                arm=arm,
                steps=args.steps,
                batch_size=args.batch,
                res=args.res,
                d_model=args.dim,
                n_layers=args.layers,
                seed=seed,
                device=args.device,
            )
            results[arm].append(res)
            print(f"  Arm: {arm:<16} Seed: {seed} -> Val Acc: {res['val_acc']*100:.2f}%, Loss: {res['val_loss']:.4f}, Tasks: { {k: f'{v*100:.1f}%' for k, v in res['task_accs'].items()} }", flush=True)

    summary = {}
    baseline_runs = {r["seed"]: r["val_acc"] for r in results["baseline"]}

    for arm, runs in results.items():
        accs = [r["val_acc"] * 100 for r in runs]
        losses = [r["val_loss"] for r in runs]
        paired_deltas = [(r["val_acc"] - baseline_runs[r["seed"]]) * 100 for r in runs]

        task_means = {}
        for t_k in ["ocr", "kinks", "color"]:
            t_vals = [r["task_accs"].get(t_k, 0.0) * 100 for r in runs]
            task_means[t_k] = float(np.mean(t_vals))

        summary[arm] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)),
            "paired_gain_mean": float(np.mean(paired_deltas)),
            "paired_gain_se": float(np.std(paired_deltas) / np.sqrt(len(paired_deltas))),
            "task_means": task_means,
            "runs": runs,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print(f"{'Arm':<18} {'Mean Acc (%)':<16} {'Paired Delta':<16} {'OCR Acc':<10} {'Kinks Acc':<10} {'Loss'}", flush=True)
    print("=" * 85, flush=True)
    for arm, s in summary.items():
        paired_str = f"{s['paired_gain_mean']:+.2f} ± {s['paired_gain_se']:.2f}%" if arm != "baseline" else "0.00% (Ref)"
        print(f"{arm:<18} {s['mean_acc']:.2f} ± {s['std_acc']:.2f}%     {paired_str:<16} {s['task_means']['ocr']:.1f}%      {s['task_means']['kinks']:.1f}%      {s['mean_loss']:.4f}", flush=True)
    print("=" * 85, flush=True)

    fig_dir = ROOT / "present" / "figs"
    render_long_plots(summary, fig_dir)


if __name__ == "__main__":
    main()
