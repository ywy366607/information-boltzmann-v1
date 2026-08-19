#!/usr/bin/env python3
"""Focused 600-Step Showdown: Baseline vs V0 JEPA vs V1 Gaussian Bayes on Deep 4-Layer MoT.

Only 3 core arms:
  1. baseline - Unmodulated standard dual-stream write-back (g=1.0)
  2. v0_jepa  - Deterministic JEPA L2 prediction error gate (U = ||S - S_hat||^2)
  3. v1_bayes - Full Gaussian Bayesian surprise gate with learned uncertainty (U = D_KL(q || p))

Features:
  - 600 steps long-horizon training
  - 4 paired random seeds (42, 143, 244, 345)
  - 960 independent held-out test samples per evaluation
  - Trajectory tracking every 20 steps (Loss(t), Acc(t), Task Breakdown over time)
  - Publication-ready showdown figures:
      - present/figs/showdown_loss_acc_trajectories.png
      - present/figs/showdown_task_evolution.png
      - present/figs/showdown_paired_gap.png

Usage:
  python scripts/run_jepa_vs_bayes_showdown.py --steps 600 --layers 4 --seeds 4 --device cuda
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


def train_single_arm_showdown(
    arm: str,
    steps: int = 600,
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
        surprise_mode=arm,
        surprise_beta=1.5,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-generate 120 training batches in memory for high GPU throughput
    n_cached = 120
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
            # Run a quick 4-batch validation probe for smooth trajectory curves
            quick_val = eval_model(model, rng_val, val_batches=4, batch_size=batch_size, res=res, device=dev)
            trajectory.append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": quick_val["loss"],
                "val_acc": quick_val["acc"],
                "val_ocr": quick_val["task_accs"].get("ocr", 0.0),
                "val_kinks": quick_val["task_accs"].get("kinks", 0.0),
                "val_color": quick_val["task_accs"].get("color", 0.0),
            })

    elapsed = time.time() - t0

    # Final comprehensive evaluation on 30 held-out batches = 960 test samples
    val_res = eval_model(model, rng_val, val_batches=30, batch_size=batch_size, res=res, device=dev)

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


def render_showdown_plots(summary: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Showdown Trajectory Plot (Validation Acc & Loss over 600 steps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    arm_styles = {
        "baseline": {"col": "#94a3b8", "name": "Baseline (Unmodulated g=1.0)", "lw": 2.0, "ls": "--"},
        "v0_jepa":  {"col": "#38bdf8", "name": "V0 JEPA (L2 Error Gate)", "lw": 2.8, "ls": "-"},
        "v1_bayes": {"col": "#2dd4bf", "name": "V1 Bayes (Gaussian KL Surprise)", "lw": 3.0, "ls": "-"},
    }

    for arm, s in summary.items():
        style = arm_styles.get(arm, {"col": "#ffffff", "name": arm, "lw": 2.0, "ls": "-"})
        runs = s["runs"]
        all_trajs = [r["trajectory"] for r in runs if "trajectory" in r and r["trajectory"]]
        if not all_trajs:
            continue
        steps = [pt["step"] for pt in all_trajs[0]]
        val_losses = np.mean([[pt["val_loss"] for pt in t] for t in all_trajs], axis=0)
        val_accs = np.mean([[pt["val_acc"] for pt in t] for t in all_trajs], axis=0)

        ax1.plot(steps, val_losses, label=style["name"], color=style["col"],
                 linewidth=style["lw"], linestyle=style["ls"])
        ax2.plot(steps, val_accs * 100, label=style["name"], color=style["col"],
                 linewidth=style["lw"], linestyle=style["ls"])

    ax1.set_title("Validation Loss Trajectory (600 Steps Convergence)", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", color="#cbd5e1")
    ax1.set_ylabel("Validation Cross Entropy Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    ax2.set_title("Validation Accuracy Trajectory (%) (600 Steps Convergence)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", color="#cbd5e1")
    ax2.set_ylabel("Validation Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    plt.tight_layout()
    traj_path = out_dir / "showdown_loss_acc_trajectories.png"
    plt.savefig(traj_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved showdown trajectory plot to {traj_path}")

    # 2. Task Evolution over Steps (OCR vs Kinks vs Color)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.patch.set_facecolor("#0b1120")

    task_keys = ["val_ocr", "val_kinks", "val_color"]
    task_titles = ["1px OCR Digit (Fine Stroke)", "Kinks Corners (Topology)", "Needle Square (Color Saccade)"]

    for ax, t_key, t_title in zip(axes, task_keys, task_titles):
        ax.set_facecolor("#0f172a")
        for arm, s in summary.items():
            style = arm_styles.get(arm, {"col": "#ffffff", "name": arm, "lw": 2.0, "ls": "-"})
            runs = s["runs"]
            all_trajs = [r["trajectory"] for r in runs if "trajectory" in r and r["trajectory"]]
            if not all_trajs:
                continue
            steps = [pt["step"] for pt in all_trajs[0]]
            t_accs = np.mean([[pt[t_key] * 100 for pt in t] for t in all_trajs], axis=0)

            ax.plot(steps, t_accs, label=style["name"], color=style["col"],
                    linewidth=style["lw"], linestyle=style["ls"])

        ax.set_title(t_title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps", color="#cbd5e1")
        ax.set_ylabel("Task Accuracy (%)", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        if ax == axes[0]:
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)

    plt.tight_layout()
    task_evo_path = out_dir / "showdown_task_evolution.png"
    plt.savefig(task_evo_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved task evolution plot to {task_evo_path}")

    # 3. Paired Delta Distribution Box / Bar Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    arms = [a for a in summary.keys() if a != "baseline"]
    delta_means = [summary[a]["paired_gain_mean"] for a in arms]
    delta_se = [summary[a]["paired_gain_se"] for a in arms]
    bar_cols = ["#38bdf8" if "jepa" in a else "#2dd4bf" for a in arms]

    bars = ax.bar(arms, delta_means, yerr=delta_se, capsize=6, color=bar_cols, alpha=0.88, width=0.45)
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.2)

    for bar, m, se in zip(bars, delta_means, delta_se):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{m:+.2f}% ± {se:.2f}%", ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold")

    ax.set_title("Steady-State Paired Gain relative to Baseline (600 Steps, N=4 Seeds)",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("Paired Gain Delta (%)", color="#cbd5e1")
    ax.set_xticklabels(["V0 JEPA (L2 Error)", "V1 Bayes (Gaussian KL)"], color="#e2e8f0", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")

    plt.tight_layout()
    gap_path = out_dir / "showdown_paired_gap.png"
    plt.savefig(gap_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved paired gap plot to {gap_path}")


def main():
    parser = argparse.ArgumentParser(description="600-Step Showdown: Baseline vs JEPA vs Bayes.")
    parser.add_argument("--steps", type=int, default=600, help="Training steps per arm")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--res", type=int, default=32, help="Resolution")
    parser.add_argument("--dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of MoT layers")
    parser.add_argument("--seeds", type=int, default=4, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/jepa_vs_bayes_table.json")
    args = parser.parse_args()

    arms = ["baseline", "v0_jepa", "v1_bayes"]
    print(f"[Showdown] Starting 600-step showdown for {arms} with {args.seeds} seeds on {args.device.upper()}...", flush=True)
    seed_list = [42 + s * 101 for s in range(args.seeds)]

    results = {}
    for arm in arms:
        results[arm] = []
        for seed in seed_list:
            res = train_single_arm_showdown(
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
            print(f"  Arm: {arm:<12} Seed: {seed} -> Val Acc: {res['val_acc']*100:.2f}%, Loss: {res['val_loss']:.4f}, OCR: {res['task_accs'].get('ocr', 0)*100:.1f}%, Kinks: {res['task_accs'].get('kinks', 0)*100:.1f}%, Needle: {res['task_accs'].get('color', 0)*100:.1f}%", flush=True)

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

    print("\n" + "=" * 90, flush=True)
    print(f"{'Arm':<15} {'Mean Acc (%)':<16} {'Paired Delta':<18} {'OCR Acc':<10} {'Kinks Acc':<10} {'Loss'}", flush=True)
    print("=" * 90, flush=True)
    for arm, s in summary.items():
        paired_str = f"{s['paired_gain_mean']:+.2f} ± {s['paired_gain_se']:.2f}%" if arm != "baseline" else "0.00% (Ref)"
        print(f"{arm:<15} {s['mean_acc']:.2f} ± {s['std_acc']:.2f}%     {paired_str:<18} {s['task_means']['ocr']:.1f}%      {s['task_means']['kinks']:.1f}%      {s['mean_loss']:.4f}", flush=True)
    print("=" * 90, flush=True)

    fig_dir = ROOT / "present" / "figs"
    render_showdown_plots(summary, fig_dir)


if __name__ == "__main__":
    main()
