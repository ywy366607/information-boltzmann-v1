#!/usr/bin/env python3
"""Long-Horizon 2000-Step Deep Training for V1 Gaussian Bayes (3 Independent Runs).

Features:
  - 2000 steps per run for asymptotic convergence & variance calibration
  - 4-layer Dual-Stream Native MoT (n_layers=4, d=128, n_slices=32, res=32)
  - Detailed trajectory tracking every 25 steps: Loss(t), Acc(t), OCR Acc(t), Kinks Acc(t), Needle Acc(t)
  - Automatic best model checkpoint saving (.pt) under checkpoints/
  - Final comprehensive evaluation on 35 held-out batches = 1120 test samples
  - Publication-ready convergence plots:
      - present/figs/bayes_2000step_trajectories.png
      - present/figs/bayes_2000step_tasks.png

Usage:
  python scripts/train_bayes_2000steps.py --steps 2000 --runs 3 --device cuda
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


def train_bayes_run(
    run_idx: int,
    seed: int,
    steps: int = 2000,
    batch_size: int = 32,
    res: int = 32,
    d_model: int = 128,
    n_layers: int = 4,
    lr: float = 1e-3,
    device: str = "cuda",
    eval_interval: int = 25,
) -> Dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 9999)

    model = DualStreamVQAModel(
        d_model=d_model,
        n_slices=32,
        n_layers=n_layers,
        res=res,
        surprise_mode="v1_bayes",
        surprise_beta=1.5,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing lr scheduler down to 1e-4 at step 2000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    # Pre-generate 150 training batches in GPU memory for ultra-fast throughput
    n_cached = 150
    cached_batches = []
    for _ in range(n_cached):
        b = make_vqa_batch(rng_train, batch=batch_size, res=res, mix=["ocr", "kinks", "color"])
        cached_batches.append({
            "imgs": b["image"].to(dev),
            "prompts": b["prompt"],
            "targets": torch.tensor([model.ans_to_idx[a] for a in b["answer"]], device=dev),
        })

    best_val_acc = -1.0
    best_state_dict = None
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
        scheduler.step()

        if step % eval_interval == 0 or step == steps:
            preds = out["logits"].argmax(dim=-1)
            train_acc = float((preds == targets).float().mean().item())
            # Quick 5-batch probe on validation distribution
            quick_val = eval_model(model, rng_val, val_batches=5, batch_size=batch_size, res=res, device=dev)

            if quick_val["acc"] > best_val_acc:
                best_val_acc = quick_val["acc"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

            if step % 200 == 0 or step == steps:
                print(f"    [Run {run_idx+1}] Step {step:4d}/{steps}: Val Acc={quick_val['acc']*100:.1f}%, Loss={quick_val['loss']:.4f}, OCR={quick_val['task_accs'].get('ocr',0)*100:.1f}%, Color={quick_val['task_accs'].get('color',0)*100:.1f}%", flush=True)

    elapsed = time.time() - t0

    # Save best checkpoint
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"v1_bayes_2000step_run{run_idx+1}_best.pt"
    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_path)
        print(f"    Saved best checkpoint ({best_val_acc*100:.2f}%) to {ckpt_path}", flush=True)

    # Load best weights for final comprehensive 35-batch evaluation (1120 samples)
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state_dict.items()})
    final_val = eval_model(model, rng_val, val_batches=35, batch_size=batch_size, res=res, device=dev)

    return {
        "run_idx": run_idx + 1,
        "seed": seed,
        "steps": steps,
        "best_probe_acc": best_val_acc,
        "final_acc": final_val["acc"],
        "final_loss": final_val["loss"],
        "final_tasks": final_val["task_accs"],
        "layer_surprise": final_val["layer_surprise"],
        "trajectory": trajectory,
        "elapsed_sec": round(elapsed, 2),
        "checkpoint": str(ckpt_path),
    }


def render_convergence_plots(runs: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    run_colors = ["#2dd4bf", "#38bdf8", "#c084fc"]

    for i, r in enumerate(runs):
        traj = r["trajectory"]
        steps = [pt["step"] for pt in traj]
        losses = [pt["val_loss"] for pt in traj]
        accs = [pt["val_acc"] * 100 for pt in traj]
        col = run_colors[i % len(run_colors)]

        ax1.plot(steps, losses, label=f"Bayes Run {r['run_idx']} (Final: {r['final_loss']:.3f})", color=col, linewidth=2.2)
        ax2.plot(steps, accs, label=f"Bayes Run {r['run_idx']} (Final: {r['final_acc']*100:.1f}%)", color=col, linewidth=2.2)

    # Plot average line
    all_losses = np.mean([[pt["val_loss"] for pt in r["trajectory"]] for r in runs], axis=0)
    all_accs = np.mean([[pt["val_acc"] * 100 for pt in r["trajectory"]] for r in runs], axis=0)
    steps = [pt["step"] for pt in runs[0]["trajectory"]]

    ax1.plot(steps, all_losses, label="Bayes Mean Trajectory", color="#ffffff", linewidth=3.0, linestyle="--")
    ax2.plot(steps, all_accs, label="Bayes Mean Trajectory", color="#ffffff", linewidth=3.0, linestyle="--")

    ax1.set_title("V1 Gaussian Bayes: 2000-Step Validation Loss Trajectory", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", color="#cbd5e1")
    ax1.set_ylabel("Validation Cross-Entropy Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    ax2.set_title("V1 Gaussian Bayes: 2000-Step Validation Accuracy Trajectory (%)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", color="#cbd5e1")
    ax2.set_ylabel("Validation Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    plt.tight_layout()
    p1 = out_dir / "bayes_2000step_trajectories.png"
    plt.savefig(p1, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved trajectory plot to {p1}", flush=True)

    # 2. Task Breakdown Evolution (OCR vs Kinks vs Needle over 2000 steps)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)
    fig.patch.set_facecolor("#0b1120")

    task_keys = ["val_ocr", "val_kinks", "val_color"]
    task_titles = ["1px OCR (Fine Digit Stroke)", "Kinks Corners (Topology Count)", "Needle Square (Color Saccade)"]

    for ax, t_key, t_title in zip(axes, task_keys, task_titles):
        ax.set_facecolor("#0f172a")
        for i, r in enumerate(runs):
            traj = r["trajectory"]
            steps = [pt["step"] for pt in traj]
            t_vals = [pt[t_key] * 100 for pt in traj]
            ax.plot(steps, t_vals, label=f"Run {r['run_idx']}", color=run_colors[i % len(run_colors)], linewidth=1.8, alpha=0.8)

        # Mean line
        t_mean = np.mean([[pt[t_key] * 100 for pt in r["trajectory"]] for r in runs], axis=0)
        ax.plot(steps, t_mean, label="Mean", color="#ffffff", linewidth=2.8, linestyle="--")

        ax.set_title(t_title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps", color="#cbd5e1")
        ax.set_ylabel("Task Accuracy (%)", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        if ax == axes[0]:
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    p2 = out_dir / "bayes_2000step_tasks.png"
    plt.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved task evolution plot to {p2}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="2000-Step Deep Bayes Training.")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per run")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/bayes_2000step_table.json")
    args = parser.parse_args()

    print(f"[2000-Step Bayes] Starting {args.runs} independent long-horizon runs ({args.steps} steps each) on {args.device.upper()}...", flush=True)

    seeds = [42, 1001, 7777][:args.runs]
    runs_data = []

    for i in range(args.runs):
        seed = seeds[i]
        print(f"\n--- Starting Bayes Run {i+1}/{args.runs} (Seed: {seed}) ---", flush=True)
        res = train_bayes_run(
            run_idx=i,
            seed=seed,
            steps=args.steps,
            device=args.device,
        )
        runs_data.append(res)
        print(f"=== Completed Run {i+1} in {res['elapsed_sec']}s: Final Val Acc={res['final_acc']*100:.2f}%, Loss={res['final_loss']:.4f}, OCR={res['final_tasks'].get('ocr',0)*100:.1f}%, Kinks={res['final_tasks'].get('kinks',0)*100:.1f}%, Color={res['final_tasks'].get('color',0)*100:.1f}% ===", flush=True)

    # Summary
    accs = [r["final_acc"] * 100 for r in runs_data]
    losses = [r["final_loss"] for r in runs_data]
    ocrs = [r["final_tasks"].get("ocr", 0.0) * 100 for r in runs_data]
    kinks = [r["final_tasks"].get("kinks", 0.0) * 100 for r in runs_data]
    colors = [r["final_tasks"].get("color", 0.0) * 100 for r in runs_data]

    summary = {
        "arm": "v1_bayes",
        "steps": args.steps,
        "n_runs": args.runs,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_loss": float(np.mean(losses)),
        "mean_ocr": float(np.mean(ocrs)),
        "mean_kinks": float(np.mean(kinks)),
        "mean_color": float(np.mean(colors)),
        "runs": runs_data,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"V1 Bayes 2000-Step Convergence Summary (N={args.runs} Independent Runs):", flush=True)
    print(f"  Mean Accuracy:   {summary['mean_acc']:.2f} ± {summary['std_acc']:.2f}%", flush=True)
    print(f"  Mean Loss:       {summary['mean_loss']:.4f}", flush=True)
    print(f"  1px OCR Acc:     {summary['mean_ocr']:.1f}%", flush=True)
    print(f"  Kinks Acc:       {summary['mean_kinks']:.1f}%", flush=True)
    print(f"  Needle Color:    {summary['mean_color']:.1f}%", flush=True)
    print("=" * 80, flush=True)

    fig_dir = ROOT / "present" / "figs"
    render_convergence_plots(runs_data, fig_dir)


if __name__ == "__main__":
    main()
