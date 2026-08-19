#!/usr/bin/env python3
"""Long-Horizon 2000-Step Deep Training & Showdown for Baseline, V0 JEPA, and V1 Bayes.

Features:
  - 2000 steps per run for asymptotic convergence & variance calibration
  - 4-layer Dual-Stream Native MoT (n_layers=4, d=128, n_slices=32, res=32)
  - Evaluates 3 arms: baseline, v0_jepa, v1_bayes (with 3 independent seeds: 42, 1001, 7777)
  - Detailed trajectory tracking every 25 steps: Loss(t), Acc(t), OCR Acc(t), Kinks Acc(t), Needle Acc(t)
  - Automatic best model checkpoint saving (.pt) under checkpoints/
  - Final comprehensive evaluation on 35 held-out batches = 1120 test samples
  - Publication-ready convergence plots:
      - present/figs/triad_2000step_trajectories.png
      - present/figs/triad_2000step_tasks.png
      - present/figs/triad_2000step_summary_bars.png
  - Output JSON: results/published/triad_2000step_table.json

Usage:
  python scripts/train_triad_2000steps.py --arms baseline,v0_jepa --steps 2000 --device cuda
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


def train_single_run(
    arm: str,
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
    prior_loss_coef: float = 0.1,
    detach_pred_target: bool = True,
    sigreg_coef: float = 0.0,
    use_stiefel: bool = True,
    deslice_topk: int = 2,
    s_update: str = "raw",
    interact_prenorm: bool = False,
    trust_rho: float = 0.1,
    evidence_decay: float = 0.0,
    ckpt_tag: str = "",
    gate_on: str = "u",
    deslice_write: str = "absolute",
    gate_h_local: bool = False,
    vfe_coef: float = 0.0,
    share_layers: bool = False,
    n_loops: int | None = None,
    deep_supervise: bool | None = None,
    lti_inject: bool = True,
    saccade: bool = False,
    saccade_gain: float = 1.0,
    saccade_inner: int = 2,
    s2a: bool = False,
    init_ckpt: str | None = None,
    s2a_ig_eps: float = 0.02,
    s2a_ent_coef: float = 0.05,
    s_kalman_update: bool = False,
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
        surprise_mode=arm,
        surprise_beta=1.5,
        prior_loss_coef=prior_loss_coef,
        detach_pred_target=detach_pred_target,
        sigreg_coef=sigreg_coef,
        use_stiefel=use_stiefel,
        deslice_topk=deslice_topk,
        s_update=s_update,
        interact_prenorm=interact_prenorm,
        trust_rho=trust_rho,
        evidence_decay=evidence_decay,
        gate_on=gate_on,
        deslice_write=deslice_write,
        gate_h_local=gate_h_local,
        vfe_coef=vfe_coef,
        share_layers=share_layers,
        n_loops=n_loops,
        deep_supervise=True if s2a else deep_supervise,
        lti_inject=lti_inject,
        saccade=saccade,
        saccade_gain=saccade_gain,
        saccade_inner=saccade_inner,
        s2a=s2a,
        s2a_ig_eps=s2a_ig_eps,
        s2a_ent_coef=s2a_ent_coef,
        s_kalman_update=s_kalman_update,
    ).to(dev)
    if init_ckpt:
        raw = torch.load(init_ckpt, map_location="cpu")
        miss = model.load_state_dict(raw, strict=False)
        print(
            f"    init {init_ckpt} miss={len(miss.missing_keys)} extra={len(miss.unexpected_keys)}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
                "rms_s": quick_val.get("layer_rms_s", []),
                "rms_delta": quick_val.get("layer_rms_delta", []),
                "rms_ratio": quick_val.get("layer_rms_ratio", []),
                "layer_u": quick_val.get("layer_surprise", []),
                "layer_gap": quick_val.get("layer_gap", []),
                "layer_F": quick_val.get("layer_F", []),
                "layer_ce": quick_val.get("layer_ce", []),
                "rho_x": quick_val.get("rho_x", 0.0),
                "rho_h": quick_val.get("rho_h", 0.0),
            })

            if step % 200 == 0 or step == steps:
                print(f"    [{arm.upper()} Run {run_idx+1}] Step {step:4d}/{steps}: Val Acc={quick_val['acc']*100:.1f}%, Loss={quick_val['loss']:.4f}, OCR={quick_val['task_accs'].get('ocr',0)*100:.1f}%, Color={quick_val['task_accs'].get('color',0)*100:.1f}%", flush=True)

    elapsed = time.time() - t0

    # Save best checkpoint
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    ckpt_path = ckpt_dir / f"{arm}_2000step{tag}_run{run_idx+1}_best.pt"
    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_path)
        print(f"    Saved best checkpoint ({best_val_acc*100:.2f}%) to {ckpt_path}", flush=True)

    # Load best weights for final comprehensive 35-batch evaluation (1120 samples)
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state_dict.items()})
    final_val = eval_model(model, rng_val, val_batches=35, batch_size=batch_size, res=res, device=dev)

    return {
        "arm": arm,
        "run_idx": run_idx + 1,
        "seed": seed,
        "steps": steps,
        "best_probe_acc": best_val_acc,
        "final_acc": final_val["acc"],
        "final_loss": final_val["loss"],
        "final_tasks": final_val["task_accs"],
        "layer_surprise": final_val.get("layer_surprise", []),
        "layer_gap": final_val.get("layer_gap", []),
        "layer_F": final_val.get("layer_F", []),
        "gate_on": gate_on,
        "deslice_write": deslice_write,
        "trajectory": trajectory,
        "elapsed_sec": round(elapsed, 2),
        "checkpoint": str(ckpt_path),
        "prior_loss_coef": prior_loss_coef,
        "detach_pred_target": detach_pred_target,
        "sigreg_coef": sigreg_coef,
        "use_stiefel": use_stiefel,
        "deslice_topk": deslice_topk,
        "s_update": s_update,
        "interact_prenorm": interact_prenorm,
        "layer_rms_s": final_val.get("layer_rms_s", []),
        "layer_rms_delta": final_val.get("layer_rms_delta", []),
        "layer_rms_ratio": final_val.get("layer_rms_ratio", []),
        "layer_ce": final_val.get("layer_ce", []),
        "rho_x": final_val.get("rho_x", 0.0),
        "rho_h": final_val.get("rho_h", 0.0),
        "share_layers": share_layers,
        "n_loops": n_loops if n_loops is not None else n_layers,
        "lti_inject": lti_inject,
        "saccade": saccade,
        "ckpt_tag": ckpt_tag,
        "s_kalman_update": s_kalman_update,
        "layer_K": final_val.get("layer_K", []),
    }


def render_triad_plots(all_results: Dict[str, Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_styles = {
        "baseline": {"name": "Baseline (Unmodulated)", "color": "#94a3b8", "light": "#cbd5e1"},
        "v0_jepa": {"name": "V0 JEPA (L2 Error)", "color": "#38bdf8", "light": "#7dd3fc"},
        "v1_bayes": {"name": "V1 Bayes (Gaussian KL)", "color": "#2dd4bf", "light": "#5eead4"},
    }

    # 1. Trajectory Loss & Accuracy Comparison (Mean + Shade)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    for arm, arm_data in all_results.items():
        style = arm_styles.get(arm, {"name": arm, "color": "#ffffff", "light": "#ffffff"})
        runs = arm_data["runs"]
        steps = [pt["step"] for pt in runs[0]["trajectory"]]

        all_losses = np.array([[pt["val_loss"] for pt in r["trajectory"]] for r in runs])
        all_accs = np.array([[pt["val_acc"] * 100 for pt in r["trajectory"]] for r in runs])

        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        mean_acc = np.mean(all_accs, axis=0)
        std_acc = np.std(all_accs, axis=0)

        # Plot individual thin lines
        for r in runs:
            r_losses = [pt["val_loss"] for pt in r["trajectory"]]
            r_accs = [pt["val_acc"] * 100 for pt in r["trajectory"]]
            ax1.plot(steps, r_losses, color=style["color"], alpha=0.25, linewidth=1.0)
            ax2.plot(steps, r_accs, color=style["color"], alpha=0.25, linewidth=1.0)

        # Plot Mean + Std Shading
        ax1.plot(steps, mean_loss, label=f"{style['name']} (Mean Loss: {arm_data['mean_loss']:.3f})", color=style["color"], linewidth=2.6)
        ax1.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, color=style["color"], alpha=0.15)

        ax2.plot(steps, mean_acc, label=f"{style['name']} ({arm_data['mean_acc']:.1f} ± {arm_data['std_acc']:.1f}%)", color=style["color"], linewidth=2.6)
        ax2.fill_between(steps, mean_acc - std_acc, mean_acc + std_acc, color=style["color"], alpha=0.15)

    ax1.set_title("2000-Step Validation Loss Trajectories (Baseline vs JEPA vs Bayes)", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Steps", color="#cbd5e1")
    ax1.set_ylabel("Validation Cross-Entropy Loss", color="#cbd5e1")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    ax2.set_title("2000-Step Validation Accuracy Trajectories (Baseline vs JEPA vs Bayes)", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Steps", color="#cbd5e1")
    ax2.set_ylabel("Validation Accuracy (%)", color="#cbd5e1")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9.5)

    plt.tight_layout()
    p1 = out_dir / "triad_2000step_trajectories.png"
    plt.savefig(p1, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved triad trajectories to {p1}", flush=True)

    # 2. Task Breakdown Evolution (OCR vs Kinks vs Color)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)
    fig.patch.set_facecolor("#0b1120")

    task_keys = ["val_ocr", "val_kinks", "val_color"]
    task_titles = ["1px OCR (Fine Digit Stroke)", "Kinks Corners (Topology Count)", "Needle Square (Color Saccade)"]

    for ax, t_key, t_title in zip(axes, task_keys, task_titles):
        ax.set_facecolor("#0f172a")
        for arm, arm_data in all_results.items():
            style = arm_styles.get(arm, {"name": arm, "color": "#ffffff"})
            runs = arm_data["runs"]
            steps = [pt["step"] for pt in runs[0]["trajectory"]]
            t_mean = np.mean([[pt[t_key] * 100 for pt in r["trajectory"]] for r in runs], axis=0)
            ax.plot(steps, t_mean, label=f"{style['name']}", color=style["color"], linewidth=2.4)

        ax.set_title(t_title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Steps", color="#cbd5e1")
        ax.set_ylabel("Task Accuracy (%)", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        if ax == axes[0]:
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    p2 = out_dir / "triad_2000step_tasks.png"
    plt.savefig(p2, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved triad task evolution to {p2}", flush=True)

    # 3. Final Summary Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    categories = ["Overall Acc", "1px OCR", "Kinks Count", "Needle Color"]
    x = np.arange(len(categories))
    width = 0.26

    offsets = [-width, 0, width]
    for i, (arm, arm_data) in enumerate(all_results.items()):
        style = arm_styles.get(arm, {"name": arm, "color": "#ffffff"})
        vals = [
            arm_data["mean_acc"],
            arm_data["mean_ocr"],
            arm_data["mean_kinks"],
            arm_data["mean_color"],
        ]
        errs = [
            arm_data["std_acc"],
            0.0,
            0.0,
            0.0,
        ]
        rects = ax.bar(x + offsets[i], vals, width, yerr=errs, capsize=4, label=style["name"], color=style["color"], alpha=0.88)
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#f8fafc", fontweight="bold")

    ax.set_title("Triad Final Asymptotic Convergence (2000 Steps, N=3 Independent Seeds)", color="white", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Evaluation Accuracy (%)", color="#cbd5e1")
    ax.set_ylim(0, 110)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)

    plt.tight_layout()
    p3 = out_dir / "triad_2000step_summary_bars.png"
    plt.savefig(p3, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved triad summary bar chart to {p3}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="2000-Step Triad Deep Training (Baseline vs JEPA vs Bayes).")
    parser.add_argument("--arms", type=str, default="baseline,v0_jepa", help="Arms to train (comma-separated)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per run")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs per arm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/published/triad_2000step_table.json")
    parser.add_argument("--prior-loss-coef", type=float, default=0.1,
                        help="λ for ||stopgrad(S)-prior||². 0 disables the aux term.")
    parser.add_argument("--tag", type=str, default="",
                        help="Checkpoint / table suffix so published ckpts are not overwritten.")
    parser.add_argument(
        "--detach-pred-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop-grad S in pred_loss (default on). --no-detach-pred-target lets pred pull S.",
    )
    parser.add_argument(
        "--sigreg-coef", type=float, default=0.0,
        help="λ_sig for SIGReg on live S. 0 disables.",
    )
    parser.add_argument(
        "--stiefel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Newton–Schulz Stiefel on visual slice directions. --no-stiefel turns it off.",
    )
    parser.add_argument(
        "--deslice-topk", type=int, default=2,
        help="Sparse deslice write. 0 = full soft scatter.",
    )
    parser.add_argument(
        "--s-update", type=str, default="raw",
        choices=["raw", "rms_dir", "trust", "momentum", "adam", "muon"],
        help="Forward state optimizer on Δ=S2-S: sgd/rms/trust/momentum/adam/muon.",
    )
    parser.add_argument(
        "--interact-prenorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="RMSNorm(S) and RMSNorm(H) before MoT; update still applied to raw S.",
    )
    parser.add_argument("--trust-rho", type=float, default=0.1,
                        help="Max RMS(Δ)/RMS(S) for --s-update trust.")
    parser.add_argument("--evidence-decay", type=float, default=0.0,
                        help="Spring S toward this layer's read of the raw field (not WD).")
    parser.add_argument("--gate-on", type=str, default="u", choices=["u", "gap", "f"],
                        help="F0: u (default). F1: gap = KL(q||q*).")
    parser.add_argument("--deslice-write", type=str, default="absolute",
                        choices=["absolute", "increment", "workspace"])
    parser.add_argument(
        "--gate-h-local",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vfe-coef", type=float, default=0.0,
        help="F2: λ for E[gap]=KL(q||q*) on amortized post_head. 0 disables.",
    )
    parser.add_argument(
        "--s-kalman-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write S ← Kalman μ* = μp+K(S−μp). Update only, no predict A.",
    )
    parser.add_argument(
        "--share-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="R0: one shared NativeMoTLayer looped n_loops times with LTI injection.",
    )
    parser.add_argument(
        "--n-loops", type=int, default=None,
        help="Recurrent depth when --share-layers. Defaults to n_layers.",
    )
    parser.add_argument(
        "--lti-inject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When sharing Φ, assemble ĀX+B̄e+Δ. --no-lti-inject is plain X←Φ(X).",
    )
    parser.add_argument(
        "--saccade",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Boost unexplained pixels in the next SliceRead pool. Q unchanged.",
    )
    parser.add_argument("--saccade-gain", type=float, default=1.0)
    parser.add_argument(
        "--saccade-inner", type=int, default=2,
        help="Looks per unique layer (time). Depth stays n_layers.",
    )
    args = parser.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [42, 1001, 7777][:args.runs]
    out_table_path = Path(args.out)

    all_results = {}
    # Load existing Bayes 2000step table if present
    bayes_table_path = ROOT / "results" / "published" / "bayes_2000step_table.json"
    if bayes_table_path.exists() and "v1_bayes" not in arm_list:
        with open(bayes_table_path, "r", encoding="utf-8") as f:
            all_results["v1_bayes"] = json.load(f)
        print(f"[Triad] Loaded pre-computed V1 Bayes 2000-step data from {bayes_table_path}", flush=True)

    # If out table exists, load previous results
    if out_table_path.exists():
        try:
            with open(out_table_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if k not in all_results:
                        all_results[k] = v
        except Exception:
            pass

    for arm in arm_list:
        print(f"\n================================================================================", flush=True)
        print(f"STARTING 2000-STEP TRAINING FOR ARM: {arm.upper()} ({args.runs} RUNS)", flush=True)
        print(f"================================================================================", flush=True)

        runs_data = []
        for i, seed in enumerate(seeds):
            print(f"\n--- Starting {arm.upper()} Run {i+1}/{args.runs} (Seed: {seed}) ---", flush=True)
            res = train_single_run(
                arm=arm,
                run_idx=i,
                seed=seed,
                steps=args.steps,
                device=args.device,
                prior_loss_coef=args.prior_loss_coef,
                detach_pred_target=args.detach_pred_target,
                sigreg_coef=args.sigreg_coef,
                use_stiefel=args.stiefel,
                deslice_topk=args.deslice_topk,
                s_update=args.s_update,
                interact_prenorm=args.interact_prenorm,
                trust_rho=args.trust_rho,
                evidence_decay=args.evidence_decay,
                ckpt_tag=args.tag,
                gate_on=args.gate_on,
                deslice_write=args.deslice_write,
                gate_h_local=args.gate_h_local,
                vfe_coef=args.vfe_coef,
                s_kalman_update=args.s_kalman_update,
                share_layers=args.share_layers,
                n_loops=args.n_loops,
                lti_inject=args.lti_inject,
                saccade=args.saccade,
                saccade_gain=args.saccade_gain,
                saccade_inner=args.saccade_inner,
            )
            runs_data.append(res)
            print(f"=== Completed {arm.upper()} Run {i+1} in {res['elapsed_sec']}s: Final Val Acc={res['final_acc']*100:.2f}%, Loss={res['final_loss']:.4f}, OCR={res['final_tasks'].get('ocr',0)*100:.1f}%, Kinks={res['final_tasks'].get('kinks',0)*100:.1f}%, Color={res['final_tasks'].get('color',0)*100:.1f}% ===", flush=True)

        accs = [r["final_acc"] * 100 for r in runs_data]
        losses = [r["final_loss"] for r in runs_data]
        ocrs = [r["final_tasks"].get("ocr", 0.0) * 100 for r in runs_data]
        kinks = [r["final_tasks"].get("kinks", 0.0) * 100 for r in runs_data]
        colors = [r["final_tasks"].get("color", 0.0) * 100 for r in runs_data]

        all_results[arm] = {
            "arm": arm,
            "steps": args.steps,
            "n_runs": args.runs,
            "prior_loss_coef": args.prior_loss_coef,
            "detach_pred_target": args.detach_pred_target,
            "sigreg_coef": args.sigreg_coef,
            "use_stiefel": args.stiefel,
            "deslice_topk": args.deslice_topk,
            "s_update": args.s_update,
            "interact_prenorm": args.interact_prenorm,
            "ckpt_tag": args.tag,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)),
            "mean_ocr": float(np.mean(ocrs)),
            "mean_kinks": float(np.mean(kinks)),
            "mean_color": float(np.mean(colors)),
            "runs": runs_data,
        }

    # Ensure ordering: baseline, v0_jepa, v1_bayes
    ordered_results = {}
    for key in ["baseline", "v0_jepa", "v1_bayes"]:
        if key in all_results:
            ordered_results[key] = all_results[key]
    for key in all_results:
        if key not in ordered_results:
            ordered_results[key] = all_results[key]

    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_table_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("TRIAD 2000-STEP FINAL ASYMPTOTIC CONVERGENCE SUMMARY:", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Arm':<24} | {'Mean Acc (%)':<16} | {'Mean Loss':<10} | {'1px OCR (%)':<12} | {'Kinks (%)':<10} | {'Needle (%)':<10}")
    print("-" * 90)
    for arm, res in ordered_results.items():
        print(f"{arm:<24} | {res['mean_acc']:>5.2f} ± {res['std_acc']:<6.2f} | {res['mean_loss']:>8.4f} | {res['mean_ocr']:>10.1f} | {res['mean_kinks']:>8.1f} | {res['mean_color']:>8.1f}")
    print("=" * 80, flush=True)

    fig_dir = ROOT / "present" / "figs"
    render_triad_plots(ordered_results, fig_dir)


if __name__ == "__main__":
    main()
