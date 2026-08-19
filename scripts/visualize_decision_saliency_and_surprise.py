#!/usr/bin/env python3
"""Generate True Decision Saliency Attribution & Layer-by-Layer Spatial Surprise Suite.

Compares:
  - Baseline vs V0 JEPA vs V1 Bayes
Visualizes:
  1. Layer-by-Layer Decision Saliency Attribution Maps (d(Logit_True) / d(X_l(x,y))) for l in [0, 1, 2, 3]
  2. Input Canvas Saliency Attribution Map (d(Logit_True) / d(X_0)) on 1px Stroke
  3. Layer-by-Layer Spatial Surprise Maps U_l(x,y) for l in [0, 1, 2, 3]
  4. Quantitative Saliency Target SNR Bar Chart across Depth

Outputs:
  - present/figs/decision_saliency_layer_evolution.png
  - present/figs/saliency_snr_comparison.png
  - present/figs/pure_surprise_layer_comparison.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.bayesian_surprise import spatial_surprise_map
from fine_grain.ocr_1px import make_ocr_1px
from fine_grain.tasks import make_kinks, make_needle
from scripts.run_v0_surprise_eval import DualStreamVQAModel
from scripts.visualize_surprise_heatmaps import run_heatmap_extraction


def compute_layer_saliency(model: DualStreamVQAModel, img: torch.Tensor, prompt: str, target_ans: str, res: int):
    model.eval()
    model.zero_grad()
    device = img.device
    tok_ids, mask = model.tokenize([prompt], device)
    text_emb = model.embed(tok_ids)

    X0 = model.mot_stack.encode_X(img)
    X0.retain_grad()
    H = model.mot_stack.text_in(text_emb)

    X1, H, _ = model.mot_stack.layers[0](X0, H, text_mask=mask, prompt_mask=mask, layer_idx=0)
    X1.retain_grad()
    X2, H, _ = model.mot_stack.layers[1](X1, H, text_mask=mask, prompt_mask=mask, layer_idx=1)
    X2.retain_grad()
    X3, H, _ = model.mot_stack.layers[2](X2, H, text_mask=mask, prompt_mask=mask, layer_idx=2)
    X3.retain_grad()
    X4, H, _ = model.mot_stack.layers[3](X3, H, text_mask=mask, prompt_mask=mask, layer_idx=3)

    mask_f = mask.unsqueeze(-1).float()
    h_pooled = (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    logits = model.head(h_pooled)
    ans_idx = model.ans_to_idx[target_ans]
    logits[0, ans_idx].backward()

    return [
        X0.grad[0].norm(dim=-1).view(res, res).cpu().numpy(),
        X1.grad[0].norm(dim=-1).view(res, res).cpu().numpy(),
        X2.grad[0].norm(dim=-1).view(res, res).cpu().numpy(),
        X3.grad[0].norm(dim=-1).view(res, res).cpu().numpy(),
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Saliency & Surprise Suite] Using device: {device}", flush=True)

    res = 48
    d_model = 128
    n_layers = 4
    out_dir = ROOT / "present" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for arm in ["baseline", "v0_jepa", "v1_bayes"]:
        m = DualStreamVQAModel(d_model=d_model, n_slices=32, n_layers=n_layers, res=res, surprise_mode=arm).to(device)
        ckpt_path = f"checkpoints/{arm}_best.pt"
        if os.path.exists(ckpt_path):
            m.load_state_dict(torch.load(ckpt_path))
        models[arm] = m

    # 1. Benchmark on Single 1px OCR Digit '7' Canvas
    rng = np.random.default_rng(42)
    img_digit, _, stroke_mask = make_ocr_1px(rng, np.array([7]), res=res)
    img_digit = img_digit.to(device)
    stroke_mask_np = stroke_mask[0].view(res, res).cpu().numpy().astype(bool)
    rgb_digit = img_digit[0].permute(1, 2, 0).cpu().numpy()
    ocr_prompt = "Question: What digit is drawn with the thin stroke? Answer:"

    arm_labels = {
        "baseline": "1. Baseline (Unmodulated)",
        "v0_jepa": "2. V0 JEPA (L2 Error)",
        "v1_bayes": "3. V1 Bayes (Gaussian KL)",
    }

    # Extract Saliency Gradients
    saliencies = {}
    for arm in ["baseline", "v0_jepa", "v1_bayes"]:
        saliencies[arm] = compute_layer_saliency(models[arm], img_digit, ocr_prompt, "7", res)

    # -------------------------------------------------------------------------
    # Plot 1: Decision Saliency Attribution Maps (d(Logit_7) / d(X_l(x,y)))
    # -------------------------------------------------------------------------
    print("[Viz] Rendering Figure 1: Layer-by-Layer Decision Saliency Maps...", flush=True)
    fig, axes = plt.subplots(3, 5, figsize=(18, 9.5), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    for row_idx, arm in enumerate(["baseline", "v0_jepa", "v1_bayes"]):
        grads = saliencies[arm]
        # Col 0: RGB Input with Ground-Truth Stroke
        axes[row_idx, 0].imshow(rgb_digit)
        axes[row_idx, 0].set_title("Input (1px Digit '7')", fontsize=10, color="white", fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1..4: Layers 0..3 Decision Saliency Attribution
        for l_idx in range(4):
            g = grads[l_idx]
            # Normalize for visualization contrast
            g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
            im = axes[row_idx, l_idx + 1].imshow(g_norm, cmap="inferno")
            snr = g[stroke_mask_np].mean() / (g[~stroke_mask_np].mean() + 1e-8)
            axes[row_idx, l_idx + 1].set_title(f"Field X_{l_idx} Saliency (SNR={snr:.2f}x)", fontsize=10, color="#38bdf8")
            axes[row_idx, l_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[row_idx, l_idx + 1], fraction=0.046, pad=0.04)

        axes[row_idx, 0].text(
            -0.12, 0.5, arm_labels[arm], transform=axes[row_idx, 0].transAxes,
            fontsize=11, fontweight="bold", color="#38bdf8", rotation=90, va="center", ha="right"
        )

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("Layer-by-Layer True Decision Saliency Attribution: ||d(Logit_7) / d(X_l(x,y))||",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path1 = out_dir / "decision_saliency_layer_evolution.png"
    plt.savefig(fig_path1, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path1}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 2: Quantitative 50-Sample Saliency SNR Bar Chart Across Depth
    # -------------------------------------------------------------------------
    print("[Viz] Computing 50-sample quantitative Saliency SNR across depth...", flush=True)
    depth_snrs = {arm: [[] for _ in range(4)] for arm in ["baseline", "v0_jepa", "v1_bayes"]}

    for i in range(40):
        d_val = int(rng.integers(0, 10))
        d_str = str(d_val)
        im_s, _, msk_s = make_ocr_1px(rng, np.array([d_val]), res=res)
        im_s = im_s.to(device)
        msk_np = msk_s[0].view(res, res).cpu().numpy().astype(bool)

        for arm in ["baseline", "v0_jepa", "v1_bayes"]:
            g_layers = compute_layer_saliency(models[arm], im_s, ocr_prompt, d_str, res)
            for l in range(4):
                g = g_layers[l]
                snr = float(g[msk_np].mean() / (g[~msk_np].mean() + 1e-8))
                depth_snrs[arm][l].append(snr)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    layers = [f"Field X_{l}" for l in range(4)]
    x = np.arange(len(layers))
    width = 0.26

    mean_base = [np.mean(depth_snrs["baseline"][l]) for l in range(4)]
    mean_jepa = [np.mean(depth_snrs["v0_jepa"][l]) for l in range(4)]
    mean_bayes = [np.mean(depth_snrs["v1_bayes"][l]) for l in range(4)]

    ax.bar(x - width, mean_base, width, label="Baseline (Unmodulated)", color="#94a3b8", alpha=0.85)
    ax.bar(x, mean_jepa, width, label="V0 JEPA (L2 Error)", color="#38bdf8", alpha=0.88)
    ax.bar(x + width, mean_bayes, width, label="V1 Bayes (Gaussian KL)", color="#2dd4bf", alpha=0.92)

    ax.set_title("Quantitative Decision Saliency SNR on 1px Stroke Target across Depth (N=40 Test Samples)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Stroke Saliency Signal-to-Noise Ratio (SNR)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b", axis="y")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)

    plt.tight_layout()
    fig_path2 = out_dir / "saliency_snr_comparison.png"
    plt.savefig(fig_path2, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path2}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 3: Pure Layer-by-Layer Spatial Surprise Comparison U_l(x,y)
    # -------------------------------------------------------------------------
    print("[Viz] Rendering Figure 3: Pure Spatial Surprise Layer Evolution...", flush=True)
    ext_j = run_heatmap_extraction(models["v0_jepa"], img_digit.cpu(), ocr_prompt, device)
    ext_by = run_heatmap_extraction(models["v1_bayes"], img_digit.cpu(), ocr_prompt, device)

    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Row 0: JEPA Surprise
    axes[0, 0].imshow(rgb_digit)
    axes[0, 0].set_title("Input (1px Digit '7')", fontsize=10, color="white", fontweight="bold")
    axes[0, 0].axis("off")

    for l_idx in range(4):
        uj = ext_j["surprises"][l_idx]
        imj = axes[0, l_idx + 1].imshow(uj, cmap="magma")
        axes[0, l_idx + 1].set_title(f"JEPA L{l_idx} Surprise U_{l_idx}", fontsize=10, color="#38bdf8")
        axes[0, l_idx + 1].axis("off")
        plt.colorbar(imj, ax=axes[0, l_idx + 1], fraction=0.046, pad=0.04)

    axes[0, 0].text(-0.12, 0.5, "V0 JEPA", transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight="bold", color="#38bdf8", rotation=90, va="center", ha="right")

    # Row 1: Bayes Surprise
    axes[1, 0].imshow(rgb_digit)
    axes[1, 0].set_title("Input (1px Digit '7')", fontsize=10, color="white", fontweight="bold")
    axes[1, 0].axis("off")

    for l_idx in range(4):
        uby = ext_by["surprises"][l_idx]
        imby = axes[1, l_idx + 1].imshow(uby, cmap="magma")
        axes[1, l_idx + 1].set_title(f"Bayes L{l_idx} Surprise U_{l_idx}", fontsize=10, color="#2dd4bf")
        axes[1, l_idx + 1].axis("off")
        plt.colorbar(imby, ax=axes[1, l_idx + 1], fraction=0.046, pad=0.04)

    axes[1, 0].text(-0.12, 0.5, "V1 Bayes", transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight="bold", color="#2dd4bf", rotation=90, va="center", ha="right")

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("Pure Spatial Surprise Comparison U_l(x,y) across Depth (Target: Single 1px Digit '7')",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path3 = out_dir / "pure_surprise_layer_comparison.png"
    plt.savefig(fig_path3, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path3}", flush=True)


if __name__ == "__main__":
    main()
