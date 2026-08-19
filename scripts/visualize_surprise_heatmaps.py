#!/usr/bin/env python3
"""Generate Calibrated Triad Showdown Heatmaps: Baseline vs V0 JEPA vs V1 Bayes.

Addresses:
  - Isolates Top-Down Net Language Writeback (Prompt - Blind) to eliminate static color patch stem bias.
  - Multi-Object Composite Canvas + Dedicated Single-Object Canvas (Pure Digit '7').
  - Direct Spatial Difference Maps: Delta(JEPA - Base), Delta(Bayes - Base), Delta(Bayes - JEPA).
  - V1 Gaussian KL Decomposition across 4 layers (U_mu vs U_sigma).

Outputs:
  - present/figs/triad_layer_heatmaps.png
  - present/figs/triad_final_difference_heatmaps.png
  - present/figs/triad_single_digit_comparison.png
  - present/figs/triad_bayes_kl_decomp.png
"""
from __future__ import annotations

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

from fine_grain.bayesian_surprise import spatial_surprise_map
from fine_grain.native_mot import NativeMoTStack
from fine_grain.ocr_1px import _DIGIT_STROKES, _bresenham, make_ocr_1px
from fine_grain.tasks import BG, SIGNAL, _canvas
from fine_grain.vlm_data import make_vqa_batch
from scripts.run_v0_surprise_eval import DualStreamVQAModel


def create_multi_object_canvas(rng: np.random.Generator, res: int = 64) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Create a composite image containing 3 distinct targets at known spatial locations."""
    img = _canvas(rng, n=1, R=res, n_blobs=2)[0]  # [3, R, R]

    # 1. Left Region: Red 1px polyline with kinks
    n_kinks = 6
    xs = np.linspace(6, res // 2 - 6, n_kinks, dtype=int)
    ys = rng.integers(8, res - 8, n_kinks)
    for i in range(n_kinks - 1):
        pts = _bresenham(xs[i], ys[i], xs[i + 1], ys[i + 1])
        for x, y in pts:
            if 0 <= x < res and 0 <= y < res:
                img[:, y, x] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # 2. Right Region: 1px Stick Digit '7'
    digit_char = "7"
    strokes = _DIGIT_STROKES[digit_char]
    dx0, dy0 = res // 2 + 6, 8
    d_size = res // 2 - 12
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            p0, p1 = stroke[i], stroke[i + 1]
            x0 = int(dx0 + p0[0] * d_size)
            y0 = int(dy0 + p0[1] * d_size)
            x1 = int(dx0 + p1[0] * d_size)
            y1 = int(dy0 + p1[1] * d_size)
            pts = _bresenham(x0, y0, x1, y1)
            for x, y in pts:
                if 0 <= x < res and 0 <= y < res:
                    img[:, y, x] = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    # 3. Bottom-Center Region: Small 3x3 Signal Square (Yellow Needle)
    nx, ny = res // 2 - 2, res - 10
    needle_col = SIGNAL[3]
    img[:, ny:ny + 3, nx:nx + 3] = needle_col[:, None, None]

    img_tensor = torch.from_numpy(np.clip(img, 0.0, 1.0)).unsqueeze(0)
    meta = {"kinks": n_kinks, "digit": digit_char, "needle_color": "yellow", "res": res}
    return img_tensor, meta


def run_heatmap_extraction(
    model: DualStreamVQAModel,
    image: torch.Tensor,
    prompt: str,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    image = image.to(device)
    res = model.res

    with torch.no_grad():
        tok_ids, mask = model.tokenize([prompt], device)
        text_emb = model.embed(tok_ids)

        X0 = model.mot_stack.encode_X(image)
        energy_0 = X0.norm(dim=-1).view(res, res).cpu().numpy()

        H = model.mot_stack.text_in(text_emb)
        X = X0

        layer_energies = [energy_0]
        layer_surprises = []
        layer_gates = []
        layer_u_mu = []
        layer_u_sigma = []

        for i, layer in enumerate(model.mot_stack.layers):
            X, H, tr = layer(X, H, text_mask=mask, prompt_mask=mask, layer_idx=i)
            energy_i = layer.last_X.norm(dim=-1).view(res, res).cpu().numpy()
            layer_energies.append(energy_i)

            if layer.last_u is not None and layer.last_w is not None:
                u_pt = spatial_surprise_map(layer.last_u, layer.last_w, res=res)
                layer_surprises.append(u_pt[0, 0].cpu().numpy())
            else:
                layer_surprises.append(np.zeros((res, res), dtype=np.float32))

            if layer.last_gate is not None and layer.last_w is not None:
                g_pt = spatial_surprise_map(layer.last_gate, layer.last_w, res=res)
                layer_gates.append(g_pt[0, 0].cpu().numpy())
            else:
                layer_gates.append(np.ones((res, res), dtype=np.float32))

            if getattr(layer, "last_u_mu", None) is not None and layer.last_w is not None:
                umu_pt = spatial_surprise_map(layer.last_u_mu, layer.last_w, res=res)
                layer_u_mu.append(umu_pt[0, 0].cpu().numpy())
            else:
                layer_u_mu.append(np.zeros((res, res), dtype=np.float32))

            if getattr(layer, "last_u_sigma", None) is not None and layer.last_w is not None:
                usig_pt = spatial_surprise_map(layer.last_u_sigma, layer.last_w, res=res)
                layer_u_sigma.append(usig_pt[0, 0].cpu().numpy())
            else:
                layer_u_sigma.append(np.zeros((res, res), dtype=np.float32))

    return {
        "energies": layer_energies,
        "surprises": layer_surprises,
        "gates": layer_gates,
        "u_mu": layer_u_mu,
        "u_sigma": layer_u_sigma,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Calibrated Viz] Using device: {device}", flush=True)

    res = 48
    d_model = 128
    n_layers = 4

    out_dir = ROOT / "present" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_rng = np.random.default_rng(123)
    img_tensor, meta = create_multi_object_canvas(canvas_rng, res=res)
    rgb_img = img_tensor[0].permute(1, 2, 0).numpy()

    # Load pre-saved best checkpoints
    models = {}
    for arm in ["baseline", "v0_jepa", "v1_bayes"]:
        m = DualStreamVQAModel(d_model=d_model, n_slices=32, n_layers=n_layers, res=res, surprise_mode=arm).to(device)
        ckpt_path = f"checkpoints/{arm}_best.pt"
        if os.path.exists(ckpt_path):
            print(f"  Loading checkpoint: {ckpt_path}", flush=True)
            m.load_state_dict(torch.load(ckpt_path))
        models[arm] = m

    ocr_prompt = "Question: What digit is drawn with the thin stroke? Answer:"
    blind_prompt = " "

    ext_ocr = {arm: run_heatmap_extraction(models[arm], img_tensor, ocr_prompt, device) for arm in models}
    ext_blind = {arm: run_heatmap_extraction(models[arm], img_tensor, blind_prompt, device) for arm in models}

    # Top-Down Net Language Writeback = Prompt Field - Blind Field (removes static stem color bias!)
    net_w = {arm: ext_ocr[arm]["energies"][-1] - ext_blind[arm]["energies"][-1] for arm in models}

    # -------------------------------------------------------------------------
    # Plot 1: Triad Layer-by-Layer Surprise & Final Fields
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 6, figsize=(18, 9.5), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    arm_labels = {
        "baseline": "1. Baseline (Unmodulated)",
        "v0_jepa": "2. V0 JEPA (L2 Error)",
        "v1_bayes": "3. V1 Bayes (Gaussian KL)",
    }

    for row_idx, arm in enumerate(models.keys()):
        data = ext_ocr[arm]
        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].set_title("Input (OCR '7')", fontsize=10, color="white")
        axes[row_idx, 0].axis("off")

        for l_idx in range(4):
            u_l = data["surprises"][l_idx]
            im_u = axes[row_idx, l_idx + 1].imshow(u_l, cmap="magma")
            axes[row_idx, l_idx + 1].set_title(f"L{l_idx} Surprise U_{l_idx}(x,y)", fontsize=10, color="white")
            axes[row_idx, l_idx + 1].axis("off")
            plt.colorbar(im_u, ax=axes[row_idx, l_idx + 1], fraction=0.046, pad=0.04)

        xf = data["energies"][-1]
        im_x = axes[row_idx, 5].imshow(xf, cmap="viridis")
        axes[row_idx, 5].set_title("Final Field ||X_4(x,y)||", fontsize=10, color="#2dd4bf", fontweight="bold")
        axes[row_idx, 5].axis("off")
        plt.colorbar(im_x, ax=axes[row_idx, 5], fraction=0.046, pad=0.04)

        axes[row_idx, 0].text(
            -0.12, 0.5, arm_labels[arm], transform=axes[row_idx, 0].transAxes,
            fontsize=11, fontweight="bold", color="#38bdf8", rotation=90, va="center", ha="right"
        )

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("Triad Comparison: Layer-by-Layer Spatial Surprise & Field Evolution (Target: 1px Stick Digit '7')",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path1 = out_dir / "triad_layer_heatmaps.png"
    plt.savefig(fig_path1, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path1}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 2: Calibrated Net Writeback & Difference Maps (Eliminates Yellow Block Bias)
    # -------------------------------------------------------------------------
    print("[Calibrated Viz] Rendering Figure 2: Calibrated Net Writeback & Differences...", flush=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Row 0: Top-Down Net Language Writeback (Prompt - Blind)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Input Canvas (RGB)", fontsize=11, color="white", fontweight="bold")
    axes[0, 0].axis("off")

    im01 = axes[0, 1].imshow(net_w["baseline"], cmap="plasma")
    axes[0, 1].set_title("1. Baseline Net Writeback", fontsize=11, color="#94a3b8", fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im02 = axes[0, 2].imshow(net_w["v0_jepa"], cmap="plasma")
    axes[0, 2].set_title("2. JEPA Net Writeback", fontsize=11, color="#38bdf8", fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im03 = axes[0, 3].imshow(net_w["v1_bayes"], cmap="plasma")
    axes[0, 3].set_title("3. Bayes Net Writeback", fontsize=11, color="#2dd4bf", fontweight="bold")
    axes[0, 3].axis("off")
    plt.colorbar(im03, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 1: Direct Spatial Differences of Net Language Writeback
    # 1,0: Spatial Gate Difference
    g_jepa = ext_ocr["v0_jepa"]["gates"][0]
    g_bayes = ext_ocr["v1_bayes"]["gates"][0]
    im10 = axes[1, 0].imshow(g_bayes - g_jepa, cmap="bwr", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title("L0 Gate Diff (Bayes - JEPA)", fontsize=11, color="#c084fc", fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 1,1: Delta Net (JEPA - Baseline)
    d_j_b = net_w["v0_jepa"] - net_w["baseline"]
    v1 = max(abs(d_j_b.min()), abs(d_j_b.max()))
    im11 = axes[1, 1].imshow(d_j_b, cmap="bwr", vmin=-v1, vmax=v1)
    axes[1, 1].set_title("Delta Net (JEPA - Baseline)", fontsize=11, color="#38bdf8", fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 1,2: Delta Net (Bayes - Baseline)
    d_by_b = net_w["v1_bayes"] - net_w["baseline"]
    v2 = max(abs(d_by_b.min()), abs(d_by_b.max()))
    im12 = axes[1, 2].imshow(d_by_b, cmap="bwr", vmin=-v2, vmax=v2)
    axes[1, 2].set_title("Delta Net (Bayes - Baseline)", fontsize=11, color="#2dd4bf", fontweight="bold")
    axes[1, 2].axis("off")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 1,3: Delta Net (Bayes - JEPA) -> Pure Language Writeback Delta!
    d_by_j = net_w["v1_bayes"] - net_w["v0_jepa"]
    v3 = max(abs(d_by_j.min()), abs(d_by_j.max()))
    im13 = axes[1, 3].imshow(d_by_j, cmap="bwr", vmin=-v3, vmax=v3)
    axes[1, 3].set_title("Delta Net (Bayes - JEPA) [Calibrated]", fontsize=11, color="#fbbf24", fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im13, ax=axes[1, 3], fraction=0.046, pad=0.04)

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("Calibrated Top-Down Net Language Writeback (Prompt - Blind) & Direct Spatial Delta Maps (OCR '7')",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path2 = out_dir / "triad_final_difference_heatmaps.png"
    plt.savefig(fig_path2, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path2}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 3: Dedicated Single-Target Canvas (Pure 1px Digit '7')
    # -------------------------------------------------------------------------
    print("[Calibrated Viz] Rendering Figure 3: Dedicated Single Digit 7 Canvas...", flush=True)
    rng_digit = np.random.default_rng(42)
    img_digit, _, _ = make_ocr_1px(rng_digit, np.array([7]), res=res)
    rgb_digit = img_digit[0].permute(1, 2, 0).numpy()

    ext_d_base = run_heatmap_extraction(models["baseline"], img_digit, ocr_prompt, device)
    ext_d_jepa = run_heatmap_extraction(models["v0_jepa"], img_digit, ocr_prompt, device)
    ext_d_bayes = run_heatmap_extraction(models["v1_bayes"], img_digit, ocr_prompt, device)

    xd_base = ext_d_base["energies"][-1]
    xd_jepa = ext_d_jepa["energies"][-1]
    xd_bayes = ext_d_bayes["energies"][-1]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=150)
    plt.subplots_adjust(wspace=0.25)

    axes[0].imshow(rgb_digit)
    axes[0].set_title("Single Input (Digit '7')", fontsize=11, color="white", fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(xd_base, cmap="viridis")
    axes[1].set_title("Baseline Field ||X_4||", fontsize=11, color="#94a3b8", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(xd_jepa, cmap="viridis")
    axes[2].set_title("JEPA Field ||X_4||", fontsize=11, color="#38bdf8", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(xd_bayes, cmap="viridis")
    axes[3].set_title("Bayes Field ||X_4||", fontsize=11, color="#2dd4bf", fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    diff_single = xd_bayes - xd_jepa
    v_s = max(abs(diff_single.min()), abs(diff_single.max()))
    im4 = axes[4].imshow(diff_single, cmap="bwr", vmin=-v_s, vmax=v_s)
    axes[4].set_title("Delta (Bayes - JEPA) [Single]", fontsize=11, color="#fbbf24", fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("Single-Target Dedicated Canvas: Isolated Direct Field Comparison (Prompt: OCR Digit '7')",
                 fontsize=13, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path3 = out_dir / "triad_single_digit_comparison.png"
    plt.savefig(fig_path3, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path3}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 4: Gaussian KL Decomposition across 4 Layers
    # -------------------------------------------------------------------------
    print("[Calibrated Viz] Rendering Figure 4: Gaussian KL Decomposition...", flush=True)
    bayes_data = ext_ocr["v1_bayes"]

    fig, axes = plt.subplots(4, 4, figsize=(15, 12), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    for l_idx in range(4):
        axes[l_idx, 0].imshow(rgb_img)
        axes[l_idx, 0].set_title(f"L{l_idx} Input RGB", fontsize=9, color="white")
        axes[l_idx, 0].axis("off")

        u_tot = bayes_data["surprises"][l_idx]
        im1 = axes[l_idx, 1].imshow(u_tot, cmap="magma")
        axes[l_idx, 1].set_title(f"L{l_idx} Total KL D_KL(q||p)", fontsize=9, color="#38bdf8", fontweight="bold")
        axes[l_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[l_idx, 1], fraction=0.046, pad=0.04)

        u_mu = bayes_data["u_mu"][l_idx]
        im2 = axes[l_idx, 2].imshow(u_mu, cmap="plasma")
        axes[l_idx, 2].set_title(f"L{l_idx} Mean Error U_mu", fontsize=9, color="#fb7185", fontweight="bold")
        axes[l_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[l_idx, 2], fraction=0.046, pad=0.04)

        u_sig = bayes_data["u_sigma"][l_idx]
        im3 = axes[l_idx, 3].imshow(u_sig, cmap="cividis")
        axes[l_idx, 3].set_title(f"L{l_idx} Uncertainty U_sigma", fontsize=9, color="#fbbf24", fontweight="bold")
        axes[l_idx, 3].axis("off")
        plt.colorbar(im3, ax=axes[l_idx, 3], fraction=0.046, pad=0.04)

    fig.patch.set_facecolor("#0b1120")
    for ax in axes.flat:
        ax.set_facecolor("#0f172a")

    plt.suptitle("V1 Gaussian Bayesian Surprise Decomposition: Mean Error (U_mu) vs Uncertainty Entropy (U_sigma)",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    fig_path4 = out_dir / "triad_bayes_kl_decomp.png"
    plt.savefig(fig_path4, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  Saved to {fig_path4}", flush=True)


if __name__ == "__main__":
    main()
