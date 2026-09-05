#!/usr/bin/env python3
"""Diagnose Spatial Address Matrix W Collapse in T2I (Blank Canvas).

Measures:
  1. Prompt Sensitivity: ||W(prompt_1) - W(prompt_2)|| on blank canvas (pi_x=0).
     Is Layer 0 assignment W identical across different prompts?
  2. Address Effective Rank: erank(W^T @ W) across M slices.
  3. Spatial Entropy & Max/Min Slot Mass: Are slices collapsed to uniform or single-slot?
  4. Visualizes the spatial footprint of slices on blank canvas vs observed canvas.

Outputs:
  - results/published/address_collapse_diagnostic.json
  - present/figs/address_collapse_slices.png
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.ocr_1px import make_ocr_1px
from scripts.run_v0_surprise_eval import DualStreamVQAModel
from scripts.probe_deslice_memory import _retrofit_fulld_prior


def erank(tokens: torch.Tensor) -> float:
    """Roy-Vetterli effective rank."""
    t = tokens.detach().float()
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])
    s = torch.linalg.svdvals(t)
    p = (s * s).clamp_min(1e-12)
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Address Diagnostic] Running on device: {device}", flush=True)

    out_dir = ROOT / "present" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run1_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    res = 32
    d_model = 128
    n_slices = 32
    n_layers = 4

    model = DualStreamVQAModel(
        d_model=d_model, n_slices=n_slices, n_layers=n_layers, res=res,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms", deslice_topk=2, n_heads=4,
    )
    _retrofit_fulld_prior(model)
    model = model.to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Capture w in layer 0
    captured_w = []

    old_layer0_read_fwd = model.mot_stack.layers[0].read.forward
    def hook_read(*args, **kwargs):
        S, w = old_layer0_read_fwd(*args, **kwargs)
        captured_w.append(w.detach())
        return S, w

    model.mot_stack.layers[0].read.forward = hook_read

    # 1. Evaluate on Blank Canvas (pi_x = 0) with two different prompts
    prompts = [
        "What digit is drawn with thin stroke ?",  # targeting digit 7
        "How many corners does red polyline have ?", # targeting polyline kinks
    ]

    blank_img = torch.zeros(1, 3, res, res, device=device)

    # Forward prompt 0 on blank
    captured_w.clear()
    with torch.no_grad():
        _ = model(blank_img, [prompts[0]], image_precision=0.0)
    w_blank_prompt0 = captured_w[0][0]  # [N, M]

    # Forward prompt 1 on blank
    captured_w.clear()
    with torch.no_grad():
        _ = model(blank_img, [prompts[1]], image_precision=0.0)
    w_blank_prompt1 = captured_w[0][0]  # [N, M]

    # 2. Evaluate on Observed Canvas (digit 7) with prompt 0
    rng = np.random.default_rng(42)
    obs_img, _, _ = make_ocr_1px(rng, np.array([7]), res=res)
    obs_img = obs_img.to(device)

    captured_w.clear()
    with torch.no_grad():
        _ = model(obs_img, [prompts[0]], image_precision=1.0)
    w_obs = captured_w[0][0]  # [N, M]

    # Quantitative Metrics
    # A) Prompt Sensitivity on blank canvas:
    diff_blank = float(torch.norm(w_blank_prompt0 - w_blank_prompt1).item())
    rel_diff_blank = diff_blank / float(torch.norm(w_blank_prompt0).item() + 1e-8)

    # B) Observed vs Blank difference:
    diff_obs_blank = float(torch.norm(w_obs - w_blank_prompt0).item())
    rel_diff_obs_blank = diff_obs_blank / float(torch.norm(w_obs).item() + 1e-8)

    # C) Effective Rank of W^T @ W across M slices:
    # gram: [M, M]
    gram_blank = w_blank_prompt0.T @ w_blank_prompt0
    gram_obs = w_obs.T @ w_obs
    erank_blank = erank(w_blank_prompt0)
    erank_obs = erank(w_obs)

    # D) Slot Mass Distribution (sum over N pixels):
    mass_blank = w_blank_prompt0.sum(dim=0).cpu().numpy()  # [M]
    mass_obs = w_obs.sum(dim=0).cpu().numpy()

    print("\n" + "="*60)
    print("=== SPATIAL ADDRESS MATRIX W COLLAPSE AUDIT ===")
    print("="*60)
    print(f"Blank Canvas Prompt Divergence ||W(p0) - W(p1)||: {diff_blank:.6f} (rel: {rel_diff_blank*100:.3f}%)")
    print(f"Observed vs Blank Divergence ||W(obs) - W(blank)||: {diff_obs_blank:.6f} (rel: {rel_diff_obs_blank*100:.3f}%)")
    print(f"Address Effective Rank (Blank Canvas):  {erank_blank:.2f} / {n_slices}")
    print(f"Address Effective Rank (Observed Image): {erank_obs:.2f} / {n_slices}")
    print(f"Blank Slot Mass: min={mass_blank.min():.2f}, max={mass_blank.max():.2f}, mean={mass_blank.mean():.2f}")
    print(f"Obs Slot Mass:   min={mass_obs.min():.2f}, max={mass_obs.max():.2f}, mean={mass_obs.mean():.2f}")

    if diff_blank < 1e-5:
        print("\n>>> CONFIRMED: W is 100% PROMPT-BLIND on blank canvas! W(H_7) == W(H_kinks)!")
    else:
        print(f"\n>>> W exhibits sensitivity: {diff_blank:.5f}")

    # Plot spatial footprints of first 8 slices on Blank Canvas vs Observed
    fig, axes = plt.subplots(3, 8, figsize=(20, 7.5), dpi=150)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    fig.patch.set_facecolor("#090d16")

    w_b0_np = w_blank_prompt0.view(res, res, n_slices).cpu().numpy()
    w_b1_np = w_blank_prompt1.view(res, res, n_slices).cpu().numpy()
    w_obs_np = w_obs.view(res, res, n_slices).cpu().numpy()

    for m in range(8):
        # Row 0: Blank Canvas with Prompt 0 ("digit 7")
        ax0 = axes[0, m]
        ax0.imshow(w_b0_np[:, :, m], cmap="viridis", vmin=0, vmax=w_b0_np.max())
        ax0.set_title(f"Blank Slice {m}\n(Prompt: '7')", fontsize=9, color="#38bdf8")
        ax0.axis("off")

        # Row 1: Blank Canvas with Prompt 1 ("kinks")
        ax1 = axes[1, m]
        ax1.imshow(w_b1_np[:, :, m], cmap="viridis", vmin=0, vmax=w_b1_np.max())
        ax1.set_title(f"Blank Slice {m}\n(Prompt: 'kinks')", fontsize=9, color="#fbbf24")
        ax1.axis("off")

        # Row 2: Observed Canvas ("digit 7")
        ax2 = axes[2, m]
        ax2.imshow(w_obs_np[:, :, m], cmap="magma", vmin=0, vmax=w_obs_np.max())
        ax2.set_title(f"Observed Slice {m}\n(Image: '7')", fontsize=9, color="#2dd4bf")
        ax2.axis("off")

    axes[0, 0].text(-0.25, 0.5, "Blank\nPrompt '7'", transform=axes[0, 0].transAxes,
                    fontsize=10, fontweight="bold", color="#38bdf8", rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.25, 0.5, "Blank\nPrompt 'kinks'", transform=axes[1, 0].transAxes,
                    fontsize=10, fontweight="bold", color="#fbbf24", rotation=90, va="center", ha="right")
    axes[2, 0].text(-0.25, 0.5, "Observed\n'7' In Ink", transform=axes[2, 0].transAxes,
                    fontsize=10, fontweight="bold", color="#2dd4bf", rotation=90, va="center", ha="right")

    plt.suptitle(
        f"Layer 0 Spatial Address Footprint W: Blank Canvas (Prompt-Blind erank={erank_blank:.2f}) vs Observed (erank={erank_obs:.2f})",
        fontsize=13, fontweight="bold", color="#f8fafc", y=0.98,
    )

    fig_path = out_dir / "address_collapse_slices.png"
    plt.savefig(fig_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved visualization to {fig_path}")

    # Save diagnostic results
    res_dict = {
        "diff_blank_prompts": diff_blank,
        "rel_diff_blank_prompts": rel_diff_blank,
        "diff_obs_blank": diff_obs_blank,
        "erank_blank": erank_blank,
        "erank_obs": erank_obs,
        "n_slices": n_slices,
        "prompt_blind_confirmed": bool(diff_blank < 1e-5),
        "mass_blank_min": float(mass_blank.min()),
        "mass_blank_max": float(mass_blank.max()),
        "mass_obs_min": float(mass_obs.min()),
        "mass_obs_max": float(mass_obs.max()),
    }
    out_json = ROOT / "results" / "published" / "address_collapse_diagnostic.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
    print(f"Saved diagnostic report to {out_json}")


if __name__ == "__main__":
    main()
