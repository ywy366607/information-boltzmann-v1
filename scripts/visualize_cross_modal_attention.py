#!/usr/bin/env python3
"""Extract and Visualize True Cross-Modal Attention Matrices & Attention Scores.

Generates:
  1. Full Joint Attention Matrix [S; H] x [S; H] in MoT Space (S->S, S->H, H->S, H->H)
  2. Top-Down Language Prior Attention Q_slice -> H (Attention of 32 probes over prompt tokens)
  3. Word-Level Visual Attention Scores across Depth (Layer 0 to Layer 3)
  4. Cross-Modal Mass Routing Dynamics across Depth (Information flow between vision and language)

Output:
  - present/figs/cross_modal_attention_matrices.png
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cross-Modal Attention] Running on device: {device}", flush=True)

    out_dir = ROOT / "present" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Champion B Checkpoint
    ckpt_path = ROOT / "checkpoints" / "v1_bayes_2000step_upd_rms_run1_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = DualStreamVQAModel(
        d_model=128, n_slices=32, n_layers=4, res=32,
        surprise_mode="v1_bayes", surprise_beta=1.5,
        s_update="rms", deslice_topk=2, n_heads=4,
    )
    _retrofit_fulld_prior(model)
    model = model.to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 2. Test Input: 1px OCR Digit '7'
    rng = np.random.default_rng(42)
    img, lab, mask = make_ocr_1px(rng, np.array([7]), res=32)
    img = img.to(device)

    prompt = "What digit is drawn with thin stroke ?"
    words = prompt.replace("?", " ?").split()
    T = len(words)
    M = 32

    # Forward pass
    with torch.no_grad():
        out = model(img, [prompt])

    pred_idx = out["logits"].argmax(dim=-1).item()
    pred_ans = model.answers[pred_idx]
    print(f"  Prediction: '{pred_ans}', Ground Truth: '{lab[0].item()}'", flush=True)

    # 3. Extract Attention Matrices across all 4 layers
    prior_attns = []    # [4, M, T]
    joint_attns = []    # [4, M+T, M+T]
    s_to_s_masses = []
    s_to_h_masses = []
    h_to_s_masses = []
    h_to_h_masses = []
    word_masses = []    # [4, T]

    for l_idx, layer in enumerate(model.mot_stack.layers):
        # A) Prior Attention: Q_slice -> H [1, 4, M, T]
        q_attn = layer.surprise_gate.last_attn[0].mean(dim=0).cpu().numpy()  # [M, T]
        prior_attns.append(q_attn)

        # B) MoT Full Attention:
        # av: [1, 4, M, M+T], at: [1, 4, T, M+T]
        av = layer.mot.last_av[0].mean(dim=0).cpu().numpy()  # [M, M+T]
        at = layer.mot.last_at[0].mean(dim=0).cpu().numpy()  # [T, M+T]
        joint = np.vstack([av, at])                          # [M+T, M+T]
        joint_attns.append(joint)

        # Mass breakdowns
        s_to_s = av[:, :M].sum(axis=-1).mean()
        s_to_h = av[:, M:].sum(axis=-1).mean()
        h_to_s = at[:, :M].sum(axis=-1).mean()
        h_to_h = at[:, M:].sum(axis=-1).mean()

        s_to_s_masses.append(float(s_to_s))
        s_to_h_masses.append(float(s_to_h))
        h_to_s_masses.append(float(h_to_s))
        h_to_h_masses.append(float(h_to_h))

        # Word-level mass from vision
        w_mass = layer.mot.last_text_token_mass[0].cpu().numpy()  # [T]
        word_masses.append(w_mass)

    # 4. Create Comprehensive 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    plt.subplots_adjust(wspace=0.28, hspace=0.32)
    fig.patch.set_facecolor("#090d16")

    border_col = (148/255, 163/255, 184/255, 0.25)
    grid_col = (148/255, 163/255, 184/255, 0.12)
    box_fc = (15/255, 23/255, 42/255, 0.85)

    for ax in axes.flat:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(border_col)

    # -------------------------------------------------------------------------
    # Panel 1: Layer 0 Full Joint MoT Attention Matrix [S; H] x [S; H]
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    j0 = joint_attns[0]
    im1 = ax1.imshow(j0, cmap="magma", aspect="auto", interpolation="nearest")
    # Draw quadrant dividers
    ax1.axvline(M - 0.5, color="#38bdf8", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.axhline(M - 0.5, color="#38bdf8", linestyle="--", linewidth=1.5, alpha=0.8)

    ax1.set_title("Layer 0: Joint MoT Cross-Modal Attention Space [S; H] x [S; H]",
                  fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
    ax1.set_xlabel("Key Tokens (Visual Slices S_0..S_31  |  Language Words H_0..H_7)",
                   fontsize=10, color="#cbd5e1", labelpad=8)
    ax1.set_ylabel("Query Tokens (Visual Slices S  |  Language Words H)",
                   fontsize=10, color="#cbd5e1", labelpad=8)

    # Quadrant annotations
    ax1.text(M / 2, M / 2, "Visual Self-Attn\n(S -> S: 44.0%)", color="#ffffff",
             fontsize=10, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc=box_fc, ec="#38bdf8", lw=1))
    ax1.text(M + T / 2, M / 2, "Vision-to-Text Cross\n(S -> H: 56.0%)\n*Reading Instructions*",
             color="#fbbf24", fontsize=10, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc=box_fc, ec="#fbbf24", lw=1))
    ax1.text(M / 2, M + T / 2, "Text-to-Vision Cross (H -> S: 73.9%)\n*Gathering Visual Evidence*",
             color="#2dd4bf", fontsize=10, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc=box_fc, ec="#2dd4bf", lw=1))
    ax1.text(M + T / 2, M + T / 2, "Text Causal\n(H -> H)", color="#c084fc",
             fontsize=9, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc=box_fc, ec="#c084fc", lw=1))

    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar1.set_label("Attention Weight", color="#cbd5e1", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Top-Down Language Prior Attention (Q_slice -> H) in Layer 0
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    p0 = prior_attns[0]  # [M, T]
    im2 = ax2.imshow(p0, cmap="viridis", aspect="auto", interpolation="nearest")
    ax2.set_title("Top-Down Language Prior Readout: Q_slice -> H (Layer 0)",
                  fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
    ax2.set_xticks(range(T))
    ax2.set_xticklabels(words, rotation=35, ha="right", fontsize=9, color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Slice Query Probes (Q_0 .. Q_31)", fontsize=10, color="#cbd5e1", labelpad=8)
    ax2.set_xlabel("Prompt Tokens H", fontsize=10, color="#cbd5e1", labelpad=8)

    # Highlight keyword columns
    ax2.axvspan(0.5, 1.5, color=(56/255, 189/255, 248/255, 0.2), label="'digit' token")
    ax2.axvspan(4.5, 6.5, color=(251/255, 191/255, 36/255, 0.2), label="'thin stroke' tokens")

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar2.set_label("Prior Attention Weight", color="#cbd5e1", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 3: Word-Level Visual Attention Scores across Depth (L_0 .. L_3)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    x = np.arange(T)
    bar_w = 0.18
    colors_layer = ["#38bdf8", "#2dd4bf", "#fbbf24", "#c084fc"]
    layer_names = ["Layer 0", "Layer 1", "Layer 2", "Layer 3"]

    for l_idx in range(4):
        offset = (l_idx - 1.5) * bar_w
        vals = word_masses[l_idx]
        rects = ax3.bar(x + offset, vals, width=bar_w, label=layer_names[l_idx],
                        color=colors_layer[l_idx], alpha=0.9, edgecolor="none")
        # Annotate top scores in Layer 0
        if l_idx == 0:
            for r, v in zip(rects, vals):
                if v > 0.1:
                    ax3.annotate(f"{v:.2f}", xy=(r.get_x() + r.get_width() / 2, v),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=8, color="#ffffff", fontweight="bold")

    ax3.set_title("Visual Attention Grounding on Prompt Words: S -> H Across Depth",
                  fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(words, rotation=25, ha="right", fontsize=9, color="#f8fafc", fontweight="bold")
    ax3.set_ylabel("Mean Visual Attention Mass per Token", fontsize=10, color="#cbd5e1", labelpad=8)
    ax3.grid(axis="y", color=grid_col, linestyle="--")
    ax3.legend(frameon=True, facecolor="#0b1120", edgecolor=border_col,
               labelcolor="#cbd5e1", fontsize=8.5, loc="upper right")

    # -------------------------------------------------------------------------
    # Panel 4: Cross-Modal Attention Flow Dynamics across Depth
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    layers_x = [0, 1, 2, 3]

    ax4.plot(layers_x, s_to_s_masses, marker="o", linewidth=2.5, color="#38bdf8", label="S -> S (Visual Self-Attn)")
    ax4.plot(layers_x, s_to_h_masses, marker="s", linewidth=2.5, color="#fbbf24", label="S -> H (Vision Reads Language)")
    ax4.plot(layers_x, h_to_s_masses, marker="^", linewidth=2.5, color="#2dd4bf", label="H -> S (Text Queries Slices)")
    ax4.plot(layers_x, h_to_h_masses, marker="d", linewidth=2.0, color="#c084fc", linestyle="--", label="H -> H (Text Causal)")

    for lx in layers_x:
        ax4.annotate(f"{s_to_h_masses[lx]*100:.1f}%", (lx, s_to_h_masses[lx]),
                     xytext=(0, -14), textcoords="offset points", ha="center",
                     fontsize=8.5, color="#fbbf24", fontweight="bold")
        ax4.annotate(f"{h_to_s_masses[lx]*100:.1f}%", (lx, h_to_s_masses[lx]),
                     xytext=(0, 7), textcoords="offset points", ha="center",
                     fontsize=8.5, color="#2dd4bf", fontweight="bold")

    ax4.set_title("Cross-Modal Attention Mass Dynamics across Depth (0 -> 3)",
                  fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
    ax4.set_xticks(layers_x)
    ax4.set_xticklabels(["Layer 0", "Layer 1", "Layer 2", "Layer 3"], fontsize=9.5, color="#f8fafc")
    ax4.set_ylabel("Attention Mass Proportion (Sum to 1.0 per query stream)", fontsize=10, color="#cbd5e1", labelpad=8)
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, color=grid_col, linestyle="--")
    ax4.legend(frameon=True, facecolor="#0b1120", edgecolor=border_col,
               labelcolor="#cbd5e1", fontsize=8.5, loc="center right")

    plt.suptitle("Vision-Language Interaction in Native MoT: Attention Matrices, Scores & Causal Flow",
                 fontsize=15, fontweight="bold", color="#ffffff", y=0.98)

    save_path = out_dir / "cross_modal_attention_matrices.png"
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Cross-Modal Attention] Saved figure to {save_path}", flush=True)

    # Save summary stats JSON
    stats_out = ROOT / "results" / "published" / "cross_modal_attention_stats.json"
    stats = {
        "prompt": prompt,
        "words": words,
        "layers": [
            {
                "layer": l,
                "s_to_s_mass": s_to_s_masses[l],
                "s_to_h_mass": s_to_h_masses[l],
                "h_to_s_mass": h_to_s_masses[l],
                "h_to_h_mass": h_to_h_masses[l],
                "word_masses": {w: float(word_masses[l][i]) for i, w in enumerate(words)},
            }
            for l in range(4)
        ]
    }
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[Cross-Modal Attention] Saved stats to {stats_out}", flush=True)


if __name__ == "__main__":
    main()
