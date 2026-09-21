"""Microstep Crystallization Visualization across Deep Pondering K in [1..256].
Shows how the 9x9 Sudoku board evolves from diffuse high-entropy superposition
into confident, sharp constraint-satisfaction digits as pondering depth increases.
"""
import os
import sys

# Pop script directory to avoid shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


@torch.no_grad()
def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=256, arm="corrective_flow").to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    # Hard extreme puzzle 122 (17 clues)
    inp = torch.as_tensor(test_in[122:123], dtype=torch.long, device=device)
    lbl = torch.as_tensor(test_lbl[122:123], dtype=torch.long, device=device)
    clues_mask = (inp.view(9, 9) > 1).cpu().numpy()

    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    curr = clue_field.clone()

    selected_horizons = [1, 4, 16, 64, 256]
    snapshots = {}

    print("[Tracing Deep Pondering Microsteps K in [1..256]]...", flush=True)
    for k in range(1, 257):
        f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
        f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
        f_star = model.model.collision(f_tr, cond=clue_field)

        cat_input = torch.cat([f_star, clue_field], dim=-1)
        v_k = model.model.corrective_net(cat_input)

        f_star_f32 = f_star.float()
        v_k_f32 = v_k.float()
        dot_prod = torch.sum(f_star_f32 * v_k_f32, dim=(1, 2, 3), keepdim=True)
        norm_sq = torch.sum(f_star_f32 ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
        proj = dot_prod / norm_sq
        u_k_f32 = v_k_f32 - proj * f_star_f32
        u_hat = u_k_f32 / (torch.linalg.vector_norm(u_k_f32, dim=(1, 2, 3), keepdim=True) + 1e-8)

        gate_logits = model.model.angle_gate(cat_input)
        alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(gate_logits, dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
        f_norm = torch.linalg.vector_norm(f_star_f32, dim=(1, 2, 3), keepdim=True).to(curr.dtype)
        curr = (torch.cos(alpha_k) * f_star) + (f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype))

        if k in selected_horizons:
            flat_k = curr.view(1, 81, model.model.d)
            logits_k = model.model.readout_mlp(flat_k)
            probs = F.softmax(logits_k, dim=-1).view(9, 9, 11)
            pred_digits = torch.argmax(probs, dim=-1).cpu().numpy()
            confidence = torch.max(probs, dim=-1).values.cpu().numpy()
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu().numpy()

            snapshots[k] = {
                "pred": pred_digits,
                "conf": confidence,
                "ent": entropy
            }

    # Render 5-Panel Evolution Figure
    print("Generating 5-Panel Crystallization Figure...", flush=True)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), facecolor="#060614")
    fig.suptitle("CBIM Deep Pondering Crystallization (Hard 17-Clue Sudoku Extreme)", fontsize=18, color="#e0e8ff", y=0.98, fontweight="bold")

    lbl_arr = lbl.view(9, 9).cpu().numpy()

    for idx, k in enumerate(selected_horizons):
        ax = axes[idx]
        ax.set_facecolor("#040312")
        snap = snapshots[k]
        conf_map = snap["conf"]
        pred_map = snap["pred"]

        im = ax.imshow(conf_map, cmap="plasma", vmin=0.2, vmax=1.0, origin="upper")
        ax.set_title(f"Ponder Depth $K={k}$\nMean Conf: {np.mean(conf_map)*100:.1f}%", color="#d0e0ff", fontsize=13, pad=10)

        # Draw 3x3 box grid lines
        for line_idx in [2.5, 5.5]:
            ax.axhline(line_idx, color="#ffffff", linewidth=1.5, alpha=0.7)
            ax.axvline(line_idx, color="#ffffff", linewidth=1.5, alpha=0.7)

        # Draw predicted digits and mark correct vs incorrect
        for r in range(9):
            for c in range(9):
                is_clue = clues_mask[r, c]
                digit = pred_map[r, c]
                is_correct = (digit == lbl_arr[r, c])

                if is_clue:
                    col = "#00ffff"  # Bright cyan for initial given clues
                    weight = "bold"
                elif is_correct:
                    col = "#ffffff"  # Pure white for correctly reasoned digits
                    weight = "bold"
                else:
                    col = "#ff5555"  # Red for erroneous digits
                    weight = "normal"

                # If digit is valid (1..9), display it (digit 2..10 in vocab -> 1..9)
                display_char = str(digit - 1) if digit > 1 else "."
                ax.text(c, r, display_char, ha="center", va="center", color=col, fontsize=11, fontweight=weight)

        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#203560")

    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Prediction Certainty (Max Softmax Probability)", color="#d0e0ff", fontsize=12)
    cbar.ax.tick_params(colors="#d0e0ff")

    out_png = out_dir / "cbim_deep_pondering_crystallization.png"
    plt.tight_layout(rect=[0.02, 0.05, 0.90, 0.94])
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved crystallization figure to {out_png}")


if __name__ == "__main__":
    main()
