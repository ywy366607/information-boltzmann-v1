"""Generate full-depth K=1..256 crystallization animations for both Easy (67.9% Acc) and Hard 17-Clue Puzzles.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


@torch.no_grad()
def generate_k256_animation(model, inp, lbl, name: str, out_path: Path, max_k: int = 256):
    print(f"\nGenerating K=1..{max_k} Animation for {name}...", flush=True)
    clues_mask = (inp.view(9, 9) > 1).cpu().numpy()
    lbl_arr = lbl.view(9, 9).cpu().numpy()

    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    curr = clue_field.clone()

    # Logarithmically / strategically sampled 40 frames across K=1..256
    sampled_steps = sorted(list(set(
        list(range(1, 17)) +
        [20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 180, 200, 220, 240, 256]
    )))

    frames = []
    confs_history = []
    entropies_history = []
    accs_history = []
    recorded_steps = []

    for k in range(1, max_k + 1):
        f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
        f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
        f_star = model.model.collision(f_tr, cond=clue_field)

        cat_in = torch.cat([f_star, clue_field], dim=-1)
        v_k = model.model.corrective_net(cat_in)
        dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
        norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
        u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
        u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
        alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
        f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
        curr = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        if k in sampled_steps:
            flat_k = curr.view(1, 81, model.model.d)
            logits_k = model.model.readout_mlp(flat_k)
            probs = F.softmax(logits_k, dim=-1).view(9, 9, 11)
            pred_digits = torch.argmax(probs, dim=-1).cpu().numpy()
            conf_map = torch.max(probs, dim=-1).values.cpu().numpy()
            ent_map = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu().numpy()

            mean_c = float(np.mean(conf_map))
            mean_e = float(np.mean(ent_map))
            cell_acc = float(np.mean(pred_digits == lbl_arr))

            confs_history.append(mean_c)
            entropies_history.append(mean_e)
            accs_history.append(cell_acc)
            recorded_steps.append(k)

            # Plot frame
            fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#060614", gridspec_kw={"width_ratios": [1.1, 1.0]})
            fig.suptitle(f"CBIM Latent Pondering Dynamics ({name}) | Time Axis K = {k:03d} / 256", fontsize=16, color="#e0e8ff", y=0.96, fontweight="bold")

            # 1. Sudoku Grid
            ax_grid.set_facecolor("#040312")
            im = ax_grid.imshow(conf_map, cmap="plasma", vmin=0.15, vmax=1.0, origin="upper")
            ax_grid.set_title(f"Certainty: {mean_c*100:.1f}% | Solved Accuracy: {cell_acc*100:.1f}%", color="#d0e0ff", fontsize=13, pad=8)

            for line_idx in [2.5, 5.5]:
                ax_grid.axhline(line_idx, color="#ffffff", linewidth=1.5, alpha=0.7)
                ax_grid.axvline(line_idx, color="#ffffff", linewidth=1.5, alpha=0.7)

            for r in range(9):
                for c in range(9):
                    is_clue = clues_mask[r, c]
                    digit = pred_digits[r, c]
                    is_correct = (digit == lbl_arr[r, c])
                    col = "#00ffff" if is_clue else ("#ffffff" if is_correct else "#ff5555")
                    weight = "bold" if (is_clue or is_correct) else "normal"
                    display_char = str(digit - 1) if digit > 1 else "."
                    ax_grid.text(c, r, display_char, ha="center", va="center", color=col, fontsize=11, fontweight=weight)

            ax_grid.set_xticks(range(9))
            ax_grid.set_yticks(range(9))
            ax_grid.tick_params(colors="#7088b0")
            for spine in ax_grid.spines.values():
                spine.set_color("#203560")

            # 2. Tracking Curves
            ax_curve.set_facecolor("#080a1c")
            ax_curve.plot(recorded_steps, [a*100 for a in accs_history], color="#38d8b8", linewidth=2.5, label="Cell Accuracy (%)")
            ax_curve.plot(recorded_steps, [c*100 for c in confs_history], color="#ffaa00", linewidth=2.0, linestyle="--", label="Mean Certainty (%)")

            ax2 = ax_curve.twinx()
            ax2.plot(recorded_steps, entropies_history, color="#ff4488", linewidth=2.0, label="Entropy S (nats)")
            ax2.set_ylabel("Shannon Entropy (nats)", color="#ff4488", fontsize=11)
            ax2.tick_params(colors="#ff4488")
            ax2.set_ylim(0.4, 2.3)

            ax_curve.set_xlim(1, 256)
            ax_curve.set_ylim(20, 100)
            ax_curve.set_xlabel("Pondering Microstep (K)", color="#90a8d0", fontsize=11)
            ax_curve.set_ylabel("Accuracy & Confidence (%)", color="#38d8b8", fontsize=11)
            ax_curve.tick_params(colors="#7088b0")
            ax_curve.grid(True, color="#182850", alpha=0.6)
            for spine in ax_curve.spines.values():
                spine.set_color("#203560")

            lines1, labels1 = ax_curve.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_curve.legend(lines1 + lines2, labels1 + labels2, loc="center right", facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff", fontsize=10)

            plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close()
            buf.seek(0)
            frames.append(Image.open(buf))

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0
    )
    print(f"Saved animation to {out_path}", flush=True)


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=256, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    # 1. Easy Puzzle (Index 909, 35 clues, reaches 67.9% accuracy)
    inp_easy = torch.as_tensor(test_in[909:910], dtype=torch.long, device=device)
    lbl_easy = torch.as_tensor(test_lbl[909:910], dtype=torch.long, device=device)
    generate_k256_animation(model, inp_easy, lbl_easy, "Easy Puzzle (35 Clues, 67.9% Acc)", out_dir / "cbim_pondering_crystallization_k256_easy.gif", max_k=256)

    # 2. Hard Extreme Puzzle (Index 122, 17 clues, 38.3% Acc)
    inp_hard = torch.as_tensor(test_in[122:123], dtype=torch.long, device=device)
    lbl_hard = torch.as_tensor(test_lbl[122:123], dtype=torch.long, device=device)
    generate_k256_animation(model, inp_hard, lbl_hard, "Hard 17-Clue Extreme (38.3% Acc)", out_dir / "cbim_pondering_crystallization_k256_hard.gif", max_k=256)


if __name__ == "__main__":
    main()
