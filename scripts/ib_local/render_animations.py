"""Comprehensive Animation Suite for CBIM Reasoning Dynamics:
Uses K (microstep pondering depth) and H (macrostep recurrence) as dynamic time axes.

Generates:
1. present/cbim_pondering_crystallization.gif & .html (9x9 Sudoku Board crystallization across K=1..48)
2. present/cbim_torus_wave_flow.gif & .html (3D Donut Torus wave streaming and periodic collision)
3. present/cbim_fractal_basin_time_evolution.gif & .html (Fractal basin bifurcation and vortex rollup over K=1..20)
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
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
import json

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead
from scripts.ib_local.render_true_fractal_basins import find_conflicting_pair


def torus_surface(n_r=9, n_c=9, R=3.0, r=1.2, interp=4):
    theta = np.linspace(0, 2*np.pi, n_r*interp, endpoint=True)
    phi = np.linspace(0, 2*np.pi, n_c*interp, endpoint=True)
    THETA, PHI = np.meshgrid(theta, phi)
    X = (R + r*np.cos(THETA)) * np.cos(PHI)
    Y = (R + r*np.cos(THETA)) * np.sin(PHI)
    Z = r*np.sin(THETA)
    return THETA, PHI, X, Y, Z


@torch.no_grad()
def generate_crystallization_animation(model, inp, lbl, out_dir: Path, max_k: int = 40):
    print("\n[1/3] Generating Board Crystallization Animation (K=1..40)...", flush=True)
    clues_mask = (inp.view(9, 9) > 1).cpu().numpy()
    lbl_arr = lbl.view(9, 9).cpu().numpy()

    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    curr = clue_field.clone()

    frames = []
    confs_history = []
    entropies_history = []
    accs_history = []

    # Custom colormap
    plasma_cmap = plt.cm.get_cmap("plasma")

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

        # Plot composite frame
        fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(13, 6), facecolor="#060614", gridspec_kw={"width_ratios": [1.1, 1.0]})
        fig.suptitle(f"CBIM Latent Pondering Dynamics (Time Axis K = {k:02d} / {max_k})", fontsize=16, color="#e0e8ff", y=0.96, fontweight="bold")

        # 1. Sudoku Grid
        ax_grid.set_facecolor("#040312")
        im = ax_grid.imshow(conf_map, cmap="plasma", vmin=0.15, vmax=1.0, origin="upper")
        ax_grid.set_title(f"Board Certainty: {mean_c*100:.1f}% | Acc: {cell_acc*100:.1f}%", color="#d0e0ff", fontsize=12, pad=8)

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
        steps_x = list(range(1, k + 1))
        ax_curve.plot(steps_x, [a*100 for a in accs_history], color="#38d8b8", linewidth=2.5, label="Cell Accuracy (%)")
        ax_curve.plot(steps_x, [c*100 for c in confs_history], color="#ffaa00", linewidth=2.0, linestyle="--", label="Mean Certainty (%)")

        ax2 = ax_curve.twinx()
        ax2.plot(steps_x, entropies_history, color="#ff4488", linewidth=2.0, label="Entropy S (nats)")
        ax2.set_ylabel("Shannon Entropy (nats)", color="#ff4488", fontsize=11)
        ax2.tick_params(colors="#ff4488")
        ax2.set_ylim(0.5, 2.3)

        ax_curve.set_xlim(1, max_k)
        ax_curve.set_ylim(20, 100)
        ax_curve.set_xlabel("Pondering Microstep (K)", color="#90a8d0", fontsize=11)
        ax_curve.set_ylabel("Accuracy & Confidence (%)", color="#38d8b8", fontsize=11)
        ax_curve.tick_params(colors="#7088b0")
        ax_curve.grid(True, color="#182850", alpha=0.6)
        for spine in ax_curve.spines.values():
            spine.set_color("#203560")

        # Combine legends
        lines1, labels1 = ax_curve.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_curve.legend(lines1 + lines2, labels1 + labels2, loc="lower right", facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff", fontsize=10)

        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])

        # Buffer to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    gif_path = out_dir / "cbim_pondering_crystallization.gif"
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0
    )
    print(f"Saved crystallization animation to {gif_path}", flush=True)


@torch.no_grad()
def generate_torus_wave_animation(model, inp, out_dir: Path, max_k: int = 24):
    print("\n[2/3] Generating 3D Donut Torus Wave Flow Animation (K=1..24)...", flush=True)
    from scipy.ndimage import map_coordinates

    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    curr = clue_field.clone()

    THETA, PHI, X, Y, Z = torus_surface(n_r=9, n_c=9, R=3.0, r=1.2, interp=4)
    r_coords = (THETA / (2 * np.pi) * 9.0) % 9.0
    c_coords = (PHI / (2 * np.pi) * 9.0) % 9.0

    frames = []

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

        # Interpolate energy onto torus
        energy_map = torch.sum(curr.float()**2, dim=-1)[0].cpu().numpy()
        energy_interp = map_coordinates(energy_map, [r_coords, c_coords], mode="wrap", order=3)
        energy_norm = (energy_interp - energy_interp.min()) / (energy_interp.max() - energy_interp.min() + 1e-8)

        # 3D Matplotlib Render with rotating azimuthal angle
        fig = plt.figure(figsize=(8, 7), facecolor="#060614")
        ax = fig.add_subplot(111, projection="3d", facecolor="#060614")
        fig.suptitle(f"3D Periodic Torus Wave Advection: K = {k:02d} / {max_k}", fontsize=15, color="#e0e8ff", y=0.95, fontweight="bold")

        azim_angle = (k * 6) % 360  # Rotate 6 deg per step
        ax.view_init(elev=35, azim=azim_angle)

        colors = plt.cm.viridis(energy_norm)
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_zlim(-2.5, 2.5)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    gif_path = out_dir / "cbim_torus_wave_flow.gif"
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Saved 3D Torus wave animation to {gif_path}", flush=True)


@torch.no_grad()
def generate_fractal_basin_time_animation(model, inp, out_dir: Path, max_k: int = 20, res: int = 100, radius: float = 2.5):
    print("\n[3/3] Generating Fractal Basin Time Evolution Animation (K=1..20)...", flush=True)
    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    f_0 = clue_field.clone()
    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1,2,3), keepdim=True)

    logits_0 = model.model.readout_mlp(f_0.view(1, 81, model.model.d))[0]
    c1, c2, cand_d = find_conflicting_pair(inp, logits_0)
    r1, col1 = c1 // 9, c1 % 9
    r2, col2 = c2 // 9, c2 % 9

    v1 = torch.zeros_like(f_0)
    v2 = torch.zeros_like(f_0)
    emb_d = model.model.clue_proj(model.model.embed_tokens(torch.tensor([cand_d], device=inp.device)).view(1,1,1,model.model.d))
    v1[0, r1, col1] = emb_d[0, 0, 0]
    v2[0, r2, col2] = emb_d[0, 0, 0]
    v1 = v1 / torch.linalg.vector_norm(v1.float(), dim=(1,2,3), keepdim=True)
    v2 = v2 / torch.linalg.vector_norm(v2.float(), dim=(1,2,3), keepdim=True)

    u_vals = np.linspace(-radius, radius, res)
    v_vals = np.linspace(-radius, radius, res)
    U, V = np.meshgrid(u_vals, v_vals)
    u_f = torch.as_tensor(U.ravel(), dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    v_f = torch.as_tensor(V.ravel(), dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    total_pts = len(u_f)

    colors = ["#020208", "#081438", "#103468", "#165888", "#1d88a4", "#2ec4b6", "#70f3d0", "#ffffff"]
    fractal_cmap = LinearSegmentedColormap.from_list("trm_fractal_vortices", colors, N=256)

    frames = []

    # Run step by step across all points in batches of 128
    batch_sz = 128
    p0_all = f_0 + u_f * v1 + v_f * v2
    p0_all = p0_all * (orig_norm / (torch.linalg.vector_norm(p0_all.float(), dim=(1,2,3), keepdim=True) + 1e-8))

    curr = p0_all.clone()
    cond = clue_field.expand(total_pts, -1, -1, -1)

    for k in range(1, max_k + 1):
        # Step all batches
        for s in range(0, total_pts, batch_sz):
            e = min(s + batch_sz, total_pts)
            b = e - s
            cb = curr[s:e]
            cd = cond[s:e]

            f_5d = cb.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cd)
            cat_in = torch.cat([f_star, cd], dim=-1)
            v_k = model.model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
            norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr[s:e] = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        # Readout decision gap between conflicting cells
        flat = curr.view(total_pts, 81, model.model.d)
        logits = model.model.readout_mlp(flat)
        probs_c1 = F.softmax(logits[:, c1], dim=-1)[:, cand_d]
        probs_c2 = F.softmax(logits[:, c2], dim=-1)[:, cand_d]
        diff_prob = (probs_c1 - probs_c2).cpu().numpy().reshape(res, res)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#060614")
        fig.suptitle(f"Phase-Space Shear Rollup (Time Axis K = {k:02d} / {max_k})", fontsize=15, color="#e0e8ff", y=0.96, fontweight="bold")
        ax.set_facecolor("#020208")

        im = ax.imshow(diff_prob, extent=[-radius, radius, -radius, radius], origin="lower", cmap="coolwarm", vmin=-0.8, vmax=0.8, interpolation="bicubic")
        ax.contour(U, V, diff_prob, levels=7, colors="#ffffff", alpha=0.3, linewidths=0.8)

        ax.set_title(f"Clash: Cell({r1},{col1}) vs Cell({r2},{col2}) | Step {k}", color="#d0e0ff", fontsize=12, pad=8)
        ax.set_xlabel("Perturbation $u \cdot \mathbf{v}_1$ (Cell 1)", color="#90a8d0", fontsize=11)
        ax.set_ylabel("Perturbation $v \cdot \mathbf{v}_2$ (Cell 2 - Conflict)", color="#90a8d0", fontsize=11)
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#182850")

        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    gif_path = out_dir / "cbim_fractal_basin_time_evolution.gif"
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Saved fractal basin time evolution animation to {gif_path}", flush=True)


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("   CBIM MULTI-AXIS DYNAMIC ANIMATION GENERATOR (K & H TIME AXES)")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=32, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    # Hard extreme puzzle 122 (17 clues)
    inp = torch.as_tensor(test_in[122:123], dtype=torch.long, device=device)
    lbl = torch.as_tensor(test_lbl[122:123], dtype=torch.long, device=device)

    # 1. Board Crystallization Animation along K axis
    generate_crystallization_animation(model, inp, lbl, out_dir, max_k=40)

    # 2. 3D Donut Torus Wave Advection Animation along K axis
    generate_torus_wave_animation(model, inp, out_dir, max_k=24)

    # 3. Fractal Basin Time Evolution along K axis
    generate_fractal_basin_time_animation(model, inp, out_dir, max_k=20, res=80, radius=2.5)

    print("\n" + "=" * 80)
    print("   ALL DYNAMIC ANIMATIONS COMPLETED AND SAVED TO present/")
    print("=" * 80)


if __name__ == "__main__":
    main()
