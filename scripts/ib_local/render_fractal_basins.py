"""Phase-Space Fractal Basins of Attraction Visualization on Sudoku-Extreme.
Reproduces and extends the 2D fractal basins from https://arxiv.org/abs/2609.04963
and creates an interactive 3D volumetric phase-space visualization.

Computes:
1. 2D Phase-space slices along top 2 PCA trajectory directions for Easy, Medium, Hard puzzles.
2. Convergence loops L(u, v), Fast Lyapunov Indicator (FLI), and Basin Entropy.
3. 3D Volumetric phase-space grid (u, v, w) and interactive 3D HTML isosurface model.
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

import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


def load_champion_model(device: str = "cuda"):
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=32, arm="corrective_flow").to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def compute_trajectory_pca(model, inp_tensor: torch.Tensor, lbl_tensor: torch.Tensor, k_steps: int = 32):
    """Run unperturbed trajectory and extract top 3 PCA directions of phase-space motion."""
    model.model.ponder_steps = k_steps
    batch = {
        "inputs": inp_tensor,
        "labels": lbl_tensor,
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=inp_tensor.device)
    }

    # Step-by-step trace
    curr = model.model.clue_proj(model.model.embed_tokens(inp_tensor).view(1, 9, 9, model.model.d))
    clue_field = curr.clone()
    f_0 = curr.clone()

    trajectory = [curr.reshape(-1).cpu().numpy()]
    for k in range(1, k_steps + 1):
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
        u_norm = torch.linalg.vector_norm(u_k_f32, dim=(1, 2, 3), keepdim=True)
        u_hat = u_k_f32 / (u_norm + 1e-8)

        gate_logits = model.model.angle_gate(cat_input)
        alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(gate_logits, dim=(1, 2, 3), keepdim=True)).to(curr.dtype)

        f_norm = torch.linalg.vector_norm(f_star_f32, dim=(1, 2, 3), keepdim=True).to(curr.dtype)
        curr = (torch.cos(alpha_k) * f_star) + (f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype))
        trajectory.append(curr.reshape(-1).cpu().numpy())

    traj_arr = np.array(trajectory)  # [K+1, 9*9*256]
    mean_traj = np.mean(traj_arr, axis=0, keepdims=True)
    centered = traj_arr - mean_traj

    # SVD for top principal components
    u_mat, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
    v1 = torch.as_tensor(vt[0], dtype=torch.float32, device=inp_tensor.device).view(1, 9, 9, model.model.d)
    v2 = torch.as_tensor(vt[1], dtype=torch.float32, device=inp_tensor.device).view(1, 9, 9, model.model.d)
    v3 = torch.as_tensor(vt[2], dtype=torch.float32, device=inp_tensor.device).view(1, 9, 9, model.model.d)

    return f_0, clue_field, v1, v2, v3


@torch.no_grad()
def scan_2d_phase_slice(model, f_0, clue_field, v1, v2, resolution: int = 140, radius: float = 15.0, max_k: int = 16, batch_size: int = 128):
    """Scan 2D perturbation slice (u, v) and record loops to converge L(u, v) and Lyapunov indicator."""
    u_vals = np.linspace(-radius, radius, resolution)
    v_vals = np.linspace(-radius, radius, resolution)
    U, V = np.meshgrid(u_vals, v_vals)
    u_flat = U.ravel()
    v_flat = V.ravel()
    total_pts = len(u_flat)

    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1, 2, 3), keepdim=True)

    # 1. Trace unperturbed baseline trajectory
    base_states = [f_0.clone()]
    curr_base = f_0.clone()
    for k in range(1, max_k + 1):
        f_5d = curr_base.view(1, 9, 9, model.model.n_v, model.model.d_c)
        f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
        f_star = model.model.collision(f_tr, cond=clue_field)
        cat_in = torch.cat([f_star, clue_field], dim=-1)
        v_k = model.model.corrective_net(cat_in)
        dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
        norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
        u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
        u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
        alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr_base.dtype)
        f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
        curr_base = torch.cos(alpha_k) * f_star + f_norm * torch.sin(alpha_k) * u_hat.to(curr_base.dtype)
        base_states.append(curr_base.clone())

    loops_to_converge = np.zeros(total_pts, dtype=np.float32)
    final_entropy = np.zeros(total_pts, dtype=np.float32)
    fli_values = np.zeros(total_pts, dtype=np.float32)

    for start in range(0, total_pts, batch_size):
        end = min(start + batch_size, total_pts)
        b = end - start
        u_b = torch.as_tensor(u_flat[start:end], dtype=torch.float32, device=f_0.device).view(b, 1, 1, 1)
        v_b = torch.as_tensor(v_flat[start:end], dtype=torch.float32, device=f_0.device).view(b, 1, 1, 1)

        # Initial perturbation
        pert_0 = f_0 + u_b * v1 + v_b * v2
        pert_norm = torch.linalg.vector_norm(pert_0.float(), dim=(1, 2, 3), keepdim=True)
        curr = (pert_0 * (orig_norm / (pert_norm + 1e-8))).to(f_0.dtype)
        cond = clue_field.expand(b, -1, -1, -1)

        converged_step = torch.full((b,), max_k, dtype=torch.float32, device=f_0.device)
        has_converged = torch.zeros((b,), dtype=torch.bool, device=f_0.device)
        init_dist = torch.linalg.vector_norm((pert_0 - f_0).float(), dim=(1, 2, 3)).clamp(min=1e-8)

        # Forward pondering
        for k in range(1, max_k + 1):
            f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cond)

            cat_input = torch.cat([f_star, cond], dim=-1)
            v_k = model.model.corrective_net(cat_input)

            f_star_f32 = f_star.float()
            v_k_f32 = v_k.float()
            dot_prod = torch.sum(f_star_f32 * v_k_f32, dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star_f32 ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            proj = dot_prod / norm_sq
            u_k_f32 = v_k_f32 - proj * f_star_f32
            u_norm = torch.linalg.vector_norm(u_k_f32, dim=(1, 2, 3), keepdim=True)
            u_hat = u_k_f32 / (u_norm + 1e-8)

            gate_logits = model.model.angle_gate(cat_input)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(gate_logits, dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star_f32, dim=(1, 2, 3), keepdim=True).to(curr.dtype)
            curr = (torch.cos(alpha_k) * f_star) + (f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype))

            # Reconvergence criterion: distance to unperturbed baseline < 1.0
            dist_to_base = torch.linalg.vector_norm((curr - base_states[k]).float(), dim=(1, 2, 3))
            is_close = (dist_to_base < 1.0) & (~has_converged)
            converged_step[is_close] = k
            has_converged = has_converged | is_close

        final_dist = torch.linalg.vector_norm((curr - f_0).float(), dim=(1, 2, 3)).clamp(min=1e-8)
        fli = torch.log10(final_dist / init_dist)

        # Entropy of final state
        flat_final = curr.view(b, 81, model.model.d)
        p_final = F.softmax(model.model.readout_mlp(flat_final), dim=-1)
        ent = -torch.sum(p_final * torch.log(p_final + 1e-12), dim=-1).mean(dim=-1)

        loops_to_converge[start:end] = converged_step.cpu().numpy()
        final_entropy[start:end] = ent.cpu().numpy()
        fli_values[start:end] = fli.cpu().numpy()

    L_grid = loops_to_converge.reshape(resolution, resolution)
    S_grid = final_entropy.reshape(resolution, resolution)
    FLI_grid = fli_values.reshape(resolution, resolution)
    return U, V, L_grid, S_grid, FLI_grid


@torch.no_grad()
def scan_3d_phase_volume(model, f_0, clue_field, v1, v2, v3, res_3d: int = 32, radius: float = 15.0, max_k: int = 16, batch_size: int = 128):
    """Scan 3D volumetric phase-space box (u, v, w) and compute 3D convergence scalar field."""
    u_vals = np.linspace(-radius, radius, res_3d)
    v_vals = np.linspace(-radius, radius, res_3d)
    w_vals = np.linspace(-radius, radius, res_3d)
    U, V, W = np.meshgrid(u_vals, v_vals, w_vals, indexing="ij")
    u_flat = U.ravel()
    v_flat = V.ravel()
    w_flat = W.ravel()
    total_pts = len(u_flat)

    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1, 2, 3), keepdim=True)

    # Trace base trajectory
    base_states = [f_0.clone()]
    curr_base = f_0.clone()
    for k in range(1, max_k + 1):
        f_5d = curr_base.view(1, 9, 9, model.model.n_v, model.model.d_c)
        f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
        f_star = model.model.collision(f_tr, cond=clue_field)
        cat_in = torch.cat([f_star, clue_field], dim=-1)
        v_k = model.model.corrective_net(cat_in)
        dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
        norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
        u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
        u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
        alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr_base.dtype)
        f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
        curr_base = torch.cos(alpha_k) * f_star + f_norm * torch.sin(alpha_k) * u_hat.to(curr_base.dtype)
        base_states.append(curr_base.clone())

    vol_loops = np.zeros(total_pts, dtype=np.float32)

    for start in range(0, total_pts, batch_size):
        end = min(start + batch_size, total_pts)
        b = end - start
        u_b = torch.as_tensor(u_flat[start:end], dtype=torch.float32, device=f_0.device).view(b, 1, 1, 1)
        v_b = torch.as_tensor(v_flat[start:end], dtype=torch.float32, device=f_0.device).view(b, 1, 1, 1)
        w_b = torch.as_tensor(w_flat[start:end], dtype=torch.float32, device=f_0.device).view(b, 1, 1, 1)

        pert_0 = f_0 + u_b * v1 + v_b * v2 + w_b * v3
        pert_norm = torch.linalg.vector_norm(pert_0.float(), dim=(1, 2, 3), keepdim=True)
        curr = (pert_0 * (orig_norm / (pert_norm + 1e-8))).to(f_0.dtype)
        cond = clue_field.expand(b, -1, -1, -1)

        converged_step = torch.full((b,), max_k, dtype=torch.float32, device=f_0.device)
        has_converged = torch.zeros((b,), dtype=torch.bool, device=f_0.device)

        for k in range(1, max_k + 1):
            f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cond)

            cat_input = torch.cat([f_star, cond], dim=-1)
            v_k = model.model.corrective_net(cat_input)

            f_star_f32 = f_star.float()
            v_k_f32 = v_k.float()
            dot_prod = torch.sum(f_star_f32 * v_k_f32, dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star_f32 ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            proj = dot_prod / norm_sq
            u_k_f32 = v_k_f32 - proj * f_star_f32
            u_norm = torch.linalg.vector_norm(u_k_f32, dim=(1, 2, 3), keepdim=True)
            u_hat = u_k_f32 / (u_norm + 1e-8)

            gate_logits = model.model.angle_gate(cat_input)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(gate_logits, dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star_f32, dim=(1, 2, 3), keepdim=True).to(curr.dtype)
            curr = (torch.cos(alpha_k) * f_star) + (f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype))

            dist_to_base = torch.linalg.vector_norm((curr - base_states[k]).float(), dim=(1, 2, 3))
            is_close = (dist_to_base < 1.0) & (~has_converged)
            converged_step[is_close] = k
            has_converged = has_converged | is_close

        vol_loops[start:end] = converged_step.cpu().numpy()

    L_3D = vol_loops.reshape(res_3d, res_3d, res_3d)
    return U, V, W, L_3D


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("   PHASE-SPACE FRACTAL BASINS OF ATTRACTION: 2D SLICES & 3D INTERACTIVE VOLUME")
    print("   Comparing Easy (35 clues), Medium (25 clues), Hard (17 clues)")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_champion_model(device)

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    puzzles = [
        ("Easy (35 Clues)", 909, 3.5),
        ("Medium (25 Clues)", 248, 3.5),
        ("Hard (17 Clues - Extreme)", 122, 3.5)
    ]

    # Custom colormap matching arXiv:2609.04963 (Black -> Deep Blue -> Cyan -> Emerald Green -> Light Mint)
    colors = ["#040312", "#0b1940", "#143c70", "#1b6d92", "#23a3a8", "#38d8b8", "#98f5d0", "#ffffff"]
    fractal_cmap = LinearSegmentedColormap.from_list("trm_fractal", colors, N=256)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#080814")
    fig.suptitle("CBIM Phase-Space Fractal Basins of Attraction (Loops to Converge L)", fontsize=18, color="#e0e8ff", y=0.98, fontweight="bold")

    results_2d = {}

    for col_idx, (puz_name, puz_id, r_val) in enumerate(puzzles):
        print(f"\n[Computing 2D Fractal Slice] {puz_name} (Index {puz_id})...", flush=True)
        inp = torch.as_tensor(test_in[puz_id:puz_id+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(test_lbl[puz_id:puz_id+1], dtype=torch.long, device=device)

        f_0, clue_field, v1, v2, v3 = compute_trajectory_pca(model, inp, lbl, k_steps=16)
        U, V, L_grid, S_grid, FLI_grid = scan_2d_phase_slice(model, f_0, clue_field, v1, v2, resolution=140, radius=r_val, max_k=16)

        results_2d[puz_name] = (U, V, L_grid, S_grid, FLI_grid, f_0, clue_field, v1, v2, v3)

        ax = axes[col_idx]
        ax.set_facecolor("#040312")
        im = ax.imshow(L_grid, extent=[-r_val, r_val, -r_val, r_val], origin="lower", cmap=fractal_cmap, vmin=1, vmax=16, interpolation="bicubic")
        ax.set_title(f"{puz_name}\nMean L: {np.mean(L_grid):.1f} | Fractal S: {np.mean(S_grid):.2f}", color="#d0e0ff", fontsize=13, pad=10)
        ax.set_xlabel("Perturbation Axis $u \cdot \mathbf{v}_1$", color="#90a8d0", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Perturbation Axis $v \cdot \mathbf{v}_2$", color="#90a8d0", fontsize=11)
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#203560")

    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Loops to Converge ($L$)", color="#d0e0ff", fontsize=12)
    cbar.ax.tick_params(colors="#d0e0ff")

    out_2d_png = out_dir / "cbim_fractal_basins_2d_easy_med_hard.png"
    plt.tight_layout(rect=[0.02, 0.05, 0.90, 0.94])
    plt.savefig(out_2d_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSaved 2D publication fractal figure to {out_2d_png}")

    # Now compute 3D Volumetric Fractal Basin for the Hard Puzzle (Extreme)
    print("\n" + "=" * 80)
    print("   COMPUTING 3D VOLUMETRIC FRACTAL BASIN (Hard 17-Clue Extreme Puzzle)")
    print("=" * 80)
    _, _, _, _, _, f_0_hard, clue_hard, v1_hard, v2_hard, v3_hard = results_2d["Hard (17 Clues - Extreme)"]
    U3, V3, W3, L_3D = scan_3d_phase_volume(model, f_0_hard, clue_hard, v1_hard, v2_hard, v3_hard, res_3d=32, radius=3.5, max_k=16)

    print("Rendering Interactive 3D Plotly Model with Isosurfaces & Volume Slices...")

    # Interactive 3D Isosurface Model
    fig_3d = go.Figure(data=go.Isosurface(
        x=U3.flatten(),
        y=V3.flatten(),
        z=W3.flatten(),
        value=L_3D.flatten(),
        isomin=2,
        isomax=16,
        surface_count=6,
        colorscale=[
            [0.0, "#0b1940"],
            [0.25, "#143c70"],
            [0.5, "#23a3a8"],
            [0.75, "#38d8b8"],
            [1.0, "#ffffff"]
        ],
        colorbar=dict(
            title=dict(text="Loops to Converge (L)", font=dict(color="#ffffff", size=14)),
            tickfont=dict(color="#d0e0ff", size=12)
        ),
        caps=dict(x_show=False, y_show=False, z_show=False),
        slices_z=dict(show=True, locations=[-0.2, 0.0, 0.2]),
        opacity=0.75
    ))

    fig_3d.update_layout(
        title=dict(
            text="CBIM 3D Phase-Space Fractal Basins of Attraction (Hard 17-Clue Extreme Puzzle)",
            font=dict(color="#ffffff", size=18)
        ),
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        scene=dict(
            xaxis=dict(title="Perturbation u (v1)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            yaxis=dict(title="Perturbation v (v2)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            zaxis=dict(title="Perturbation w (v3)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    out_3d_html = out_dir / "cbim_fractal_basins_3d_interactive.html"
    fig_3d.write_html(str(out_3d_html), include_plotlyjs="cdn")
    print(f"Saved interactive 3D volumetric model to {out_3d_html}")


if __name__ == "__main__":
    main()
