"""True Phase-Space Fractal Basins of Attraction via Competing Constraint Slices.
Implements the exact non-linear phase-space slicing from arXiv:2609.04963:
Instead of perturbing along the trivial relaxation direction (which produces concentric circles),
we perturb along the COMPETING CONSTRAINT MANIFOLD:
- v1: Promotes Candidate A at bottleneck cell c1
- v2: Promotes the conflicting Candidate A at shared-unit cell c2 (row/col/box constraint clash)

This unleashes the full nonlinear fluid-like vortices, shear boundaries, and heteroclinic tangles!
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
import plotly.graph_objects as go

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


def find_conflicting_pair(inp_tensor: torch.Tensor, logits_0: torch.Tensor):
    """Find a pair of unassigned cells sharing a row, column, or 3x3 box with maximum ambiguity."""
    unassigned = (inp_tensor.view(81) <= 1).cpu().numpy()
    top2_vals, top2_idx = torch.topk(logits_0[:, 1:], k=2, dim=-1)
    gap = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
    gap[~unassigned] = 1e9

    # Sort unassigned cells by ambiguity (lowest gap)
    ambiguous_cells = np.argsort(gap)

    # Find the first pair that shares a row, column, or box
    for i in range(len(ambiguous_cells)):
        c1 = ambiguous_cells[i]
        if not unassigned[c1]:
            continue
        r1, col1 = c1 // 9, c1 % 9
        b1 = (r1 // 3) * 3 + (col1 // 3)

        for j in range(i + 1, len(ambiguous_cells)):
            c2 = ambiguous_cells[j]
            if not unassigned[c2]:
                continue
            r2, col2 = c2 // 9, c2 % 9
            b2 = (r2 // 3) * 3 + (col2 // 3)

            # Check if they share row, col, or box
            if r1 == r2 or col1 == col2 or b1 == b2:
                cand_digit = int(top2_idx[c1, 0].item() + 1)  # vocab index 2..10
                return c1, c2, cand_digit

    # Fallback to first two unassigned
    un_idx = np.where(unassigned)[0]
    return int(un_idx[0]), int(un_idx[1]), 2


@torch.no_grad()
def scan_conflict_slice(model, inp, lbl, resolution: int = 150, radius: float = 2.5, max_k: int = 24, batch_size: int = 128):
    """Scan the 2D competing constraint slice (u, v) and record loops to resolve L(u, v)."""
    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    f_0 = clue_field.clone()
    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1, 2, 3), keepdim=True)

    # Initial logits to identify conflicting pair
    logits_0 = model.model.readout_mlp(f_0.view(1, 81, model.model.d))[0]
    c1, c2, cand_digit = find_conflicting_pair(inp, logits_0)
    r1, col1 = c1 // 9, c1 % 9
    r2, col2 = c2 // 9, c2 % 9

    # Construct conflicting perturbation vectors v1, v2
    v1 = torch.zeros_like(f_0)
    v2 = torch.zeros_like(f_0)
    emb_digit = model.model.clue_proj(model.model.embed_tokens(torch.tensor([cand_digit], device=inp.device)).view(1, 1, 1, model.model.d))
    v1[0, r1, col1] = emb_digit[0, 0, 0]
    v2[0, r2, col2] = emb_digit[0, 0, 0]

    v1 = v1 / torch.linalg.vector_norm(v1.float(), dim=(1, 2, 3), keepdim=True)
    v2 = v2 / torch.linalg.vector_norm(v2.float(), dim=(1, 2, 3), keepdim=True)

    u_vals = np.linspace(-radius, radius, resolution)
    v_vals = np.linspace(-radius, radius, resolution)
    U, V = np.meshgrid(u_vals, v_vals)
    u_f = torch.as_tensor(U.ravel(), dtype=torch.float32, device=inp.device)
    v_f = torch.as_tensor(V.ravel(), dtype=torch.float32, device=inp.device)
    total_pts = len(u_f)

    p0_all = f_0 + u_f.view(total_pts, 1, 1, 1) * v1 + v_f.view(total_pts, 1, 1, 1) * v2
    p0_all = p0_all * (orig_norm / (torch.linalg.vector_norm(p0_all.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))
    cond_all = clue_field.expand(total_pts, -1, -1, -1)

    loops_grid = np.zeros(total_pts, dtype=np.float32)
    entropy_grid = np.zeros(total_pts, dtype=np.float32)

    for start in range(0, total_pts, batch_size):
        end = min(start + batch_size, total_pts)
        b = end - start
        curr = p0_all[start:end].clone()
        cond = cond_all[start:end]

        loop_counts = torch.full((b,), max_k, dtype=torch.float32, device=inp.device)
        has_decided = torch.zeros(b, dtype=torch.bool, device=inp.device)

        for k in range(1, max_k + 1):
            f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cond)

            cat_in = torch.cat([f_star, cond], dim=-1)
            v_k = model.model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            curr = torch.cos(alpha_k) * f_star + f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype)

            flat_k = curr.view(b, 81, model.model.d)
            logits_k = model.model.readout_mlp(flat_k)

            # Check if conflict between c1 and c2 is resolved
            pred_c1 = torch.argmax(logits_k[:, c1], dim=-1)
            pred_c2 = torch.argmax(logits_k[:, c2], dim=-1)
            probs_c1 = F.softmax(logits_k[:, c1], dim=-1)
            probs_c2 = F.softmax(logits_k[:, c2], dim=-1)
            conf_c1 = torch.max(probs_c1, dim=-1).values
            conf_c2 = torch.max(probs_c2, dim=-1).values

            # Resolution condition: predictions must not clash and confidence exceeds 0.50
            resolved = (conf_c1 >= 0.50) & (conf_c2 >= 0.50) & (pred_c1 != pred_c2) & (~has_decided)
            loop_counts[resolved] = k
            has_decided = has_decided | resolved

        # Final entropy of the conflicting pair
        ent_c1 = -torch.sum(probs_c1 * torch.log(probs_c1 + 1e-12), dim=-1)
        ent_c2 = -torch.sum(probs_c2 * torch.log(probs_c2 + 1e-12), dim=-1)
        mean_ent = 0.5 * (ent_c1 + ent_c2)

        loops_grid[start:end] = loop_counts.cpu().numpy()
        entropy_grid[start:end] = mean_ent.cpu().numpy()

    L = loops_grid.reshape(resolution, resolution)
    S = entropy_grid.reshape(resolution, resolution)
    return U, V, L, S, c1, c2, cand_digit, f_0, clue_field, v1, v2


@torch.no_grad()
def scan_3d_conflict_volume(model, inp, f_0, clue_field, v1, v2, c1, c2, res_3d: int = 32, radius: float = 2.5, max_k: int = 20, batch_size: int = 128):
    """Scan 3D conflict volume (u*v1, v*v2, w*v3) where v3 is a 3rd conflicting cell in the same box."""
    # Find a 3rd unassigned cell c3 in the same box or unit
    r1, col1 = c1 // 9, c1 % 9
    b1 = (r1 // 3) * 3 + (col1 // 3)
    unassigned = (inp.view(81) <= 1).cpu().numpy()

    c3 = c1
    for idx in range(81):
        if unassigned[idx] and idx != c1 and idx != c2:
            r, c = idx // 9, idx % 9
            b = (r // 3) * 3 + (c // 3)
            if b == b1 or r == r1 or c == col1:
                c3 = idx
                break

    r3, col3 = c3 // 9, c3 % 9
    v3 = torch.zeros_like(f_0)
    emb_digit = model.model.clue_proj(model.model.embed_tokens(torch.tensor([2], device=inp.device)).view(1, 1, 1, model.model.d))
    v3[0, r3, col3] = emb_digit[0, 0, 0]
    v3 = v3 / torch.linalg.vector_norm(v3.float(), dim=(1, 2, 3), keepdim=True)

    orig_norm = torch.linalg.vector_norm(f_0.float(), dim=(1, 2, 3), keepdim=True)

    u_vals = np.linspace(-radius, radius, res_3d)
    v_vals = np.linspace(-radius, radius, res_3d)
    w_vals = np.linspace(-radius, radius, res_3d)
    U, V, W = np.meshgrid(u_vals, v_vals, w_vals, indexing="ij")
    u_flat = U.ravel()
    v_flat = V.ravel()
    w_flat = W.ravel()
    total_pts = len(u_flat)

    u_f = torch.as_tensor(u_flat, dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    v_f = torch.as_tensor(v_flat, dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)
    w_f = torch.as_tensor(w_flat, dtype=torch.float32, device=inp.device).view(-1, 1, 1, 1)

    vol_loops = np.zeros(total_pts, dtype=np.float32)

    for start in range(0, total_pts, batch_size):
        end = min(start + batch_size, total_pts)
        b = end - start
        ub = u_f[start:end]
        vb = v_f[start:end]
        wb = w_f[start:end]
        p0_b = f_0 + ub * v1 + vb * v2 + wb * v3
        p0_b = p0_b * (orig_norm / (torch.linalg.vector_norm(p0_b.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))
        curr = p0_b.to(f_0.dtype)
        cond = clue_field.expand(b, -1, -1, -1)

        loop_counts = torch.full((b,), max_k, dtype=torch.float32, device=inp.device)
        has_decided = torch.zeros(b, dtype=torch.bool, device=inp.device)

        for k in range(1, max_k + 1):
            f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
            f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
            f_star = model.model.collision(f_tr, cond=cond)

            cat_in = torch.cat([f_star, cond], dim=-1)
            v_k = model.model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
            alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            curr = torch.cos(alpha_k) * f_star + f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype)

            flat_k = curr.view(b, 81, model.model.d)
            logits_k = model.model.readout_mlp(flat_k)

            pred_c1 = torch.argmax(logits_k[:, c1], dim=-1)
            pred_c2 = torch.argmax(logits_k[:, c2], dim=-1)
            probs_c1 = F.softmax(logits_k[:, c1], dim=-1)
            probs_c2 = F.softmax(logits_k[:, c2], dim=-1)
            conf_c1 = torch.max(probs_c1, dim=-1).values
            conf_c2 = torch.max(probs_c2, dim=-1).values

            resolved = (conf_c1 >= 0.50) & (conf_c2 >= 0.50) & (pred_c1 != pred_c2) & (~has_decided)
            loop_counts[resolved] = k
            has_decided = has_decided | resolved

        vol_loops[start:end] = loop_counts.cpu().numpy()

    L_3D = vol_loops.reshape(res_3d, res_3d, res_3d)
    return U, V, W, L_3D


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("   TRUE COMPETING-CONSTRAINT FRACTAL BASINS OF ATTRACTION")
    print("   Reveals Nonlinear Vortices, Shear Separators, and Multilevel Folding")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=24, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    puzzles = [
        ("Easy (35 Clues)", 909, 2.5),
        ("Medium (25 Clues)", 248, 2.5),
        ("Hard (17 Clues - Extreme)", 122, 2.5)
    ]

    # Exact colormap matching arXiv:2609.04963 (Black -> Navy -> Royal Blue -> Cyan -> Mint -> White)
    colors = ["#020208", "#081438", "#103468", "#165888", "#1d88a4", "#2ec4b6", "#70f3d0", "#ffffff"]
    fractal_cmap = LinearSegmentedColormap.from_list("trm_fractal_vortices", colors, N=256)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#060614")
    fig.suptitle("CBIM Nonlinear Fractal Basins of Attraction (Competing Constraint Manifold)", fontsize=18, color="#e0e8ff", y=0.98, fontweight="bold")

    results_data = {}

    for col_idx, (puz_name, puz_id, r_val) in enumerate(puzzles):
        print(f"\n[Scanning Competing Constraint Slice] {puz_name} (Index {puz_id})...", flush=True)
        inp = torch.as_tensor(test_in[puz_id:puz_id+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(test_lbl[puz_id:puz_id+1], dtype=torch.long, device=device)

        U, V, L, S, c1, c2, cand_d, f_0, clue_field, v1, v2 = scan_conflict_slice(model, inp, lbl, resolution=150, radius=r_val, max_k=20)
        results_data[puz_name] = (inp, f_0, clue_field, v1, v2, c1, c2)

        ax = axes[col_idx]
        ax.set_facecolor("#020208")

        # Smooth cubic interpolation for continuous fluid-like fractal lines
        im = ax.imshow(L, extent=[-r_val, r_val, -r_val, r_val], origin="lower", cmap=fractal_cmap, vmin=2, vmax=20, interpolation="bicubic")

        # Overlay contour streamlines to highlight the fluid-like vortex shear lines
        ax.contour(U, V, L, levels=6, colors="#ffffff", alpha=0.18, linewidths=0.75)

        r1, col1 = c1 // 9, c1 % 9
        r2, col2 = c2 // 9, c2 % 9
        ax.set_title(f"{puz_name}\nClash: Cell({r1},{col1}) vs Cell({r2},{col2})\nMean L: {np.mean(L):.1f} | Basin Entropy: {np.mean(S):.2f}", color="#d0e0ff", fontsize=12, pad=10)
        ax.set_xlabel("Perturbation $u \cdot \mathbf{v}_1$ (Promotes Candidate)", color="#90a8d0", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Perturbation $v \cdot \mathbf{v}_2$ (Conflicting Constraint)", color="#90a8d0", fontsize=11)
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#182850")

    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Loops to Converge ($L$)", color="#d0e0ff", fontsize=12)
    cbar.ax.tick_params(colors="#d0e0ff")

    out_png = out_dir / "cbim_fractal_basins_fli_final.png"
    plt.tight_layout(rect=[0.02, 0.05, 0.90, 0.94])
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSaved true fluid-like fractal basins figure to {out_png}")

    # Now compute 3D Volumetric Conflict Model for Hard Puzzle
    print("\n" + "=" * 80)
    print("   COMPUTING 3D VOLUMETRIC CONFLICT BASIN (Hard 17-Clue Extreme Puzzle)")
    print("=" * 80)
    inp_hard, f_0_h, clue_h, v1_h, v2_h, c1_h, c2_h = results_data["Hard (17 Clues - Extreme)"]
    U3, V3, W3, L_3D = scan_3d_conflict_volume(model, inp_hard, f_0_h, clue_h, v1_h, v2_h, c1_h, c2_h, res_3d=32, radius=2.5, max_k=20)

    print("Rendering Interactive 3D Plotly Model with Swirling Isosurfaces...")
    fig_3d = go.Figure(data=go.Isosurface(
        x=U3.flatten(),
        y=V3.flatten(),
        z=W3.flatten(),
        value=L_3D.flatten(),
        isomin=3,
        isomax=20,
        surface_count=8,
        colorscale=[
            [0.0, "#081438"],
            [0.2, "#103468"],
            [0.4, "#1d88a4"],
            [0.7, "#2ec4b6"],
            [1.0, "#ffffff"]
        ],
        colorbar=dict(
            title=dict(text="Loops to Converge (L)", font=dict(color="#ffffff", size=14)),
            tickfont=dict(color="#d0e0ff", size=12)
        ),
        caps=dict(x_show=False, y_show=False, z_show=False),
        slices_z=dict(show=True, locations=[-1.5, 0.0, 1.5]),
        opacity=0.70
    ))

    fig_3d.update_layout(
        title=dict(
            text="CBIM 3D Phase-Space Competing Constraint Vortices & Fractal Basins",
            font=dict(color="#ffffff", size=18)
        ),
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        scene=dict(
            xaxis=dict(title="Perturbation u (Cell 1)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            yaxis=dict(title="Perturbation v (Cell 2 - Conflict)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            zaxis=dict(title="Perturbation w (Cell 3 - Triplet)", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    out_3d_html = out_dir / "cbim_fractal_basins_3d_interactive.html"
    fig_3d.write_html(str(out_3d_html), include_plotlyjs="cdn")
    print(f"Saved interactive 3D conflict model to {out_3d_html}")


if __name__ == "__main__":
    main()
