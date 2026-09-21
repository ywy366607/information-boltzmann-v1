"""3D Periodic Torus Wave Propagation and Intra-Step Operator Dynamics Visualization.
Embeds the 9x9 discrete periodic spatial grid onto a continuous 3D Donut Torus:
  X = (R + r*cos(theta)) * cos(phi)
  Y = (R + r*cos(theta)) * sin(phi)
  Z = r * sin(theta)
Visualizes:
1. Transport T: 8-velocity wave packet streaming across periodic boundaries
2. Collision C: Local Lie algebra Givens rotations & nonlinear interference
3. Corrective Flow: Tangent relaxation vector and adaptive geodesic rotation
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
import plotly.graph_objects as go

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


def torus_surface_coordinates(n_r: int = 9, n_c: int = 9, R: float = 3.0, r: float = 1.2, interp_factor: int = 4):
    """Generate smooth 3D coordinates for the periodic donut torus."""
    theta_vals = np.linspace(0, 2 * np.pi, n_r * interp_factor, endpoint=True)
    phi_vals = np.linspace(0, 2 * np.pi, n_c * interp_factor, endpoint=True)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals)

    X = (R + r * np.cos(THETA)) * np.cos(PHI)
    Y = (R + r * np.cos(THETA)) * np.sin(PHI)
    Z = r * np.sin(THETA)
    return THETA, PHI, X, Y, Z


@torch.no_grad()
def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=16, arm="corrective_flow").to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    # Hard extreme puzzle 122
    inp = torch.as_tensor(test_in[122:123], dtype=torch.long, device=device)
    lbl = torch.as_tensor(test_lbl[122:123], dtype=torch.long, device=device)

    # Embed initial clues
    clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
    curr = clue_field.clone()

    # Step 1: Evolve to a contentious microstep k=8
    print("[Extracting Intra-Step Wave Fields at Microstep k=8]...", flush=True)
    for k in range(1, 8):
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

    # At microstep 8: Capture the 4 internal operator sub-stages
    # Stage 1: Prior State F_{k-1}
    state_0_energy = torch.sum(curr.float() ** 2, dim=-1)[0].cpu().numpy()  # [9, 9]

    # Stage 2: Transport T
    f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
    f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
    state_tr_energy = torch.sum(f_tr.float() ** 2, dim=-1)[0].cpu().numpy()  # [9, 9]

    # Directional momentum along row and col
    v_dirs = torch.tensor([
        [0, 1], [1, 1], [1, 0], [1, -1],
        [0, -1], [-1, -1], [-1, 0], [-1, 1]
    ], dtype=torch.float32, device=device)  # [8, 2]
    v_energies = torch.sum(f_5d.float() ** 2, dim=-1)[0]  # [9, 9, 8]
    p_row = torch.sum(v_energies * v_dirs[:, 0].view(1, 1, 8), dim=-1).cpu().numpy()
    p_col = torch.sum(v_energies * v_dirs[:, 1].view(1, 1, 8), dim=-1).cpu().numpy()

    # Stage 3: Collision C
    f_star = model.model.collision(f_tr, cond=clue_field)
    state_col_energy = torch.sum(f_star.float() ** 2, dim=-1)[0].cpu().numpy()
    collision_shear = torch.abs(f_star.float() - f_tr.float()).mean(dim=-1)[0].cpu().numpy()

    # Stage 4: Corrective Flow
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
    curr_next = (torch.cos(alpha_k) * f_star) + (f_norm * torch.sin(alpha_k) * u_hat.to(curr.dtype))

    corrective_magnitude = torch.linalg.vector_norm(u_k_f32, dim=-1)[0].cpu().numpy()

    # 1. Publication Figure: 4-Panel Intra-Step Wave Dynamics
    print("Generating 4-Panel 2D Intra-Step Figure...", flush=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor="#060614")
    fig.suptitle("CBIM Intra-Microstep Operator Wave Dynamics (Microstep k=8)", fontsize=18, color="#e0e8ff", y=0.98, fontweight="bold")

    cmap = "magma"

    # Panel 1: State Energy F_{k-1}
    im0 = axes[0].imshow(state_0_energy, cmap=cmap, origin="upper")
    axes[0].set_title("1. Prior Wave Field $F_{k-1}$\n(Total Spatial Energy)", color="#d0e0ff", fontsize=12)
    axes[0].set_facecolor("#040312")

    # Panel 2: Transport Wave Momentum Vectors
    im1 = axes[1].imshow(state_tr_energy, cmap=cmap, origin="upper")
    axes[1].quiver(np.arange(9), np.arange(9), p_col, p_row, color="#00ffcc", alpha=0.85, scale=15.0, width=0.012)
    axes[1].set_title("2. Advection $\mathcal{T}(F)$\n(8-Velocity Wave Streaming)", color="#d0e0ff", fontsize=12)
    axes[1].set_facecolor("#040312")

    # Panel 3: Collision Shear / Nonlinear Mixing
    im2 = axes[2].imshow(collision_shear, cmap="viridis", origin="upper")
    axes[2].set_title("3. Collision $\mathcal{C}(F_{\\rm tr}; P)$\n(Lie Algebra Nonlinear Shear)", color="#d0e0ff", fontsize=12)
    axes[2].set_facecolor("#040312")

    # Panel 4: Corrective Flow Relaxation Force
    im3 = axes[3].imshow(corrective_magnitude, cmap="plasma", origin="upper")
    axes[3].set_title(f"4. Corrective Flow $G_\\phi$\n(Tangent Relaxation $\\alpha={float(alpha_k.item()):.3f}$)", color="#d0e0ff", fontsize=12)
    axes[3].set_facecolor("#040312")

    for ax in axes:
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.tick_params(colors="#7088b0")
        for spine in ax.spines.values():
            spine.set_color("#203560")

    out_png = out_dir / "cbim_intra_step_wave_dynamics.png"
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved intra-step wave dynamics figure to {out_png}")

    # 2. Interactive 3D Donut Torus Embedding in Plotly
    print("Generating Interactive 3D Donut Torus Embedding...", flush=True)
    THETA, PHI, X, Y, Z = torus_surface_coordinates(n_r=9, n_c=9, R=3.0, r=1.2, interp_factor=6)

    # Interpolate 9x9 field onto the fine torus mesh (periodic wrap)
    from scipy.ndimage import map_coordinates
    n_theta, n_phi = THETA.shape
    r_coords = (THETA / (2 * np.pi) * 9.0) % 9.0
    c_coords = (PHI / (2 * np.pi) * 9.0) % 9.0

    energy_interp = map_coordinates(state_tr_energy, [r_coords, c_coords], mode="wrap", order=3)

    fig_torus = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=energy_interp,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="Wave Energy |F_tr|^2", font=dict(color="#ffffff", size=14)),
            tickfont=dict(color="#d0e0ff", size=12)
        ),
        lighting=dict(ambient=0.65, diffuse=0.8, specular=0.5, roughness=0.3)
    )])

    fig_torus.update_layout(
        title=dict(
            text="CBIM 3D Periodic Torus Embedding: Wave Streaming on T^2 = S^1 x S^1",
            font=dict(color="#ffffff", size=18)
        ),
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            yaxis=dict(title="Y", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            zaxis=dict(title="Z", backgroundcolor="#080a1c", color="#a0b8e0", gridcolor="#182850"),
            aspectratio=dict(x=1.2, y=1.2, z=0.6),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    out_torus_html = out_dir / "cbim_torus_wave_propagation_3d.html"
    fig_torus.write_html(str(out_torus_html), include_plotlyjs="cdn")
    print(f"Saved interactive 3D Donut Torus model to {out_torus_html}")


if __name__ == "__main__":
    main()
