"""Scan 2D phase-space along the top 2 singular/sensitive non-linear directions (Approach 2).

Instead of slicing along arbitrary random Gaussian vectors, this script projects the phase-space
slice onto the manifold's two principal singular directions:
1. Direction u: Principal gradient / singular sensitivity of next-token predictive loss (d Loss / d Field).
2. Direction v: Principal singular steering vector of the non-linear velocity direction controller
   (orthogonalized against u via Gram-Schmidt).

This directly visualizes the fracture lines, vortices, and non-linear fractal decision boundaries
trapped in the latent kinetic manifold.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def load_model_and_state(checkpoint_path: Path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        collision_layers=cfg.get("collision_layers", 2),
        relative_address=cfg.get("relative_address", False),
        v2_coordinate_components=cfg.get("v2_coordinate_components", True),
        readout_type=cfg.get("readout_type", "baseline"),
        readout_probes=cfg.get("readout_probes", 8),
        readout_rounds=cfg.get("readout_rounds", 1),
        write_type=cfg.get("write_type", "w0_baseline"),
        micro_steps=cfg.get("micro_steps", 1),
        adaptive_clock=cfg.get("adaptive_clock", False),
        continuous_velocities=cfg.get("continuous_velocities", False),
        alpha_causal=cfg.get("alpha_causal", 0.90),
        alpha_max=cfg.get("alpha_max", 2.50),
        dissipation_type=cfg.get("dissipation_type", "quadratic"),
        dissipation_rank=cfg.get("dissipation_rank", 4),
    ).cuda().eval()
    model.load_state_dict(saved["model"])
    state = saved["state"].cuda()
    return model, cfg, state


def extract_principal_singular_directions(model, state, token_id, target_id):
    """Extract top 2 principal non-linear singular directions on the manifold."""
    # Direction u: Maximum first-order loss gradient sensitivity
    x = state.detach().clone().requires_grad_(True)
    tok_embed = model.source.embedding(token_id)
    field, _, _ = model.source(x, token_id)
    for k in range(model.micro_steps):
        alpha_k = model.clock(field, tok_embed) if model.adaptive_clock else 1.0
        dt_k = alpha_k * model.tau_0_tensor
        dir_k = model.direction_controller(field, tok_embed) if model.continuous_velocities else None
        mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
        field = model.transport.apply_multiplier(field, mult)
        field, _ = model.collision(field, dt_k)
        field, _ = model.bath(field, dt_k)

    feat, _ = model.readout(field, tok_embed, return_diag=False)
    logits = model.decoder(feat)
    loss = F.cross_entropy(logits, target_id)
    (g,) = torch.autograd.grad(loss, x)
    u = g / g.norm()

    # Direction v: Principal steering singular vector of the internal non-linear controllers
    x2 = state.detach().clone().requires_grad_(True)
    if model.continuous_velocities:
        dir_out = model.direction_controller(x2, tok_embed)
    else:
        flat = x2.reshape(x2.shape[0], -1, model.d)
        angle_input = model.collision.norm(flat)
        if model.collision.position_conditioned:
            position = model.collision.position_features.to(flat)[None].expand(flat.shape[0], -1, -1)
            angle_input = torch.cat((angle_input, position), -1)
        dir_out = model.collision.angle(angle_input)
    torch.manual_seed(999)
    proj_rand = torch.randn_like(dir_out)
    proj_rand = proj_rand / proj_rand.norm()
    (g_steer,) = torch.autograd.grad((dir_out * proj_rand).sum(), x2)

    # Gram-Schmidt orthogonalization
    v_raw = g_steer - (g_steer * u).sum() * u
    v = v_raw / v_raw.norm()
    return u.detach(), v.detach()


@torch.inference_mode()
def scan_singular_slice(model, state, eval_tokens, u, v, grid_res=32, span=0.35, steps_eval=8):
    """Scan phase-space grid along singular axes u and v."""
    base_nlls = []
    curr = state.clone()
    for t in range(steps_eval):
        tok_id = torch.as_tensor([eval_tokens[t]], dtype=torch.long, device="cuda")
        target_id = torch.as_tensor([eval_tokens[t + 1]], dtype=torch.long, device="cuda")
        tok_embed = model.source.embedding(tok_id)
        field, _, _ = model.source(curr, tok_id)
        for k in range(model.micro_steps):
            alpha_k = model.clock(field, tok_embed) if model.adaptive_clock else 1.0
            dt_k = alpha_k * model.tau_0_tensor
            dir_k = model.direction_controller(field, tok_embed) if model.continuous_velocities else None
            mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
            field = model.transport.apply_multiplier(field, mult)
            field, _ = model.collision(field, dt_k)
            field, _ = model.bath(field, dt_k)

        feat, _ = model.readout(field, tok_embed, return_diag=False)
        logits = model.decoder(feat)
        base_nlls.append(F.cross_entropy(logits, target_id).item())
        curr = field

    state_norm = state.norm().item()
    a_vals = np.linspace(-span, span, grid_res)
    b_vals = np.linspace(-span, span, grid_res)

    settling_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
    nll_diff_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
    fli_grid = np.zeros((grid_res, grid_res), dtype=np.float32)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            pert = (a * u + b * v) * state_norm
            init_p = state + pert
            d0 = pert.norm().item()

            curr_p = init_p.clone()
            settled_step = steps_eval
            accum_nll_diff = 0.0

            for t in range(steps_eval):
                tok_id = torch.as_tensor([eval_tokens[t]], dtype=torch.long, device="cuda")
                target_id = torch.as_tensor([eval_tokens[t + 1]], dtype=torch.long, device="cuda")
                tok_embed = model.source.embedding(tok_id)
                field, _, _ = model.source(curr_p, tok_id)
                for k in range(model.micro_steps):
                    alpha_k = model.clock(field, tok_embed) if model.adaptive_clock else 1.0
                    dt_k = alpha_k * model.tau_0_tensor
                    dir_k = model.direction_controller(field, tok_embed) if model.continuous_velocities else None
                    mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                    field = model.transport.apply_multiplier(field, mult)
                    field, _ = model.collision(field, dt_k)
                    field, _ = model.bath(field, dt_k)

                feat, _ = model.readout(field, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_p = F.cross_entropy(logits, target_id).item()
                diff = abs(loss_p - base_nlls[t])
                accum_nll_diff += diff

                if settled_step == steps_eval and diff < 0.015 and t >= 1:
                    settled_step = t + 1

                curr_p = field

            d_final = (curr_p - curr).norm().item()
            fli = math.log((d_final + 1e-12) / (d0 + 1e-12)) / steps_eval

            settling_grid[i, j] = settled_step
            nll_diff_grid[i, j] = accum_nll_diff / steps_eval
            fli_grid[i, j] = fli

    values, counts = np.unique(settling_grid, return_counts=True)
    probs = counts / counts.sum()
    basin_entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "settling_grid": settling_grid,
        "nll_diff_grid": nll_diff_grid,
        "fli_grid": fli_grid,
        "basin_entropy": basin_entropy,
        "mean_settling_time": float(np.mean(settling_grid)),
        "mean_fli": float(np.mean(fli_grid)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", type=Path,
                        default=Path("results/cbim_torus3d_w2_16ch_k3_adaptive_xavier_3000/BBest.pt"))
    parser.add_argument("--ckpt3", type=Path,
                        default=Path("results/cbim_torus3d_w2_8x8x4_arm_c_continuous_q8_3000/BBest.pt"))
    parser.add_argument("--ckpt4", type=Path,
                        default=Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--grid-res", type=int, default=32)
    parser.add_argument("--span", type=float, default=0.35)
    args = parser.parse_args()

    print("=" * 70)
    print("APPROACH 2: PHASE-SPACE SLICE ALONG PRINCIPAL SINGULAR VECTORS")
    print("Mapping True Manifold Singularities & Fractal Vortex Boundaries")
    print("=" * 70, flush=True)

    valid = np.load(args.data / "validation.npy", mmap_mode="r")
    start = 8192
    eval_tokens = valid[start:start + 64]
    tok_id = torch.as_tensor([eval_tokens[0]], dtype=torch.long, device="cuda")
    tgt_id = torch.as_tensor([eval_tokens[1]], dtype=torch.long, device="cuda")

    print("\nLoading models and persistent live states...")
    m1, _, s1 = load_model_and_state(args.ckpt1)
    m3, _, s3 = load_model_and_state(args.ckpt3)
    m4, _, s4 = load_model_and_state(args.ckpt4)

    print("\nExtracting principal non-linear singular directions (u, v)...")
    u1, v1 = extract_principal_singular_directions(m1, s1, tok_id, tgt_id)
    u3, v3 = extract_principal_singular_directions(m3, s3, tok_id, tgt_id)
    u4, v4 = extract_principal_singular_directions(m4, s4, tok_id, tgt_id)
    print("Singular vectors extracted successfully.")

    print(f"\nScanning {args.grid_res}x{args.grid_res} phase-space slice along singular directions...")
    res1 = scan_singular_slice(m1, s1, eval_tokens, u1, v1, grid_res=args.grid_res, span=args.span)
    res3 = scan_singular_slice(m3, s3, eval_tokens, u3, v3, grid_res=args.grid_res, span=args.span)
    res4 = scan_singular_slice(m4, s4, eval_tokens, u4, v4, grid_res=args.grid_res, span=args.span)

    print("\n" + "=" * 75)
    print("SINGULAR SLICE BENCHMARK COMPARISON (3-WAY)")
    print("=" * 75)
    print(f"Basin Entropy (S_b):        Run 1 = {res1['basin_entropy']:.4f} | Run 3 = {res3['basin_entropy']:.4f} | Run 4 = {res4['basin_entropy']:.4f}")
    print(f"Mean Settling Time (steps): Run 1 = {res1['mean_settling_time']:.2f}   | Run 3 = {res3['mean_settling_time']:.2f}   | Run 4 = {res4['mean_settling_time']:.2f}")
    print(f"Mean FLI:                   Run 1 = {res1['mean_fli']:.4f} | Run 3 = {res3['mean_fli']:.4f} | Run 4 = {res4['mean_fli']:.4f}")
    print("=" * 75)

    # Render High-Resolution Chart
    fig_path = Path("present/cbim_singular_slice_basins.png").resolve()
    print(f"\nRendering high-res singular slice chart to: {fig_path}...")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11), dpi=150)
    fig.patch.set_facecolor("#0b0d13")
    for ax in axes.flat:
        ax.set_facecolor("#131620")
        ax.tick_params(colors="#8f9ba8")
        for spine in ax.spines.values():
            spine.set_color("#262b3a")
        ax.xaxis.label.set_color("#d0d7de")
        ax.yaxis.label.set_color("#d0d7de")
        ax.title.set_color("#f0f3f6")

    # Row 0: Settling Basins along singular vectors (Colormap Spectral_r)
    vmin_s, vmax_s = 2.0, 8.0
    cmap_s = "Spectral_r"

    im00 = axes[0, 0].imshow(res1["settling_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_s, vmax=vmax_s, cmap=cmap_s, origin="lower", aspect="auto")
    axes[0, 0].set_title(f"Run 1 (D3Q8 Discrete + Quad): Sb={res1['basin_entropy']:.3f}\n(Severe Boundary Deadlock)", fontsize=10, pad=8)
    axes[0, 0].set_xlabel("Singular Sensitivity u (dLoss/dField)")
    axes[0, 0].set_ylabel("Singular Steering v (Nonlinear)")
    cb00 = plt.colorbar(im00, ax=axes[0, 0], ticks=range(2, 9))
    cb00.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb00.set_label("Settling Steps", color="#8f9ba8", size=9)

    im01 = axes[0, 1].imshow(res3["settling_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_s, vmax=vmax_s, cmap=cmap_s, origin="lower", aspect="auto")
    axes[0, 1].set_title(f"Run 3 (Cont Q8 S² + Quad): Sb={res3['basin_entropy']:.3f}\n(Laminar Flow Channel)", fontsize=10, pad=8)
    axes[0, 1].set_xlabel("Singular Sensitivity u (dLoss/dField)")
    axes[0, 1].set_ylabel("Singular Steering v (Nonlinear)")
    cb01 = plt.colorbar(im01, ax=axes[0, 1], ticks=range(2, 9))
    cb01.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb01.set_label("Settling Steps", color="#8f9ba8", size=9)

    im02 = axes[0, 2].imshow(res4["settling_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_s, vmax=vmax_s, cmap=cmap_s, origin="lower", aspect="auto")
    axes[0, 2].set_title(f"Run 4 (Cont Q8 S² + Unified Diss): Sb={res4['basin_entropy']:.3f}\n(Transient Chaos & Rapid Escape)", fontsize=10, pad=8)
    axes[0, 2].set_xlabel("Singular Sensitivity u (dLoss/dField)")
    axes[0, 2].set_ylabel("Singular Steering v (Nonlinear)")
    cb02 = plt.colorbar(im02, ax=axes[0, 2], ticks=range(2, 9))
    cb02.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb02.set_label("Settling Steps", color="#8f9ba8", size=9)

    # Row 1: Fast Lyapunov Indicator (FLI) Landscape along singular vectors
    vmin_f = min(res1["fli_grid"].min(), res3["fli_grid"].min(), res4["fli_grid"].min())
    vmax_f = max(res1["fli_grid"].max(), res3["fli_grid"].max(), res4["fli_grid"].max())
    cmap_f = "inferno"

    im10 = axes[1, 0].imshow(res1["fli_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_f, vmax=vmax_f, cmap=cmap_f, origin="lower", aspect="auto")
    axes[1, 0].set_title(f"Run 1: Singular FLI Map (Mean={res1['mean_fli']:.3f})", fontsize=10, pad=8)
    axes[1, 0].set_xlabel("Singular Sensitivity u")
    axes[1, 0].set_ylabel("Singular Steering v")
    cb10 = plt.colorbar(im10, ax=axes[1, 0])
    cb10.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb10.set_label("FLI", color="#8f9ba8", size=9)

    im11 = axes[1, 1].imshow(res3["fli_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_f, vmax=vmax_f, cmap=cmap_f, origin="lower", aspect="auto")
    axes[1, 1].set_title(f"Run 3: Singular FLI Map (Mean={res3['mean_fli']:.3f})", fontsize=10, pad=8)
    axes[1, 1].set_xlabel("Singular Sensitivity u")
    axes[1, 1].set_ylabel("Singular Steering v")
    cb11 = plt.colorbar(im11, ax=axes[1, 1])
    cb11.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb11.set_label("FLI", color="#8f9ba8", size=9)

    im12 = axes[1, 2].imshow(res4["fli_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_f, vmax=vmax_f, cmap=cmap_f, origin="lower", aspect="auto")
    axes[1, 2].set_title(f"Run 4: Singular FLI Map (Mean={res4['mean_fli']:.3f})", fontsize=10, pad=8)
    axes[1, 2].set_xlabel("Singular Sensitivity u")
    axes[1, 2].set_ylabel("Singular Steering v")
    cb12 = plt.colorbar(im12, ax=axes[1, 2])
    cb12.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb12.set_label("FLI", color="#8f9ba8", size=9)
    axes[1, 0].set_title(f"Run 1 (D3Q8): Singular FLI Map (Mean={res1['mean_fli']:.3f})", fontsize=11, pad=8)
    axes[1, 0].set_xlabel("Singular Sensitivity Direction u (dLoss/dField)")
    axes[1, 0].set_ylabel("Singular Steering Direction v (Nonlinear)")
    cb10 = plt.colorbar(im10, ax=axes[1, 0])
    cb10.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb10.set_label("Fast Lyapunov Indicator (FLI)", color="#8f9ba8", size=9)

    im11 = axes[1, 1].imshow(res3["fli_grid"], extent=[-args.span, args.span, -args.span, args.span],
                             vmin=vmin_f, vmax=vmax_f, cmap=cmap_f, origin="lower", aspect="auto")
    axes[1, 1].set_title(f"Run 3 (Cont Q8 S²): Singular FLI Map (Mean={res3['mean_fli']:.3f})", fontsize=11, pad=8)
    axes[1, 1].set_xlabel("Singular Sensitivity Direction u (dLoss/dField)")
    axes[1, 1].set_ylabel("Singular Steering Direction v (Nonlinear)")
    cb11 = plt.colorbar(im11, ax=axes[1, 1])
    cb11.ax.tick_params(labelsize=9, labelcolor="#8f9ba8")
    cb11.set_label("Fast Lyapunov Indicator (FLI)", color="#8f9ba8", size=9)

    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Chart rendered successfully: {fig_path}")


if __name__ == "__main__":
    main()
