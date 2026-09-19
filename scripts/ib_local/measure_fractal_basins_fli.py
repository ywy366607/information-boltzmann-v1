"""Fractal basins and Fast Lyapunov Indicator (FLI) diagnostic for CBIM.

Implements the experimental protocol from 'Fractal basins trap latent reasoning' (Lai et al., 2026):
1. Fast Lyapunov Indicator (FLI): local divergence rate between perturbed field trajectories.
2. 2D Phase-space slice grid scan: z(a, b) = z_0 + a*u + b*v across a 2D orthonormal slice.
3. Settling-time landscape: number of micro/macro steps required for NLL/state to relax.
4. Basin entropy (S_b) and boundary complexity metrics.
5. Visual rendering to high-resolution PNG charts.
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


def load_model(checkpoint_path: Path):
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
    ).cuda().eval()
    model.load_state_dict(saved["model"])
    saved_state = saved["state"].cuda() if "state" in saved else None
    return model, cfg, saved_state


@torch.inference_mode()
def compute_fli_and_settling_curve(model, mature_state, eval_tokens, delta_scales=None, steps_horizon=32):
    """Compute Fast Lyapunov Indicator (FLI) and Settling Time vs initial perturbation magnitude.

    FLI_T = log( ||F_perturbed(T) - F_unperturbed(T)|| / ||F_perturbed(0) - F_unperturbed(0)|| )
    """
    if delta_scales is None:
        delta_scales = np.logspace(-5, -1, 10)  # from 1e-5 to 1e-1

    results = []
    # Base unperturbed trajectory
    base_states = [mature_state.clone()]
    base_nlls = []
    curr = mature_state.clone()
    for t in range(steps_horizon):
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
        if model.readout_type == "baseline":
            feat = model.readout(field)
        else:
            feat, _ = model.readout(field, tok_embed, return_diag=False)
        logits = model.decoder(feat)
        loss = F.cross_entropy(logits, target_id).item()
        base_nlls.append(loss)
        base_states.append(field.clone())
        curr = field

    # Scan perturbation magnitudes
    for delta in delta_scales:
        # Generate random perturbation on unit sphere in field space
        torch.manual_seed(42)
        perturb_dir = torch.randn_like(mature_state)
        perturb_dir = perturb_dir / perturb_dir.norm()
        perturbed_init = mature_state + delta * perturb_dir * mature_state.norm()
        d0 = (perturbed_init - mature_state).norm().item()

        curr_p = perturbed_init.clone()
        fli_curve = []
        nll_diff_curve = []
        settling_step = steps_horizon  # default to max horizon if never settled

        for t in range(steps_horizon):
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

            if model.readout_type == "baseline":
                feat = model.readout(field)
            else:
                feat, _ = model.readout(field, tok_embed, return_diag=False)
            logits = model.decoder(feat)
            loss_p = F.cross_entropy(logits, target_id).item()

            dt_norm = (field - base_states[t + 1]).norm().item()
            fli_val = math.log((dt_norm + 1e-12) / d0) / (t + 1)
            fli_curve.append(fli_val)

            nll_diff = abs(loss_p - base_nlls[t])
            nll_diff_curve.append(nll_diff)

            # Settling criterion: NLL discrepancy drops below 0.01 nats
            if settling_step == steps_horizon and nll_diff < 0.01 and t >= 3:
                settling_step = t + 1

            curr_p = field

        results.append({
            "delta": float(delta),
            "d0": float(d0),
            "max_fli": float(np.max(fli_curve)),
            "final_fli": float(fli_curve[-1]),
            "settling_step": int(settling_step),
            "fli_curve": [float(v) for v in fli_curve],
            "nll_diff_curve": [float(v) for v in nll_diff_curve],
        })

    return results


@torch.inference_mode()
def scan_2d_phase_space_slice(model, mature_state, eval_tokens, grid_res=40, span=0.30, steps_eval=8):
    """Scan a 2D orthonormal slice z(a, b) = z_0 + a*u + b*v in persistent field space.

    Computes for each (a, b) grid point:
    1. Settling Time: number of steps until NLL matches the unperturbed baseline within epsilon.
    2. Final NLL deviation: |NLL_perturbed - NLL_unperturbed|.
    3. Fast Lyapunov Indicator FLI.
    """
    # 1. Unperturbed baseline trajectory
    base_nlls = []
    curr = mature_state.clone()
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
        if model.readout_type == "baseline":
            feat = model.readout(field)
        else:
            feat, _ = model.readout(field, tok_embed, return_diag=False)
        logits = model.decoder(feat)
        base_nlls.append(F.cross_entropy(logits, target_id).item())
        curr = field

    # 2. Orthonormal directions u, v via Gram-Schmidt
    torch.manual_seed(101)
    u_raw = torch.randn_like(mature_state)
    u = u_raw / u_raw.norm()
    v_raw = torch.randn_like(mature_state)
    v_proj = v_raw - (v_raw * u).sum() * u
    v = v_proj / v_proj.norm()

    state_norm = mature_state.norm().item()
    a_vals = np.linspace(-span, span, grid_res)
    b_vals = np.linspace(-span, span, grid_res)

    settling_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
    nll_diff_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
    fli_grid = np.zeros((grid_res, grid_res), dtype=np.float32)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            pert = (a * u + b * v) * state_norm
            init_p = mature_state + pert
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

                if model.readout_type == "baseline":
                    feat = model.readout(field)
                else:
                    feat, _ = model.readout(field, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_p = F.cross_entropy(logits, target_id).item()
                diff = abs(loss_p - base_nlls[t])
                accum_nll_diff += diff

                if settled_step == steps_eval and diff < 0.02 and t >= 1:
                    settled_step = t + 1

                curr_p = field

            d_final = (curr_p - curr).norm().item()
            fli = math.log((d_final + 1e-12) / (d0 + 1e-12)) / steps_eval

            settling_grid[i, j] = settled_step
            nll_diff_grid[i, j] = accum_nll_diff / steps_eval
            fli_grid[i, j] = fli

    # Compute Basin Entropy S_b: Shannon entropy of discrete settling steps in the 2D grid
    values, counts = np.unique(settling_grid, return_counts=True)
    probs = counts / counts.sum()
    basin_entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "a_vals": a_vals.tolist(),
        "b_vals": b_vals.tolist(),
        "settling_grid": settling_grid.tolist(),
        "nll_diff_grid": nll_diff_grid.tolist(),
        "fli_grid": fli_grid.tolist(),
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
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, default=Path("results/published/cbim_fractal_basins_fli_report.json"))
    parser.add_argument("--grid-res", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("FRACTAL BASINS & FAST LYAPUNOV INDICATOR (FLI) BENCHMARK")
    print("Run 1 (D3Q8 Discrete) vs Matched Run 3 (Continuous Q8 on S^2)")
    print("=" * 70, flush=True)

    valid = np.load(args.data / "validation.npy", mmap_mode="r")
    start = 8192

    # Load both models and their mature persistent states from the infinite training stream
    print("\nLoading models and inheriting live mature stream states...")
    m1, _, s1_checkpoint = load_model(args.ckpt1)
    m3, _, s3_checkpoint = load_model(args.ckpt3)

    if s1_checkpoint is not None and s3_checkpoint is not None:
        print(f"Inheriting live persistent state directly from checkpoints (No Zero Reset):")
        print(f"  Run 1 state norm: {s1_checkpoint.norm().item():.4f}, energy: {0.5 * s1_checkpoint.square().sum(-1).mean().item():.4f}")
        print(f"  Run 3 state norm: {s3_checkpoint.norm().item():.4f}, energy: {0.5 * s3_checkpoint.square().sum(-1).mean().item():.4f}")
        s1 = s1_checkpoint.detach().clone()
        s3 = s3_checkpoint.detach().clone()
    else:
        print("Fallback: Warming up models over OWT context stream...")
        s1 = m1.initial_state(1, "cuda")
        s3 = m3.initial_state(1, "cuda")
        for t in range(args.warmup):
            x = torch.as_tensor([valid[start + t]], dtype=torch.long, device="cuda")[None]
            y = torch.as_tensor([valid[start + t + 1]], dtype=torch.long, device="cuda")[None]
            _, s1, _ = m1(x, y, s1)
            _, s3, _ = m3(x, y, s3)

    eval_tokens = valid[start:start + 64]

    # Part 1: Fast Lyapunov Indicator (FLI) vs Perturbation Scale
    print("\n[Part 1] Computing Fast Lyapunov Indicator (FLI) across perturbation scales (1e-5 to 1e-1)...")
    fli_res1 = compute_fli_and_settling_curve(m1, s1, eval_tokens, steps_horizon=24)
    fli_res3 = compute_fli_and_settling_curve(m3, s3, eval_tokens, steps_horizon=24)

    # Part 2: 2D Phase-Space Slice & Settling-Time Landscape
    print(f"\n[Part 2] Scanning 2D Phase-Space Slices ({args.grid_res}x{args.grid_res} grid, span=0.25)...")
    slice_res1 = scan_2d_phase_space_slice(m1, s1, eval_tokens, grid_res=args.grid_res, span=0.25, steps_eval=8)
    slice_res3 = scan_2d_phase_space_slice(m3, s3, eval_tokens, grid_res=args.grid_res, span=0.25, steps_eval=8)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS & METRICS COMPARISON")
    print("=" * 70)
    print(f"Metric                               | Run 1 (D3Q8 Discrete) | Matched Run 3 (Cont Q8 S^2) | Delta")
    print("-" * 75)
    print(f"Basin Entropy (S_b)                  | {slice_res1['basin_entropy']:.4f}                | {slice_res3['basin_entropy']:.4f}                    | {slice_res3['basin_entropy'] - slice_res1['basin_entropy']:+.4f}")
    print(f"Mean Settling Time (steps)           | {slice_res1['mean_settling_time']:.2f}                  | {slice_res3['mean_settling_time']:.2f}                      | {slice_res3['mean_settling_time'] - slice_res1['mean_settling_time']:+.2f}")
    print(f"Mean FLI (Lyapunov divergence)       | {slice_res1['mean_fli']:.4f}                | {slice_res3['mean_fli']:.4f}                    | {slice_res3['mean_fli'] - slice_res1['mean_fli']:+.4f}")
    print("-" * 75)

    # Save detailed JSON report
    report = {
        "protocol": {
            "source_paper": "Fractal basins trap latent reasoning (Lai et al., 2026)",
            "grid_res": args.grid_res,
            "warmup_tokens": args.warmup,
            "slice_span": 0.25,
        },
        "run1_d3q8": {
            "basin_entropy": slice_res1["basin_entropy"],
            "mean_settling_time": slice_res1["mean_settling_time"],
            "mean_fli": slice_res1["mean_fli"],
            "fli_vs_delta": fli_res1,
            "slice_2d": slice_res1,
        },
        "run3_continuous_q8": {
            "basin_entropy": slice_res3["basin_entropy"],
            "mean_settling_time": slice_res3["mean_settling_time"],
            "mean_fli": slice_res3["mean_fli"],
            "fli_vs_delta": fli_res3,
            "slice_2d": slice_res3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {args.output}")

    # Part 3: Generate Publication Figure (PNG)
    fig_path = Path("present/cbim_fractal_basins_fli.png").resolve()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering high-res visualization chart to: {fig_path}...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#0f111a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#a0a8b7")
        for spine in ax.spines.values():
            spine.set_color("#2e3440")
        ax.xaxis.label.set_color("#d8dee9")
        ax.yaxis.label.set_color("#d8dee9")
        ax.title.set_color("#eceff4")

    # (0, 0) Run 1: Settling Time Basin Landscape
    im00 = axes[0, 0].imshow(slice_res1["settling_grid"], extent=[-0.25, 0.25, -0.25, 0.25],
                             cmap="magma", origin="lower", aspect="auto")
    axes[0, 0].set_title(f"Run 1 (D3Q8): Settling Basin (Sb={slice_res1['basin_entropy']:.3f})")
    axes[0, 0].set_xlabel("Latent Slice Coordinate a")
    axes[0, 0].set_ylabel("Latent Slice Coordinate b")
    cb00 = plt.colorbar(im00, ax=axes[0, 0])
    cb00.ax.yaxis.set_tick_params(color="#a0a8b7")
    cb00.ax.tick_params(labelsize=8, labelcolor="#a0a8b7")

    # (0, 1) Matched Run 3: Settling Time Basin Landscape
    im01 = axes[0, 1].imshow(slice_res3["settling_grid"], extent=[-0.25, 0.25, -0.25, 0.25],
                             cmap="magma", origin="lower", aspect="auto")
    axes[0, 1].set_title(f"Run 3 (Cont Q8 S²): Settling Basin (Sb={slice_res3['basin_entropy']:.3f})")
    axes[0, 1].set_xlabel("Latent Slice Coordinate a")
    axes[0, 1].set_ylabel("Latent Slice Coordinate b")
    cb01 = plt.colorbar(im01, ax=axes[0, 1])
    cb01.ax.tick_params(labelsize=8, labelcolor="#a0a8b7")

    # (0, 2) Settling Time vs Initial Perturbation
    deltas = [r["delta"] for r in fli_res1]
    st1 = [r["settling_step"] for r in fli_res1]
    st3 = [r["settling_step"] for r in fli_res3]
    axes[0, 2].plot(deltas, st1, "o-", color="#ff5370", label="Run 1 (D3Q8 Discrete)", lw=2)
    axes[0, 2].plot(deltas, st3, "s-", color="#82aaff", label="Run 3 (Cont Q8 on S²)", lw=2)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_title("Perturbation Magnitude vs Settling Horizon")
    axes[0, 2].set_xlabel("Initial Perturbation ||δF(0)|| / ||F(0)||")
    axes[0, 2].set_ylabel("Settling Horizon (Steps to Reconverge)")
    axes[0, 2].grid(True, alpha=0.2, color="#4c566a")
    axes[0, 2].legend(facecolor="#1a1d27", edgecolor="#2e3440", labelcolor="#eceff4")

    # (1, 0) Run 1: Local FLI Landscape
    im10 = axes[1, 0].imshow(slice_res1["fli_grid"], extent=[-0.25, 0.25, -0.25, 0.25],
                             cmap="viridis", origin="lower", aspect="auto")
    axes[1, 0].set_title(f"Run 1 (D3Q8): Fast Lyapunov (FLI={slice_res1['mean_fli']:.3f})")
    axes[1, 0].set_xlabel("Latent Slice Coordinate a")
    axes[1, 0].set_ylabel("Latent Slice Coordinate b")
    cb10 = plt.colorbar(im10, ax=axes[1, 0])
    cb10.ax.tick_params(labelsize=8, labelcolor="#a0a8b7")

    # (1, 1) Matched Run 3: Local FLI Landscape
    im11 = axes[1, 1].imshow(slice_res3["fli_grid"], extent=[-0.25, 0.25, -0.25, 0.25],
                             cmap="viridis", origin="lower", aspect="auto")
    axes[1, 1].set_title(f"Run 3 (Cont Q8 S²): Fast Lyapunov (FLI={slice_res3['mean_fli']:.3f})")
    axes[1, 1].set_xlabel("Latent Slice Coordinate a")
    axes[1, 1].set_ylabel("Latent Slice Coordinate b")
    cb11 = plt.colorbar(im11, ax=axes[1, 1])
    cb11.ax.tick_params(labelsize=8, labelcolor="#a0a8b7")

    # (1, 2) Max FLI vs Delta
    fli1_vals = [r["max_fli"] for r in fli_res1]
    fli3_vals = [r["max_fli"] for r in fli_res3]
    axes[1, 2].plot(deltas, fli1_vals, "o-", color="#ff5370", label="Run 1 (D3Q8 Discrete)", lw=2)
    axes[1, 2].plot(deltas, fli3_vals, "s-", color="#82aaff", label="Run 3 (Cont Q8 on S²)", lw=2)
    axes[1, 2].set_xscale("log")
    axes[1, 2].axhline(0.0, color="#4c566a", linestyle="--", alpha=0.7)
    axes[1, 2].set_title("Fast Lyapunov Indicator vs Perturbation")
    axes[1, 2].set_xlabel("Initial Perturbation ||δF(0)|| / ||F(0)||")
    axes[1, 2].set_ylabel("Max Local FLI (Divergence Exponent)")
    axes[1, 2].grid(True, alpha=0.2, color="#4c566a")
    axes[1, 2].legend(facecolor="#1a1d27", edgecolor="#2e3440", labelcolor="#eceff4")

    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Visualization figure generated successfully at: {fig_path}")


if __name__ == "__main__":
    main()
