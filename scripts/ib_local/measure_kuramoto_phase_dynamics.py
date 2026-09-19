"""Measure Kuramoto Phase Order Parameters and Coherence Dynamics on CBIM (BBest.pt).

Directly tests the Neural Kuramoto Synchronization Hypothesis:
1. Does the internal field F(x, v, c) exhibit spontaneous Kuramoto phase synchronization?
   - Global Kuramoto order parameter R_global(tau) = |(1/M) sum e^{i theta}| in [0, 1].
   - Macroscopic collective phase Psi_global(tau) = angle(sum e^{i theta}).
2. Does Psi_global rotate at the exact characteristic period T ~ 128 microsteps?
   - dPsi / dtau = omega_carrier ~ 2*pi / 128 = 0.049 rad/microstep.
3. Is Readout Loss L(k) a direct function of the macroscopic phase Psi(k)?
   - L(Psi): does a specific phase angle Psi* define the optimal observation window?
4. Spatial and velocity phase coherence fields:
   - Does spatial coherence R_space(v) or velocity coherence R_vel(x) form chimera-like
     coherent / incoherent spatial patterns on the 3D torus?
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
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def compute_kuramoto_order_parameters(field):
    """Extract complex phase field and Kuramoto order parameters.
    field: [1, Nx, Ny, Nz, V, C] where V=8, C=16 (or [1, 8, 8, 4, 128])
    """
    # Reshape to [Nx*Ny*Nz, V, C_pairs, 2]
    # In CBIMTorus3D, field is [1, 8, 8, 4, 128] where 128 = 8 velocities * 16 content
    B, Nx, Ny, Nz, d = field.shape
    V = 8
    C = d // V  # 16
    f_vc = field.view(B, Nx * Ny * Nz, V, C)  # [1, 256, 8, 16]

    # Pair adjacent content channels (2c, 2c+1) into complex numbers
    # f_complex: [256, 8, 8] complex tensor
    f_real = f_vc[0, :, :, 0::2]  # [256, 8, 8]
    f_imag = f_vc[0, :, :, 1::2]  # [256, 8, 8]
    z = torch.complex(f_real, f_imag)  # [256, 8, 8]

    # Phase angles on S^1
    # Handle zero amplitudes safely
    amp = z.abs().clamp_min(1e-12)
    phasors = z / amp  # e^{i theta}, [256, 8, 8]

    # 1. Global Kuramoto order parameter: R_global * e^{i Psi_global}
    global_phasor = phasors.mean()
    r_global = global_phasor.abs().item()
    psi_global = torch.angle(global_phasor).item()

    # 2. Velocity Kuramoto order parameter at each spatial location:
    # R_vel(x): average over the 8 velocities, shape [256, 8]
    vel_phasors = phasors.mean(dim=1)  # average over V=8 -> [256, 8]
    r_vel = vel_phasors.abs().mean().item()

    # 3. Spatial Kuramoto order parameter for each velocity:
    # R_space(v): average over the 256 spatial nodes -> [8, 8]
    space_phasors = phasors.mean(dim=0)  # average over N=256 -> [8, 8]
    r_space = space_phasors.abs().mean().item()

    return {
        "r_global": r_global,
        "psi_global": psi_global,
        "r_vel": r_vel,
        "r_space": r_space,
    }


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...", flush=True)
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 64
    warmup_tokens = 128
    max_k = 256

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    # Accumulators: [num_tokens, max_k]
    traj_r_global = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_psi_global = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_r_vel = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_r_space = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_loss = np.zeros((num_tokens, max_k), dtype=np.float32)

    print(f"Executing Kuramoto Phase Diagnostic across {num_tokens} tokens out to K={max_k}...", flush=True)
    t0 = time.perf_counter()

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        # Warmup
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        for tok_idx in range(num_tokens):
            t = warmup_tokens + tok_idx
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            tok_embed = model.source.embedding(inp_id)

            f_sim, _, _ = model.source(state, inp_id)
            next_state = None

            for k in range(1, max_k + 1):
                alpha_k = model.clock(f_sim, tok_embed)
                dt_k = alpha_k * model.tau_0_tensor
                dir_k = model.direction_controller(f_sim, tok_embed)

                # T + C + D
                mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                f_sim = model.transport.apply_multiplier(f_sim, mult)
                f_sim, _ = model.collision(f_sim, dt_k)
                f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed)

                if k == 3:
                    next_state = f_sim.clone()

                # Measure Kuramoto phase order parameters
                kuramoto = compute_kuramoto_order_parameters(f_sim)
                traj_r_global[tok_idx, k - 1] = kuramoto["r_global"]
                traj_psi_global[tok_idx, k - 1] = kuramoto["psi_global"]
                traj_r_vel[tok_idx, k - 1] = kuramoto["r_vel"]
                traj_r_space[tok_idx, k - 1] = kuramoto["r_space"]

                # Readout loss
                feat, _ = model.readout(f_sim, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_k = F.cross_entropy(logits, tgt_id).item()
                traj_loss[tok_idx, k - 1] = loss_k

            state = next_state
            if (tok_idx + 1) % 16 == 0 or tok_idx == num_tokens - 1:
                elapsed = time.perf_counter() - t0
                spd = (tok_idx + 1) / elapsed
                print(f"  Token {tok_idx + 1:2d} / {num_tokens} | Elapsed: {elapsed:.1f}s | Speed: {spd:.2f} tok/s", flush=True)

    # Statistical Analysis
    mean_r_glob = np.mean(traj_r_global, axis=0)  # [256]
    mean_r_vel = np.mean(traj_r_vel, axis=0)      # [256]
    mean_r_space = np.mean(traj_r_space, axis=0)  # [256]
    mean_loss = np.mean(traj_loss, axis=0)        # [256]

    # Unwrap phase across steps to compute continuous phase velocity dPsi/dtau
    unwrapped_psi = np.unwrap(traj_psi_global, axis=1)  # [num_tokens, 256]
    mean_unwrapped_psi = np.mean(unwrapped_psi, axis=0)  # [256]

    # Compute phase rotation velocity (rad per microstep)
    dpsi_dt = np.gradient(mean_unwrapped_psi)  # [256]
    mean_omega = float(np.mean(np.abs(dpsi_dt[10:200])))
    estimated_t_period = (2.0 * np.pi) / (mean_omega + 1e-12)

    # Correlation between Kuramoto Coherence R_global and Validation NLL
    corr_r_loss = float(np.corrcoef(mean_r_glob, mean_loss)[0, 1])

    # Phase binning: bin L(k) into phase sectors Psi in [-pi, pi] across all tokens
    flat_psi = traj_psi_global.flatten()
    flat_loss = traj_loss.flatten()
    num_bins = 16
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_losses = []
    bin_counts = []
    for b_idx in range(num_bins):
        mask = (flat_psi >= bins[b_idx]) & (flat_psi < bins[b_idx + 1])
        if np.any(mask):
            bin_losses.append(float(np.mean(flat_loss[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_losses.append(float("nan"))
            bin_counts.append(0)

    print("\n" + "=" * 90)
    print("      KURAMOTO PHASE SYNCHRONIZATION AND COHERENCE DIAGNOSTIC")
    print("=" * 90)
    print(f"Mean Global Kuramoto Order Parameter R_global:  {float(np.mean(mean_r_glob)):.4f}  (Range: {float(np.min(mean_r_glob)):.4f} .. {float(np.max(mean_r_glob)):.4f})")
    print(f"Velocity Space Coherence R_vel (across V=8):    {float(np.mean(mean_r_vel)):.4f}")
    print(f"Spatial Domain Coherence R_space (across N=256): {float(np.mean(mean_r_space)):.4f}")
    print("-" * 90)
    print(f"Macroscopic Phase Angular Velocity |dPsi/dt|:   {mean_omega:.5f} rad/microstep")
    print(f"Estimated Phase Rotation Period T_phase:        {estimated_t_period:.1f} microsteps")
    print(f"Correlation between R_global and NLL:           {corr_r_loss:+.4f}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("      SAMPLE STEP DYNAMICS ACROSS K in [1 .. 256]")
    print("=" * 90)
    print(f"{'K':<6} | {'Mean NLL':<10} | {'R_global':<12} | {'R_vel (local)':<14} | {'R_space (global)':<16} | {'Psi (deg)':<10}")
    print("-" * 90)
    sample_ks = [1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    for k in sample_ks:
        idx = k - 1
        deg = float(np.rad2deg(mean_unwrapped_psi[idx]) % 360.0)
        print(f"K={k:<4d} | {mean_loss[idx]:<10.4f} | {mean_r_glob[idx]:<12.4f} | {mean_r_vel[idx]:<14.4f} | {mean_r_space[idx]:<16.4f} | {deg:<10.1f}")

    print("\n" + "=" * 90)
    print("      PHASE-SECTOR LOSS MODULATION L(Psi)")
    print("=" * 90)
    print(f"{'Phase Sector Psi (deg)':<25} | {'Mean Validation NLL':<22} | {'Sample Count'}")
    print("-" * 90)
    best_phase = None
    min_loss = float("inf")
    worst_phase = None
    max_loss = float("-inf")
    for b_idx in range(num_bins):
        deg = float(np.rad2deg(bin_centers[b_idx]))
        loss_val = bin_losses[b_idx]
        count_val = bin_counts[b_idx]
        if not np.isnan(loss_val):
            if loss_val < min_loss:
                min_loss = loss_val
                best_phase = deg
            if loss_val > max_loss:
                max_loss = loss_val
                worst_phase = deg
            print(f"Psi in [{deg - 11.25:+.1f}, {deg + 11.25:+.1f}] deg | {loss_val:<22.4f} | {count_val}")
    print("-" * 90)
    print(f"Optimal Observation Phase: Psi* = {best_phase:+.1f} deg  (NLL = {min_loss:.4f} nats)")
    print(f"Worst Observation Phase:   Psi_w = {worst_phase:+.1f} deg  (NLL = {max_loss:.4f} nats)")
    print(f"Phase Modulation Depth:    Delta L = {max_loss - min_loss:.4f} nats!")
    print("=" * 90)

    out = {
        "mean_r_global": float(np.mean(mean_r_glob)),
        "mean_r_vel": float(np.mean(mean_r_vel)),
        "mean_r_space": float(np.mean(mean_r_space)),
        "phase_velocity_omega": mean_omega,
        "estimated_period_t": estimated_t_period,
        "corr_r_loss": corr_r_loss,
        "optimal_phase_deg": best_phase,
        "optimal_nll": min_loss,
        "worst_phase_deg": worst_phase,
        "worst_nll": max_loss,
        "phase_modulation_depth": max_loss - min_loss,
        "sample_ks": {str(k): {
            "nll": float(mean_loss[k - 1]),
            "r_global": float(mean_r_glob[k - 1]),
            "r_vel": float(mean_r_vel[k - 1]),
            "r_space": float(mean_r_space[k - 1]),
            "psi_deg": float(np.rad2deg(mean_unwrapped_psi[k - 1]) % 360.0),
        } for k in sample_ks},
        "phase_bins": {
            "centers_deg": [float(np.rad2deg(c)) for c in bin_centers],
            "losses": bin_losses,
            "counts": bin_counts,
        }
    }
    out_file = Path("results/kuramoto_phase_dynamics_diagnostic.json")
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
