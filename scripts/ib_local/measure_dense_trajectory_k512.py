"""Measure Dense Internal Trajectory across K in [1 .. 512] at Single-Step Resolution.

Hypotheses to test:
1. Long-horizon internal dynamics is NOT monotonic relaxation to a trivial static fixed point,
   nor is it uniform white-noise heat death.
2. Low-frequency coherent modes on the 3D torus (q = 1, 2) execute slow phase rotations / beatings,
   causing periodic or quasi-periodic oscillations in readout accessibility (e.g. T_slow ~ 128 steps).
3. The observation that K=64 (9.005) -> K=128 (9.236) -> K=192 (9.012) represents a slow
   recurrence / phase cycle where optimal readout time is an optimal observation phase problem.

Outputs:
- Step-by-step L(k), E(k), E_q(k) for k = 1 .. 512.
- FFT power spectrum of detrended L(k) and dominant spectral peak period T*.
- Autocorrelation function R_L(Delta k) identifying recurrence periods.
- Modal energy breakdown (DC q=0, fundamental q=1, harmonic q=2, high q>=3).
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


def compute_modal_energies(field, wave_sq):
    """Decompose field energy into spatial Fourier wavenumber shells.
    field: [1, Nx, Ny, Nz, V, C]
    wave_sq: [Nx, Ny, Nz] continuous |k_phys|^2
    """
    # FFT over spatial dimensions (axes 1, 2, 3)
    f_hat = torch.fft.fftn(field, dim=(1, 2, 3))
    # Energy density per spatial mode summed over channel dimension d
    mode_energy = (f_hat.real.square() + f_hat.imag.square()).sum(dim=-1).squeeze(0)  # [Nx, Ny, Nz]
    total_e = mode_energy.sum().item() + 1e-12

    # Shell definitions:
    # q=0: DC mode (|k_phys| == 0)
    # q=1: fundamental modes (|k_phys|^2 <= (2*pi)^2 * 1.5)
    # q=2: harmonic modes ((2*pi)^2 * 1.5 < |k_phys|^2 <= (2*pi)^2 * 4.5)
    # q>=3: high-frequency modes
    k1_thresh = (2.0 * np.pi) ** 2 * 1.5
    k2_thresh = (2.0 * np.pi) ** 2 * 4.5

    mask_dc = (wave_sq == 0)
    mask_q1 = (wave_sq > 0) & (wave_sq <= k1_thresh)
    mask_q2 = (wave_sq > k1_thresh) & (wave_sq <= k2_thresh)
    mask_q3 = (wave_sq > k2_thresh)

    e_dc = (mode_energy[mask_dc].sum().item()) / total_e
    e_q1 = (mode_energy[mask_q1].sum().item()) / total_e
    e_q2 = (mode_energy[mask_q2].sum().item()) / total_e
    e_q3 = (mode_energy[mask_q3].sum().item()) / total_e

    return {"e_dc": e_dc, "e_q1": e_q1, "e_q2": e_q2, "e_q3": e_q3}


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 128
    warmup_tokens = 128
    max_k = 512

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    # Precompute continuous |k_phys|^2 for modal energy analysis
    wave_axes = [torch.fft.fftfreq(n, d=1.0 / n, device="cuda") * 2.0 * np.pi for n in model.shape]
    wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
    wave_sq = wave.square().sum(-1)  # [8, 8, 4]

    # Arrays to accumulate trajectories: [num_tokens, max_k]
    traj_loss = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_energy = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_entropy = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_eq1 = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_eq2 = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_eq3 = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_edc = np.zeros((num_tokens, max_k), dtype=np.float32)

    print(f"Executing dense trajectory analysis (K = 1 .. {max_k}) across {num_tokens} tokens...", flush=True)
    start_time = time.perf_counter()

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

            # Standard write
            f_sim, _, _ = model.source(state, inp_id)

            # Store the state for canonical progression (at K=3)
            next_state = None

            # Counterfactual single-step rollout out to max_k
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

                # Measurements at step k:
                # 1. Total energy
                e_tot = float(f_sim.square().mean().item())
                traj_energy[tok_idx, k - 1] = e_tot

                # 2. Modal energy decomposition
                mod_e = compute_modal_energies(f_sim, wave_sq)
                traj_edc[tok_idx, k - 1] = mod_e["e_dc"]
                traj_eq1[tok_idx, k - 1] = mod_e["e_q1"]
                traj_eq2[tok_idx, k - 1] = mod_e["e_q2"]
                traj_eq3[tok_idx, k - 1] = mod_e["e_q3"]

                # 3. Readout & NLL
                feat, _ = model.readout(f_sim, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_k = F.cross_entropy(logits, tgt_id).item()
                traj_loss[tok_idx, k - 1] = loss_k

                probs = F.softmax(logits, dim=-1)
                entropy_k = -(probs * (probs + 1e-12).log()).sum(-1).item()
                traj_entropy[tok_idx, k - 1] = entropy_k

            # Advance canonical stream state
            state = next_state

            if (tok_idx + 1) % 16 == 0 or tok_idx == num_tokens - 1:
                elapsed = time.perf_counter() - start_time
                rate = (tok_idx + 1) / elapsed
                rem = (num_tokens - (tok_idx + 1)) / (rate + 1e-6)
                print(f"  Token {tok_idx + 1:3d} / {num_tokens} | Elapsed: {elapsed:.1f}s | Speed: {rate:.2f} tok/s | Rem: {rem:.1f}s", flush=True)

    # Compute population statistics across tokens
    mean_l = np.mean(traj_loss, axis=0)        # [512]
    std_l = np.std(traj_loss, axis=0)          # [512]
    mean_e = np.mean(traj_energy, axis=0)      # [512]
    mean_edc = np.mean(traj_edc, axis=0)      # [512]
    mean_eq1 = np.mean(traj_eq1, axis=0)      # [512]
    mean_eq2 = np.mean(traj_eq2, axis=0)      # [512]
    mean_eq3 = np.mean(traj_eq3, axis=0)      # [512]
    mean_ent = np.mean(traj_entropy, axis=0)  # [512]

    # Time-series analysis on mean_l(k):
    # 1. Detrend using smooth polynomial / moving average to isolate oscillatory component
    k_axis = np.arange(1, max_k + 1)
    poly_fit = np.poly1d(np.polyfit(k_axis, mean_l, deg=3))(k_axis)
    detrended_l = mean_l - poly_fit

    # 2. Autocorrelation of detrended loss
    detrended_norm = (detrended_l - np.mean(detrended_l)) / (np.std(detrended_l) + 1e-12)
    autocorr = np.correlate(detrended_norm, detrended_norm, mode="full")[max_k - 1:]
    autocorr = autocorr / autocorr[0]  # normalized R(0) = 1.0

    # Find peaks in autocorrelation (excluding lag 0)
    lags = np.arange(len(autocorr))
    peaks = []
    for lag in range(10, len(autocorr) - 10):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1] and autocorr[lag] > 0.15:
            peaks.append((int(lag), float(autocorr[lag])))
    peaks.sort(key=lambda x: x[1], reverse=True)

    # 3. FFT Power Spectrum
    fft_vals = np.fft.rfft(detrended_l)
    fft_power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(max_k, d=1.0)  # cycles per microstep

    # Dominant non-DC frequency peaks
    fft_peaks = []
    for idx in range(1, len(freqs) - 1):
        if fft_power[idx] > fft_power[idx - 1] and fft_power[idx] > fft_power[idx + 1]:
            period = 1.0 / freqs[idx] if freqs[idx] > 0 else float("inf")
            fft_peaks.append({
                "freq_cycles_per_step": float(freqs[idx]),
                "period_microsteps": float(period),
                "power": float(fft_power[idx])
            })
    fft_peaks.sort(key=lambda x: x["power"], reverse=True)

    print("\n" + "=" * 90)
    print("      DENSE INTERNAL TRAJECTORY ANALYSIS (K = 1 .. 512, N = 128 TOKENS)")
    print("=" * 90)
    print(f"{'K':<6} | {'Mean NLL':<10} | {'Total Energy':<14} | {'DC q=0':<10} | {'Fund q=1':<10} | {'Harm q=2':<10} | {'High q>=3':<10}")
    print("-" * 90)
    sample_ks = [1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    for k in sample_ks:
        idx = k - 1
        print(f"K={k:<4d} | {mean_l[idx]:<10.4f} | {mean_e[idx]:<14.6f} | {mean_edc[idx]:<10.3f} | {mean_eq1[idx]:<10.3f} | {mean_eq2[idx]:<10.3f} | {mean_eq3[idx]:<10.3f}")

    print("\n" + "=" * 90)
    print("      SPECTRAL AND RECURRENCE / BEATING ANALYSIS")
    print("=" * 90)
    print("Top Autocorrelation Peaks (Candidate Recurrence Lags):")
    for lag, r_val in peaks[:5]:
        print(f"  Lag Delta K = {lag:3d} microsteps | Autocorrelation R = {r_val:+.4f}")
    if not peaks:
        print("  No strong discrete autocorrelation peak detected above threshold 0.15.")

    print("\nTop FFT Power Spectrum Peaks (Candidate Oscillatory Modes):")
    for p_info in fft_peaks[:5]:
        print(f"  Period T = {p_info['period_microsteps']:6.1f} microsteps | Freq = {p_info['freq_cycles_per_step']:.5f} | Power = {p_info['power']:.2e}")

    # Specific check: difference and recurrence between K=64 and K=192
    diff_64_192 = mean_l[191] - mean_l[63]
    diff_64_128 = mean_l[127] - mean_l[63]
    print("\nSpecific Cycle Check (64 -> 128 -> 192):")
    print(f"  L(K=64)  = {mean_l[63]:.4f}")
    print(f"  L(K=128) = {mean_l[127]:.4f}  (Delta vs 64: {diff_64_128:+.4f})")
    print(f"  L(K=192) = {mean_l[191]:.4f}  (Delta vs 64: {diff_64_192:+.4f})")
    print("=" * 90)

    # Save complete JSON
    results = {
        "num_tokens": num_tokens,
        "max_k": max_k,
        "sample_points": {str(k): {
            "nll": float(mean_l[k - 1]),
            "total_energy": float(mean_e[k - 1]),
            "e_dc": float(mean_edc[k - 1]),
            "e_q1": float(mean_eq1[k - 1]),
            "e_q2": float(mean_eq2[k - 1]),
            "e_q3": float(mean_eq3[k - 1]),
            "entropy": float(mean_ent[k - 1]),
        } for k in sample_ks},
        "autocorr_peaks": [{"lag": p[0], "r": p[1]} for p in peaks[:10]],
        "fft_peaks": fft_peaks[:10],
        "cycle_64_128_192": {
            "l_64": float(mean_l[63]),
            "l_128": float(mean_l[127]),
            "l_192": float(mean_l[191]),
            "delta_128_vs_64": float(diff_64_128),
            "delta_192_vs_64": float(diff_64_192),
        },
        "full_trajectory_mean_l": [float(x) for x in mean_l],
        "full_trajectory_mean_e": [float(x) for x in mean_e],
        "full_trajectory_eq1": [float(x) for x in mean_eq1],
        "full_trajectory_eq2": [float(x) for x in mean_eq2],
    }

    out_file = Path("results/dense_trajectory_k512_analysis.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed analysis to {out_file}")


if __name__ == "__main__":
    main()
