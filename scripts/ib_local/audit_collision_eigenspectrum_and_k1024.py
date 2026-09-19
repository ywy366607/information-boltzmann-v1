"""Audit Collision Generator Eigen-Spectrum and 1024-step FFT Peak Stability.

Directly verifies GPT's mathematical hypothesis:
1. Collision Generator Eigenvalues:
   - On spatial DC mode (k=0), Transport does not move (omega=0).
   - Thus, long-horizon rotation MUST originate from the Collision Lie group SO(K_null).
   - We construct the full Collision transfer matrix M_coll = M_layer1 * M_layer0 in SO(d_null),
     compute its complex eigenvalues lambda_j = e^{i phi_j},
     and extract the exact eigenperiods T_j = 2*pi / |phi_j|.
   - Check if the eigenperiods predict T ~ 60, 128, 256 microsteps!

2. 1024-Step FFT Peak Stability:
   - Extends the dense trajectory window from 512 to 1024 steps across 16 validation tokens.
   - Tests whether the T=128 and T=256 peaks stay fixed at 128 and 256 microsteps
     (which would fall at FFT bin j=8 and j=4 for N=1024),
     or whether they shift with window length.
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
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def audit_collision_eigenspectrum(model, sample_state, tok_embed):
    """Compute the exact orthogonal transfer matrix of the Collision layers
    and extract its complex eigenvalues lambda_j = e^{i phi_j}.
    """
    flat = sample_state.reshape(1, -1, model.d)
    normed = model.collision.norm(flat)
    if model.collision.position_conditioned:
        pos = model.collision.position_features.to(flat)[None]
        ang_in = torch.cat((normed, pos), -1)
    else:
        ang_in = normed
    angles = model.collision.angle(ang_in).reshape(
        1, flat.shape[1], model.collision.layers, model.collision.nullity // 2)

    alpha = model.clock(sample_state, tok_embed)
    dt = alpha * model.tau_0_tensor
    scaled_angles = angles * dt.view(1, 1, 1, 1)

    # Average angle across spatial nodes for DC mode
    mean_scaled_angles = scaled_angles.mean(dim=1).squeeze(0)  # [layers, nullity // 2]
    K_null = model.collision.nullity  # dimension of nullspace

    # Construct the orthogonal matrix M for the two Givens rotation layers
    M_total = torch.eye(K_null, device="cuda", dtype=torch.float32)

    layer_periods = []
    for l in range(model.collision.layers):
        pairs = model.collision.schedules[l]
        thetas = mean_scaled_angles[l]
        M_l = torch.eye(K_null, device="cuda", dtype=torch.float32)
        for p_idx, pair in enumerate(pairs):
            th = thetas[p_idx].item()
            c = float(np.cos(th))
            s = float(np.sin(th))
            i, j = int(pair[0]), int(pair[1])
            # Givens rotation block
            R_block = torch.eye(K_null, device="cuda", dtype=torch.float32)
            R_block[i, i] = c
            R_block[i, j] = -s
            R_block[j, i] = s
            R_block[j, j] = c
            M_l = R_block @ M_l
        M_total = M_l @ M_total

        # Layer-average angle
        mean_th = thetas.abs().mean().item()
        layer_periods.append(float(2.0 * np.pi / (mean_th + 1e-12)))

    # Compute eigenvalues of M_total (orthogonal matrix in SO(K_null))
    eigenvalues = torch.linalg.eigvals(M_total).cpu().numpy()
    eigen_angles = np.angle(eigenvalues)
    # Filter positive angles
    pos_angles = eigen_angles[eigen_angles > 1e-4]
    pos_angles = np.sort(pos_angles)

    eigen_periods = 2.0 * np.pi / pos_angles

    return {
        "nullity": K_null,
        "layer_periods": layer_periods,
        "eigen_angles_rad": [float(a) for a in pos_angles],
        "eigen_periods_microsteps": [float(p) for p in eigen_periods],
    }


def audit_1024_step_fft(model, val_data, num_tokens=16, max_k=1024):
    """Run dense trajectory out to K=1024 to verify FFT peak invariance."""
    warmup_tokens = 128
    traj_loss = np.zeros((num_tokens, max_k), dtype=np.float32)

    print(f"Running 1024-step trajectory across {num_tokens} tokens for FFT peak audit...", flush=True)

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
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

                mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                f_sim = model.transport.apply_multiplier(f_sim, mult)
                f_sim, _ = model.collision(f_sim, dt_k)
                f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed)

                if k == 3:
                    next_state = f_sim.clone()

                feat, _ = model.readout(f_sim, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_k = F.cross_entropy(logits, tgt_id).item()
                traj_loss[tok_idx, k - 1] = loss_k

            state = next_state
            print(f"  Token {tok_idx + 1:2d} / {num_tokens} completed out to K={max_k}", flush=True)

    mean_l = np.mean(traj_loss, axis=0)  # [1024]
    k_axis = np.arange(1, max_k + 1)
    poly_fit = np.poly1d(np.polyfit(k_axis, mean_l, deg=3))(k_axis)
    detrended = mean_l - poly_fit

    # 1024-point FFT
    fft_vals = np.fft.rfft(detrended)
    fft_power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(max_k, d=1.0)

    peaks_1024 = []
    for idx in range(1, len(freqs) - 1):
        if fft_power[idx] > fft_power[idx - 1] and fft_power[idx] > fft_power[idx + 1]:
            period = 1.0 / freqs[idx] if freqs[idx] > 0 else float("inf")
            peaks_1024.append({
                "bin_idx": int(idx),
                "freq": float(freqs[idx]),
                "period_microsteps": float(period),
                "power": float(fft_power[idx]),
            })
    peaks_1024.sort(key=lambda x: x["power"], reverse=True)

    return peaks_1024[:10], mean_l


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    # 1. Collision Eigenspectrum Audit
    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        inp = torch.tensor([val_data[128]], dtype=torch.long, device="cuda")
        tok_emb = model.source.embedding(inp)
        coll_audit = audit_collision_eigenspectrum(model, state, tok_emb)

    print("\n" + "=" * 90)
    print("      COLLISION GENERATOR EIGEN-SPECTRUM AUDIT")
    print("=" * 90)
    print(f"Collision Nullspace Dimension: {coll_audit['nullity']}")
    print(f"Layer-wise average rotation periods: Layer 0 = {coll_audit['layer_periods'][0]:.1f} steps, Layer 1 = {coll_audit['layer_periods'][1]:.1f} steps")
    print("-" * 90)
    print("Top Collision Eigenperiods T_j = 2*pi / |phi_j| (microsteps):")
    for p_val in coll_audit["eigen_periods_microsteps"][:10]:
        print(f"  Eigenperiod T = {p_val:7.2f} microsteps")
    print("=" * 90)

    # 2. 1024-step FFT Peak Audit
    peaks_1024, mean_l = audit_1024_step_fft(model, val_data, num_tokens=16, max_k=1024)

    print("\n" + "=" * 90)
    print("      1024-STEP FFT POWER SPECTRUM PEAKS AUDIT")
    print("=" * 90)
    print(f"{'FFT Bin':<10} | {'Frequency (cyc/step)':<24} | {'Period (microsteps)':<24} | {'Power'}")
    print("-" * 90)
    for p in peaks_1024[:8]:
        print(f"Bin {p['bin_idx']:<6d} | {p['freq']:<24.6f} | {p['period_microsteps']:<24.1f} | {p['power']:.2e}")
    print("=" * 90)

    report = {
        "collision_audit": coll_audit,
        "fft_1024_peaks": peaks_1024,
        "sample_nll_at_k": {
            "64": float(mean_l[63]),
            "128": float(mean_l[127]),
            "192": float(mean_l[191]),
            "256": float(mean_l[255]),
            "512": float(mean_l[511]),
            "1024": float(mean_l[1023]),
        }
    }
    out_file = Path("results/collision_eigenspectrum_and_1024fft_audit.json")
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}", flush=True)


if __name__ == "__main__":
    main()
