"""Diagnose Intrinsic SO(124) Eigenplane Phase Dynamics and Kuramoto Order.

Theoretical Foundation (GPT Formulation):
Rather than arbitrary adjacent channel pairing (F_{2c}, F_{2c+1}), the true
intrinsic phases of the continuous Boltzmann system are defined by the 62 real
2D invariant rotation planes of the orthogonal Collision matrix M_coll in SO(124).

For each conjugate eigenvalue pair lambda_j = e^{pm i phi_j}:
w_j = u_j + i v_j, where u_j, v_j in R^{124} span the 2D invariant plane.

In each plane j in {1 .. 62}:
a_j(k) = u_j^T F_null(k)
b_j(k) = v_j^T F_null(k)
rho_j(k) = sqrt(a_j^2 + b_j^2)  (energy in eigenplane j)
theta_j(k) = atan2(b_j, a_j)     (intrinsic dynamic phase in eigenplane j)

Energy-weighted intrinsic Kuramoto order parameter:
R_eig(k) * e^{i Psi_eig(k)} = (sum_j rho_j^2 e^{i theta_j}) / (sum_j rho_j^2)

Tests:
1. Does R_eig correlate strongly with validation NLL L(k)?
2. Do dominant eigenplanes (e.g. Mode 6, T=248.9 steps) phase-lock with other modes?
3. Phase-NLL modulation profile L(Psi_eig) on the intrinsic Lie group manifold.
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


def extract_so124_eigenplanes(model, sample_state, tok_embed):
    """Extract the 62 real orthonormal 2D invariant subspaces (u_j, v_j) of M_coll."""
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
    mean_scaled_angles = scaled_angles.mean(dim=1).squeeze(0)  # [layers, nullity // 2]
    K_null = model.collision.nullity  # 124

    M_total = torch.eye(K_null, device="cuda", dtype=torch.float32)
    for l in range(model.collision.layers):
        pairs = model.collision.schedules[l]
        thetas = mean_scaled_angles[l]
        M_l = torch.eye(K_null, device="cuda", dtype=torch.float32)
        for p_idx, pair in enumerate(pairs):
            th = thetas[p_idx].item()
            c = float(np.cos(th))
            s = float(np.sin(th))
            i, j = int(pair[0]), int(pair[1])
            R_block = torch.eye(K_null, device="cuda", dtype=torch.float32)
            R_block[i, i] = c
            R_block[i, j] = -s
            R_block[j, i] = s
            R_block[j, j] = c
            M_l = R_block @ M_l
        M_total = M_l @ M_total

    # Eigen-decomposition of M_total
    evals, evecs = torch.linalg.eig(M_total)  # [124], [124, 124]
    angles_rad = torch.angle(evals).cpu().numpy()
    evecs_np = evecs.cpu().numpy()

    # Pair complex conjugate eigenvalues (phi > 0)
    pos_indices = np.where(angles_rad > 1e-4)[0]
    # Sort by frequency
    pos_indices = pos_indices[np.argsort(angles_rad[pos_indices])]

    u_planes = []
    v_planes = []
    eigen_periods = []
    eigen_angles = []

    for idx in pos_indices:
        phi = float(angles_rad[idx])
        vec = evecs_np[:, idx]
        # Real and imaginary parts span the 2D invariant plane
        u = vec.real
        v = vec.imag
        # Orthonormalize (u, v) using Gram-Schmidt
        u = u / (np.linalg.norm(u) + 1e-12)
        v = v - np.dot(u, v) * u
        v = v / (np.linalg.norm(v) + 1e-12)

        u_planes.append(torch.tensor(u, dtype=torch.float32, device="cuda"))
        v_planes.append(torch.tensor(v, dtype=torch.float32, device="cuda"))
        eigen_periods.append(float(2.0 * np.pi / phi))
        eigen_angles.append(phi)

    # Stack into [62, 124]
    U_mat = torch.stack(u_planes, dim=0)  # [62, 124]
    V_mat = torch.stack(v_planes, dim=0)  # [62, 124]

    return U_mat, V_mat, eigen_periods, eigen_angles


def compute_intrinsic_kuramoto(f_sim, model, U_mat, V_mat):
    """Project field onto the 62 real 2D eigenplanes and compute weighted Kuramoto order parameter."""
    nullspace = model.collision.nullspace.to(device="cuda", dtype=torch.float32)  # [128, 124]
    flat = f_sim.reshape(1, -1, model.d)  # [1, 256, 128]
    # Project to nullspace: coefficient in R^{124}
    coeff = torch.einsum("dk,bnd->bnk", nullspace, flat).squeeze(0)  # [256, 124]
    # Average across space to get DC mode
    coeff_dc = coeff.mean(dim=0)  # [124]

    # Project onto 62 eigenplanes: a_j = U @ coeff_dc, b_j = V @ coeff_dc
    a = torch.mv(U_mat, coeff_dc)  # [62]
    b = torch.mv(V_mat, coeff_dc)  # [62]

    # In-plane polar coordinates
    rho_sq = a.square() + b.square()  # [62] energy per eigenplane
    total_energy = rho_sq.sum().item() + 1e-12
    weights = rho_sq / total_energy  # [62]

    theta = torch.atan2(b, a)  # [62]
    phasors = torch.complex(torch.cos(theta), torch.sin(theta))  # [62]

    # Energy-weighted Kuramoto order parameter
    order_param = (weights * phasors).sum()
    r_eig = order_param.abs().item()
    psi_eig = torch.angle(order_param).item()

    # Track Mode 6 (T=248.9 steps) phase
    theta_mode6 = theta[6].item() if len(theta) > 6 else 0.0

    return {
        "r_eig": r_eig,
        "psi_eig": psi_eig,
        "theta_mode6": theta_mode6,
        "weights": weights.cpu().numpy(),
    }


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
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

    # 1. Extract the 62 real orthonormal 2D eigenplanes of SO(124)
    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        inp = torch.tensor([val_data[128]], dtype=torch.long, device="cuda")
        tok_emb = model.source.embedding(inp)
        U_mat, V_mat, eigen_periods, eigen_angles = extract_so124_eigenplanes(model, state, tok_emb)

    print(f"Extracted {len(eigen_periods)} invariant 2D eigenplanes in SO(124).", flush=True)

    traj_r_eig = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_psi_eig = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_th6 = np.zeros((num_tokens, max_k), dtype=np.float32)
    traj_loss = np.zeros((num_tokens, max_k), dtype=np.float32)

    print(f"Evaluating Intrinsic SO(124) Kuramoto Dynamics across {num_tokens} tokens out to K={max_k}...", flush=True)
    t0 = time.perf_counter()

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

                # Intrinsic Kuramoto calculation
                k_diag = compute_intrinsic_kuramoto(f_sim, model, U_mat, V_mat)
                traj_r_eig[tok_idx, k - 1] = k_diag["r_eig"]
                traj_psi_eig[tok_idx, k - 1] = k_diag["psi_eig"]
                traj_th6[tok_idx, k - 1] = k_diag["theta_mode6"]

                feat, _ = model.readout(f_sim, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_k = F.cross_entropy(logits, tgt_id).item()
                traj_loss[tok_idx, k - 1] = loss_k

            state = next_state
            if (tok_idx + 1) % 16 == 0 or tok_idx == num_tokens - 1:
                elapsed = time.perf_counter() - t0
                spd = (tok_idx + 1) / elapsed
                print(f"  Token {tok_idx + 1:2d} / {num_tokens} | Elapsed: {elapsed:.1f}s | Speed: {spd:.2f} tok/s", flush=True)

    mean_r_eig = np.mean(traj_r_eig, axis=0)  # [256]
    mean_loss = np.mean(traj_loss, axis=0)    # [256]
    corr_r_loss = float(np.corrcoef(mean_r_eig, mean_loss)[0, 1])

    # Phase binning over Psi_eig
    flat_psi = traj_psi_eig.flatten()
    flat_loss = traj_loss.flatten()
    num_bins = 16
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_losses = []
    for b_idx in range(num_bins):
        mask = (flat_psi >= bins[b_idx]) & (flat_psi < bins[b_idx + 1])
        if np.any(mask):
            bin_losses.append(float(np.mean(flat_loss[mask])))
        else:
            bin_losses.append(float("nan"))

    valid_losses = [l for l in bin_losses if not np.isnan(l)]
    min_loss = min(valid_losses)
    max_loss = max(valid_losses)
    best_phase_deg = float(np.rad2deg(bin_centers[bin_losses.index(min_loss)]))
    worst_phase_deg = float(np.rad2deg(bin_centers[bin_losses.index(max_loss)]))

    print("\n" + "=" * 90)
    print("      INTRINSIC SO(124) EIGENPLANE KURAMOTO SYNCHRONIZATION DIAGNOSTIC")
    print("=" * 90)
    print(f"Mean Intrinsic Kuramoto Order Parameter R_eig:  {float(np.mean(mean_r_eig)):.4f} (Range: {float(np.min(mean_r_eig)):.4f} .. {float(np.max(mean_r_eig)):.4f})")
    print(f"Correlation between R_eig and Validation NLL:    {corr_r_loss:+.4f}")
    print(f"Optimal Intrinsic Observation Phase:             Psi_eig* = {best_phase_deg:+.1f} deg (NLL = {min_loss:.4f} nats)")
    print(f"Worst Intrinsic Observation Phase:               Psi_eig_w = {worst_phase_deg:+.1f} deg (NLL = {max_loss:.4f} nats)")
    print(f"Intrinsic Phase Modulation Depth:                Delta L = {max_loss - min_loss:.4f} nats!")
    print("=" * 90)

    report = {
        "mean_r_eig": float(np.mean(mean_r_eig)),
        "corr_r_loss": corr_r_loss,
        "optimal_phase_deg": best_phase_deg,
        "optimal_nll": min_loss,
        "worst_phase_deg": worst_phase_deg,
        "worst_nll": max_loss,
        "phase_modulation_depth": max_loss - min_loss,
        "sample_ks": {
            str(k): {
                "nll": float(mean_loss[k - 1]),
                "r_eig": float(mean_r_eig[k - 1]),
            } for k in [1, 3, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
        }
    }
    out_file = Path("results/intrinsic_so124_eigenplane_kuramoto.json")
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
