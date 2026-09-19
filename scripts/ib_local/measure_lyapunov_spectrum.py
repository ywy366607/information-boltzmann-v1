"""Measure the Finite-Time Lyapunov Spectrum of CBIM Three-Clock v1.

Algorithm: Benettin QR algorithm for Lyapunov Exponents (1980):
- Given a discrete-time continuous-state dynamical system: F_{t+1} = G(F_t, x_t)
- Tangent map: J_t = dG/dF
- Track r orthonormal tangent vectors V_t = [v_1, ..., v_r] in R^{D x r} (D = 32,768)
- At each step t:
    W_t = J_t V_{t-1} = [J_t v_1, ..., J_t v_r] (computed via exact autograd JVP)
    Q_t, R_t = qr(W_t)
    lambda_i += (1/T) * log(R_{ii})
    V_t = Q_t
- Measures both:
    1. Full coupled system (with state-dependent collision controller)
    2. Decoupled collision tangent system (where tangent feedback d(theta)/dF is detached)
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
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def patch_collision_tangent_decoupling(model):
    """Detach field from angle_input inside collision."""
    orig_collision_forward = model.collision.forward

    def decoupled_collision_forward(field, delta_tau=1.0):
        batch = field.shape[0]
        flat = field.reshape(batch, -1, model.collision.d)
        nullspace = model.collision.nullspace.to(dtype=flat.dtype)
        coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
        conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)

        # DETACH flat from angle_input to make tangent map strictly orthogonal
        angle_input = model.collision.norm(flat.detach())
        if model.collision.position_conditioned:
            position = model.collision.position_features.to(flat)[None].expand(batch, -1, -1)
            angle_input = torch.cat((angle_input, position), -1)
        angles = model.collision.angle(angle_input).reshape(
            batch, flat.shape[1], model.collision.layers, model.collision.nullity // 2)

        if isinstance(delta_tau, torch.Tensor):
            dt = delta_tau.view(batch, 1, 1, 1)
        else:
            dt = float(delta_tau)
        scaled_angles = angles * dt
        value = coefficient
        for layer in range(model.collision.layers):
            pair = model.collision.schedules[layer]
            left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
            theta = scaled_angles[:, :, layer]
            cosine, sine = theta.cos(), theta.sin()
            updated = value.clone()
            updated[..., pair[:, 0]] = cosine * left - sine * right
            updated[..., pair[:, 1]] = sine * left + cosine * right
            value = updated
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, value)
        coll_in_power = coefficient.square().sum(-1).mean()
        cons_in_power = conserved.square().sum(-1).mean()
        coll_out_power = value.square().sum(-1).mean()
        return output.reshape_as(field), {
            "collision_angle_abs_mean": scaled_angles.detach().abs().mean(),
            "collision_angle_abs_max": scaled_angles.detach().abs().amax(),
            "collision_input_snr": (coll_in_power / cons_in_power.clamp_min(1e-8)).detach(),
            "collision_output_snr": (coll_out_power / cons_in_power.clamp_min(1e-8)).detach(),
        }

    model.collision.forward = decoupled_collision_forward


def compute_lyapunov_spectrum(model, val_data, mature_state_base, start_offset, warmup_len, eval_steps, r_subspace=8):
    # 1. Warm up
    state = mature_state_base.clone()
    with torch.no_grad():
        for t in range(warmup_len):
            inp = torch.as_tensor([val_data[start_offset + t]], dtype=torch.long, device="cuda")
            _, state, _ = model.step(state, inp, micro_steps=3)

    D = state.numel()  # 32,768
    # Initialize random orthonormal tangent basis V_0 in R^{D x r}
    torch.manual_seed(42)
    V_flat = torch.randn(D, r_subspace, device="cuda", dtype=state.dtype)
    Q, _ = torch.linalg.qr(V_flat)
    V = Q.reshape(r_subspace, *state.shape[1:])  # [r, 8, 8, 4, 128]

    lyapunov_sums = torch.zeros(r_subspace, device="cuda", dtype=torch.float64)

    t0 = time.perf_counter()
    for step_i in range(eval_steps):
        inp = torch.as_tensor([val_data[start_offset + warmup_len + step_i]], dtype=torch.long, device="cuda")

        # Step forward state
        with torch.no_grad():
            _, next_state, _ = model.step(state, inp, micro_steps=3)

        # Compute JVP for each basis vector v_i
        def step_field(s):
            return model.step(s, inp, micro_steps=3)[1]

        W_list = []
        for i in range(r_subspace):
            v_i = V[i:i+1]
            _, w_i = torch.autograd.functional.jvp(step_field, state, v_i)
            W_list.append(w_i.flatten())

        # QR decomposition of W
        W_mat = torch.stack(W_list, dim=1).to(dtype=torch.float32)  # [D, r]
        Q_mat, R_mat = torch.linalg.qr(W_mat)

        # Extract expansion rates from diagonal of R
        # R_ii > 0
        diag_R = torch.diagonal(R_mat).abs().clamp_min(1e-12).to(dtype=torch.float64)
        lyapunov_sums += torch.log(diag_R)

        # Re-orthonormalize basis vectors
        V = Q_mat.T.reshape(r_subspace, *state.shape[1:]).to(dtype=state.dtype)
        state = next_state

        if (step_i + 1) % 50 == 0 or (step_i + 1) == eval_steps:
            current_lambda = (lyapunov_sums / (step_i + 1)).cpu().numpy()
            print(f" Step {step_i+1:<4d}/{eval_steps}: lambda_1 = {current_lambda[0]:+.6f}, lambda_{r_subspace} = {current_lambda[-1]:+.6f}", flush=True)

    final_spectrum = (lyapunov_sums / eval_steps).cpu().numpy()
    return final_spectrum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_bptt128_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_lyapunov_spectrum.json"))
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--subspace-dim", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading champion model from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    def create_model():
        m = CBIMTorus3D(
            shape=tuple(cfg["shape"]),
            velocities=cfg["velocities"],
            content_dim=cfg["content_dim"],
            v2_coordinate_components=True,
            readout_type=cfg.get("readout_type", "kernel_r1"),
            write_type=cfg.get("write_type", "w2_impedance"),
            micro_steps=3,
            adaptive_clock=True,
            continuous_velocities=True,
            dissipation_type="unified",
            dissipation_rank=4,
            three_clock=True,
            tau_mem=3.0,
            nu_s_init=0.020,
            decouple_source_feedback=True
        ).cuda()
        m.load_state_dict(saved["model"])
        m.eval()
        return m

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()
    start_offset = 8192

    print("\n" + "=" * 90)
    print("   MEASURING FINITE-TIME LYAPUNOV SPECTRUM (JVP + BENETTIN QR ALGORITHM)")
    print("=" * 90)

    # 1. Coupled Model (Normal Collision Controller with state feedback)
    print("\n[Mode 1] Coupled System (Full Collision Controller State-Dependency):", flush=True)
    model_coupled = create_model()
    spectrum_coupled = compute_lyapunov_spectrum(
        model_coupled, val_data, mature_state_base, start_offset,
        args.warmup, args.eval_steps, args.subspace_dim)

    # 2. Decoupled Tangent Model (Tangent feedback d(theta)/dF detached)
    print("\n[Mode 2] Decoupled Tangent System (Collision Controller Tangent Decoupled):", flush=True)
    model_decoupled = create_model()
    patch_collision_tangent_decoupling(model_decoupled)
    spectrum_decoupled = compute_lyapunov_spectrum(
        model_decoupled, val_data, mature_state_base, start_offset,
        args.warmup, args.eval_steps, args.subspace_dim)

    print("\n" + "=" * 90)
    print("                      CBIM THREE-CLOCK V1: LYAPUNOV SPECTRUM COMPARISON")
    print("=" * 90)
    print(f"{'Mode Index i':<14} | {'Coupled lambda_i (nats/tok)':<32} | {'Decoupled lambda_i (nats/tok)':<32}")
    print("-" * 90)
    for i in range(args.subspace_dim):
        c_val = spectrum_coupled[i]
        d_val = spectrum_decoupled[i]
        print(f"Mode {i+1:<9d} | {c_val:<+32.6f} | {d_val:<+32.6f}")
    print("-" * 90)

    report = {
        "checkpoint": str(args.checkpoint),
        "warmup": args.warmup,
        "eval_steps": args.eval_steps,
        "subspace_dim": args.subspace_dim,
        "coupled_spectrum": spectrum_coupled.tolist(),
        "decoupled_spectrum": spectrum_decoupled.tolist(),
        "coupled_max_lambda": float(spectrum_coupled[0]),
        "decoupled_max_lambda": float(spectrum_decoupled[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Lyapunov spectrum report to {args.output}")


if __name__ == "__main__":
    main()
