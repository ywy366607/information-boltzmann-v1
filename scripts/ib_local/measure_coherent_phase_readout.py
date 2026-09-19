"""Measure Coherent Phase Readout Demodulation across K in [1 .. 256].

Theoretical Hypothesis:
The T=128 microstep oscillation and K-dependent cognitive valley are caused by
a static Readout probe measuring an oscillating/rotating complex wave field.
The phase modulation depth is Delta L = 1.5255 nats between optimal phase
Psi* = -56.2 deg and anti-phase Psi_w = -168.8 deg.

If Readout performs Coherent Phase Demodulation:
1. Global Phase Locking: Rotate field by Delta theta = Psi* - Psi_global before Readout.
2. Channel-wise IQ Demodulation: Invariant quadrature envelope A = sqrt(I^2 + Q^2).
3. Phase-Optimal Readout: Field is always observed at the constructive interference phase.

Tests whether coherent demodulation:
- Eliminates the 128-step oscillation.
- Flattens the cognitive valley.
- Locks NLL permanently into the optimal regime across all microsteps K.
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


def rotate_field_phase(field, delta_theta):
    """Rotate complex paired channels of field by angle delta_theta.
    field: [1, Nx, Ny, Nz, d] where d = 128 (paired as 64 complex channels).
    delta_theta: scalar or tensor broadcastable.
    """
    f_real = field[..., 0::2]
    f_imag = field[..., 1::2]
    cos_t = torch.cos(delta_theta)
    sin_t = torch.sin(delta_theta)
    f_real_rot = f_real * cos_t - f_imag * sin_t
    f_imag_rot = f_real * sin_t + f_imag * cos_t
    out = torch.empty_like(field)
    out[..., 0::2] = f_real_rot
    out[..., 1::2] = f_imag_rot
    return out


def compute_iq_envelope_field(field):
    """Convert paired channels into invariant IQ amplitude envelope:
    A = sqrt(real^2 + imag^2).
    """
    f_real = field[..., 0::2]
    f_imag = field[..., 1::2]
    amp = torch.sqrt(f_real.square() + f_imag.square() + 1e-12)
    # Replicate amplitude across both real and imaginary slots
    out = torch.empty_like(field)
    out[..., 0::2] = amp
    out[..., 1::2] = amp
    return out


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...", flush=True)
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 64
    warmup_tokens = 128
    horizons = [1, 3, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    # Optimal phase from Kuramoto diagnostic
    psi_optimal = float(np.deg2rad(-56.2))

    branches = [
        "1. Static Baseline Readout",
        "2. Coherent Lock-in (Global Psi*)",
        "3. Invariant IQ Envelope Readout",
    ]
    nll_records = {b: {k: [] for k in horizons} for b in branches}

    print(f"Executing Coherent Phase Readout Sweep across {num_tokens} tokens...", flush=True)
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

            for k in range(1, max(horizons) + 1):
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

                if k in horizons:
                    # 1. Static Baseline Readout
                    feat_base, _ = model.readout(f_sim, tok_embed, return_diag=False)
                    loss_base = F.cross_entropy(model.decoder(feat_base), tgt_id).item()
                    nll_records["1. Static Baseline Readout"][k].append(loss_base)

                    # 2. Coherent Lock-in: Rotate field by Delta theta = Psi* - Psi_current
                    f_real = f_sim[..., 0::2]
                    f_imag = f_sim[..., 1::2]
                    z = torch.complex(f_real, f_imag)
                    global_phasor = (z / z.abs().clamp_min(1e-12)).mean()
                    current_psi = torch.angle(global_phasor)
                    delta_theta = psi_optimal - current_psi
                    f_coherent = rotate_field_phase(f_sim, delta_theta)

                    feat_lock, _ = model.readout(f_coherent, tok_embed, return_diag=False)
                    loss_lock = F.cross_entropy(model.decoder(feat_lock), tgt_id).item()
                    nll_records["2. Coherent Lock-in (Global Psi*)"][k].append(loss_lock)

                    # 3. Invariant IQ Envelope Readout
                    f_env = compute_iq_envelope_field(f_sim)
                    feat_env, _ = model.readout(f_env, tok_embed, return_diag=False)
                    loss_env = F.cross_entropy(model.decoder(feat_env), tgt_id).item()
                    nll_records["3. Invariant IQ Envelope Readout"][k].append(loss_env)

            state = next_state
            if (tok_idx + 1) % 16 == 0 or tok_idx == num_tokens - 1:
                elapsed = time.perf_counter() - t0
                spd = (tok_idx + 1) / elapsed
                print(f"  Token {tok_idx + 1:2d} / {num_tokens} | Elapsed: {elapsed:.1f}s | Speed: {spd:.2f} tok/s", flush=True)

    print("\n" + "=" * 92)
    print("      COHERENT PHASE READOUT DEMODULATION ACROSS K in [1 .. 256]")
    print("=" * 92)
    print(f"{'K':<5} | {'1. Static Base NLL':<20} | {'2. Coherent Lock-in':<20} | {'3. IQ Envelope':<20} | {'Lock-in Gain'}")
    print("-" * 92)

    summary = {b: {} for b in branches}
    for k in horizons:
        v_base = float(np.mean(nll_records["1. Static Baseline Readout"][k]))
        v_lock = float(np.mean(nll_records["2. Coherent Lock-in (Global Psi*)"][k]))
        v_env = float(np.mean(nll_records["3. Invariant IQ Envelope Readout"][k]))
        summary["1. Static Baseline Readout"][str(k)] = v_base
        summary["2. Coherent Lock-in (Global Psi*)"][str(k)] = v_lock
        summary["3. Invariant IQ Envelope Readout"][str(k)] = v_env
        gain = v_lock - v_base
        print(f"{k:<5d} | {v_base:<20.4f} | {v_lock:<20.4f} | {v_env:<20.4f} | {gain:+.4f} nats")

    print("=" * 92)

    out_file = Path("results/coherent_phase_readout_diagnostic.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
