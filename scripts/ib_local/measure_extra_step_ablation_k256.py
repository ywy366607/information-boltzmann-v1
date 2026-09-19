"""Extra-Step Operator Component Ablation across K in [1, 3, 8, 16, 32, 64, 128, 256].

Theoretical Hypotheses (GPT Decomposition):
1. Computation = T + C (Information Advection + Nonlinear Lie-Algebra Collision Reorganization).
2. Regularization / Forgetting = D (Dissipation Operator).
   - Viscosity nu * |k|^2: kills high-frequency acoustic shock ripples.
   - Subspace forgetting U Lambda U^T: selective feature erasure.
   - Uniform floor gamma_0: global retention leak.

Tests counterfactually which operator components actually drive the deep pondering gains
as K scales out to K=256:
- Branch 1 (TC): Pure computation (zero dissipation).
- Branch 2 (TC + Visc): Computation + spectral viscosity (no subspace forgetting, no retention leak).
- Branch 3 (TC + Visc + Sub): Computation + spectral viscosity + subspace forgetting (no retention leak).
- Branch 4 (Full TCD): Standard complete unified dissipation operator.
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


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 128
    warmup_tokens = 128
    horizons = [1, 3, 8, 16, 32, 64, 128, 256]

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    branches = ["TC (Pure Comp)", "TC + Viscosity", "TC + Visc + Sub", "Full TCD"]
    branch_nlls = {b: {k: [] for k in horizons} for b in branches}

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)
        # Warm up state over 128 tokens
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        print(f"Executing operator component sweeps across {num_tokens} tokens...")
        for t in range(warmup_tokens, warmup_tokens + num_tokens):
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            tok_embed = model.source.embedding(inp_id)
            f_written, _, _ = model.source(state, inp_id)

            # Evolve each branch counterfactually
            for b in branches:
                f_sim = f_written.clone()
                next_canonical = None
                for k_step in range(1, max(horizons) + 1):
                    alpha_k = model.clock(f_sim, tok_embed)
                    dt_k = alpha_k * model.tau_0_tensor
                    dir_k = model.direction_controller(f_sim, tok_embed)

                    # T
                    mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                    f_sim = model.transport.apply_multiplier(f_sim, mult)
                    # C
                    f_sim, _ = model.collision(f_sim, dt_k)

                    # D according to branch
                    if b == "TC (Pure Comp)":
                        pass  # zero dissipation
                    elif b == "TC + Viscosity":
                        f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed,
                                              gamma0_factor=0.0, disable_viscosity=False, disable_subspace=True)
                    elif b == "TC + Visc + Sub":
                        f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed,
                                              gamma0_factor=0.0, disable_viscosity=False, disable_subspace=False)
                    elif b == "Full TCD":
                        f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed,
                                              gamma0_factor=1.0, disable_viscosity=False, disable_subspace=False)

                    if b == "Full TCD" and k_step == 3:
                        next_canonical = f_sim.clone()

                    if k_step in horizons:
                        feat, _ = model.readout(f_sim, tok_embed, return_diag=False)
                        loss = F.cross_entropy(model.decoder(feat), tgt_id).item()
                        branch_nlls[b][k_step].append(loss)

            # Advance canonical background state
            state = next_canonical

            if (t - warmup_tokens + 1) % 16 == 0 or (t - warmup_tokens + 1) == num_tokens:
                print(f"  Processed {t - warmup_tokens + 1} / {num_tokens} tokens...", flush=True)

    print("\n" + "=" * 90)
    print(f"      EXTRA-STEP OPERATOR ABLATION ACROSS K in [1 .. 256] (N = {num_tokens} TOKENS)")
    print("=" * 90)
    print(f"{'K':<5} | {'TC (Pure Comp)':<18} | {'TC + Viscosity':<18} | {'TC + Visc + Sub':<18} | {'Full TCD':<18}")
    print("-" * 90)

    summary = {b: {} for b in branches}
    for k in horizons:
        vals = [float(np.mean(branch_nlls[b][k])) for b in branches]
        for b, v in zip(branches, vals):
            summary[b][str(k)] = v
        print(f"{k:<5d} | {vals[0]:<18.4f} | {vals[1]:<18.4f} | {vals[2]:<18.4f} | {vals[3]:<18.4f}")

    print("=" * 90)

    out_file = Path("results/extra_step_operator_ablation_k256.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
