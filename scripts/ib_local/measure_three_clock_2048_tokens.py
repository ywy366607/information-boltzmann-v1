"""Verify Long-Horizon Three-Clock NESS across 2048 Tokens for K in [3, 8, 16, 32, 64].

Theoretical Question (GPT Review #3):
"Is 7.9315 nats at K=64 merely a 512-token short-range sweet spot,
or is it a true long-horizon Non-Equilibrium Steady State (NESS)?"

Evaluates:
- 2048 continuous tokens with zero state resets.
- 4 temporal quarter-windows (0-512, 512-1024, 1024-1536, 1536-2048) to prove stationarity.
- Continuous tracking of field energy E(t) to confirm non-zero steady state (NESS).
- Horizons K in [3, 8, 16, 32, 64].
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


def main():
    ckpt_path = Path("results/cbim_torus3d_w2_8x8x4_arm_c_cont_q8_unified_dissipation_3000/BBest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading reference model from {ckpt_path}...", flush=True)
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 2048
    warmup_tokens = 128
    horizons = [3, 8, 16, 32, 64]

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    def run_stream_2048(k_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            state = model.initial_state(1, "cuda", warm_start=True)
            # Warmup
            for offset in range(0, warmup_tokens, 128):
                ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
                targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
                _, state, _ = model(ids, targets, state)

            token_losses = []
            energy_history = []
            dt_nominal = 3.0 * model.tau_0_tensor

            for t in range(warmup_tokens, warmup_tokens + num_tokens):
                inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                tok_embed = model.source.embedding(inp_id)

                # 1. Event Write
                field, _, _ = model.source(state, inp_id)

                # 2. Source Spectral Viscosity (applied ONCE to kill injection ringing)
                field, _ = model.bath(
                    field, model.tau_0_tensor, tok_embed=tok_embed,
                    gamma0_factor=0.0, disable_viscosity=False, disable_subspace=True
                )

                # 3. Pure Conservative Pondering Core (TC)^K
                for _ in range(k_steps):
                    alpha_k = model.clock(field, tok_embed)
                    dt_k = alpha_k * model.tau_0_tensor
                    dir_k = model.direction_controller(field, tok_embed)

                    mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                    field = model.transport.apply_multiplier(field, mult)
                    field, _ = model.collision(field, dt_k)

                # 4. Readout
                feat, _ = model.readout(field, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                loss_t = F.cross_entropy(logits, tgt_id).item()
                token_losses.append(loss_t)

                # 5. Memory Retention Clock (applied ONCE per token)
                state, _ = model.bath(
                    field, dt_nominal, tok_embed=tok_embed,
                    gamma0_factor=1.0, disable_viscosity=True, disable_subspace=False
                )

                if (t - warmup_tokens + 1) % 64 == 0:
                    energy_history.append(float(state.square().mean().item()))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        speed = num_tokens / elapsed

        losses_arr = np.array(token_losses)
        q1_loss = float(np.mean(losses_arr[0:512]))
        q2_loss = float(np.mean(losses_arr[512:1024]))
        q3_loss = float(np.mean(losses_arr[1024:1536]))
        q4_loss = float(np.mean(losses_arr[1536:2048]))
        total_loss = float(np.mean(losses_arr))
        final_e = float(state.square().mean().item())
        mean_e = float(np.mean(energy_history))

        return {
            "total_loss": total_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "q3_loss": q3_loss,
            "q4_loss": q4_loss,
            "final_e": final_e,
            "mean_e": mean_e,
            "speed": speed,
            "drift": q4_loss - q1_loss,
        }

    print("\n" + "=" * 110)
    print("      LONG-HORIZON THREE-CLOCK NESS VERIFICATION (2048 TOKENS)")
    print("=" * 110)
    print(f"{'K':<5} | {'2048 NLL':<12} | {'Q1 (0-512)':<12} | {'Q2 (512-1k)':<12} | {'Q3 (1k-1.5k)':<12} | {'Q4 (1.5k-2k)':<12} | {'Drift (Q4-Q1)':<14} | {'Speed'}")
    print("-" * 110)

    all_results = {}
    for k in horizons:
        res = run_stream_2048(k)
        all_results[str(k)] = res
        print(f"K={k:<3d} | {res['total_loss']:<12.4f} | {res['q1_loss']:<12.4f} | {res['q2_loss']:<12.4f} | {res['q3_loss']:<12.4f} | {res['q4_loss']:<12.4f} | {res['drift']:<+14.4f} | {res['speed']:.1f} tok/s", flush=True)

    print("=" * 110)

    out_file = Path("results/three_clock_2048_tokens_ness_diagnostic.json")
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
