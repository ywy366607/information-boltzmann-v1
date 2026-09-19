"""Measure Three-Clock Decoupled Closed-Loop Stream across K in [0, 1, 3, 8, 16, 32, 64].

Theoretical Architecture (Three-Clock Separation):
1. Event Clock (t): Arrival of external token x_t.
   F_t^+ = Write(F_t, x_t).
2. Boundary Shaping (once):
   tilde{F}_t = S_nu(F_t^+)  (Spectral viscosity applied ONCE to kill injection ringing).
3. Pondering Computation Clock (tau):
   F_{t, k+1} = C_{dt_k} T_{dt_k} F_{t, k}  (Pure conservative Lie-algebra rotation for K steps).
4. Readout:
   y_t = Readout(F_{t, K_t}, x_t).
5. Memory Retention Clock (once per token):
   F_{t+1} = D_mem(F_{t, K_t}) = exp(-3 * tau_0 * (gamma_0 I + U Lambda U^T)) F_{t, K_t}.
   (Memory aging executed ONCE per token, totally decoupled from computation steps K).

Evaluates on 512 tokens:
- Three-Clock Architecture across K in [0, 1, 3, 8, 16, 32, 64].
- Compares with Old Microstep-D Baseline and Heuristic Matched Retention.
- Measures NLL, energy stability (E_final), and token speed (tok/s).
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

    num_tokens = 512
    warmup_tokens = 128
    horizons = [0, 1, 3, 8, 16, 32, 64]

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    def run_three_clock_stream(k_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            state = model.initial_state(1, "cuda", warm_start=True)
            # Warmup with standard model forward (3 steps)
            for offset in range(0, warmup_tokens, 128):
                ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
                targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
                _, state, _ = model(ids, targets, state)

            total_loss = 0.0
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

                    # T
                    mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                    field = model.transport.apply_multiplier(field, mult)
                    # C
                    field, _ = model.collision(field, dt_k)
                    # (NO dissipation in microsteps!)

                # 4. Readout
                feat, _ = model.readout(field, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                total_loss += F.cross_entropy(logits, tgt_id).item()

                # 5. Memory Retention Clock (applied ONCE per token)
                # D_mem = exp(-3 * tau_0 * (gamma_0 + U Lambda U^T))
                state, _ = model.bath(
                    field, dt_nominal, tok_embed=tok_embed,
                    gamma0_factor=1.0, disable_viscosity=True, disable_subspace=False
                )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_nll = total_loss / num_tokens
        speed = num_tokens / elapsed
        final_e = float(state.square().mean().item())
        return avg_nll, final_e, speed

    print("\n" + "=" * 90)
    print("      THREE-CLOCK DECOUPLED CLOSED-LOOP STREAM SWEEP (N = 512 TOKENS)")
    print("=" * 90)
    print(f"{'K':<5} | {'Three-Clock NLL':<18} | {'Field Energy E':<16} | {'Speed (tok/s)':<16} | {'Status'}")
    print("-" * 90)

    results = {}

    for k in horizons:
        nll_val, e_val, spd_val = run_three_clock_stream(k)
        status = "HEALTHY" if e_val > 0.005 else ("WEAK" if e_val > 0.001 else "FROZEN")
        results[str(k)] = {
            "nll": nll_val,
            "energy": e_val,
            "speed_tok_sec": spd_val,
            "status": status,
        }
        print(f"K={k:<3d} | {nll_val:<18.4f} | {e_val:<16.6f} | {spd_val:<16.1f} | {status}", flush=True)

    print("=" * 90)

    out_file = Path("results/three_clock_decoupled_closed_loop_diagnostic.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
