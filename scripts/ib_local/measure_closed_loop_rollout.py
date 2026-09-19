"""Closed-Loop Persistent Rollout across Fixed Horizons K in {1, 3, 8, 16, 32}.

Theoretical Question:
Does unforced recurrent depth extrapolation hold in true closed-loop state rollouts?
In this test, each horizon K evolves its own continuous state trajectory across 1024 tokens:
    F_{t+1} = Phi_K( W(F_t, x_t) )
No branch resets; evaluates true persistent dynamical memory over long horizons.
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

    print(f"Loading reference model from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location="cuda")
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")

    num_tokens = 1024
    warmup_tokens = 256
    fixed_k_list = [1, 3, 8, 16, 24, 32]

    results = {}

    print("\n" + "=" * 80)
    print(f"   CLOSED-LOOP PERSISTENT STREAM ROLLOUT (N = {num_tokens} TOKENS)")
    print("=" * 80)
    print(f"{'Horizon K':<12} | {'Validation NLL':<22} | {'Delta vs K=3':<16} | {'Final Energy E':<16} | {'Speed'}")
    print("-" * 80)

    l_3_baseline = None

    for k in fixed_k_list:
        model = CBIMTorus3D(
            shape=(8, 8, 4), velocities=8, content_dim=16,
            v2_coordinate_components=True,
            readout_type="kernel_r1", write_type="w2_impedance",
            micro_steps=k, adaptive_clock=True, continuous_velocities=True,
            dissipation_type="unified", dissipation_rank=4
        ).cuda()
        model.load_state_dict(saved["model"], strict=True)
        model.eval()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            # Warm up state over 256 tokens using horizon k
            state = model.initial_state(1, "cuda", warm_start=True)
            for offset in range(0, warmup_tokens, 128):
                ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
                targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
                _, state, _ = model(ids, targets, state)

            # Continuous closed-loop streaming over 1024 tokens
            total_loss = 0.0
            for t in range(warmup_tokens, warmup_tokens + num_tokens):
                inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                logits, state, diag = model.step(state, inp_id)
                loss = F.cross_entropy(logits, tgt_id).item()
                total_loss += loss

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_nll = total_loss / num_tokens
        speed = num_tokens / elapsed
        final_energy = float(diag.get("energy", 0.0))

        if k == 3:
            l_3_baseline = avg_nll
            delta_str = "0.0000 (Base)"
        elif l_3_baseline is not None:
            delta_str = f"{avg_nll - l_3_baseline:+.4f}"
        else:
            delta_str = "N/A"

        results[str(k)] = {
            "nll": avg_nll,
            "final_energy": final_energy,
            "speed_tok_sec": speed,
            "elapsed_sec": elapsed,
        }
        print(f"K = {k:<8d} | {avg_nll:<22.4f} | {delta_str:<16} | {final_energy:<16.4f} | {speed:.1f} tok/s")

    print("=" * 80)

    # Save results
    out_file = Path("results/closed_loop_rollout_k_sweep.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
