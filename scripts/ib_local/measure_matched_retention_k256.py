"""Matched-Retention Closed-Loop Stream Rollout across K in [1, 3, 8, 16, 32, 64, 128, 256].

Theoretical Hypotheses:
1. Two-Clock Decoupling:
   - External Token Lifetime Clock: controls global memory retention leak D_retention = exp(-gamma_0 * Delta t_token).
   - Internal Computation Clock: controls pondering microsteps Phi_tau = (T + C + D_visc)^K.
2. In the unscaled baseline, running K=32..256 multiplies the retention leak by K times,
   starving persistent memory (E -> 0.001) and degrading closed-loop NLL.
3. In Matched Retention, scaling gamma_{0, K} = gamma_{0, ref} * (3 / K) keeps the total
   token-level retention budget constant: sum_{k=1}^K gamma_{0, K} = 3 * gamma_0 = const.
   Tests whether deep pondering (K=16..256) unlocks closed-loop gains once energy starvation is eliminated.
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

    num_tokens = 512
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

    def run_stream(k_steps, gamma0_fac):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            state = model.initial_state(1, "cuda", warm_start=True)
            # Warmup
            for offset in range(0, warmup_tokens, 128):
                ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
                targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
                _, state, _ = model(ids, targets, state)

            total_loss = 0.0
            for t in range(warmup_tokens, warmup_tokens + num_tokens):
                inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
                tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
                logits, state, diag = model.step(
                    state, inp_id,
                    micro_steps=k_steps,
                    gamma0_factor=gamma0_fac
                )
                total_loss += F.cross_entropy(logits, tgt_id).item()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_nll = total_loss / num_tokens
        speed = num_tokens / elapsed
        final_energy = float(diag.get("energy", 0.0))
        return avg_nll, final_energy, speed

    print("\n" + "=" * 92)
    print("      MATCHED-RETENTION CLOSED-LOOP STREAM SWEEP (K in [1 .. 256], N = 512 TOKENS)")
    print("=" * 92)
    print(f"{'K':<5} | {'Unscaled NLL':<14} | {'Unscaled E':<12} | {'Matched NLL':<14} | {'Matched E':<12} | {'Delta (NLL)':<12} | {'Speed'}")
    print("-" * 92)

    results = {}

    for k in horizons:
        # 1. Unscaled baseline: gamma0_factor = 1.0
        nll_unscaled, e_unscaled, spd_unscaled = run_stream(k, gamma0_fac=1.0)

        # 2. Matched retention: gamma0_factor = 3.0 / k
        gamma0_matched_fac = 3.0 / float(k)
        nll_matched, e_matched, spd_matched = run_stream(k, gamma0_fac=gamma0_matched_fac)

        delta_nll = nll_matched - nll_unscaled
        delta_str = f"{delta_nll:+.4f}"

        results[str(k)] = {
            "unscaled_nll": nll_unscaled, "unscaled_e": e_unscaled,
            "matched_nll": nll_matched, "matched_e": e_matched,
            "delta_nll": delta_nll,
            "speed_tok_sec": spd_matched,
        }

        print(f"{k:<5d} | {nll_unscaled:<14.4f} | {e_unscaled:<12.4f} | {nll_matched:<14.4f} | {e_matched:<12.4f} | {delta_str:<12} | {spd_matched:.1f} tok/s")

    print("=" * 92)

    out_file = Path("results/matched_retention_k256_diagnostic.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
