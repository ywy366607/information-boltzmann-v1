"""Deep Counterfactual Pondering Probe out to K=64.

Theoretical Questions:
1. Does the slow branch continue to monotonically improve beyond K=16 (K=24, 32, 48, 64)?
2. Where is the true natural saturation peak of internal fluid pondering?
3. What is the expanded Oracle headroom when K in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]?
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

    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        v2_coordinate_components=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True, continuous_velocities=True,
        dissipation_type="unified", dissipation_rank=4
    ).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")
    num_tokens = 1024
    warmup_tokens = 256

    probe_depths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    max_k = max(probe_depths)
    print(f"Executing Deep Pondering Probe out to K={max_k} over {num_tokens} tokens...")

    token_nlls = {k: [] for k in probe_depths}
    best_k_list = []

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)

        print(f"Warming up state over {warmup_tokens} tokens...")
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        print(f"Executing deep counterfactual sweeps...")
        for t in range(warmup_tokens, warmup_tokens + num_tokens):
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")

            tok_embed = model.source.embedding(inp_id)
            f_written, _, _ = model.source(state, inp_id)

            f_sim = f_written.clone()
            k_losses = {}
            next_canonical_state = None

            for k_step in range(1, max_k + 1):
                if model.adaptive_clock:
                    alpha_k = model.clock(f_sim, tok_embed)
                    dt_k = alpha_k * model.tau_0_tensor
                else:
                    dt_k = model.tau_0_tensor

                if model.continuous_velocities:
                    dir_k = model.direction_controller(f_sim, tok_embed)
                else:
                    dir_k = None

                mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                f_sim = model.transport.apply_multiplier(f_sim, mult)
                f_sim, _ = model.collision(f_sim, dt_k)
                f_sim, _ = model.bath(f_sim, dt_k, tok_embed=tok_embed)

                if k_step == 3:
                    next_canonical_state = f_sim.clone()

                if k_step in probe_depths:
                    feat_k, _ = model.readout(f_sim, tok_embed, return_diag=False)
                    logits_k = model.decoder(feat_k)
                    loss_k = F.cross_entropy(logits_k, tgt_id).item()
                    k_losses[k_step] = loss_k
                    token_nlls[k_step].append(loss_k)

            best_k = min(probe_depths, key=lambda k: k_losses[k])
            best_k_list.append(best_k)
            state = next_canonical_state

            if (t - warmup_tokens + 1) % 128 == 0:
                print(f"  Processed {t - warmup_tokens + 1} / {num_tokens} tokens...")

    # Statistics
    mean_nll = {k: float(np.mean(token_nlls[k])) for k in probe_depths}
    l_3 = mean_nll[3]
    best_fixed_k = min(probe_depths, key=lambda k: mean_nll[k])
    best_fixed_nll = mean_nll[best_fixed_k]

    oracle_nlls = [min(token_nlls[k][i] for k in probe_depths) for i in range(num_tokens)]
    l_oracle = float(np.mean(oracle_nlls))
    headroom = l_3 - l_oracle

    k_counts = {k: best_k_list.count(k) for k in probe_depths}
    k_pcts = {k: k_counts[k] / num_tokens for k in probe_depths}

    print("\n" + "=" * 80)
    print("      DEEP COUNTERFACTUAL PONDERING DIAGNOSTIC (K = 1 .. 64, N = 1024 TOKENS)")
    print("=" * 80)
    print(f"{'Internal Depth K':<18} | {'Mean Validation NLL':<24} | {'Delta vs K=3':<16} | {'K* Optimal %'}")
    print("-" * 80)
    for k in probe_depths:
        delta = mean_nll[k] - l_3
        d_str = f"{delta:+.4f}" if k != 3 else "0.0000 (Base)"
        print(f"K = {k:<14d} | {mean_nll[k]:<24.4f} | {d_str:<16} | {k_pcts[k]:<10.1%}")
    print("-" * 80)
    print(f"Baseline Fixed Depth:     K = 3  (NLL = {l_3:.4f} nats)")
    print(f"Best Fixed Depth:         K = {best_fixed_k} (NLL = {best_fixed_nll:.4f} nats, Delta = {best_fixed_nll - l_3:+.4f} nats)")
    print(f"Expanded Oracle Headroom: L_oracle = {l_oracle:.4f} nats (Net Gain = {headroom:+.4f} nats)")
    print("=" * 80)

    out_file = Path("results/deep_pondering_k64_diagnostic.json")
    results = {
        "probe_depths": probe_depths,
        "mean_nll": mean_nll,
        "best_fixed_k": best_fixed_k,
        "best_fixed_nll": best_fixed_nll,
        "l_oracle": l_oracle,
        "headroom": headroom,
        "k_counts": k_counts,
        "k_pcts": k_pcts,
    }
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
