"""Oracle Adaptive Pondering Horizon Diagnostic.

Evaluates whether varying internal thinking depth K in {1, 2, 3, 4, 6, 8, 12, 16}
unlocks hidden predictive headroom beyond the fixed K=3 training ceiling.

Theoretical Questions Answered:
1. Baseline Fixed Depth: L_{K=3} = E[L_3]
2. Best Global Fixed Depth: min_K E[L_K]
3. Oracle Adaptive Depth: L_oracle = E[min_K L_K]
4. Oracle Headroom: E[L_3] - L_oracle (in nats)
5. Distribution of Optimal Depth K*(x) across validation tokens
6. Correlation between token difficulty L_3 and optimal thinking depth K*(x)
7. Linguistic inspection of tokens requiring shallow (K=1,2) vs deep (K=8,12,16) deliberation.
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
import math

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
    print(f"Evaluating Oracle Pondering across {num_tokens} validation tokens...")

    # Horizons to probe
    probe_depths = [1, 2, 3, 4, 6, 8, 12, 16]
    max_k = max(probe_depths)

    # Storage
    token_nlls = {k: [] for k in probe_depths}
    best_k_list = []
    tokens_text = []
    target_ids = []

    # Simple gpt2 tokenizer decoder if available
    try:
        from transformers import AutoTokenizer
        tok_obj = AutoTokenizer.from_pretrained("gpt2")
        decode_fn = lambda tid: tok_obj.decode([tid])
    except Exception:
        decode_fn = lambda tid: f"id_{tid}"

    with torch.no_grad():
        state = model.initial_state(1, "cuda", warm_start=True)

        # Warm up state over 256 tokens to ensure fully mature background field
        warmup_tokens = 256
        print(f"Warming up state over {warmup_tokens} tokens...")
        for offset in range(0, warmup_tokens, 128):
            ids = torch.as_tensor(np.array(val_data[offset:offset + 128]), dtype=torch.long, device="cuda")[None]
            targets = torch.as_tensor(np.array(val_data[offset + 1:offset + 129]), dtype=torch.long, device="cuda")[None]
            _, state, _ = model(ids, targets, state)

        print(f"Executing counterfactual depth sweeps (K in {probe_depths}) on {num_tokens} tokens...")
        for t in range(warmup_tokens, warmup_tokens + num_tokens):
            inp_id = torch.as_tensor([val_data[t]], dtype=torch.long, device="cuda")
            tgt_id = torch.as_tensor([val_data[t + 1]], dtype=torch.long, device="cuda")
            target_ids.append(int(tgt_id.item()))
            tokens_text.append(decode_fn(int(tgt_id.item())))

            tok_embed = model.source.embedding(inp_id)

            # 1. Injection via Write (W)
            f_written, _, _ = model.source(state, inp_id)

            # 2. Counterfactual multi-depth simulation from f_written
            f_sim = f_written.clone()
            k_losses = {}

            # Canonical K=3 state to advance the background stream
            next_canonical_state = None

            for k_step in range(1, max_k + 1):
                # Adaptive clock and continuous direction
                if model.adaptive_clock:
                    alpha_k = model.clock(f_sim, tok_embed)
                    dt_k = alpha_k * model.tau_0_tensor
                else:
                    dt_k = model.tau_0_tensor

                if model.continuous_velocities:
                    dir_k = model.direction_controller(f_sim, tok_embed)
                else:
                    dir_k = None

                # Transport -> Collision -> Dissipation
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

            # Determine best depth K* for this token
            best_k = min(probe_depths, key=lambda k: k_losses[k])
            best_k_list.append(best_k)

            # Advance background state using canonical K=3
            state = next_canonical_state

            if (t - warmup_tokens + 1) % 256 == 0:
                print(f"  Processed {t - warmup_tokens + 1} / {num_tokens} tokens...")

    # Calculate statistics
    mean_nll = {k: float(np.mean(token_nlls[k])) for k in probe_depths}
    l_3 = mean_nll[3]
    best_fixed_k = min(probe_depths, key=lambda k: mean_nll[k])
    best_fixed_nll = mean_nll[best_fixed_k]

    oracle_nlls = [min(token_nlls[k][i] for k in probe_depths) for i in range(num_tokens)]
    l_oracle = float(np.mean(oracle_nlls))
    headroom = l_3 - l_oracle

    # Distribution of K*
    k_counts = {k: best_k_list.count(k) for k in probe_depths}
    k_pcts = {k: k_counts[k] / num_tokens for k in probe_depths}

    # Correlation between token difficulty L3 and optimal depth K*
    l3_arr = np.array(token_nlls[3])
    k_arr = np.array(best_k_list)
    r_corr = float(np.corrcoef(l3_arr, k_arr)[0, 1])

    # Mean difficulty per K* group
    diff_per_k = {}
    for k in probe_depths:
        idxs = [i for i, bk in enumerate(best_k_list) if bk == k]
        if idxs:
            diff_per_k[k] = float(np.mean(l3_arr[idxs]))
        else:
            diff_per_k[k] = 0.0

    print("\n" + "=" * 76)
    print("        ORACLE ADAPTIVE PONDERING DIAGNOSTIC RESULTS (N = 1024 TOKENS)")
    print("=" * 76)
    print(f"{'Internal Depth K':<20} | {'Mean Validation NLL (nats)':<28} | {'Delta vs K=3'}")
    print("-" * 76)
    for k in probe_depths:
        delta = mean_nll[k] - l_3
        d_str = f"{delta:+.4f}" if k != 3 else "0.0000 (Baseline)"
        print(f"K = {k:<16d} | {mean_nll[k]:<28.4f} | {d_str}")
    print("-" * 76)
    print(f"Best Fixed Depth:        K = {best_fixed_k} (NLL = {best_fixed_nll:.4f} nats, Delta = {best_fixed_nll - l_3:+.4f})")
    print(f"Oracle Adaptive Depth:   L_oracle = {l_oracle:.4f} nats")
    print(f"Total Oracle Headroom:   {headroom:+.4f} nats (Drop of {headroom:.4f} nats)")
    print("=" * 76)

    print("\n" + "=" * 76)
    print("           OPTIMAL THINKING DEPTH K*(x) DISTRIBUTION & DIFFICULTY")
    print("=" * 76)
    print(f"{'Depth K*':<12} | {'Token Count':<14} | {'Percentage':<14} | {'Mean Baseline Difficulty (L_3)'}")
    print("-" * 76)
    for k in probe_depths:
        print(f"K* = {k:<8d} | {k_counts[k]:<14d} | {k_pcts[k]:<13.1%} | {diff_per_k[k]:.4f} nats")
    print("-" * 76)
    print(f"Correlation r(L_3, K*):   {r_corr:+.4f} ({'Positive correlation: harder tokens demand more depth' if r_corr > 0 else 'Uncorrelated'})")
    print("=" * 76)

    # Sample qualitative inspection: 5 examples of K*=1 vs K*=16
    print("\n" + "=" * 76)
    print("             QUALITATIVE TOKEN DELIBERATION EXAMPLES")
    print("=" * 76)
    shallow_examples = [(tokens_text[i], token_nlls[3][i], token_nlls[1][i]) for i, k in enumerate(best_k_list) if k == 1][:6]
    deep_examples = [(tokens_text[i], token_nlls[3][i], token_nlls[max(probe_depths)][i]) for i, k in enumerate(best_k_list) if k in (12, 16)][:6]

    print("Shallow Tokens (K*=1: rapid intuitive response, no deliberation needed):")
    for tok_repr, l3_val, lk_val in shallow_examples:
        print(f"  Token: {tok_repr!r:<18} | L_3: {l3_val:.3f} -> L_1: {lk_val:.3f} (saved {l3_val - lk_val:+.3f} nats)")

    print("\nDeep Tokens (K* in {12, 16}: complex deliberation / dispute resolution):")
    for tok_repr, l3_val, lk_val in deep_examples:
        print(f"  Token: {tok_repr!r:<18} | L_3: {l3_val:.3f} -> L_deep: {lk_val:.3f} (saved {l3_val - lk_val:+.3f} nats)")
    print("=" * 76)

    out_file = Path("results/oracle_pondering_diagnostic_1024tokens.json")
    results = {
        "num_tokens": num_tokens,
        "probe_depths": probe_depths,
        "mean_nll_per_k": mean_nll,
        "baseline_l3": l_3,
        "best_fixed_k": best_fixed_k,
        "best_fixed_nll": best_fixed_nll,
        "l_oracle": l_oracle,
        "oracle_headroom": headroom,
        "k_counts": k_counts,
        "k_percentages": k_pcts,
        "correlation_l3_k": r_corr,
        "mean_difficulty_per_k": diff_per_k,
    }
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_file}")


if __name__ == "__main__":
    main()
