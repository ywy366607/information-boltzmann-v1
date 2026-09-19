from __future__ import annotations

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
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.gated_deltanet_2 import GatedDeltaNet2LM


@torch.no_grad()
def evaluate_long_sequence(model, tokens, chunk_size=128, reset_interval=None):
    """Run an un-reset long sequence and track Frobenius norm, operator sigma_max, and Lyapunov exponents."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tokens = len(tokens)
    state = model.initial_state(1, device=device)

    state_norms = []
    sigma_max_list = []
    rho_list = []

    # Track Lyapunov exponent via perturbed trajectory
    eps = 1e-6
    state_pert = state.clone()
    perturbation = torch.randn_like(state_pert)
    perturbation = perturbation / perturbation.norm() * eps
    state_pert = state_pert + perturbation
    log_divergences = []

    # Gate stats
    decay_stats, erase_stats, write_stats = [], [], []

    for start in range(0, num_tokens, chunk_size):
        if reset_interval and start > 0 and start % reset_interval == 0:
            state = model.initial_state(1, device=device)
            state_pert = state.clone() + (torch.randn_like(state) / torch.randn_like(state).norm() * eps)

        chunk = torch.as_tensor(tokens[start:start+chunk_size], dtype=torch.long, device=device)[None]
        if chunk.shape[1] < chunk_size:
            break

        # Unperturbed step
        _, next_state, diags = model(chunk, state=state)
        # Perturbed step
        _, next_state_pert, _ = model(chunk, state=state_pert)

        # Measure divergence for Lyapunov exponent
        diff = (next_state_pert - next_state).norm()
        log_divergences.append(math.log(max(float(diff) / eps, 1e-12)) / chunk_size)
        # Re-normalize perturbation
        state_pert = next_state + (next_state_pert - next_state) / diff * eps

        state = next_state
        current_norm = float(state.square().sum().sqrt())
        state_norms.append((start + chunk_size, current_norm))

        if "decay_mean" in diags:
            decay_stats.append(float(diags["decay_mean"]))
            erase_stats.append(float(diags["erase_mean"]))
            write_stats.append(float(diags["write_mean"]))
        elif "alpha_mean" in diags:
            decay_stats.append(float(diags["alpha_mean"]))
            write_stats.append(float(diags["beta_mean"]))

    mle = float(np.mean(log_divergences)) if log_divergences else 0.0
    return {
        "final_state_norm": state_norms[-1][1] if state_norms else 0.0,
        "max_state_norm": max(x[1] for x in state_norms) if state_norms else 0.0,
        "state_norm_trajectory": state_norms,
        "lyapunov_exponent": mle,
        "decay_mean": float(np.mean(decay_stats)) if decay_stats else 0.0,
        "erase_mean": float(np.mean(erase_stats)) if erase_stats else 0.0,
        "write_mean": float(np.mean(write_stats)) if write_stats else 0.0,
    }


def main():
    val_data = np.load("data/ib_owt_gpt2/validation.npy", mmap_mode="r")
    lengths = [2048, 8192, 32768, 65536] # up to 65k continuous tokens

    # Checkpoints
    gdn_ckpt = torch.load("results/gated_deltanet_d128_3000/BBest.pt", map_location="cuda")
    gdn_model = GatedDeltaNetLM(vocab_size=50257, d=128, layers=4, heads=4).cuda()
    gdn_model.load_state_dict(gdn_ckpt["model"])

    # Load GDN-2 (or current last/BBest)
    gdn2_ckpt_path = Path("results/gated_deltanet_2_d128_3000/last.pt")
    if not gdn2_ckpt_path.exists():
        gdn2_ckpt_path = Path("results/gated_deltanet_2_d128_3000/BBest.pt")
    gdn2_ckpt = torch.load(gdn2_ckpt_path, map_location="cuda")
    gdn2_model = GatedDeltaNet2LM(vocab_size=50257, d=128, layers=3, heads=4).cuda()
    gdn2_model.load_state_dict(gdn2_ckpt["model"])

    results = {}
    print("=== Long Sequence Audit: GDN vs GDN-2 ===")
    for length in lengths:
        sub_tokens = val_data[:length]
        print(f"\nEvaluating Length = {length:,} tokens...")

        # 1. GDN (Standard)
        res_gdn = evaluate_long_sequence(gdn_model, sub_tokens)
        print(f"  [GDN]  Max Norm: {res_gdn['max_state_norm']:.4f}, Final Norm: {res_gdn['final_state_norm']:.4f}, Lyapunov: {res_gdn['lyapunov_exponent']:+.4f}")

        # 2. GDN-2 (Un-reset)
        res_gdn2 = evaluate_long_sequence(gdn2_model, sub_tokens)
        print(f"  [GDN2] Max Norm: {res_gdn2['max_state_norm']:.4f}, Final Norm: {res_gdn2['final_state_norm']:.4f}, Lyapunov: {res_gdn2['lyapunov_exponent']:+.4f}")

        # 3. GDN-2 (Reset every 2048)
        res_gdn2_reset = evaluate_long_sequence(gdn2_model, sub_tokens, reset_interval=2048)
        print(f"  [GDN2 Reset-2048] Max Norm: {res_gdn2_reset['max_state_norm']:.4f}, Final Norm: {res_gdn2_reset['final_state_norm']:.4f}")

        results[length] = {
            "gdn": res_gdn,
            "gdn2": res_gdn2,
            "gdn2_reset2048": res_gdn2_reset,
        }

    with open("results/gdn_stability_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAudit saved to results/gdn_stability_audit.json")


if __name__ == "__main__":
    main()
