"""Test Timescale Fusion Readout: Alpha Blend of [h_motor, z1, z2] sharing output_read.

Instead of an overparameterized 6999->128 matrix that overfits on short windows,
uses physically motivated timescale fusion:
  h_fused = alpha_fast * h_motor + alpha_mid * z1 + alpha_slow * z2
  feat = model.output_read(h_fused)

Tests:
1. Grid sweep of [alpha_fast, alpha_mid, alpha_slow] across 4 continuous windows (128 tokens)
   to find the optimal physical timescale combination for prequential OWT language modeling.
2. Online learned alpha (3 parameters only - cannot overfit!).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    advance_fly_input_event,
)
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt")
    saved, learner = load_checkpoint(ckpt_path, device=device)
    model = learner.model
    s_init = clone_state(learner.state)

    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])
    prev_tok = int(saved["learner"]["previous_token"])

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    W = 32
    n_windows = 4
    total_tokens = W * n_windows
    all_tokens = torch.as_tensor(data[cursor:cursor + total_tokens], device=device, dtype=torch.long)
    input_tokens = torch.cat([all_tokens.new_tensor([prev_tok]), all_tokens[:-1]])

    # Pre-extract physical states along the single true physical trajectory
    state = clone_state(s_init)
    h_m_all = []
    z1_all = []
    z2_all = []

    with torch.no_grad():
        for t in range(total_tokens):
            tok = input_tokens[t:t+1]
            state = advance_fly_input_event(
                model, state, tok, settle_ticks=0, writer_baseline_clock="input", **options
            )
            h_m = (state.h - state.h_mean)[:, model.read_indices] if model.read_centering else state.h[:, model.read_indices]
            h_m_all.append(h_m)
            z1_all.append(state.gamma_z1)
            z2_all.append(state.gamma_z2)

    H_fast = torch.cat(h_m_all, dim=0) # [128, 2333]
    H_mid = torch.cat(z1_all, dim=0)   # [128, 2333]
    H_slow = torch.cat(z2_all, dim=0)  # [128, 2333]

    bias_logits = model.decoder.bias.unsqueeze(0).expand(total_tokens, -1)
    unigram_nll = F.cross_entropy(bias_logits, all_tokens).item()

    print("=" * 75)
    print("GRID EVALUATION: Timescale Mixtures across 128 Continuous OWT Tokens")
    print("=" * 75)
    print(f"Unigram (Decoder Bias) 128-token Mean NLL: {unigram_nll:.4f} nats\n")

    grid_mixtures = [
        ("Current Baseline (100% z2)",           0.0,  0.0,  1.0),
        ("Pure Mid (100% z1)",                   0.0,  1.0,  0.0),
        ("Pure Fast (100% h_motor)",             1.0,  0.0,  0.0),
        ("Equal Blend (1/3 each)",               0.33, 0.33, 0.34),
        ("Dominant Slow + 10% Fast",             0.10, 0.0,  0.90),
        ("Dominant Slow + 20% Fast",             0.20, 0.0,  0.80),
        ("Dominant Slow + 10% Mid",              0.0,  0.10, 0.90),
        ("Slow + Mid (50/50)",                   0.0,  0.50, 0.50),
        ("Fast + Slow (50/50)",                  0.50, 0.0,  0.50),
        ("Tri-Scale (10% Fast, 20% Mid, 70% Slow)", 0.10, 0.20, 0.70),
        ("Tri-Scale (5% Fast, 15% Mid, 80% Slow)",  0.05, 0.15, 0.80),
    ]

    print(f"{'Mixture Name':<38} | {'Fast':<5} | {'Mid':<5} | {'Slow':<5} | {'128-Token NLL':<12} | {'vs Base':<9}")
    print("-" * 80)

    best_nll = float("inf")
    best_name = None

    with torch.no_grad():
        for name, a_f, a_m, a_s in grid_mixtures:
            H_fused = a_f * H_fast + a_m * H_mid + a_s * H_slow
            feat = model.output_read(H_fused)
            logits = model.decoder(model.read_norm(feat))
            loss = F.cross_entropy(logits, all_tokens).item()
            delta = loss - grid_mixtures[0][4] if "grid_mixtures[0][4]" in locals() else 0.0

            if name == grid_mixtures[0][0]:
                base_loss = loss
                delta = 0.0
            else:
                delta = loss - base_loss

            if loss < best_nll:
                best_nll = loss
                best_name = name

            marker = " <== BEST" if loss == best_nll else ""
            print(f"{name:<38} | {a_f:<5.2f} | {a_m:<5.2f} | {a_s:<5.2f} | {loss:<12.4f} | {delta:+9.4f}{marker}")

    # Now let's optimize alpha directly with a 3-parameter continuous softmax!
    print("\n" + "=" * 75)
    print("CONTINUOUS OPTIMIZATION: Learning Optimal Alpha (3 Parameters)")
    print("=" * 75)

    alpha_logits = nn.Parameter(torch.tensor([0.0, 0.0, 2.0], device=device)) # biased towards z2 initially
    opt_alpha = torch.optim.Adam([alpha_logits], lr=0.05)

    for step in range(30):
        weights = F.softmax(alpha_logits, dim=0)
        H_fused = weights[0] * H_fast + weights[1] * H_mid + weights[2] * H_slow
        feat = model.output_read(H_fused)
        logits = model.decoder(model.read_norm(feat))
        loss = F.cross_entropy(logits, all_tokens)
        opt_alpha.zero_grad()
        loss.backward()
        opt_alpha.step()
        if (step + 1) % 5 == 0 or step == 0:
            w_np = weights.detach().cpu().numpy()
            print(f"  Step {step+1:2d} | NLL: {loss.item():.4f} (Delta: {loss.item() - base_loss:+.4f}) | Weights: Fast={w_np[0]:.3f}, Mid={w_np[1]:.3f}, Slow={w_np[2]:.3f}")

    opt_weights = F.softmax(alpha_logits, dim=0).detach().cpu().numpy()
    final_loss = loss.item()
    print(f"\nOptimal Timescale Weights: Fast={opt_weights[0]:.3f}, Mid={opt_weights[1]:.3f}, Slow={opt_weights[2]:.3f}")
    print(f"Total NLL Improvement: {final_loss - base_loss:+.4f} nats across 128 continuous tokens.")


if __name__ == "__main__":
    main()
