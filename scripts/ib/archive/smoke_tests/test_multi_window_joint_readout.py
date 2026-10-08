"""Multi-Window Continuous Prequential Evaluation of Fast/Slow Joint Readout.

Runs 3 consecutive updates across 4 continuous OWT windows (128 tokens total):
W1 (Train 1) -> W2 (Score & Train 2) -> W3 (Score & Train 3) -> W4 (Score 4)

Compares:
1. Frozen Baseline (z2 only)
2. Standard Single-Scale Readout (z2 only, online AdamW)
3. Fast/Slow Joint Readout ([h_motor, z1, z2], online AdamW)
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
from scripts.ib.test_fast_slow_alignment import FastSlowJointReadout


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

    # Pre-extract physical states along the single true physical trajectory
    print(f"Extracting physical states across {n_windows} windows ({total_tokens} tokens)...", flush=True)
    state = clone_state(s_init)
    input_tokens = torch.cat([all_tokens.new_tensor([prev_tok]), all_tokens[:-1]])

    h_m_windows = []
    z1_windows = []
    z2_windows = []
    unigram_windows = []

    bias_logits = model.decoder.bias.unsqueeze(0).expand(W, -1)

    with torch.no_grad():
        for w in range(n_windows):
            w_h_m = []
            w_z1 = []
            w_z2 = []
            for t in range(W):
                idx = w * W + t
                tok = input_tokens[idx:idx+1]
                state = advance_fly_input_event(
                    model, state, tok, settle_ticks=0, writer_baseline_clock="input", **options
                )
                h_m = (state.h - state.h_mean)[:, model.read_indices] if model.read_centering else state.h[:, model.read_indices]
                w_h_m.append(h_m)
                w_z1.append(state.gamma_z1)
                w_z2.append(state.gamma_z2)

            h_m_windows.append(torch.cat(w_h_m, dim=0).detach())
            z1_windows.append(torch.cat(w_z1, dim=0).detach())
            z2_windows.append(torch.cat(w_z2, dim=0).detach())

            target_w = all_tokens[w * W:(w + 1) * W]
            unigram_windows.append(F.cross_entropy(bias_logits, target_w).item())

    # 1. Arm A: Frozen Baseline
    frozen_scores = []
    with torch.no_grad():
        for w in range(n_windows):
            feat = model.output_read(z2_windows[w])
            logits = model.decoder(model.read_norm(feat))
            loss = F.cross_entropy(logits, all_tokens[w * W:(w + 1) * W]).item()
            frozen_scores.append(loss)

    # 2. Arm B: Online Standard Readout (z2 only)
    # Clone output_read
    model_b_read = nn.Linear(len(model.read_indices), model.output_read.out_features, bias=model.output_read.bias is not None).to(device)
    with torch.no_grad():
        model_b_read.weight.copy_(model.output_read.weight)
        if model.output_read.bias is not None:
            model_b_read.bias.copy_(model.output_read.bias)

    opt_b = torch.optim.AdamW(model_b_read.parameters(), lr=1e-3, weight_decay=1e-2)
    arm_b_prequential = []

    for w in range(n_windows):
        target_w = all_tokens[w * W:(w + 1) * W]
        # Prequential score FIRST (before update on this window)
        with torch.no_grad():
            feat = model_b_read(z2_windows[w])
            logits = model.decoder(model.read_norm(feat))
            score = F.cross_entropy(logits, target_w).item()
            arm_b_prequential.append(score)

        # Update on window w
        feat = model_b_read(z2_windows[w])
        logits = model.decoder(model.read_norm(feat))
        loss = F.cross_entropy(logits, target_w)
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()

    # 3. Arm C: Online Fast/Slow Joint Readout ([h_m, z1, z2])
    d_model = model.output_read.out_features
    joint_readout = FastSlowJointReadout(len(model.read_indices), d_model, model.output_read).to(device)
    opt_c = torch.optim.AdamW(joint_readout.parameters(), lr=1e-3, weight_decay=1e-2)
    arm_c_prequential = []

    for w in range(n_windows):
        target_w = all_tokens[w * W:(w + 1) * W]
        # Prequential score FIRST
        with torch.no_grad():
            feat = joint_readout(h_m_windows[w], z1_windows[w], z2_windows[w])
            logits = model.decoder(model.read_norm(feat))
            score = F.cross_entropy(logits, target_w).item()
            arm_c_prequential.append(score)

        # Update on window w
        feat = joint_readout(h_m_windows[w], z1_windows[w], z2_windows[w])
        logits = model.decoder(model.read_norm(feat))
        loss = F.cross_entropy(logits, target_w)
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()

    print("\n" + "=" * 78)
    print("PREQUENTIAL EVALUATION ACROSS 4 CONTINUOUS OWT WINDOWS (128 TOKENS)")
    print("=" * 78)
    print(f"{'Window':<8} | {'Unigram':<10} | {'Frozen (z2)':<12} | {'Standard z2':<13} | {'Fast/Slow Joint':<15} | {'Joint vs Std':<12}")
    print("-" * 78)

    for w in range(n_windows):
        u = unigram_windows[w]
        f = frozen_scores[w]
        b = arm_b_prequential[w]
        c = arm_c_prequential[w]
        d_cb = c - b
        tag = " (Init)" if w == 0 else f" (Follow {w})"
        print(f"W{w+1:<6} | {u:<10.4f} | {f:<12.4f} | {b:<13.4f} | {c:<15.4f} | {d_cb:+12.4f}{tag}")

    print("-" * 78)
    # Average across follow windows (W2..W4, strictly unseen before being scored)
    mean_follow_b = np.mean(arm_b_prequential[1:])
    mean_follow_c = np.mean(arm_c_prequential[1:])
    mean_follow_f = np.mean(frozen_scores[1:])
    print(f"Mean Unseen Follow NLL (W2..W4):")
    print(f"  Frozen Baseline    : {mean_follow_f:.4f} nats")
    print(f"  Standard z2 Readout: {mean_follow_b:.4f} nats (Delta: {mean_follow_b - mean_follow_f:+.4f})")
    print(f"  Fast/Slow Joint    : {mean_follow_c:.4f} nats (Delta: {mean_follow_c - mean_follow_f:+.4f})")
    print(f"  Net Joint Advantage: {mean_follow_c - mean_follow_b:+.4f} nats")


if __name__ == "__main__":
    main()
