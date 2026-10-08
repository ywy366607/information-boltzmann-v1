"""Test Fast/Slow Joint Readout & Causal Alignment on Continuous Follow Windows.

Prequential Evaluation Protocol:
1. Window 1 (cursor 1,600,000..1,600,032):
   - Measure initial NLL.
   - Run 1 realistic AdamW update (lr=1e-3, matching production learner lr_decoder).
2. Window 2 (cursor 1,600,032..1,600,064 - STRICT UNSEEN FOLLOW STREAM):
   - Measure Prequential NLL (scored strictly before any Window 2 update):
     Compare:
     A. Frozen Baseline (z2 only, original weights)
     B. Standard Baseline (z2 only, updated on Window 1)
     C. Fast/Slow Joint Readout (h_motor + z1 + z2, updated on Window 1)
     D. Joint Readout + 1-tick Causal Delay Alignment
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


def run_window_forward(model, readout_fn, state, input_tokens, target_tokens, options):
    """Executes a 32-token window forward pass, returning per-token NLL and next state."""
    nll_list = []
    h_m_list = []
    z1_list = []
    z2_list = []

    with torch.no_grad():
        for t in range(len(input_tokens)):
            tok = input_tokens[t:t+1]
            state = advance_fly_input_event(
                model, state, tok, settle_ticks=0, writer_baseline_clock="input", **options
            )
            h_m = (state.h - state.h_mean)[:, model.read_indices] if model.read_centering else state.h[:, model.read_indices]
            h_m_list.append(h_m)
            z1_list.append(state.gamma_z1)
            z2_list.append(state.gamma_z2)

        h_m_all = torch.cat(h_m_list, dim=0).detach()
        z1_all = torch.cat(z1_list, dim=0).detach()
        z2_all = torch.cat(z2_list, dim=0).detach()

    features = readout_fn(h_m_all, z1_all, z2_all)
    logits = model.decoder(model.read_norm(features))
    scores = F.cross_entropy(logits, target_tokens, reduction="none")
    return scores, state, (h_m_all, z1_all, z2_all)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}...")

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

    # Prepare Window 1 and Window 2 tokens
    W = 32
    w1_tokens = torch.as_tensor(data[cursor:cursor + W], device=device, dtype=torch.long)
    w2_tokens = torch.as_tensor(data[cursor + W:cursor + 2 * W], device=device, dtype=torch.long)

    w1_inputs = torch.cat([w1_tokens.new_tensor([prev_tok]), w1_tokens[:-1]])
    w2_inputs = torch.cat([w1_tokens[-1:], w2_tokens[:-1]])

    # Unigram baselines
    bias_logits = model.decoder.bias.unsqueeze(0)
    u_w1 = F.cross_entropy(bias_logits.expand(W, -1), w1_tokens).item()
    u_w2 = F.cross_entropy(bias_logits.expand(W, -1), w2_tokens).item()

    print("\n" + "=" * 70)
    print("EXPERIMENTAL PROTOCOL: Window 1 (Train) -> Window 2 (Prequential Follow)")
    print("=" * 70)
    print(f"Window 1 Unigram NLL: {u_w1:.4f} nats")
    print(f"Window 2 Unigram NLL: {u_w2:.4f} nats")

    # -------------------------------------------------------------------------
    # ARM A: Frozen Baseline (z2 only, original weights)
    # -------------------------------------------------------------------------
    def base_readout(h, z1, z2):
        return model.output_read(z2)

    with torch.no_grad():
        w1_scores_a, s_after_w1_a, _ = run_window_forward(model, base_readout, clone_state(s_init), w1_inputs, w1_tokens, options)
        w2_scores_a, _, _ = run_window_forward(model, base_readout, clone_state(s_after_w1_a), w2_inputs, w2_tokens, options)

    nll_w1_base = w1_scores_a.mean().item()
    nll_w2_base = w2_scores_a.mean().item()
    print(f"\n[Arm A: Frozen Baseline (z2 only)]")
    print(f"  Window 1 NLL : {nll_w1_base:.4f} (vs unigram: {nll_w1_base - u_w1:+.4f})")
    print(f"  Window 2 NLL : {nll_w2_base:.4f} (vs unigram: {nll_w2_base - u_w2:+.4f})")

    # -------------------------------------------------------------------------
    # ARM B: Standard Baseline Readout (z2 only) with 1 Adam Update on Window 1
    # -------------------------------------------------------------------------
    # Clone original output_read
    orig_output_read_w = model.output_read.weight.detach().clone()
    orig_output_read_b = model.output_read.bias.detach().clone() if model.output_read.bias is not None else None

    # Forward Window 1 with autograd on output_read
    model.output_read.weight.requires_grad_(True)
    if model.output_read.bias is not None:
        model.output_read.bias.requires_grad_(True)

    scores_w1_b, s_w1_b, (_, _, z2_b) = run_window_forward(model, base_readout, clone_state(s_init), w1_inputs, w1_tokens, options)
    loss_b = scores_w1_b.mean()

    opt_b = torch.optim.AdamW(model.output_read.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_b.zero_grad()
    loss_b.backward()
    opt_b.step()

    # Now evaluate on Window 2 with updated output_read
    with torch.no_grad():
        w2_scores_b, _, _ = run_window_forward(model, base_readout, clone_state(s_w1_b), w2_inputs, w2_tokens, options)
    nll_w2_arm_b = w2_scores_b.mean().item()

    print(f"\n[Arm B: Standard z2 Readout Updated on W1]")
    print(f"  Window 2 Follow NLL: {nll_w2_arm_b:.4f} (Delta from frozen: {nll_w2_arm_b - nll_w2_base:+.4f})")

    # Restore original output_read
    with torch.no_grad():
        model.output_read.weight.copy_(orig_output_read_w)
        if orig_output_read_b is not None:
            model.output_read.bias.copy_(orig_output_read_b)

    # -------------------------------------------------------------------------
    # ARM C: Fast/Slow Joint Readout ([h_motor, z1, z2]) with 1 Adam Update on W1
    # -------------------------------------------------------------------------
    d_model = model.output_read.out_features
    joint_readout = FastSlowJointReadout(len(model.read_indices), d_model, model.output_read).to(device)

    def joint_readout_fn(h, z1, z2):
        return joint_readout(h, z1, z2)

    # Window 1 forward
    scores_w1_c, s_w1_c, _ = run_window_forward(model, joint_readout_fn, clone_state(s_init), w1_inputs, w1_tokens, options)
    loss_c = scores_w1_c.mean()

    opt_c = torch.optim.AdamW(joint_readout.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()

    # Window 2 follow evaluation
    with torch.no_grad():
        w2_scores_c, _, _ = run_window_forward(model, joint_readout_fn, clone_state(s_w1_c), w2_inputs, w2_tokens, options)
    nll_w2_arm_c = w2_scores_c.mean().item()

    print(f"\n[Arm C: Fast/Slow Joint Readout Updated on W1]")
    print(f"  Window 2 Follow NLL: {nll_w2_arm_c:.4f} (Delta from frozen: {nll_w2_arm_c - nll_w2_base:+.4f})")

    # -------------------------------------------------------------------------
    # ARM D: Fast/Slow Joint Readout with 1-Tick Settle / Causal Conduction
    # -------------------------------------------------------------------------
    # In each step, run 1 settle tick so h_motor receives current token's 1-hop spikes!
    options_settle = dict(options)
    def run_window_settle(readout_fn, state, input_tokens, target_tokens):
        h_m_list = []
        z1_list = []
        z2_list = []
        with torch.no_grad():
            for t in range(len(input_tokens)):
                tok = input_tokens[t:t+1]
                state = advance_fly_input_event(
                    model, state, tok, settle_ticks=1, writer_baseline_clock="input", **options_settle
                )
                h_m = (state.h - state.h_mean)[:, model.read_indices] if model.read_centering else state.h[:, model.read_indices]
                h_m_list.append(h_m)
                z1_list.append(state.gamma_z1)
                z2_list.append(state.gamma_z2)

            h_m_all = torch.cat(h_m_list, dim=0).detach()
            z1_all = torch.cat(z1_list, dim=0).detach()
            z2_all = torch.cat(z2_list, dim=0).detach()
        features = readout_fn(h_m_all, z1_all, z2_all)
        logits = model.decoder(model.read_norm(features))
        scores = F.cross_entropy(logits, target_tokens, reduction="none")
        return scores, state

    joint_readout_d = FastSlowJointReadout(len(model.read_indices), d_model, model.output_read).to(device)
    scores_w1_d, s_w1_d = run_window_settle(joint_readout_d, clone_state(s_init), w1_inputs, w1_tokens)
    loss_d = scores_w1_d.mean()

    opt_d = torch.optim.AdamW(joint_readout_d.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    with torch.no_grad():
        w2_scores_d, _ = run_window_settle(joint_readout_d, clone_state(s_w1_d), w2_inputs, w2_tokens)
    nll_w2_arm_d = w2_scores_d.mean().item()

    print(f"\n[Arm D: Joint Readout + 1 Settle Tick (Causal Arrival)]")
    print(f"  Window 2 Follow NLL: {nll_w2_arm_d:.4f} (Delta from frozen: {nll_w2_arm_d - nll_w2_base:+.4f})")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE ON UNSEEN FOLLOW STREAM (WINDOW 2)")
    print("=" * 70)
    print(f"{'Condition':<45} | {'W2 Follow NLL':<13} | {'vs Frozen':<10}")
    print("-" * 72)
    print(f"{'Unigram (Decoder Bias) Baseline':<45} | {u_w2:<13.4f} | {u_w2 - nll_w2_base:+10.4f}")
    print(f"{'Arm A: Frozen Baseline (z2 only)':<45} | {nll_w2_base:<13.4f} | {'+0.0000':<10}")
    print(f"{'Arm B: Standard z2 Readout (1 Adam Update)':<45} | {nll_w2_arm_b:<13.4f} | {nll_w2_arm_b - nll_w2_base:+10.4f}")
    print(f"{'Arm C: Fast/Slow Joint Readout (1 Adam Update)':<45} | {nll_w2_arm_c:<13.4f} | {nll_w2_arm_c - nll_w2_base:+10.4f}")
    print(f"{'Arm D: Joint Readout + 1 Settle Tick':<45} | {nll_w2_arm_d:<13.4f} | {nll_w2_arm_d - nll_w2_base:+10.4f}")


if __name__ == "__main__":
    main()
