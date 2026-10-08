"""Measure Silent Relaxation Dynamics: Mutual Information & Conduction Latency Spectrum.

Measures:
1. Stream Delayed Prediction Curve:
   Across 32 real consecutive OWT tokens (cursor 1,600,000):
   Injects x_t at tau=0, evolves silently for tau=0..32 ticks.
   Measures mean NLL(x_{t+1}; tau), NLL(x_t; tau), and optimal latency distribution.

2. Information-Theoretic Mutual Information Spectrum:
   Across K=64 diverse vocabulary tokens:
   Injects x^(k) at tau=0, evolves silently for tau=0..48 ticks.
   Measures:
   - Exact Jensen-Shannon Mutual Information I(X; P_gamma(tau)) and I(X; P_raw(tau))
   - Pairwise motor state distinction D_motor(tau)
   - Anatomical wave propagation (sensory -> intrinsic -> motor -> gamma)
   - Input reconstruction top-k accuracy over delay tau
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event, FlyPhysicalState
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state


def quiet_step(model, s, dummy_token, options):
    """Executes one quiet tick (zero external sensory drive)."""
    h, spike, ring, ge, gi, b, x, u = model.step(
        s.h, dummy_token, spike_ring=s.ring, ge=s.ge, gi=s.gi, b=s.b,
        x=s.x, u=s.u, sensory_drive=torch.zeros_like(s.h), **options
    )
    hm = 0.99 * s.h_mean + 0.01 * h
    read = (h - hm)[:, model.read_indices] if model.read_centering else h[:, model.read_indices]
    a = model.get_read_gamma_decay()
    z1 = a * s.gamma_z1 + (1.0 - a) * read
    z2 = a * s.gamma_z2 + (1.0 - a) * z1
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, s.baseline, hm, s.dan_gate, z1, z2), spike


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/fly_silent_relaxation_information.json"))
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} on {args.device}...", flush=True)
    saved, learner = load_checkpoint(args.checkpoint, device=args.device)
    model = learner.model
    s_init = clone_state(learner.state)

    graph_data = np.load(ROOT / saved["config"]["graph"])
    s_names = list(graph_data["superclass_names"])
    superclass_id = graph_data["superclass_id"]

    motor_indices = np.flatnonzero(np.isin(superclass_id, [s_names.index(c) for c in model.OUTPUT_CLASSES if c in s_names]))
    sensory_indices = model.topographic_writer.injection_index.cpu().numpy()
    intrinsic_indices = np.setdiff1d(np.arange(model.n_neurons), np.union1d(motor_indices, sensory_indices))

    motor_idx_t = torch.as_tensor(motor_indices, device=args.device)
    sensory_idx_t = torch.as_tensor(sensory_indices, device=args.device)
    intrinsic_idx_t = torch.as_tensor(intrinsic_indices, device=args.device)

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    bias_logits = model.decoder.bias.unsqueeze(0)
    bias_probs = F.softmax(bias_logits, dim=-1)
    bias_entropy = -(bias_probs * F.log_softmax(bias_logits, dim=-1)).sum().item()
    print(f"Decoder Bias Unigram Entropy: {bias_entropy:.4f} nats ({bias_entropy / np.log(2):.2f} bits)", flush=True)

    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])

    results = {
        "unigram_entropy_nats": bias_entropy,
        "stream_cursor": cursor,
        "part1_stream_ensemble": {},
        "part2_mutual_information_spectrum": {},
    }

    # =========================================================================
    # PART 1: Stream Delayed Prediction Curve (32 real consecutive OWT tokens)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Stream Delayed Prediction Curve (32 Consecutive OWT Transitions)")
    print("=" * 70, flush=True)

    n_stream_tokens = 32
    max_tau_stream = 24
    stream_tokens = [int(data[cursor + i]) for i in range(n_stream_tokens + 1)]

    # We need to maintain continuous state along the stream for accurate contexts!
    # Let's run the continuous stream, and at each token, branch out a silent probe for max_tau_stream ticks.
    stream_state = clone_state(s_init)
    dummy_tok = torch.tensor([stream_tokens[0]], device=args.device)

    all_nll_future = np.zeros((n_stream_tokens, max_tau_stream + 1), dtype=np.float32)
    all_nll_stimulus = np.zeros((n_stream_tokens, max_tau_stream + 1), dtype=np.float32)
    all_unigram_future = np.zeros(n_stream_tokens, dtype=np.float32)

    with torch.no_grad():
        for step in range(n_stream_tokens):
            x_cur = stream_tokens[step]
            x_next = stream_tokens[step + 1]
            tok_tensor = torch.tensor([x_cur], device=args.device)
            target_next = torch.tensor([x_next], device=args.device)
            target_cur = torch.tensor([x_cur], device=args.device)

            all_unigram_future[step] = -F.log_softmax(bias_logits, dim=-1)[0, x_next].item()

            # Advance stream state with real token x_cur at tau=0
            stream_state = advance_fly_input_event(
                model, stream_state, tok_tensor, settle_ticks=0, writer_baseline_clock="input", **options
            )

            # Probe branch: clone stream state and run silence for tau=0..max_tau_stream
            probe_state = clone_state(stream_state)
            for tau in range(max_tau_stream + 1):
                if tau > 0:
                    probe_state, _ = quiet_step(model, probe_state, dummy_tok, options)

                feat = model.output_read(probe_state.gamma_z2)
                logits = model.decoder(model.read_norm(feat))

                loss_next = F.cross_entropy(logits, target_next).item()
                loss_cur = F.cross_entropy(logits, target_cur).item()

                all_nll_future[step, tau] = loss_next
                all_nll_stimulus[step, tau] = loss_cur

            if (step + 1) % 8 == 0 or step == 0:
                print(f"  Processed stream token {step + 1:2d}/{n_stream_tokens} (x={x_cur} -> next={x_next})...", flush=True)

    mean_nll_future = np.mean(all_nll_future, axis=0)
    std_nll_future = np.std(all_nll_future, axis=0)
    mean_nll_stimulus = np.mean(all_nll_stimulus, axis=0)
    mean_unigram = float(np.mean(all_unigram_future))

    best_tau_per_token = np.argmin(all_nll_future, axis=1)
    tau_counts = {int(t): int(np.sum(best_tau_per_token == t)) for t in range(max_tau_stream + 1)}

    print("\n--- Summary of Stream Delayed Prediction across 32 Transitions ---")
    print(f"Unigram (Decoder Bias) Mean NLL: {mean_unigram:.4f} nats")
    print("Lag tau | Mean NLL(next) | Delta from tau=0 | vs Unigram | Mean NLL(stimulus) | Best-Lag Count")
    print("-" * 75)
    for tau in range(max_tau_stream + 1):
        d_tau0 = mean_nll_future[tau] - mean_nll_future[0]
        d_uni = mean_nll_future[tau] - mean_unigram
        count = tau_counts.get(tau, 0)
        marker = " <== OPTIMAL" if tau == np.argmin(mean_nll_future) else ""
        print(f"{tau:7d} | {mean_nll_future[tau]:14.4f} | {d_tau0:+16.4f} | {d_uni:+10.4f} | {mean_nll_stimulus[tau]:18.4f} | {count:14d}{marker}")

    opt_tau = int(np.argmin(mean_nll_future))
    results["part1_stream_ensemble"] = {
        "mean_unigram_nll": mean_unigram,
        "mean_nll_future": mean_nll_future.tolist(),
        "std_nll_future": std_nll_future.tolist(),
        "mean_nll_stimulus": mean_nll_stimulus.tolist(),
        "optimal_lag_tau": opt_tau,
        "delta_at_optimal_lag": float(mean_nll_future[opt_tau] - mean_nll_future[0]),
        "tokens_improving_at_optimal": int(np.sum(all_nll_future[:, opt_tau] < all_nll_future[:, 0])),
        "best_tau_distribution": tau_counts,
    }

    # =========================================================================
    # PART 2: Mutual Information Spectrum I(X; Output(tau)) (K=64 Diverse Tokens)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Mutual Information Spectrum & Wave Propagation (K=64 Diverse Tokens)")
    print("=" * 70, flush=True)

    # Select K=64 diverse tokens across log-frequency bins from decoder bias
    # to guarantee representativeness (function words, nouns, rare words)
    with torch.no_grad():
        bias_sorted_idx = torch.argsort(model.decoder.bias, descending=True).cpu().numpy()
        # Take 16 ultra-frequent, 24 mid-frequent, 24 lower-frequent
        idx_high = bias_sorted_idx[10:26]
        idx_mid = bias_sorted_idx[np.linspace(100, 5000, 24, dtype=int)]
        idx_low = bias_sorted_idx[np.linspace(5000, 40000, 24, dtype=int)]
        test_tokens = np.concatenate([idx_high, idx_mid, idx_low])
        K = len(test_tokens)
        print(f"Selected K={K} diverse tokens across vocabulary ranks 10..40,000.", flush=True)

    max_tau_mi = 36

    # 1. First run a quiet control (sensory_drive = 0) from s_init
    s_control = clone_state(s_init)
    ctrl_motor_h = []
    ctrl_gamma_z2 = []
    dummy = torch.tensor([test_tokens[0]], device=args.device)

    with torch.no_grad():
        for tau in range(max_tau_mi + 1):
            s_control, _ = quiet_step(model, s_control, dummy, options)
            ctrl_motor_h.append(s_control.h[:, motor_idx_t].clone())
            ctrl_gamma_z2.append(s_control.gamma_z2.clone())

    # 2. For each token x_k: inject at tau=0, evolve silently for tau=1..max_tau_mi
    # Store:
    # - motor state h_motor(k, tau)
    # - sensory norm dH_sens(k, tau)
    # - intrinsic norm dH_int(k, tau)
    # - gamma z2(k, tau)
    # - output distribution p_gamma(k, tau)
    # - output logits L_gamma(k, tau)
    motor_trajectories = torch.zeros((K, max_tau_mi + 1, len(motor_indices)), device="cpu", dtype=torch.float32)
    sensory_waves = np.zeros((K, max_tau_mi + 1), dtype=np.float32)
    intrins_waves = np.zeros((K, max_tau_mi + 1), dtype=np.float32)
    motor_waves = np.zeros((K, max_tau_mi + 1), dtype=np.float32)
    gamma_waves = np.zeros((K, max_tau_mi + 1), dtype=np.float32)

    # To compute mutual information without holding K x (max_tau) x 50257 in RAM/GPU:
    # We can accumulate the mixture distribution sum_k p_k(tau) on GPU directly,
    # and compute sum_k H(p_k(tau)) online!
    mixture_probs = torch.zeros((max_tau_mi + 1, 50257), device=args.device, dtype=torch.float64)
    sum_individual_entropy = np.zeros(max_tau_mi + 1, dtype=np.float64)
    reconstruction_top1 = np.zeros(max_tau_mi + 1, dtype=int)
    reconstruction_top5 = np.zeros(max_tau_mi + 1, dtype=int)

    t0_start = time.perf_counter()
    with torch.no_grad():
        for k_idx, tok_val in enumerate(test_tokens):
            tok_t = torch.tensor([tok_val], device=args.device)
            state = clone_state(s_init)
            # Inject at tau=0
            state = advance_fly_input_event(
                model, state, tok_t, settle_ticks=0, writer_baseline_clock="input", **options
            )

            for tau in range(max_tau_mi + 1):
                if tau > 0:
                    state, _ = quiet_step(model, state, dummy, options)

                # Store motor states on CPU
                motor_trajectories[k_idx, tau] = state.h[:, motor_idx_t].cpu()

                # Waves relative to control
                dh_sens = (state.h[:, sensory_idx_t] - s_init.h[:, sensory_idx_t]).norm().item()
                dh_int = state.h[:, intrinsic_idx_t].norm().item()
                dh_mot = (state.h[:, motor_idx_t] - ctrl_motor_h[tau]).norm().item()
                dz2 = (state.gamma_z2 - ctrl_gamma_z2[tau]).norm().item()

                sensory_waves[k_idx, tau] = dh_sens
                intrins_waves[k_idx, tau] = dh_int
                motor_waves[k_idx, tau] = dh_mot
                gamma_waves[k_idx, tau] = dz2

                # Output distribution & entropy
                feat = model.output_read(state.gamma_z2)
                logits = model.decoder(model.read_norm(feat))
                probs = F.softmax(logits, dim=-1).double()

                # Shannon entropy of p_k(tau): H(p) = -sum p log p
                ent = -(probs * F.log_softmax(logits, dim=-1).double()).sum().item()
                sum_individual_entropy[tau] += ent
                mixture_probs[tau] += probs[0]

                # Check reconstruction of injected token
                top5_preds = torch.topk(logits, k=5, dim=-1).indices[0].cpu().numpy()
                if tok_val == top5_preds[0]:
                    reconstruction_top1[tau] += 1
                if tok_val in top5_preds:
                    reconstruction_top5[tau] += 1

            if (k_idx + 1) % 8 == 0 or k_idx == 0:
                print(f"  Stimulus {k_idx + 1:2d}/{K} (token={tok_val}) simulated across {max_tau_mi} ticks...", flush=True)

    elapsed = time.perf_counter() - t0_start
    print(f"Simulated {K} stimuli x {max_tau_mi} ticks in {elapsed:.2f}s ({elapsed / (K * max_tau_mi) * 1000:.2f} ms/tick).", flush=True)

    # Compute Mutual Information: I(X; Output) = H(Mixture) - Mean(H(Individual))
    mixture_probs /= K
    mixture_entropy = np.zeros(max_tau_mi + 1, dtype=np.float64)
    for tau in range(max_tau_mi + 1):
        p = mixture_probs[tau]
        p_nz = p[p > 1e-12]
        mixture_entropy[tau] = -(p_nz * torch.log(p_nz)).sum().item()

    mean_individual_entropy = sum_individual_entropy / K
    mutual_information_nats = mixture_entropy - mean_individual_entropy
    mutual_information_bits = mutual_information_nats / np.log(2)

    # Compute Pairwise Motor Distance D_motor(tau)
    # Sample 200 pairs to compute mean distance efficiently
    motor_contrast = np.zeros(max_tau_mi + 1, dtype=np.float32)
    n_pairs = 0
    for i in range(min(K, 30)):
        for j in range(i + 1, min(K, 30)):
            diff = (motor_trajectories[i] - motor_trajectories[j]).norm(dim=-1).numpy()
            motor_contrast += diff
            n_pairs += 1
    motor_contrast /= max(n_pairs, 1)

    print("\n--- Mutual Information Spectrum & Anatomical Wave Propagation ---")
    print("Lag tau | Mutual Info (bits) | Motor Contrast | Sensory dH | Intrins dH | Motor dH | Gamma dZ2 | Top-1 Recov")
    print("-" * 88)
    for tau in range(max_tau_mi + 1):
        mi_b = mutual_information_bits[tau]
        d_mot = motor_contrast[tau]
        s_w = np.mean(sensory_waves[:, tau])
        i_w = np.mean(intrins_waves[:, tau])
        m_w = np.mean(motor_waves[:, tau])
        g_w = np.mean(gamma_waves[:, tau])
        top1 = reconstruction_top1[tau]
        marker = " <== MAX INFO" if tau == np.argmax(mutual_information_bits) else ""
        print(f"{tau:7d} | {mi_b:18.5f} | {d_mot:14.3f} | {s_w:10.2f} | {i_w:10.2f} | {m_w:8.2f} | {g_w:9.3f} | {top1:7d}/{K}{marker}")

    max_mi_tau = int(np.argmax(mutual_information_bits))
    max_d_mot_tau = int(np.argmax(motor_contrast))

    print(f"\nMutual Information Peak : tau = {max_mi_tau} ticks (I = {mutual_information_bits[max_mi_tau]:.5f} bits)")
    print(f"Motor Contrast Peak     : tau = {max_d_mot_tau} ticks (Distance = {motor_contrast[max_d_mot_tau]:.3f})")

    results["part2_mutual_information_spectrum"] = {
        "max_info_tau": max_mi_tau,
        "max_info_bits": float(mutual_information_bits[max_mi_tau]),
        "max_contrast_tau": max_d_mot_tau,
        "max_contrast": float(motor_contrast[max_d_mot_tau]),
        "mutual_information_bits": mutual_information_bits.tolist(),
        "motor_contrast": motor_contrast.tolist(),
        "sensory_wave_mean": np.mean(sensory_waves, axis=0).tolist(),
        "intrins_wave_mean": np.mean(intrins_waves, axis=0).tolist(),
        "motor_wave_mean": np.mean(motor_waves, axis=0).tolist(),
        "gamma_wave_mean": np.mean(gamma_waves, axis=0).tolist(),
        "reconstruction_top1": reconstruction_top1.tolist(),
        "reconstruction_top5": reconstruction_top5.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output} successfully!", flush=True)


if __name__ == "__main__":
    main()
