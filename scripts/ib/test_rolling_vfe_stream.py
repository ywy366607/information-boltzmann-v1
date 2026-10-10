"""Pilot GPU experiment for Variational Rolling Stream Lifelong Learning.

Evaluates the continuous lifelong learning mechanism on real OpenWebText data:
- Overlapping rolling window (Window W=128, Stride S=64): Context A -> Target B
- Inner assimilation loop optimizing posterior q(z) against frozen prior p_A^-(z)
  until plateau (|Delta F| < epsilon) or budget limit
- Prequential scoring before adaptation vs converged plateau scoring
- Single-counting outer update on model weights
- Ornstein-Uhlenbeck state transition to carry belief forward
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.variational_rolling_stream import (
    RollingStreamTransformer,
    VariationalRollingStreamLearner,
)


def run_rolling_stream_pilot(args: argparse.Namespace) -> dict:
    # Keep ablation arms comparable: model initialization and any stochastic
    # tensor operations must start from the same state for a given seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== Variational Rolling Stream Pilot Experiment ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Dataset: {args.data_path}")
    print(f"Window Size W: {args.window_size}, Stride S: {args.stride} (Prompt: {args.window_size - args.stride}, Target: {args.stride})")
    print(f"Inner Max Steps: {args.inner_max_steps}, Inner LR: {args.inner_lr}")
    print(f"Plateau Delta: {args.plateau_delta}, Patience: {args.plateau_patience}")
    print(f"KL Weight: {args.kl_weight}, Retention Rho: {args.retention}")
    print(f"Seed: {args.seed}")
    print(f"Number of Windows: {args.num_windows} (Total stream length: {args.num_windows * args.stride + (args.window_size - args.stride)} tokens)\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Load real OpenWebText tokens
    data_mmap = np.load(args.data_path, mmap_mode="r")
    start_pos = args.start_cursor
    needed_tokens = args.num_windows * args.stride + (args.window_size - args.stride)
    raw_slice = np.array(data_mmap[start_pos : start_pos + needed_tokens], dtype=np.int64)
    tokens_stream = torch.from_numpy(raw_slice).to(device)

    # Initialize compact causal transformer
    model = RollingStreamTransformer(
        vocab_size=args.vocab_size,
        dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.window_size,
        latent_dim=args.latent_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    learner = VariationalRollingStreamLearner(
        model=model,
        latent_dim=args.latent_dim,
        window_size=args.window_size,
        stride=args.stride,
        inner_lr=args.inner_lr,
        inner_max_steps=args.inner_max_steps,
        plateau_delta=args.plateau_delta,
        plateau_patience=args.plateau_patience,
        kl_weight=args.kl_weight,
        retention=args.retention,
        adaptive_retention=args.adaptive_retention,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        error_sensitivity=args.error_sensitivity,
        multi_scale_retention=args.multi_scale_retention,
        outer_lr=args.outer_lr,
        device=device,
    )

    print("-" * 125)
    print(
        f"{'Win':>4} | {'Tokens':>13} | {'Preq NLL':>9} | {'Plat NLL':>9} | {'Gain (nats)':>11} | "
        f"{'KL (nats)':>9} | {'FE':>8} | {'Steps':>5} | {'Plat?':>5} | {'Prec':>6} | {'Rho':>6} | {'Time':>7}"
    )
    print("-" * 125)

    reports = []
    t0_total = time.perf_counter()

    for w_idx in range(args.num_windows):
        offset = w_idx * args.stride
        window_tokens = tokens_stream[offset : offset + args.window_size]

        rep = learner.step(window_tokens)
        reports.append(rep)

        tok_range = f"{rep.token_start}:{rep.token_end}"
        plat_str = "YES" if rep.plateau_reached else "NO"
        print(
            f"{rep.window_idx:4d} | {tok_range:>13} | {rep.prequential_nll:9.4f} | {rep.plateau_nll:9.4f} | "
            f"{rep.adaptation_gain:+11.4f} | {rep.final_kl:9.4f} | {rep.final_free_energy:8.4f} | "
            f"{rep.inner_steps_taken:5d} | {plat_str:>5} | {rep.precision_ratio:6.2f} | {rep.retention_mean:6.3f} | {rep.wall_time_ms:6.1f}ms"
        )

    t_total = time.perf_counter() - t0_total

    # Summary statistics
    preq_nlls = [r.prequential_nll for r in reports]
    plat_nlls = [r.plateau_nll for r in reports]
    gains = [r.adaptation_gain for r in reports]
    kls = [r.final_kl for r in reports]
    steps = [r.inner_steps_taken for r in reports]
    plateau_counts = sum(1 for r in reports if r.plateau_reached)

    peak_vram_mib = 0.0
    if device.type == "cuda":
        peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    total_tokens_processed = args.num_windows * args.stride
    tokens_per_sec = total_tokens_processed / max(t_total, 1e-6)

    print("-" * 105)
    print(f"\n=== Experiment Summary ===")
    print(f"Total Windows: {args.num_windows}")
    print(f"Total Stream Tokens Processed: {total_tokens_processed} tokens")
    print(f"Total Elapsed Time: {t_total:.2f} s ({tokens_per_sec:.1f} tokens/s)")
    print(f"Peak VRAM Usage: {peak_vram_mib:.2f} MiB")
    print(f"Mean Prequential NLL: {np.mean(preq_nlls):.4f} nats/token")
    print(f"Mean Converged Plateau NLL: {np.mean(plat_nlls):.4f} nats/token")
    print(f"Mean Adaptation Gain: {np.mean(gains):+.4f} nats/token ({np.mean(gains) / np.mean(preq_nlls) * 100:.2f}%)")
    print(f"Mean Belief KL Divergence: {np.mean(kls):.4f} nats")
    print(f"Mean Inner Steps to Plateau: {np.mean(steps):.2f} / {args.inner_max_steps} steps")
    print(f"Plateau Trigger Rate: {plateau_counts}/{args.num_windows} ({plateau_counts / args.num_windows * 100:.1f}%)")
    rhos = [r.retention_mean for r in reports]
    precs = [r.precision_ratio for r in reports]
    print(f"Mean Retention Rho: {np.mean(rhos):.4f} (range: [{min(rhos):.4f}, {max(rhos):.4f}])")
    print(f"Mean Precision Ratio Pi_rel: {np.mean(precs):.4f} (range: [{min(precs):.4f}, {max(precs):.4f}])")

    summary_result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_windows": args.num_windows,
        "window_size": args.window_size,
        "stride": args.stride,
        "prompt_size": args.window_size - args.stride,
        "target_size": args.stride,
        "latent_dim": args.latent_dim,
        "inner_max_steps": args.inner_max_steps,
        "inner_lr": args.inner_lr,
        "plateau_delta": args.plateau_delta,
        "kl_weight": args.kl_weight,
        "retention": args.retention,
        "seed": args.seed,
        "adaptive_retention": args.adaptive_retention,
        "error_sensitivity": args.error_sensitivity,
        "multi_scale_retention": args.multi_scale_retention,
        "total_tokens_processed": total_tokens_processed,
        "total_elapsed_s": t_total,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_mib": peak_vram_mib,
        "mean_prequential_nll": float(np.mean(preq_nlls)),
        "mean_plateau_nll": float(np.mean(plat_nlls)),
        "mean_adaptation_gain": float(np.mean(gains)),
        "mean_kl_divergence": float(np.mean(kls)),
        "mean_inner_steps": float(np.mean(steps)),
        "plateau_trigger_rate": float(plateau_counts / args.num_windows),
        "mean_retention": float(np.mean(rhos)),
        "mean_precision_ratio": float(np.mean(precs)),
        "window_telemetry": [
            {
                "window": r.window_idx,
                "token_start": r.token_start,
                "token_end": r.token_end,
                "prequential_nll": r.prequential_nll,
                "plateau_nll": r.plateau_nll,
                "adaptation_gain": r.adaptation_gain,
                "kl_divergence": r.final_kl,
                "free_energy": r.final_free_energy,
                "inner_steps": r.inner_steps_taken,
                "plateau_reached": r.plateau_reached,
                "wall_time_ms": r.wall_time_ms,
                "retention_mean": r.retention_mean,
                "precision_ratio": r.precision_ratio,
            }
            for r in reports
        ],
    }

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, indent=2)
        print(f"\nReport written to: {out_path}")

    return summary_result


def main():
    parser = argparse.ArgumentParser(description="Pilot GPU test for Variational Rolling Stream Learner")
    parser.add_argument("--data-path", type=str, default="data/ib_owt_gpt2/train.npy")
    parser.add_argument("--start-cursor", type=int, default=0)
    parser.add_argument("--num-windows", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--inner-max-steps", type=int, default=12)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--plateau-delta", type=float, default=1e-3)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--retention", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--adaptive-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retention-min", type=float, default=0.10)
    parser.add_argument("--retention-max", type=float, default=0.98)
    parser.add_argument("--error-sensitivity", type=float, default=1.0)
    parser.add_argument("--multi-scale-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report-out", type=str, default="results/published/rolling_vfe_pilot_verification_20261009.json")

    args = parser.parse_args()
    run_rolling_stream_pilot(args)


if __name__ == "__main__":
    main()
