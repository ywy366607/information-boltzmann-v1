"""Four-pillar AG evaluation for the rolling variational learner.

Train each arm on environment A, switch to a disjoint OWT segment B, and
score B pre-update predictions with the shared adaptation/generalization
summary. The reference arm is frozen after initialization so AG quality uses
the same B targets for every arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from information_boltzmann.core.variational_rolling_stream import (
    RollingStreamTransformer,
    VariationalRollingStreamLearner,
)
from information_boltzmann.runtime.lifelong_evaluation import (
    adaptation_generalization_summary,
)


def run_arm(args, tokens_a: torch.Tensor, tokens_b: torch.Tensor, *,
            adaptive: bool, multiscale: bool, inner_steps: int,
            outer_lr: float):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = RollingStreamTransformer(
        vocab_size=args.vocab_size,
        dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.window_size,
        latent_dim=args.latent_dim,
    ).to(device)
    learner = VariationalRollingStreamLearner(
        model=model,
        latent_dim=args.latent_dim,
        window_size=args.window_size,
        stride=args.stride,
        inner_lr=args.inner_lr,
        inner_max_steps=inner_steps,
        plateau_delta=args.plateau_delta,
        plateau_patience=args.plateau_patience,
        kl_weight=args.kl_weight,
        retention=args.retention,
        adaptive_retention=adaptive,
        multi_scale_retention=multiscale,
        outer_lr=outer_lr,
        device=device,
    )
    for index in range(args.windows_a):
        learner.step(tokens_a[index * args.stride:index * args.stride + args.window_size])
    b_scores = []
    for index in range(args.windows_b):
        report = learner.step(tokens_b[index * args.stride:index * args.stride + args.window_size])
        b_scores.append(report.prequential_nll)
    return b_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/ib_owt_gpt2/train.npy")
    parser.add_argument("--start-a", type=int, default=0)
    parser.add_argument("--start-b", type=int, default=1_000_000)
    parser.add_argument("--windows-a", type=int, default=80)
    parser.add_argument("--windows-b", type=int, default=80)
    parser.add_argument("--repeat-b-block-windows", type=int, default=0,
                        help="Repeat a real OWT environment block; 0 keeps the natural B stream.")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--inner-max-steps", type=int, default=12)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--plateau-delta", type=float, default=0.003)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--retention", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-out", default="results/published/rolling_vfe_ag_20261009.json")
    args = parser.parse_args()

    mmap = np.load(args.data_path, mmap_mode="r")
    needed = args.windows_a * args.stride + args.window_size - args.stride
    raw_a = np.asarray(mmap[args.start_a:args.start_a + needed], dtype=np.int64)
    b_length = args.windows_b * args.stride + args.window_size - args.stride
    if args.repeat_b_block_windows > 0:
        block_length = args.repeat_b_block_windows * args.stride + args.window_size - args.stride
        block = np.asarray(mmap[args.start_b:args.start_b + block_length], dtype=np.int64)
        raw_b = np.resize(block, b_length)
    else:
        raw_b = np.asarray(mmap[args.start_b:args.start_b + b_length], dtype=np.int64)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tokens_a = torch.from_numpy(raw_a).to(device)
    tokens_b = torch.from_numpy(raw_b).to(device)

    reference = run_arm(args, tokens_a, tokens_b, adaptive=False, multiscale=False,
                        inner_steps=0, outer_lr=0.0)
    scalar = run_arm(args, tokens_a, tokens_b, adaptive=False, multiscale=False,
                     inner_steps=args.inner_max_steps, outer_lr=1e-4)
    gdn = run_arm(args, tokens_a, tokens_b, adaptive=True, multiscale=True,
                  inner_steps=args.inner_max_steps, outer_lr=1e-4)

    reports = {}
    for name, values in (("reference", reference), ("scalar", scalar), ("gdn", gdn)):
        if name == "reference":
            reports[name] = {"mean_nll": float(np.mean(values)), "scores": values}
            continue
        summary = adaptation_generalization_summary(
            values,
            block_tokens=1,
            hold_blocks=3,
            plateau_blocks=8,
            plateau_tolerance_nll=0.1,
            reference_nll=reference,
        )
        summary["recovery_tokens_in_stream"] = (
            None if summary.get("recovery_tokens") is None
            else summary["recovery_tokens"] * args.stride
        )
        summary["mean_nll"] = float(np.mean(values))
        summary["scores"] = values
        reports[name] = summary

    result = {
        "protocol": "rolling_vfe_four_pillar_ag",
        "seed": args.seed,
        "environment_a_start": args.start_a,
        "environment_b_start": args.start_b,
        "windows_a": args.windows_a,
        "windows_b": args.windows_b,
        "repeat_b_block_windows": args.repeat_b_block_windows,
        "stride": args.stride,
        "reports": reports,
    }
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ("scalar", "gdn"):
        r = reports[name]
        print(name, "AG", r.get("ag"), "recovery_tokens", r.get("recovery_tokens_in_stream"),
              "plateau_nll", r.get("plateau_nll"), "status", r.get("status"))
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
