"""Canonical Master Baseline Training Pipeline:
Combines Explicit Branch Multi-Path Search (M in {1, 2, 4})
with Unbounded Expanding Frontier 1/K Training (K from 4 growing to 1024).
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_master_baseline import CBIMMasterBaselineModel
from scripts.ib_local.train_adaptive_expanding_frontier import get_equal_flops_distribution
from scripts.ib_local.train_explicit_branch_multipath import evaluate_explicit_branch_grid
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


def main():
    parser = argparse.ArgumentParser(description="CBIM Master Baseline Training Pipeline")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--initial-j", type=int, default=2, help="Initial frontier exponent (K_max = 2^2 = 4)")
    parser.add_argument("--max-j-cap", type=int, default=10, help="Maximum allowed frontier exponent (2^10 = 1024)")
    parser.add_argument("--expansion-interval", type=int, default=350, help="Expand frontier every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/cbim_master_baseline_canonical_3000", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 115)
    print("   CBIM MASTER CANONICAL BASELINE: EXPLICIT BRANCH MULTI-PATH + UNBOUNDED 1/K HORIZON")
    print(f"   Architecture: Explicit Branch Axis M in {{1, 2, 4}} + Hadamard Exploration Seeds + Mass-Norm Readout")
    print(f"   K Cognitive Horizon: Growing dynamically from K=4 to K=1024 under Equal-FLOPs p(K) ~ 1/K!")
    print(f"   Life Stream: Strictly Never-Reset single-stream execution | Steps: {args.steps} | LR: {args.lr:.1e} -> {args.min_lr:.1e}")
    print("=" * 115)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = CBIMMasterBaselineModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    init_sample = torch.as_tensor(train_in[0:1], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(1, 9, 9, args.d_channels))

    current_j = args.initial_j
    k_horizons, k_probs = get_equal_flops_distribution(current_j)
    m_choices = [1, 2, 4]
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_cell_acc = 0.0
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        model.train()

        progress = step / args.steps
        current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        # Check for Adaptive Frontier Growth
        if step > 0 and step % args.expansion_interval == 0 and current_j < args.max_j_cap:
            current_j += 1
            k_horizons, k_probs = get_equal_flops_distribution(current_j)
            print(f"\n>>> [Cognitive Frontier Expansion Step {step}]: J={current_j} -> K_max={2**current_j} (E[K]={sum(k*p for k,p in zip(k_horizons, k_probs)):.1f} steps) <<<\n", flush=True)

        m_step = int(rng.choice(m_choices))
        k_step = int(rng.choice(k_horizons, p=k_probs))

        puzzle_idx = (step - 1) % num_train
        inp = torch.as_tensor(train_in[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            state_next, log_probs, diag_last = model.forward_stream_step(persistent_state, inp, M=m_step, k_step=k_step)
            loss = F.nll_loss(log_probs.view(-1, 11), lbl.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        with torch.no_grad():
            persistent_state.copy_(state_next.detach())

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if step % 50 == 0 or step == 1:
            fps = 1.0 / (step_time_ms / 1000.0)
            print(f"Step {step:4d}/{args.steps} [K_max={2**current_j:4d}] | M: {m_step} | K: {k_step:4d} | Loss: {loss.item():6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:5.1f}ms | {fps:5.1f} puz/s | VRAM: {vram_mb:5.1f}MB", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "J": current_j, "M": m_step, "k": k_step, "loss": loss.item(), "lr": current_lr,
                    "step_time_ms": step_time_ms, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Evaluation across 2D grid
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_explicit_branch_grid(
                model, test_in[:200], test_lbl[:200],
                mature_state=persistent_state,
                m_list=[1, 2, 4],
                k_list=[2, 8, 32]
            )
            print(f"\n[INTERIM 2D EVALUATION Step {step} | Frontier K_max={2**current_j}]")
            for m_val in [1, 2, 4]:
                line_str = f"  M={m_val} | " + " | ".join([f"K={k_val:2d}: {val_results[f'M{m_val}_K{k_val}']['cell_accuracy']*100:5.2f}%" for k_val in [2, 8, 32]])
                print(line_str)

            acc_m2_k8 = val_results["M2_K8"]["cell_accuracy"]
            if acc_m2_k8 > best_cell_acc:
                best_cell_acc = acc_m2_k8
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "val_results": val_results
                }, out_path / "Best_Master_Baseline_d256.pt")
                print(f"Saved new best master baseline model to {out_path / 'Best_Master_Baseline_d256.pt'}\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nMaster Baseline Training completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time 2D grid evaluation across all 1,000 test puzzles
    print("\n" + "=" * 115)
    print("   FINAL POST-TRAINING 2D RESPONSE MATRIX Q(M, K) (FULL 1,000 TEST PUZZLES)")
    print("=" * 115)
    final_grid = evaluate_explicit_branch_grid(
        model, test_in, test_lbl,
        mature_state=persistent_state,
        m_list=[1, 2, 4, 8],
        k_list=[1, 2, 4, 8, 16, 32, 64]
    )

    header = f"{'Width M':^10} | " + " | ".join([f"K={k:<4d}" for k in [1, 2, 4, 8, 16, 32, 64]])
    print(header)
    print("-" * len(header))
    for M in [1, 2, 4, 8]:
        row_str = f" M = {M:2d}    | " + " | ".join([f"{final_grid[f'M{M}_K{K}']['cell_accuracy']*100:5.2f}%" for K in [1, 2, 4, 8, 16, 32, 64]])
        print(row_str)

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "steps": args.steps, "d_channels": args.d_channels, "total_time_min": total_time_min,
            "final_grid": final_grid
        }, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "persistent_state": persistent_state.detach().cpu(),
        "final_grid": final_grid
    }, out_path / "Final_Master_Baseline_d256.pt")
    print(f"\nAll completed! Master report saved to {out_path / 'final_report.json'}")


if __name__ == "__main__":
    main()
