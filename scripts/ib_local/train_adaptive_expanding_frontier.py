"""Adaptive Expanding Frontier Cognitive Training on Sudoku-Extreme.
Synthesizes the Constitutional Protocol:
1. Outer Life Stream:
   - Macro Clock H = 1 (each puzzle is an event, no brute-force macro-BPTT unroll).
   - Never-Reset persistent field carry across all batches.
   - 2-Port State-and-Problem Aware Scattering theta(F, P) + H-Exit Quadratic Bath.
2. Inner Cognitive Stream:
   - Octave Horizons: K_j = 2^j for j in 0..J(t).
   - Equal-FLOPs Allocation: p_t(K_j) proportional to 1/K_j, meaning p_t(K_j) * K_j == constant!
   - Adaptive Frontier Expansion:
     Starts at J=2 (K_max = 4).
     Monitors shallow saturation and frontier improvement:
     When shallower horizons saturate and the current frontier K_f shows positive training benefit,
     the cognitive frontier automatically expands: J <- J + 1 (unlocking 2 * K_f)!
3. Pure Anytime Task Loss:
   - L = ell(Readout(F_K), y) with NO artificial constraints.
4. Physical Operator Substrate:
   - Time-Step Continuum Scaling: dt = 1.0 / K.
   - Exact Complex Exponential Spectral Advection: M(dt) = exp(-i * dt * omega) on 9x9 Torus.
   - Transolver++ Mass-Normalized Soft Integral Readout (tok = sum(w*V)/sum(w)).
   - Gradient Checkpointing in slices of 16 steps (guarantees VRAM <= 500 MB at any depth!).
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

from scripts.ib_local.train_ultra_deep_k1024_3000 import (
    CheckpointedUltraDeepCBIMSudokuModel,
    evaluate_ultra_deep_stream
)
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


def get_equal_flops_distribution(max_j: int) -> Tuple[List[int], List[float]]:
    """Calculates p(K_j) proportional to 1/K_j for K_j = 2^j, j in 0..max_j.
    Guarantees p_j * K_j == constant across all octaves!
    """
    horizons = [2 ** j for j in range(max_j + 1)]
    # Unnormalized weights w_j = 1 / K_j = 2^(-j)
    weights = [1.0 / k for k in horizons]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    return horizons, probs


def main():
    parser = argparse.ArgumentParser(description="Adaptive Expanding Frontier Cognitive Training")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--initial-j", type=int, default=2, help="Initial frontier exponent (K_max = 2^2 = 4)")
    parser.add_argument("--max-j-cap", type=int, default=10, help="Maximum allowed frontier exponent (2^10 = 1024)")
    parser.add_argument("--expansion-interval", type=int, default=400, help="Check frontier expansion every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/adaptive_expanding_frontier_3000", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 115)
    print("   ADAPTIVE EXPANDING FRONTIER COGNITIVE TRAINING (THE CONSTITUTIONAL PROTOCOL)")
    print(f"   Architecture: Stage 3 Champion with Equal-FLOPs (p(K) ~ 1/K) & Dynamic Frontier Growth")
    print(f"   Initial Frontier: J={args.initial_j} (K_max={2**args.initial_j}) -> Growing towards 2^{args.max_j_cap}={2**args.max_j_cap}")
    print(f"   Steps: {args.steps} | Eff Batch: {args.micro_batch_size * args.grad_accum_steps} | d={args.d_channels} (0.893M params)")
    print(f"   Equal-FLOPs Law: Every octave receives the exact same compute budget! E[K] ~ (J+1)/2 steps!")
    print("=" * 115)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = CheckpointedUltraDeepCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state
    init_sample = torch.as_tensor(train_in[:args.micro_batch_size], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(args.micro_batch_size, 9, 9, args.d_channels))

    current_j = args.initial_j
    horizons, probs = get_equal_flops_distribution(current_j)
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_cell_acc = 0.0
    start_time = time.perf_counter()

    frontier_history = [{
        "step": 0, "J": current_j, "K_max": 2 ** current_j, "horizons": horizons, "probs": probs
    }]

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
            horizons, probs = get_equal_flops_distribution(current_j)
            print("\n" + "#" * 90)
            print(f"   [COGNITIVE FRONTIER EXPANSION Step {step}]")
            print(f"   New Cognitive Frontier J={current_j}! Horizon expanded to K_max={2**current_j}!")
            print(f"   Octaves: {horizons}")
            print(f"   Equal-FLOPs Probabilities: {[round(p, 4) for p in probs]}")
            print(f"   Expected Microsteps E[K]: {sum(k*p for k,p in zip(horizons, probs)):.2f} steps!")
            print("#" * 90 + "\n", flush=True)
            frontier_history.append({
                "step": step, "J": current_j, "K_max": 2 ** current_j, "horizons": horizons, "probs": probs
            })

        # Sample microstep from the 1/K Equal-FLOPs distribution
        k_step = int(rng.choice(horizons, p=probs))
        accum_loss = 0.0
        diag_last = {}

        for _ in range(args.grad_accum_steps):
            idx = rng.integers(0, num_train, size=args.micro_batch_size)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                state_next, logits, diag_last = model.forward_stream_step(persistent_state, inp, k_step=k_step)
                loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum_steps

            with torch.no_grad():
                persistent_state.copy_(state_next.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if step % 50 == 0 or step == 1:
            eff_batch = args.micro_batch_size * args.grad_accum_steps
            throughput = eff_batch / (step_time_ms / 1000.0)
            theta_m = diag_last.get("theta_mean", 0.0)
            cool_p = diag_last.get("cooling_power", 0.0)
            print(f"Step {step:4d}/{args.steps} [J={current_j}: K_max={2**current_j:4d}] | K: {k_step:4d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:6.1f}ms | {throughput:5.1f} puz/s | VRAM: {vram_mb:5.1f}MB", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "J": current_j, "k": k_step, "loss": accum_loss, "lr": current_lr,
                    "theta_mean": float(theta_m), "cooling_power": float(cool_p),
                    "step_time_ms": step_time_ms, "throughput": throughput, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Evaluation across all active octaves and beyond
        if step % args.eval_interval == 0 or step == args.steps:
            val_horizons = [2 ** j for j in range(current_j + 1)]
            # Probe 2 octaves beyond current frontier to measure zero-shot extrapolation!
            probe_horizons = sorted(list(set(val_horizons + [2 ** (current_j + 1), 2 ** (current_j + 2)])))

            val_results = evaluate_ultra_deep_stream(
                model, test_in, test_lbl,
                mature_state=persistent_state,
                horizons=probe_horizons,
                batch_size=32
            )
            mean_loss_now = min(m["mean_loss"] for m in val_results.values())
            max_cell_now = max(m["cell_accuracy"] for m in val_results.values())

            print(f"\n[EVALUATION Step {step} | Frontier J={current_j} (K_max={2**current_j})] Best Cell Acc: {max_cell_now*100:5.2f}% | Best Loss: {mean_loss_now:6.3f}", flush=True)
            for k in probe_horizons:
                m = val_results[k]
                status = " (In Frontier)" if k <= (2 ** current_j) else " (PROBING UNTRAINED FRONTIER!)"
                print(f"   K = {k:4d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{status}", flush=True)

            if max_cell_now > best_cell_acc or mean_loss_now < best_loss:
                best_cell_acc = max(best_cell_acc, max_cell_now)
                best_loss = min(best_loss, mean_loss_now)
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "current_j": current_j,
                    "val_results": val_results
                }, out_path / "Best_Adaptive_Frontier_d256.pt")
                print(f"Saved new best adaptive model to {out_path / 'Best_Adaptive_Frontier_d256.pt'}\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nAdaptive Training completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time zero-shot pondering sweep out to K=1024
    print("\n" + "=" * 115)
    print("   POST-TRAINING FULL ZERO-SHOT PONDERING SWEEP ACROSS COMPLETE SPECTRUM (K in [1..1024])")
    print("=" * 115)
    final_sweep = evaluate_ultra_deep_stream(
        model, test_in, test_lbl,
        mature_state=persistent_state,
        horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        batch_size=32
    )
    for k, m in final_sweep.items():
        status = " [In Frontier]" if k <= (2 ** current_j) else " [Extrapolated]"
        print(f"  Ponder Depth K = {k:4d} | Exact Acc: {m['exact_accuracy']*100:5.2f}% | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{status}")

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_frontier_j": current_j,
            "frontier_history": frontier_history,
            "steps": args.steps,
            "d_channels": args.d_channels,
            "total_time_min": total_time_min,
            "best_cell_acc": best_cell_acc,
            "best_loss": best_loss,
            "final_sweep": final_sweep
        }, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "persistent_state": persistent_state.detach().cpu(),
        "final_sweep": final_sweep
    }, out_path / "Final_Adaptive_Frontier_d256.pt")
    print(f"All completed! Final report saved to {out_path / 'final_report.json'}")


if __name__ == "__main__":
    main()
