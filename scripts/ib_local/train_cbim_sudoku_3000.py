"""CBIM Sudoku 3000-Step Multi-Horizon Training & Head-to-Head Comparison.
Applies Jeffreys scale-invariant multi-horizon sampling across 8 octaves (K in [2..256])
with strided multi-step auxiliary loss supervision under 4GB GTX 1650 constraints.
"""
import os
import sys

# Pop script directory to avoid shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Also add external/TinyRecursiveModels for imports
trm_dir = os.path.join(repo_root, "external", "TinyRecursiveModels")
if trm_dir not in sys.path:
    sys.path.insert(0, trm_dir)

import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku import CBIMSudokuModel
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from adam_atan2 import AdamATan2


def load_dataset(data_dir: str):
    train_in = np.load(os.path.join(data_dir, "train", "all__inputs.npy"), mmap_mode="r")
    train_lbl = np.load(os.path.join(data_dir, "train", "all__labels.npy"), mmap_mode="r")
    test_in = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")
    test_lbl = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")
    return {
        "train_in": train_in, "train_lbl": train_lbl,
        "test_in": test_in, "test_lbl": test_lbl
    }


@torch.no_grad()
def evaluate_multi_k(model, test_in: np.ndarray, test_lbl: np.ndarray,
                     eval_horizons: List[int], batch_size: int = 64) -> Dict[str, Any]:
    model.eval()
    num_samples = len(test_in)
    results = {}

    for k in eval_horizons:
        orig_k = model.model.ponder_steps
        model.model.ponder_steps = k

        total_correct_cells = 0
        total_cells = 0
        total_exact_boards = 0
        total_loss = 0.0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            B = end_idx - start_idx
            inp = torch.as_tensor(test_in[start_idx:end_idx], dtype=torch.long, device="cuda")
            lbl = torch.as_tensor(test_lbl[start_idx:end_idx], dtype=torch.long, device="cuda")
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device="cuda")}
            carry = model.model.initial_carry(batch)

            # Evaluate with single ACT step
            carry, loss, metrics, preds, _ = model(carry=carry, batch=batch, return_keys=["logits"])
            logits = preds["logits"]
            pred_digits = torch.argmax(logits, dim=-1)

            cell_match = (pred_digits == lbl)
            total_correct_cells += int(cell_match.sum().item())
            total_cells += int(lbl.numel())
            total_exact_boards += int(cell_match.all(dim=-1).sum().item())
            total_loss += float(loss.item()) * B

        model.model.ponder_steps = orig_k
        results[k] = {
            "exact_accuracy": total_exact_boards / num_samples,
            "cell_accuracy": total_correct_cells / total_cells,
            "mean_loss": total_loss / num_samples
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="CBIM 3000-step Multi-Horizon Training on Sudoku-Extreme")
    parser.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-100")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--output-dir", default="results/cbim_sudoku_3000")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 85)
    print("   CBIM 3000-STEP MULTI-HORIZON TRAINING ON SUDOKU-EXTREME")
    print(f"   Steps: {args.steps} | Batch: {args.batch_size} | Base LR: {args.lr} | Data: {args.data_dir}")
    print("=" * 85)

    data = load_dataset(args.data_dir)
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 8 Temporal Octaves
    horizons = [2, 4, 8, 16, 32, 64, 128, 256]
    # Jeffreys scale-invariant compute quota: p(k) * k = const => p(k) \propto 1/k
    raw_probs = [1.0 / k for k in horizons]
    sum_probs = sum(raw_probs)
    probs = [p / sum_probs for p in raw_probs]

    print("Temporal Octaves & Sampling Probabilities:")
    for k, p in zip(horizons, probs):
        print(f"  K = {k:3d} | p(K) = {p*100:5.2f}% | Expected Steps = {args.steps * p:.1f}")

    # Initialize model
    inner_model = CBIMSudokuModel(
        vocab_size=11, d_channels=128, n_velocities=8,
        ponder_steps=16, halt_max_steps=16, multi_step_loss=True
    ).to(device)
    model = ACTLossHead(inner_model, loss_type="stablemax_cross_entropy")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized | Parameter Count: {num_params:,} ({num_params/1e6:.3f}M)")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)

    # Initial Step 0 Eval
    print("\n--- Initial Step 0 Evaluation (Testing on K in [2, 4, 8, 16, 32, 64, 128]) ---")
    val_metrics = evaluate_multi_k(model, test_in, test_lbl, eval_horizons=[2, 4, 8, 16, 32, 64, 128], batch_size=args.batch_size)
    for k, m in val_metrics.items():
        print(f"  K = {k:3d} | Exact Acc: {m['exact_accuracy']*100:.2f}% | Cell Acc: {m['cell_accuracy']*100:.2f}% | Loss: {m['mean_loss']:.4f}")

    best_loss = min(m["mean_loss"] for m in val_metrics.values())
    best_cell_acc = max(m["cell_accuracy"] for m in val_metrics.values())
    best_exact_acc = max(m["exact_accuracy"] for m in val_metrics.values())

    start_time = time.perf_counter()
    rng = np.random.default_rng(42)
    carry = None

    for step in range(1, args.steps + 1):
        model.train()
        t0 = time.perf_counter()

        # Cosine LR schedule
        progress = step / args.steps
        current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        # Sample horizon K for this step
        k_step = int(rng.choice(horizons, p=probs))
        model.model.ponder_steps = k_step

        # Sample batch
        idx = rng.integers(0, num_train, size=args.batch_size)
        inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
        batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(args.batch_size, dtype=torch.long, device=device)}

        if carry is None:
            carry = model.model.initial_carry(batch)

        carry, loss, loss_metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        if step % 25 == 0 or step == 1:
            throughput = args.batch_size / (step_time_ms / 1000.0)
            print(f"Step {step:4d}/{args.steps} | K: {k_step:3d} | Loss: {loss.item():6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:5.1f}ms | Throughput: {throughput:5.1f} puz/s | VRAM: {vram_mb:6.1f}MB", flush=True)
            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "k": k_step,
                    "loss": float(loss.item()), "lr": current_lr,
                    "step_time_ms": step_time_ms, "throughput": throughput,
                    "vram_mb": vram_mb
                }) + "\n")

        # Periodic Validation
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_multi_k(model, test_in, test_lbl, eval_horizons=[2, 4, 8, 16, 32, 64, 128], batch_size=args.batch_size)
            mean_loss_now = min(m["mean_loss"] for m in val_results.values())
            max_cell_now = max(m["cell_accuracy"] for m in val_results.values())
            max_exact_now = max(m["exact_accuracy"] for m in val_results.values())

            print(f"\n[VALIDATION Step {step}] Best Loss: {mean_loss_now:.4f} | Best Cell Acc: {max_cell_now*100:.2f}% | Exact Acc: {max_exact_now*100:.2f}%", flush=True)
            for k in [4, 16, 64, 128]:
                if k in val_results:
                    m = val_results[k]
                    print(f"   K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:.2f}% | Loss: {m['mean_loss']:.4f}", flush=True)

            if mean_loss_now < best_loss or max_cell_now > best_cell_acc:
                best_loss = min(best_loss, mean_loss_now)
                best_cell_acc = max(best_cell_acc, max_cell_now)
                torch.save({"model": model.state_dict(), "step": step, "val_results": val_results}, out_path / "Best_CBIM_Sudoku_3000.pt")

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "val", "step": step,
                    "best_loss": mean_loss_now,
                    "best_cell_acc": max_cell_now,
                    "exact_acc": max_exact_now,
                    "horizons": val_results
                }) + "\n")

    total_time_min = (time.perf_counter() - start_time) / 60.0
    torch.save({"model": model.state_dict(), "step": args.steps}, out_path / "Final_CBIM_Sudoku_3000.pt")
    print(f"\nTraining 3000 steps completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time pondering sweep
    print("\n" + "=" * 85)
    print("   POST-TRAINING FULL TEST-TIME PONDERING SWEEP (K in [1..256])")
    print("=" * 85)
    final_sweep = evaluate_multi_k(
        model, test_in, test_lbl,
        eval_horizons=[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        batch_size=args.batch_size
    )
    for k, m in final_sweep.items():
        print(f"  Ponder Depth K = {k:3d} | Exact Acc: {m['exact_accuracy']*100:5.2f}% | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:.4f}")

    final_report = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_params": num_params,
        "total_time_min": total_time_min,
        "best_cell_acc": best_cell_acc,
        "best_loss": best_loss,
        "final_ponder_sweep": final_sweep,
        "trm_1000_baseline": {
            "num_params": 5028866,
            "cell_accuracy": 0.5156,
            "mean_loss": 44.04,
            "training_time_min": 35.58
        }
    }
    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFinal report written to {out_path / 'final_report.json'}")


if __name__ == "__main__":
    main()
