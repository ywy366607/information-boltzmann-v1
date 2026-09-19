"""Unified Benchmark Harness: CBIM vs TRM on Sudoku-Extreme.
Runs head-to-head training, evaluation, and test-time pondering depth scaling
under 4GB single-GPU constraints (strict 0 MB Windows shared memory).
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
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config
)
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


def create_trm_model(batch_size: int = 32, hidden_size: int = 512, forward_dtype: str = "float32"):
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=batch_size,
        seq_len=81,
        vocab_size=11,
        puzzle_emb_ndim=hidden_size,
        num_puzzle_identifiers=1,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=hidden_size,
        expansion=4,
        num_heads=8,
        pos_encodings="none",
        forward_dtype=forward_dtype,
        mlp_t=True,
        puzzle_emb_len=16,
        no_ACT_continue=True,
        halt_max_steps=16,
        halt_exploration_prob=0.1
    )
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        inner_model = TinyRecursiveReasoningModel_ACTV1(config.model_dump())
        model = ACTLossHead(inner_model, loss_type="stablemax_cross_entropy")
    return model


def create_cbim_model(batch_size: int = 32, d_channels: int = 128, ponder_steps: int = 6):
    inner_model = CBIMSudokuModel(
        vocab_size=11,
        d_channels=d_channels,
        n_velocities=8,
        ponder_steps=ponder_steps,
        halt_max_steps=16
    )
    model = ACTLossHead(inner_model, loss_type="stablemax_cross_entropy")
    return model


@torch.no_grad()
def evaluate_exact_accuracy(model, test_in: np.ndarray, test_lbl: np.ndarray,
                            batch_size: int = 64, max_steps: int = 16,
                            cbim_ponder_k: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    num_samples = len(test_in)
    total_correct_cells = 0
    total_cells = 0
    total_exact_boards = 0
    total_steps_taken = 0
    total_loss = 0.0
    num_batches = 0

    # If testing CBIM with custom ponder depth K
    if cbim_ponder_k is not None and hasattr(model.model, "ponder_steps"):
        orig_k = model.model.ponder_steps
        model.model.ponder_steps = cbim_ponder_k

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        B = end_idx - start_idx
        inp = torch.as_tensor(test_in[start_idx:end_idx], dtype=torch.long, device="cuda")
        lbl = torch.as_tensor(test_lbl[start_idx:end_idx], dtype=torch.long, device="cuda")

        batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device="cuda")}
        carry = model.model.initial_carry(batch)

        # Run ACT until all sequences halt or max_steps reached
        for step in range(max_steps):
            carry, loss, metrics, preds, _ = model(carry=carry, batch=batch, return_keys=["logits"])
            if carry.halted.all():
                break

        logits = preds["logits"]  # [B, 81, 11]
        pred_digits = torch.argmax(logits, dim=-1)  # [B, 81]

        # Cell-level accuracy
        cell_match = (pred_digits == lbl)
        total_correct_cells += int(cell_match.sum().item())
        total_cells += int(lbl.numel())

        # Exact board match (all 81 cells correct)
        board_match = cell_match.all(dim=-1)
        total_exact_boards += int(board_match.sum().item())

        total_steps_taken += float(carry.steps.float().mean().item()) * B
        total_loss += float(loss.item()) * B
        num_batches += 1

    if cbim_ponder_k is not None and hasattr(model.model, "ponder_steps"):
        model.model.ponder_steps = orig_k

    return {
        "exact_accuracy": total_exact_boards / num_samples,
        "cell_accuracy": total_correct_cells / total_cells,
        "mean_loss": total_loss / num_samples,
        "mean_steps": total_steps_taken / num_samples,
        "num_puzzles": num_samples
    }


def train_benchmark(arch: str, data_dir: str, steps: int = 1000,
                    batch_size: int = 32, lr: float = 1e-4,
                    eval_interval: int = 250, output_dir: str = "results/sudoku_benchmark",
                    cbim_k: int = 6):
    print("=" * 80)
    print(f"   STARTING BENCHMARK RUN: {arch.upper()} on Sudoku-Extreme")
    print(f"   Steps: {steps} | Batch Size: {batch_size} | LR: {lr} | CBIM-K: {cbim_k} | Data: {data_dir}")
    print("=" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / f"{arch}_metrics.jsonl"

    data = load_dataset(data_dir)
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    if arch == "trm":
        model = create_trm_model(batch_size=batch_size, hidden_size=512, forward_dtype="float32").to(device)
    elif arch == "cbim":
        model = create_cbim_model(batch_size=batch_size, d_channels=128, ponder_steps=cbim_k).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch.upper()} | Parameter Count: {num_params:,} ({num_params/1e6:.3f}M)")

    # Optimizer
    optimizer = AdamATan2(model.parameters(), lr=lr, weight_decay=1.0)

    carry = None
    best_exact_acc = 0.0
    history = []

    # Initial evaluation
    print("\n--- Initial Step 0 Evaluation ---")
    val_metrics = evaluate_exact_accuracy(model, test_in, test_lbl, batch_size=batch_size)
    print(f"Step 0 | Test Exact Acc: {val_metrics['exact_accuracy']*100:.2f}% | Cell Acc: {val_metrics['cell_accuracy']*100:.2f}% | Loss: {val_metrics['mean_loss']:.4f}")

    start_time = time.perf_counter()
    rng = np.random.default_rng(42)

    for step in range(1, steps + 1):
        model.train()
        t0 = time.perf_counter()

        # Sample batch
        idx = rng.integers(0, num_train, size=batch_size)
        inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
        batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)}

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
            throughput = batch_size / (step_time_ms / 1000.0)
            print(f"Step {step:5d}/{steps} | Loss: {loss.item():.4f} | Time: {step_time_ms:5.1f}ms | Throughput: {throughput:5.1f} puz/s | VRAM: {vram_mb:6.1f}MB", flush=True)
            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "arch": arch, "step": step,
                    "loss": float(loss.item()), "step_time_ms": step_time_ms,
                    "throughput": throughput, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Evaluation
        if step % eval_interval == 0 or step == steps:
            val_metrics = evaluate_exact_accuracy(model, test_in, test_lbl, batch_size=batch_size)
            exact_acc = val_metrics["exact_accuracy"]
            cell_acc = val_metrics["cell_accuracy"]
            print(f"\n[VAL Step {step}] Exact Acc: {exact_acc*100:.2f}% | Cell Acc: {cell_acc*100:.2f}% | Loss: {val_metrics['mean_loss']:.4f}", flush=True)

            if exact_acc > best_exact_acc:
                best_exact_acc = exact_acc
                torch.save({"model": model.state_dict(), "step": step, "exact_acc": exact_acc}, out_path / f"{arch}_best.pt")

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "val", "arch": arch, "step": step,
                    "exact_accuracy": exact_acc, "cell_accuracy": cell_acc,
                    "mean_loss": val_metrics["mean_loss"]
                }) + "\n")

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\n{arch.upper()} Training completed in {total_time_min:.2f} minutes! Best Exact Acc: {best_exact_acc*100:.2f}%")

    # Final Test-Time Pondering Depth Scaling Sweep
    print("\n" + "=" * 80)
    print(f"   TEST-TIME PONDERING DEPTH SCALING SWEEP ({arch.upper()})")
    print("=" * 80)
    ponder_results = {}

    if arch == "cbim":
        horizons_k = [1, 2, 4, 6, 8, 12, 16, 24, 32]
        for k in horizons_k:
            res = evaluate_exact_accuracy(model, test_in, test_lbl, batch_size=batch_size, cbim_ponder_k=k)
            ponder_results[k] = res
            print(f"  CBIM Depth K = {k:2d} | Exact Acc: {res['exact_accuracy']*100:5.2f}% | Cell Acc: {res['cell_accuracy']*100:5.2f}% | Loss: {res['mean_loss']:.4f}")
    elif arch == "trm":
        max_steps_list = [1, 2, 4, 8, 12, 16, 24]
        for s in max_steps_list:
            res = evaluate_exact_accuracy(model, test_in, test_lbl, batch_size=batch_size, max_steps=s)
            ponder_results[s] = res
            print(f"  TRM Max Steps = {s:2d} | Exact Acc: {res['exact_accuracy']*100:5.2f}% | Cell Acc: {res['cell_accuracy']*100:5.2f}% | Loss: {res['mean_loss']:.4f}")

    with open(out_path / f"{arch}_ponder_scaling.json", "w", encoding="utf-8") as f:
        json.dump(ponder_results, f, indent=2)

    return {
        "arch": arch, "num_params": num_params,
        "best_exact_acc": best_exact_acc,
        "final_val_metrics": val_metrics,
        "total_time_min": total_time_min,
        "ponder_results": ponder_results
    }


def main():
    parser = argparse.ArgumentParser(description="CBIM vs TRM Sudoku-Extreme Benchmark")
    parser.add_argument("--arch", choices=["trm", "cbim", "both"], default="both")
    parser.add_argument("--data-dir", default="data/sudoku-extreme-1k-aug-100")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--cbim-k", type=int, default=6)
    parser.add_argument("--output-dir", default="results/sudoku_benchmark")
    args = parser.parse_args()

    results = {}
    if args.arch in ("trm", "both"):
        results["trm"] = train_benchmark(
            arch="trm", data_dir=args.data_dir, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, eval_interval=args.eval_interval,
            output_dir=args.output_dir
        )
    if args.arch in ("cbim", "both"):
        results["cbim"] = train_benchmark(
            arch="cbim", data_dir=args.data_dir, steps=args.steps,
            batch_size=args.batch_size, lr=args.lr, eval_interval=args.eval_interval,
            output_dir=args.output_dir, cbim_k=args.cbim_k
        )

    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results saved to {summary_file}")


if __name__ == "__main__":
    main()
