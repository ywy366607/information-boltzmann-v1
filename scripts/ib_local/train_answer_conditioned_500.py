"""Controlled 500-Step Test: Answer Self-Conditioning (P_problem + P_answer) on Sudoku-Extreme.
Tests whether conditioning on the previous microstep's draft answer improves
constraint satisfaction, error-correction, and cell accuracy.
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
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku import CBIMSudokuModel
from scripts.ib_local.cbim_sudoku_answer_conditioned import AnswerConditionedCBIMSudokuModel
from models.losses import ACTLossHead
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
def evaluate_model(model, test_in: np.ndarray, test_lbl: np.ndarray,
                   horizons: List[int] = [1, 2, 4, 8, 16, 32, 64, 128], batch_size: int = 32):
    model.eval()
    num_samples = len(test_in)
    results = {}

    for k in horizons:
        orig_k = model.model.ponder_steps
        model.model.ponder_steps = k

        total_correct = 0
        total_cells = 0
        total_exact = 0
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            B = end - start
            inp = torch.as_tensor(test_in[start:end], dtype=torch.long, device="cuda")
            lbl = torch.as_tensor(test_lbl[start:end], dtype=torch.long, device="cuda")
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device="cuda")}
            carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, outputs = model.model(carry, batch)
            logits = outputs["logits"].float()
            pred = torch.argmax(logits, dim=-1)

            cell_match = (pred == lbl)
            total_correct += int(cell_match.sum().item())
            total_cells += int(lbl.numel())
            total_exact += int(cell_match.all(dim=-1).sum().item())
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1), ignore_index=0)
            total_loss += float(loss.item()) * B

        model.model.ponder_steps = orig_k
        results[k] = {
            "exact_accuracy": total_exact / num_samples,
            "cell_accuracy": total_correct / total_cells,
            "mean_loss": total_loss / num_samples
        }

    return results


def train_500_test():
    print("=" * 90)
    print("   CONTROLLED 500-STEP TEST: ANSWER SELF-CONDITIONING (P_problem + P_answer)")
    print("=" * 90)

    data_dir = "data/sudoku-extreme-1k-aug-100"
    data = load_dataset(data_dir)
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Multi-horizon sampling scales
    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)

    configs = [
        ("Branch_B_Problem_Only", lambda: CBIMSudokuModel(vocab_size=11, d_channels=128, ponder_steps=16, conditioned=True)),
        ("Branch_B_Plus_Answer_Conditioned", lambda: AnswerConditionedCBIMSudokuModel(vocab_size=11, d_channels=128, ponder_steps=16))
    ]

    all_results = {}

    for name, model_fn in configs:
        print("\n" + "=" * 80)
        print(f"   TRAINING 500 STEPS: {name}")
        print("=" * 80)

        inner = model_fn().to(device)
        model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
        optim = AdamATan2(model.parameters(), lr=3e-4, weight_decay=1.0)
        scaler = torch.amp.GradScaler("cuda")

        carry = None
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()

        for step in range(1, 501):
            model.train()
            k_step = int(rng.choice(horizons, p=probs))
            model.model.ponder_steps = k_step

            idx = rng.integers(0, num_train, size=32)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(32, dtype=torch.long, device=device)}

            if carry is None:
                carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            if step % 50 == 0 or step == 1:
                vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                print(f"[{name}] Step {step:3d}/500 | K: {k_step:2d} | Loss: {loss.item():6.3f} | VRAM: {vram_mb:6.1f}MB", flush=True)

        elapsed = (time.perf_counter() - t0) / 60.0
        print(f"{name} completed in {elapsed:.2f} min. Running evaluation on 1,000 test puzzles...")

        eval_res = evaluate_model(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128], batch_size=32)
        all_results[name] = eval_res

        print(f"\nResults for {name}:")
        for k, r in eval_res.items():
            print(f"  K = {k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}")

    # Print Comparative Summary
    print("\n" + "=" * 90)
    print("   HEAD-TO-HEAD COMPARISON: Problem-Only (Branch B) vs Answer-Conditioned (Branch B+)")
    print("=" * 90)
    print(f"{'K':>4s} | {'Branch B Acc':>14s} | {'Branch B+ Acc':>14s} | {'Delta Acc':>11s} | {'Branch B Loss':>14s} | {'Branch B+ Loss':>14s}")
    print("-" * 85)
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        b = all_results["Branch_B_Problem_Only"][k]
        bp = all_results["Branch_B_Plus_Answer_Conditioned"][k]
        delta_acc = (bp["cell_accuracy"] - b["cell_accuracy"]) * 100
        print(f"{k:4d} | {b['cell_accuracy']*100:13.2f}% | {bp['cell_accuracy']*100:13.2f}% | {delta_acc:+10.2f}% | {b['mean_loss']:14.3f} | {bp['mean_loss']:14.3f}")

    out_file = Path("results/ablation_answer_conditioning_500.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    train_500_test()
