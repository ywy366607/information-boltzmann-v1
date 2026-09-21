"""Controlled 500-Step Benchmark: Internal Phase-Space Momentum on Sudoku-Extreme.
Compares:
1. Branch_B_No_Momentum (beta=0 baseline)
2. Branch_B_HeavyBall_0.7 (Polyak heavy-ball momentum with spherical projection)
3. Branch_B_Adam_Momentum (AdamW-style coordinate-wise adaptive damping)

Strictly identical seeds, data order, and multi-horizon training protocol.
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
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku_momentum import MomentumCBIMSudokuModel
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


def train_momentum_experiment():
    print("=" * 95)
    print("   CONTROLLED 500-STEP BENCHMARK: INTERNAL PHASE-SPACE MOMENTUM ON SUDOKU-EXTREME")
    print("=" * 95)

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
        ("No_Momentum", "none", 0.0, 0.99),
        ("HeavyBall_0.7", "heavy_ball", 0.7, 0.99),
        ("Adam_Momentum", "adam", 0.7, 0.99)
    ]

    all_results = {}
    checkpoint_dir = Path("checkpoints/momentum_500")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name, mode, beta, beta2 in configs:
        print("\n" + "=" * 85)
        print(f"   TRAINING 500 STEPS: {name} (Mode: {mode}, beta: {beta}, beta2: {beta2})")
        print("=" * 85)

        inner = MomentumCBIMSudokuModel(
            vocab_size=11, d_channels=128, ponder_steps=16,
            momentum_mode=mode, beta=beta, beta2=beta2
        ).to(device)

        model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
        optim = AdamATan2(model.parameters(), lr=3e-4, weight_decay=1.0)
        scaler = torch.amp.GradScaler("cuda")

        carry = None
        # Strict identical random seed for perfect data and horizon comparability
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
                print(f"[{name:15s}] Step {step:3d}/500 | K: {k_step:2d} | Loss: {loss.item():6.3f} | VRAM: {vram_mb:6.1f}MB", flush=True)

        elapsed = (time.perf_counter() - t0) / 60.0
        print(f"{name} training finished in {elapsed:.2f} min. Evaluating on 1,000 test puzzles across K in [1..128]...")

        # Save checkpoint (.pt)
        ckpt_path = checkpoint_dir / f"{name}_step500.pt"
        torch.save({"model": model.state_dict(), "name": name, "mode": mode, "beta": beta, "beta2": beta2}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        eval_res = evaluate_model(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128], batch_size=32)
        all_results[name] = eval_res

        print(f"\nResults for {name}:")
        for k, r in eval_res.items():
            print(f"  K = {k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}")

    # Comparative Summary Table
    print("\n" + "=" * 105)
    print("   HEAD-TO-HEAD COMPARISON: NO MOMENTUM vs HEAVY-BALL vs ADAM ADAPTIVE MOMENTUM")
    print("=" * 105)
    header = f"{'K':>4s} | {'No_Mom Acc':>11s} | {'HeavyBall Acc':>14s} | {'AdamMom Acc':>12s} | {'HB Delta':>9s} | {'No_Mom Loss':>12s} | {'HB Loss':>8s} | {'Adam Loss':>10s}"
    print(header)
    print("-" * 105)

    eval_k_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for k in eval_k_list:
        nom = all_results["No_Momentum"][k]
        hb = all_results["HeavyBall_0.7"][k]
        adm = all_results["Adam_Momentum"][k]

        hb_delta = (hb["cell_accuracy"] - nom["cell_accuracy"]) * 100
        row = (f"{k:4d} | {nom['cell_accuracy']*100:10.2f}% | {hb['cell_accuracy']*100:13.2f}% | "
               f"{adm['cell_accuracy']*100:11.2f}% | {hb_delta:+8.2f}% | {nom['mean_loss']:12.3f} | "
               f"{hb['mean_loss']:8.3f} | {adm['mean_loss']:10.3f}")
        print(row)

    out_file = Path("results/ablation_momentum_500.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    train_momentum_experiment()
