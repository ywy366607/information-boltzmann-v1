"""Arm C Capacity Scaling Benchmark: d=128 (0.18M) vs d=256 (0.53M) vs d=384 (1.04M).
Tests whether expanding parameter capacity and Lie algebra rotational dimensions
breaks the 45% plateau and improves exact-match and cell accuracy on Sudoku-Extreme.

Strictly identical seeds, data order, batch size, and multi-horizon training protocol.
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

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
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


def compute_field_diagnostics(field: torch.Tensor) -> Dict[str, float]:
    f32 = field.float()
    fft = torch.fft.fftn(f32, dim=(1, 2))
    power = torch.abs(fft) ** 2
    power_spatial = torch.mean(power, dim=(0, 3))

    kx = torch.fft.fftfreq(9, d=1.0, device=field.device)
    ky = torch.fft.fftfreq(9, d=1.0, device=field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    R = torch.sqrt(KX**2 + KY**2)

    total_p = torch.sum(power_spatial) + 1e-12
    e_dc = float((power_spatial[0, 0] / total_p).item())
    e_low = float((torch.sum(power_spatial[(R > 0) & (R <= 0.25)]) / total_p).item())
    e_mid = float((torch.sum(power_spatial[(R > 0.25) & (R <= 0.45)]) / total_p).item())
    e_high = float((torch.sum(power_spatial[R > 0.45]) / total_p).item())

    p_norm = (power_spatial / total_p).clamp(min=1e-12)
    s_spec = float((-torch.sum(p_norm * torch.log(p_norm))).item())
    norm_val = float(torch.mean(torch.linalg.vector_norm(f32, dim=(1, 2, 3))).item())

    return {
        "E_DC": e_dc,
        "E_low": e_low,
        "E_mid": e_mid,
        "E_high": e_high,
        "S_spec": s_spec,
        "field_norm": norm_val
    }


@torch.no_grad()
def evaluate_model(model, test_in: np.ndarray, test_lbl: np.ndarray,
                   horizons: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size: int = 32):
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
        sample_field = None

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

            if sample_field is None:
                sample_field = carry.inner_carry.field.clone()

        diagnostics = compute_field_diagnostics(sample_field)
        model.model.ponder_steps = orig_k
        results[k] = {
            "exact_accuracy": total_exact / num_samples,
            "cell_accuracy": total_correct / total_cells,
            "mean_loss": total_loss / num_samples,
            "diagnostics": diagnostics
        }

    return results


def run_capacity_benchmark(steps: int = 500):
    print("=" * 110)
    print("   ARM C CAPACITY SCALING BENCHMARK: d=128 (0.18M) vs d=256 (0.53M) vs d=384 (1.04M)")
    print(f"   Steps per Scale: {steps} | Task: Sudoku-Extreme 1,000 Test Puzzles")
    print("=" * 110)

    data_dir = "data/sudoku-extreme-1k-aug-100"
    data = load_dataset(data_dir)
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)

    # We test d=256 and d=384 live, and load/compare with d=128 from master tournament
    scales = [
        ("Arm_C_d256", 256),
        ("Arm_C_d384", 384)
    ]

    all_results = {}
    checkpoint_dir = Path("checkpoints/capacity_scaling")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline d=128 if available
    d128_json = Path("results/master_tournament_results_500steps.json")
    if d128_json.exists():
        with open(d128_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
            if "Arm_C_Corrective_Flow" in prev:
                all_results["Arm_C_d128"] = prev["Arm_C_Corrective_Flow"]
                print("\n[Loaded baseline Arm_C_d128 from master tournament results]")

    for name, d_val in scales:
        print("\n" + "=" * 95)
        print(f"   TRAINING {steps} STEPS: {name} (d_channels={d_val})")
        print("=" * 95)

        inner = CorrectiveCBIMSudokuModel(
            vocab_size=11, d_channels=d_val, ponder_steps=16,
            arm="corrective_flow", alpha_max=0.25
        ).to(device)

        params = sum(p.numel() for p in inner.parameters())
        print(f"[{name}] Parameter Count: {params:,} ({params/1e6:.3f}M)", flush=True)

        model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
        optim = AdamATan2(model.parameters(), lr=3e-4, weight_decay=1.0)
        scaler = torch.amp.GradScaler("cuda")

        carry = None
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
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

            if step % 100 == 0 or step == 1:
                vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"[{name:15s}] Step {step:3d}/{steps} | K: {k_step:2d} | Loss: {loss.item():6.3f} | Alloc: {alloc_mb:5.1f}MB | Reserved: {vram_mb:5.1f}MB", flush=True)

        elapsed = (time.perf_counter() - t0) / 60.0
        print(f"{name} completed in {elapsed:.2f} min. Running evaluation across K in [1..256]...", flush=True)

        ckpt_path = checkpoint_dir / f"{name}_step{steps}.pt"
        torch.save({"model": model.state_dict(), "name": name, "d_channels": d_val, "params": params}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}", flush=True)

        eval_res = evaluate_model(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size=32)
        all_results[name] = eval_res

        print(f"\nResults for {name}:")
        for k, r in eval_res.items():
            diag = r["diagnostics"]
            print(f"  K = {k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f} | S_spec: {diag['S_spec']:5.3f} | E_high: {diag['E_high']*100:5.1f}% | Norm: {diag['field_norm']:5.2f}")

    # Comparative Summary Table
    print("\n" + "=" * 125)
    print("   CAPACITY SCALING COMPARISON: d=128 (0.18M) vs d=256 (0.53M) vs d=384 (1.04M)")
    print("=" * 125)
    header = f"{'K':>4s} | {'d=128 Acc':>11s} | {'d=256 Acc':>11s} | {'d=384 Acc':>11s} | {'256 vs 128':>11s} | {'384 vs 128':>11s} | {'d=128 Loss':>11s} | {'d=384 Loss':>11s}"
    print(header)
    print("-" * 125)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        k_str = str(k) if str(k) in all_results["Arm_C_d128"] else k
        c128 = all_results["Arm_C_d128"][k_str]
        c256 = all_results["Arm_C_d256"][k]
        c384 = all_results["Arm_C_d384"][k]
        d256 = (c256["cell_accuracy"] - c128["cell_accuracy"]) * 100
        d384 = (c384["cell_accuracy"] - c128["cell_accuracy"]) * 100
        row = (f"{k:4d} | {c128['cell_accuracy']*100:10.2f}% | {c256['cell_accuracy']*100:10.2f}% | "
               f"{c384['cell_accuracy']*100:10.2f}% | {d256:+10.2f}% | {d384:+10.2f}% | "
               f"{c128['mean_loss']:11.3f} | {c384['mean_loss']:11.3f}")
        print(row)

    out_file = Path("results/capacity_scaling_500.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull capacity scaling results written to {out_file}", flush=True)


if __name__ == "__main__":
    run_capacity_benchmark(steps=500)
