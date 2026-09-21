"""Master Three-Way Tournament Benchmark on Sudoku-Extreme:
Arm A: TC_Baseline (Pure Problem-Conditioned CBIM)
Arm B: Spectral_Viscosity (Residual Scale-Selective Torus Viscosity)
Arm C: Corrective_Flow (Learned Isoenergetic Corrective Flow on Riemannian Tangent Space)

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
import argparse
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
    """Compute 2D spatial Fourier energy distribution and spectral entropy on 9x9 torus."""
    # field: [B, 9, 9, D]
    f32 = field.float()
    fft = torch.fft.fftn(f32, dim=(1, 2))
    power = torch.abs(fft) ** 2  # [B, 9, 9, D]
    power_spatial = torch.mean(power, dim=(0, 3))  # [9, 9]

    # Coordinate frequency radius
    kx = torch.fft.fftfreq(9, d=1.0, device=field.device)
    ky = torch.fft.fftfreq(9, d=1.0, device=field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    R = torch.sqrt(KX**2 + KY**2)

    total_p = torch.sum(power_spatial) + 1e-12
    e_dc = float((power_spatial[0, 0] / total_p).item())
    e_low = float((torch.sum(power_spatial[(R > 0) & (R <= 0.25)]) / total_p).item())
    e_mid = float((torch.sum(power_spatial[(R > 0.25) & (R <= 0.45)]) / total_p).item())
    e_high = float((torch.sum(power_spatial[R > 0.45]) / total_p).item())

    # Spectral entropy
    p_norm = (power_spatial / total_p).clamp(min=1e-12)
    s_spec = float((-torch.sum(p_norm * torch.log(p_norm))).item())

    # Average field norm
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


def run_tournament(steps: int = 1000):
    print("=" * 105)
    print("   MASTER THREE-WAY TOURNAMENT BENCHMARK ON SUDOKU-EXTREME")
    print(f"   Steps per Arm: {steps} | Arms: TC_Baseline vs Spectral_Viscosity vs Corrective_Flow")
    print("=" * 105)

    data_dir = "data/sudoku-extreme-1k-aug-100"
    data = load_dataset(data_dir)
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)

    arms = [
        ("Arm_A_TC_Baseline", "tc_baseline"),
        ("Arm_B_Spectral_Viscosity", "spectral_viscosity"),
        ("Arm_C_Corrective_Flow", "corrective_flow")
    ]

    all_results = {}
    checkpoint_dir = Path("checkpoints/master_tournament")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name, arm_type in arms:
        print("\n" + "=" * 95)
        print(f"   TRAINING {steps} STEPS: {name} ({arm_type})")
        print("=" * 95)

        inner = CorrectiveCBIMSudokuModel(
            vocab_size=11, d_channels=128, ponder_steps=16,
            arm=arm_type, alpha_max=0.25, viscosity_nu=0.05
        ).to(device)

        params = sum(p.numel() for p in inner.parameters())
        print(f"[{name}] Parameter Count: {params:,} ({params/1e6:.3f}M)")

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
                print(f"[{name:24s}] Step {step:4d}/{steps} | K: {k_step:2d} | Loss: {loss.item():6.3f} | VRAM: {vram_mb:6.1f}MB", flush=True)

        elapsed = (time.perf_counter() - t0) / 60.0
        print(f"{name} completed in {elapsed:.2f} min. Running evaluation across K in [1..256]...")

        # Save checkpoint (.pt)
        ckpt_path = checkpoint_dir / f"{name}_step{steps}.pt"
        torch.save({"model": model.state_dict(), "name": name, "arm": arm_type, "params": params}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        eval_res = evaluate_model(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size=32)
        all_results[name] = eval_res

        print(f"\nResults for {name}:")
        for k, r in eval_res.items():
            diag = r["diagnostics"]
            print(f"  K = {k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f} | S_spec: {diag['S_spec']:5.3f} | E_high: {diag['E_high']*100:5.1f}% | Norm: {diag['field_norm']:5.2f}")

    # Head-to-Head Comparison Summary
    print("\n" + "=" * 120)
    print("   MASTER THREE-WAY TOURNAMENT RESULTS: CELL ACCURACY & LOSS COMPARISON")
    print("=" * 120)
    print(f"{'K':>4s} | {'Arm A (TC) Acc':>14s} | {'Arm B (Visc) Acc':>16s} | {'Arm C (Flow) Acc':>16s} | {'C vs A Delta':>12s} | {'Arm A Loss':>11s} | {'Arm C Loss':>11s}")
    print("-" * 120)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        a = all_results["Arm_A_TC_Baseline"][k]
        b = all_results["Arm_B_Spectral_Viscosity"][k]
        c = all_results["Arm_C_Corrective_Flow"][k]
        delta_ca = (c["cell_accuracy"] - a["cell_accuracy"]) * 100
        print(f"{k:4d} | {a['cell_accuracy']*100:13.2f}% | {b['cell_accuracy']*100:15.2f}% | {c['cell_accuracy']*100:15.2f}% | {delta_ca:+11.2f}% | {a['mean_loss']:11.3f} | {c['mean_loss']:11.3f}")

    out_file = Path(f"results/master_tournament_results_{steps}steps.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull tournament results written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    run_tournament(steps=args.steps)
