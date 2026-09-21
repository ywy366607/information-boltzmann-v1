"""Arm C Champion (d=256, 0.532M Params) 3,000-Step Full Convergence Training.
Features:
- Architecture: Conditioned Conservative TC + Learned Isoenergetic Corrective Flow on Riemannian Tangent Space
- Capacity: d=256, 8 velocities, 32 channels/vel, 252 active Lie algebra Givens rotation dimensions
- Gradient Accumulation: micro-batch 16, accum 2 (effective batch 32), rock-solid <2.0 GB VRAM
- Optimizer: AdamATan2 with Cosine LR schedule (3e-4 -> 1e-5)
- Evaluation: Every 500 steps on 1,000 test puzzles across K in [1, 2, 4, 8, 16, 32, 64, 128, 256]
- Post-training deep pondering sweep up to K=512
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


def main():
    parser = argparse.ArgumentParser(description="Train Arm C Champion 3,000 steps")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--d-channels", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)  # Effective batch = 32
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--output-dir", default="results/arm_c_champion_3000")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 105)
    print("   ARM C CHAMPION 3,000-STEP CONVERGENCE TRAINING ON SUDOKU-EXTREME")
    print(f"   Architecture: Conditioned TC + Learned Isoenergetic Corrective Flow (d={args.d_channels})")
    print(f"   Steps: {args.steps} | Micro-Batch: {args.micro_batch_size} | Accum: {args.grad_accum_steps} (Eff Batch: {args.micro_batch_size * args.grad_accum_steps})")
    print(f"   Base LR: {args.lr} | Min LR: {args.min_lr} | Output: {args.output_dir}")
    print("=" * 105)

    data = load_dataset("data/sudoku-extreme-1k-aug-100")
    train_in, train_lbl = data["train_in"], data["train_lbl"]
    test_in, test_lbl = data["test_in"], data["test_lbl"]
    num_train = len(train_in)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Multi-horizon sampling scales
    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)

    inner_model = CorrectiveCBIMSudokuModel(
        vocab_size=11, d_channels=args.d_channels, ponder_steps=16,
        arm="corrective_flow", alpha_max=0.25
    ).to(device)

    num_params = sum(p.numel() for p in inner_model.parameters())
    print(f"Model Initialized | Parameter Count: {num_params:,} ({num_params/1e6:.3f}M)\n", flush=True)

    model = ACTLossHead(inner_model, loss_type="stablemax_cross_entropy")
    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    best_cell_acc = 0.0
    best_exact_acc = 0.0

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

        # Sample horizon K uniformly
        k_step = int(rng.choice(horizons, p=probs))
        model.model.ponder_steps = k_step

        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            idx = rng.integers(0, num_train, size=args.micro_batch_size)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(args.micro_batch_size, dtype=torch.long, device=device)}

            if carry is None:
                carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, loss, loss_metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.item()) / args.grad_accum_steps

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        if step % 50 == 0 or step == 1:
            eff_batch = args.micro_batch_size * args.grad_accum_steps
            throughput = eff_batch / (step_time_ms / 1000.0)
            print(f"Step {step:4d}/{args.steps} | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:6.1f}ms | Throughput: {throughput:5.1f} puz/s | Alloc: {alloc_mb:5.1f}MB | VRAM: {vram_mb:5.1f}MB", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "k": k_step,
                    "loss": accum_loss, "lr": current_lr,
                    "step_time_ms": step_time_ms, "throughput": throughput,
                    "alloc_mb": alloc_mb, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Validation
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_model(model, test_in, test_lbl, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size=32)
            mean_loss_now = min(m["mean_loss"] for m in val_results.values())
            max_cell_now = max(m["cell_accuracy"] for m in val_results.values())
            max_exact_now = max(m["exact_accuracy"] for m in val_results.values())

            print(f"\n[VALIDATION Step {step}] Best Loss: {mean_loss_now:6.3f} | Best Cell Acc: {max_cell_now*100:5.2f}% | Exact Acc: {max_exact_now*100:5.2f}%", flush=True)
            for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                m = val_results[k]
                print(f"   K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f} | S_spec: {m['diagnostics']['S_spec']:5.3f}", flush=True)

            if mean_loss_now < best_loss or max_cell_now > best_cell_acc or max_exact_now > best_exact_acc:
                best_loss = min(best_loss, mean_loss_now)
                best_cell_acc = max(best_cell_acc, max_cell_now)
                best_exact_acc = max(best_exact_acc, max_exact_now)
                torch.save({"model": model.state_dict(), "step": step, "val_results": val_results, "d_channels": args.d_channels}, out_path / "Best_Arm_C_Champion_d256.pt")
                print(f"Saved new best model checkpoint to {out_path / 'Best_Arm_C_Champion_d256.pt'}\n", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "val", "step": step,
                    "best_loss": mean_loss_now,
                    "best_cell_acc": max_cell_now,
                    "exact_acc": max_exact_now,
                    "horizons": val_results
                }) + "\n")

    total_time_min = (time.perf_counter() - start_time) / 60.0
    torch.save({"model": model.state_dict(), "step": args.steps, "d_channels": args.d_channels}, out_path / "Final_Arm_C_Champion_d256.pt")
    print(f"\nTraining completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time pondering sweep
    print("\n" + "=" * 105)
    print("   POST-TRAINING FULL TEST-TIME PONDERING SWEEP (K in [1..512])")
    print("=" * 105)
    final_sweep = evaluate_model(
        model, test_in, test_lbl,
        horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512],
        batch_size=32
    )
    for k, m in final_sweep.items():
        print(f"  Ponder Depth K = {k:3d} | Exact Acc: {m['exact_accuracy']*100:5.2f}% | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f} | S_spec: {m['diagnostics']['S_spec']:5.3f}")

    final_report = {
        "steps": args.steps,
        "d_channels": args.d_channels,
        "num_params": num_params,
        "total_time_min": total_time_min,
        "best_cell_acc": best_cell_acc,
        "best_loss": best_loss,
        "best_exact_acc": best_exact_acc,
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

    print(f"\nFinal report written to {out_path / 'final_report.json'}", flush=True)


if __name__ == "__main__":
    main()
