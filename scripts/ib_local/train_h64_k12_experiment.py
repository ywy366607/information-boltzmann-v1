"""Recurrent Horizon Experiment (STRICT SINGLE-STREAM, BATCH=1, NO PARALLEL SLOTS).
Core Principles:
1. Strict Single-Stream: Batch=1! Zero cross-slot random puzzle interference!
   Only ONE physical field [1, 9, 9, 256].
2. Sequential Stream: Puzzles stream in sequentially (puzzle 0 -> puzzle 1 -> puzzle 2...).
3. Dual-Clock Architecture:
   - For each puzzle, the model unrolls across H=64 macro-steps with K in [1, 2] microsteps.
   - BPTT backpropagates across the H=64 trajectory with deep supervision milestones at h in [8, 16, 32, 48, 64].
   - State is carried to the next puzzle (Never Reset!).
4. Post-Training Evaluation Suite:
   - Suite A: Zero-Shot K-Extrapolation at H=1 (K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).
   - Suite B: Cognitive Horizon Interchangeability Test (H * K == 64).
   - Suite C: Deep Dual-Clock Scaling (H=64 with K scaling).
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

from scripts.ib_local.train_stage3_mass_norm_readout_3000 import Stage3MassNormCBIMSudokuModel
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


@torch.no_grad()
def evaluate_single_stream_dual_clock(model, test_in: np.ndarray, test_lbl: np.ndarray,
                                      mature_state: torch.Tensor, h_val: int, k_val: int) -> Dict[str, float]:
    """Strict Single-Stream (Batch=1) evaluation with state carried across all test puzzles!"""
    model.eval()
    device = mature_state.device
    state = mature_state.detach().clone()  # [1, 9, 9, d]
    num_test = len(test_in)

    total_cells = 0
    correct_cells = 0
    exact_matches = 0
    losses = []

    for idx in range(num_test):
        inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device=device)

        curr_state = state
        final_logits = None

        # Unroll for H macro-steps
        for h in range(h_val):
            curr_state, logits, _ = model.forward_stream_step(curr_state, inp, k_step=k_val)
            final_logits = logits

        loss = F.cross_entropy(final_logits.view(-1, 11), lbl.view(-1)).item()
        losses.append(loss)
        preds = torch.argmax(final_logits, dim=-1)
        correct_cells += (preds == lbl).sum().item()
        total_cells += 81
        exact_matches += ((preds == lbl).sum().item() == 81)

        # Single-stream Never-Reset carry!
        state = curr_state.detach()

    return {
        "cell_accuracy": correct_cells / total_cells,
        "exact_accuracy": exact_matches / num_test,
        "mean_loss": float(np.mean(losses)),
        "H": h_val,
        "K": k_val,
        "total_depth": h_val * k_val
    }


def main():
    parser = argparse.ArgumentParser(description="H=64, K in [1, 2] Single-Stream Experiment")
    parser.add_argument("--puzzles", type=int, default=5000, help="Total sequential puzzles to train on")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--H-macro", type=int, default=64, help="Macro-steps per puzzle during training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/h64_k12_single_stream_experiment", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 115)
    print("   STRICT SINGLE-STREAM RECURRENT HORIZON EXPERIMENT: BATCH=1, H=64 MACRO-STEPS, K in [1, 2]")
    print(f"   Architecture: Stage 3 Champion (Exact Exp Transport + dt=1/K + Mass-Norm Readout)")
    print(f"   Single Stream: BATCH=1! Zero cross-slot random puzzle mixing! Only ONE physical field [1, 9, 9, 256]")
    print(f"   Training Horizon: {args.puzzles} sequential puzzles | H={args.H_macro} macro-steps/puzzle | K in [1, 2]")
    print("=" * 115)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = Stage3MassNormCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state: single world [1, 9, 9, d]
    init_sample = torch.as_tensor(train_in[0:1], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(1, 9, 9, args.d_channels))

    training_horizons = [1, 2]
    probs = [0.5, 0.5]
    rng = np.random.default_rng(42)

    supervision_milestones = [8, 16, 32, 48, 64]
    start_time = time.perf_counter()

    for step in range(1, args.puzzles + 1):
        t0 = time.perf_counter()
        model.train()

        progress = step / args.puzzles
        current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        # Strictly sequential stream (puzzle 0 -> puzzle 1 -> puzzle 2...)
        puzzle_idx = (step - 1) % num_train
        inp = torch.as_tensor(train_in[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            curr_state = persistent_state
            traj_losses = []

            for h in range(1, args.H_macro + 1):
                k_step = int(rng.choice(training_horizons, p=probs))
                curr_state, logits, _ = model.forward_stream_step(curr_state, inp, k_step=k_step)

                if h in supervision_milestones:
                    loss_h = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))
                    traj_losses.append(loss_h)

            total_loss = sum(traj_losses) / len(traj_losses)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Never Reset: single stream carries forward!
        with torch.no_grad():
            persistent_state.copy_(curr_state.detach())

        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000.0

        if step % 100 == 0 or step == 1:
            elapsed = time.perf_counter() - start_time
            fps = step / elapsed
            print(f"Puzzle {step:5d}/{args.puzzles} ({step/args.puzzles*100:4.1f}%) | Loss: {total_loss.item():6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:5.1f}ms | {fps:5.1f} puz/s", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "puzzle": step, "loss": total_loss.item(), "lr": current_lr,
                    "step_time_ms": step_time_ms, "fps": fps
                }) + "\n")

        # Periodic Diagnostic Check
        if step % args.eval_interval == 0:
            print(f"\n[INTERIM EVALUATION Puzzle {step}] Testing (H=1, K in [1, 2, 4, 8, 16, 32, 64]) & (H=64, K=1)...", flush=True)
            for test_k in [1, 2, 4, 8, 16, 32, 64]:
                r = evaluate_single_stream_dual_clock(model, test_in[:200], test_lbl[:200], persistent_state, h_val=1, k_val=test_k)
                tag = " (In-Domain)" if test_k in [1, 2] else " (Zero-Shot Extrapolated!)"
                print(f"   H= 1, K={test_k:2d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}{tag}", flush=True)

            r_rec = evaluate_single_stream_dual_clock(model, test_in[:200], test_lbl[:200], persistent_state, h_val=64, k_val=1)
            print(f"   H=64, K= 1 | Cell Acc: {r_rec['cell_accuracy']*100:5.2f}% | Loss: {r_rec['mean_loss']:6.3f} (Trained H=64 Recurrence!)\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nTraining completed in {total_time_min:.2f} minutes!")

    # =========================================================================
    # POST-TRAINING MASTER SUITE: TESTING ALL 3 HYPOTHESES UNDER BATCH=1
    # =========================================================================
    print("\n" + "=" * 115)
    print("   POST-TRAINING COMPREHENSIVE DUAL-CLOCK EVALUATION SUITE (BATCH=1)")
    print("=" * 115)

    report = {"training_params": vars(args), "total_time_min": total_time_min}

    # Suite A: Zero-Shot K-Extrapolation at H=1
    print("\n[SUITE A] Zero-Shot K-Extrapolation at H=1 (Model trained on H=64 BPTT with K in [1, 2], Batch=1):")
    sweep_a = {}
    for test_k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]:
        r = evaluate_single_stream_dual_clock(model, test_in, test_lbl, persistent_state, h_val=1, k_val=test_k)
        sweep_a[test_k] = r
        tag = " (In-Domain K)" if test_k in [1, 2] else " (Zero-Shot Extrapolated K!)"
        print(f"  H=  1, K = {test_k:3d} | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}{tag}")
    report["suite_a_zero_shot_k_sweep"] = sweep_a

    # Suite B: Interchangeability Test (H * K == 64)
    print("\n[SUITE B] Cognitive Horizon Interchangeability Test (Target Total Depth = 64):")
    configs_b = [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)]
    sweep_b = {}
    for h_val, k_val in configs_b:
        r = evaluate_single_stream_dual_clock(model, test_in, test_lbl, persistent_state, h_val=h_val, k_val=k_val)
        sweep_b[f"H{h_val}_K{k_val}"] = r
        print(f"  (H={h_val:2d}, K={k_val:2d}) [Total={h_val*k_val:2d}] | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}")
    report["suite_b_interchangeability"] = sweep_b

    # Suite C: Deep Recurrent Pondering (H=64 with K scaling)
    print("\n[SUITE C] Deep Dual-Clock Recurrent Pondering (H=64 with K scaling):")
    sweep_c = {}
    for k_val in [1, 2, 4, 8]:
        r = evaluate_single_stream_dual_clock(model, test_in, test_lbl, persistent_state, h_val=64, k_val=k_val)
        sweep_c[f"H64_K{k_val}"] = r
        print(f"  H=64, K={k_val:2d} [Total Depth={64*k_val:3d}] | Cell Acc: {r['cell_accuracy']*100:5.2f}% | Loss: {r['mean_loss']:6.3f}")
    report["suite_c_deep_dual_clock"] = sweep_c

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "persistent_state": persistent_state.detach().cpu(),
        "report": report
    }, out_path / "Final_H64_K12_Batch1_Model.pt")
    print(f"\nAll suites completed! Final report saved to {out_path / 'final_report.json'}")


if __name__ == "__main__":
    main()
