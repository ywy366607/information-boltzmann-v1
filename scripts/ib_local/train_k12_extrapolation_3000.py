"""Train Master Unified Never-Reset CBIM on ONLY K in [1, 2] to test Zero-Shot Pondering Extrapolation.
Core Research Question:
Can a continuous neural-operator model, having ONLY EVER SEEN K=1 and K=2 during training,
zero-shot extrapolate to K=4, 8, 16, 32, 64, 128, 256, 512 at inference time (just like Language did)?

Configuration:
- Architecture: MasterUnifiedCBIMSudokuModel (d=256, 0.630M params)
- Training Horizons: STRICTLY K in [1, 2] (Model never sees K >= 3 during training!)
- Persistence: Never-Reset continuous stream carry
- Schedule: 3,000 steps with Cosine LR (3e-4 -> 1e-5)
- Post-training Evaluation: Full test-time sweep K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
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
import torch.nn.functional as F

from scripts.ib_local.train_never_reset_unified_3000 import (
    MasterUnifiedCBIMSudokuModel,
    evaluate_never_reset_stream
)
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


def main():
    parser = argparse.ArgumentParser(description="Test K=1,2 Training with Zero-Shot Extrapolation to K=512")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--d-channels", type=int, default=256, help="Phase space feature channels")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--output-dir", type=str, default="results/k12_extrapolation_experiment_3000", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_log = out_path / "metrics.jsonl"

    print("=" * 110)
    print("   ZERO-SHOT PONDERING EXTRAPOLATION EXPERIMENT: TRAIN ON ONLY K in [1, 2]")
    print(f"   Architecture: Master Unified Never-Reset CBIM (d={args.d_channels}, 0.630M params)")
    print(f"   Training Horizons: STRICTLY K in [1, 2] | Steps: {args.steps} | Eff Batch: {args.micro_batch_size * args.grad_accum_steps}")
    print(f"   Inference Evaluation: Zero-Shot Extrapolation to K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]")
    print("=" * 110)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = MasterUnifiedCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params / 1e6:.3f}M)\n")

    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state
    init_sample = torch.as_tensor(train_in[:args.micro_batch_size], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(args.micro_batch_size, 9, 9, args.d_channels))

    # STRICTLY K in [1, 2] during training!
    training_horizons = [1, 2]
    probs = [0.5, 0.5]
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

        k_step = int(rng.choice(training_horizons, p=probs))
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

        if step % 100 == 0 or step == 1:
            eff_batch = args.micro_batch_size * args.grad_accum_steps
            throughput = eff_batch / (step_time_ms / 1000.0)
            theta_m = diag_last.get("theta_mean", 0.0)
            cool_p = diag_last.get("cooling_power", 0.0)
            print(f"Step {step:4d}/{args.steps} | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | theta: {theta_m:.3f} | cool: {cool_p:.2e} | Time: {step_time_ms:5.1f}ms | Throughput: {throughput:5.1f} puz/s", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "k": k_step, "loss": accum_loss, "lr": current_lr,
                    "theta_mean": float(theta_m), "cooling_power": float(cool_p),
                    "step_time_ms": step_time_ms, "throughput": throughput, "vram_mb": vram_mb
                }) + "\n")

        # Periodic Zero-Shot Extrapolation Check
        if step % args.eval_interval == 0 or step == args.steps:
            val_results = evaluate_never_reset_stream(
                model, test_in, test_lbl,
                mature_state=persistent_state,
                horizons=[1, 2, 4, 8, 16, 32, 64],
                batch_size=32
            )
            print(f"\n[INTERIM ZERO-SHOT EXTRAPOLATION Step {step}] Trained ONLY on K in [1, 2]! Testing K in [1..64]:", flush=True)
            for k in [1, 2, 4, 8, 16, 32, 64]:
                m = val_results[k]
                tag = " (In-Domain)" if k in [1, 2] else " (Zero-Shot Extrapolated!)"
                print(f"   K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{tag}", flush=True)

            acc_k16 = val_results[16]["cell_accuracy"]
            if acc_k16 > best_cell_acc:
                best_cell_acc = acc_k16
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "val_results": val_results
                }, out_path / "Best_K12_Extrapolated_d256.pt")
                print(f"Saved new best extrapolated model to {out_path / 'Best_K12_Extrapolated_d256.pt'}\n", flush=True)

    total_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"\nTraining completed in {total_time_min:.2f} minutes!")

    # Post-training full test-time zero-shot pondering sweep out to K=512
    print("\n" + "=" * 105)
    print("   POST-TRAINING FULL ZERO-SHOT PONDERING EXTRAPOLATION SWEEP (K in [1..512])")
    print("   (Model was trained ONLY on K=1 and K=2! Testing depth generalization out to 512!)")
    print("=" * 105)
    final_sweep = evaluate_never_reset_stream(
        model, test_in, test_lbl,
        mature_state=persistent_state,
        horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512],
        batch_size=32
    )
    for k, m in final_sweep.items():
        tag = " (In-Domain)" if k in [1, 2] else " (Zero-Shot Extrapolation!)"
        print(f"  Ponder Depth K = {k:3d} | Exact Acc: {m['exact_accuracy']*100:5.2f}% | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}{tag}")

    with open(out_path / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "trained_horizons": [1, 2], "steps": args.steps, "d_channels": args.d_channels,
            "total_time_min": total_time_min, "final_sweep": final_sweep
        }, f, indent=2)


if __name__ == "__main__":
    main()
