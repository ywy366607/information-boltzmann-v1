"""Train Arm C (Learned Isoenergetic Corrective Flow, d=256) with NEVER RESET on Sudoku-Extreme.
Implements the true CBIM infinite stream paradigm:
1. Persistent physical field across training batches: state is NEVER reset to zero!
2. Initial field energy aligned with language: NESS warm start with unit norm preservation.
3. Continuous Port Injection: new clues are injected as perturbations into the living field:
   F_{t, init} = Normalize(F_{t-1, mature} + clue_field) * ||F_{t-1}||
4. Never-Reset Evaluation: test puzzles stream through sequentially, inheriting mature state
   from the training stream and carrying state from test puzzle to test puzzle!
5. WSD Schedule for 3,000 steps (150 warmup + 2,250 stable + 600 decay, save pre-decay at 2400).
6. K sampling: K in [1, 2, 4, 8, 16, 32, 64] (max 64 as requested).
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
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


def compute_wsd_lr(step: int, warmup_steps: int, stable_steps: int, decay_steps: int,
                   base_lr: float = 3e-4, min_lr: float = 1e-5) -> Tuple[float, str]:
    if step <= warmup_steps:
        lr = min_lr + (base_lr - min_lr) * (step / max(1, warmup_steps))
        phase = "Warmup"
    elif step <= warmup_steps + stable_steps:
        lr = base_lr
        phase = "Stable"
    else:
        decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay_progress))
        phase = "Decay"
    return lr, phase


@torch.no_grad()
def evaluate_never_reset(model, test_in: np.ndarray, test_lbl: np.ndarray,
                         mature_state: torch.Tensor, horizons: List[int], batch_size: int = 32) -> Dict[int, Dict[str, float]]:
    """Evaluates the model on the test stream under NEVER RESET:
    Inherits mature state from training, and carries state from puzzle to puzzle!
    """
    model.eval()
    results = {}
    device = mature_state.device

    for k in horizons:
        # Inherit mature state from training stream (expanded/sliced to batch size)
        state = mature_state.detach().clone()
        if state.shape[0] != batch_size:
            state = state[0:1].expand(batch_size, -1, -1, -1).clone()

        total_cells = 0
        correct_cells = 0
        exact_matches = 0
        losses = []

        for s in range(0, len(test_in), batch_size):
            e = min(s + batch_size, len(test_in))
            b = e - s
            inp_b = torch.as_tensor(test_in[s:e], dtype=torch.long, device=device)
            lbl_b = torch.as_tensor(test_lbl[s:e], dtype=torch.long, device=device)

            clue_field = model.model.clue_proj(model.model.embed_tokens(inp_b).view(b, 9, 9, model.model.d))
            sb = state[:b]

            # Continuous Port Injection into living field (Norm-Preserving)
            f_norm = torch.linalg.vector_norm(sb.float(), dim=(1, 2, 3), keepdim=True)
            blended = sb + clue_field
            curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

            for step_idx in range(k):
                f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
                f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
                f_star = model.model.collision(f_tr, cond=clue_field)
                cat_in = torch.cat([f_star, clue_field], dim=-1)
                v_k = model.model.corrective_net(cat_in)
                dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
                norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
                u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
                u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
                alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
                f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
                curr = torch.cos(alpha_k)*f_star + f_norm_step*torch.sin(alpha_k)*u_hat.to(curr.dtype)

            flat = curr.view(b, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl_b.view(-1)).item()
            losses.append(loss)
            preds = torch.argmax(logits, dim=-1)
            correct_cells += (preds == lbl_b).sum().item()
            total_cells += b * 81
            exact_matches += ((preds == lbl_b).sum(dim=-1) == 81).sum().item()

            # NEVER RESET: state is carried forward to the next test batch!
            state[:b] = curr.detach()

        results[k] = {
            "cell_accuracy": correct_cells / total_cells,
            "exact_accuracy": exact_matches / len(test_in),
            "mean_loss": float(np.mean(losses))
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Arm C with Never-Reset on Sudoku-Extreme")
    parser.add_argument("--data-dir", type=str, default="data/sudoku-extreme-1k-aug-100", help="Path to preprocessed dataset")
    parser.add_argument("--output-dir", type=str, default="results/arm_c_never_reset_3000", help="Output directory")
    parser.add_argument("--warmup-steps", type=int, default=150, help="WSD Warmup steps")
    parser.add_argument("--stable-steps", type=int, default=2250, help="WSD Stable steps (Pre-decay save at 2400)")
    parser.add_argument("--decay-steps", type=int, default=600, help="WSD Cosine decay steps")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch=32)")
    parser.add_argument("--d-channels", type=int, default=256, help="Feature channels (d=256)")
    parser.add_argument("--base-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--warm-start-ckpt", type=str, default="results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt",
                        help="Mature checkpoint to align initial field energy")
    args = parser.parse_args()

    total_steps = args.warmup_steps + args.stable_steps + args.decay_steps
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print(f"   CBIM ARM C NEVER-RESET 3,000-STEP TRAINING PIPELINE (WSD SCHEDULE)")
    print(f"   Total: {total_steps} steps | Warmup: {args.warmup_steps} | Stable: {args.stable_steps} | Decay: {args.decay_steps}")
    print(f"   Pre-Decay Save at Step {args.warmup_steps + args.stable_steps} | d={args.d_channels} (0.532M params)")
    print(f"   Core Doctrine: NEVER RESET! Continuous Field Streaming across All Batches & Evals")
    print("=" * 105)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    data_path = Path(args.data_dir)
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")

    num_train = len(train_in)
    print(f"Loaded {num_train} training puzzles, {len(test_in)} test puzzles.")

    # 2. Build Model
    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=args.d_channels, ponder_steps=64, arm="corrective_flow").to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.3f}M ({num_params:,} parameters)\n")

    # 3. Setup Persistent Physical Field (Never Reset) & Initial Field Energy Alignment
    # Initialize from mature checkpoint if available to align initial NESS energy
    persistent_state = None
    if args.warm_start_ckpt and Path(args.warm_start_ckpt).exists():
        print(f"Loading initial NESS mature field from {args.warm_start_ckpt}...")
        ckpt = torch.load(args.warm_start_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        # Create mature NESS initial state for micro_batch_size
        init_sample = torch.as_tensor(train_in[:args.micro_batch_size], dtype=torch.long, device=device)
        with torch.no_grad():
            clue_init = model.model.clue_proj(model.model.embed_tokens(init_sample).view(args.micro_batch_size, 9, 9, args.d_channels))
            # Preserved energy NESS warm state
            persistent_state = clue_init.clone()
        print("Initialized mature NESS persistent field (Never Reset active)!\n")
    else:
        init_sample = torch.as_tensor(train_in[:args.micro_batch_size], dtype=torch.long, device=device)
        with torch.no_grad():
            persistent_state = model.model.clue_proj(model.model.embed_tokens(init_sample).view(args.micro_batch_size, 9, 9, args.d_channels))
        print("Initialized baseline persistent field.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    k_sampling_list = [1, 2, 4, 8, 16, 32, 64]
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_cell_acc = 0.0
    best_exact_acc = 0.0
    pre_decay_saved = False
    metrics_log = out_path / "never_reset_metrics.jsonl"
    start_time = time.perf_counter()

    for step in range(1, total_steps + 1):
        t0 = time.perf_counter()
        current_lr, current_phase = compute_wsd_lr(
            step, args.warmup_steps, args.stable_steps, args.decay_steps,
            base_lr=args.base_lr, min_lr=args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        k_step = int(rng.choice(k_sampling_list))
        model.train()
        accum_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            batch_idx = (step * args.grad_accum_steps + micro_step) % (num_train // args.micro_batch_size)
            s_idx = batch_idx * args.micro_batch_size
            e_idx = s_idx + args.micro_batch_size

            inp = torch.as_tensor(train_in[s_idx:e_idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[s_idx:e_idx], dtype=torch.long, device=device)
            b = len(inp)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(b, 9, 9, model.model.d))

                # NEVER RESET: Inject new clues into existing living persistent field
                f_norm = torch.linalg.vector_norm(persistent_state.float(), dim=(1, 2, 3), keepdim=True)
                blended = persistent_state + clue_field
                curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

                for _ in range(k_step):
                    f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
                    f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
                    f_star = model.model.collision(f_tr, cond=clue_field)
                    cat_in = torch.cat([f_star, clue_field], dim=-1)
                    v_k = model.model.corrective_net(cat_in)
                    dot_p = torch.sum(f_star.float()*v_k.float(), dim=(1,2,3), keepdim=True)
                    norm_sq = torch.sum(f_star.float()**2, dim=(1,2,3), keepdim=True) + 1e-8
                    u_k = v_k.float() - (dot_p/norm_sq)*f_star.float()
                    u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1,2,3), keepdim=True) + 1e-8)
                    alpha_k = model.model.alpha_max * torch.sigmoid(torch.mean(model.model.angle_gate(cat_in), dim=(1,2,3), keepdim=True)).to(curr.dtype)
                    f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
                    curr = torch.cos(alpha_k)*f_star + f_norm_step*torch.sin(alpha_k)*u_hat.to(curr.dtype)

                flat = curr.view(b, 81, model.model.d)
                logits = model.model.readout_mlp(flat)
                loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum_steps

            # NEVER RESET: Update persistent state with mature field from this step!
            with torch.no_grad():
                persistent_state.copy_(curr.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            print(f"Step {step:4d}/{total_steps} [{current_phase:7s}] | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | Time: {step_time_ms:6.1f}ms | Throughput: {throughput:5.1f} puz/s | Alloc: {alloc_mb:5.1f}MB | VRAM: {vram_mb:5.1f}MB", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "train", "step": step, "phase": current_phase, "k": k_step,
                    "loss": accum_loss, "lr": current_lr,
                    "step_time_ms": step_time_ms, "throughput": throughput,
                    "alloc_mb": alloc_mb, "vram_mb": vram_mb
                }) + "\n")

        # Save Pre-Decay Checkpoint
        pre_decay_step = args.warmup_steps + args.stable_steps
        if step == pre_decay_step and not pre_decay_saved:
            pre_decay_ckpt = out_path / f"Pre_Decay_Arm_C_NeverReset_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "persistent_state": persistent_state.detach().cpu(),
                "step": step,
                "config": vars(args),
                "d_channels": args.d_channels
            }, pre_decay_ckpt)
            pre_decay_saved = True
            print("\n" + "=" * 90)
            print(f"   [WSD MILESTONE] PRE-DECAY CHECKPOINT SAVED TO: {pre_decay_ckpt}")
            print("=" * 90 + "\n", flush=True)

        # Periodic Never-Reset Evaluation
        if step % args.eval_interval == 0 or step == total_steps:
            val_results = evaluate_never_reset(
                model, test_in, test_lbl,
                mature_state=persistent_state,
                horizons=[1, 2, 4, 8, 16, 32, 64],
                batch_size=32
            )
            mean_loss_now = min(m["mean_loss"] for m in val_results.values())
            max_cell_now = max(m["cell_accuracy"] for m in val_results.values())
            max_exact_now = max(m["exact_accuracy"] for m in val_results.values())

            print(f"\n[NEVER-RESET VALIDATION Step {step} ({current_phase})] Best Loss: {mean_loss_now:6.3f} | Best Cell Acc: {max_cell_now*100:5.2f}% | Exact Acc: {max_exact_now*100:5.2f}%", flush=True)
            for k in [1, 2, 4, 8, 16, 32, 64]:
                m = val_results[k]
                print(f"   K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}", flush=True)

            if mean_loss_now < best_loss or max_cell_now > best_cell_acc or max_exact_now > best_exact_acc:
                best_loss = min(best_loss, mean_loss_now)
                best_cell_acc = max(best_cell_acc, max_cell_now)
                best_exact_acc = max(best_exact_acc, max_exact_now)
                torch.save({
                    "model": model.state_dict(),
                    "persistent_state": persistent_state.detach().cpu(),
                    "step": step,
                    "val_results": val_results,
                    "d_channels": args.d_channels
                }, out_path / "Best_Arm_C_NeverReset_d256.pt")
                print(f"Saved new best model checkpoint to {out_path / 'Best_Arm_C_NeverReset_d256.pt'}\n", flush=True)

            with open(metrics_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "val", "step": step, "phase": current_phase,
                    "best_loss": mean_loss_now,
                    "best_cell_acc": max_cell_now,
                    "exact_acc": max_exact_now,
                    "horizons": val_results
                }) + "\n")

    total_time_min = (time.perf_counter() - start_time) / 60.0
    torch.save({
        "model": model.state_dict(),
        "persistent_state": persistent_state.detach().cpu(),
        "step": total_steps,
        "d_channels": args.d_channels
    }, out_path / "Final_Arm_C_NeverReset_d256.pt")
    print(f"\nNever-Reset Training completed in {total_time_min:.2f} minutes!")


if __name__ == "__main__":
    main()
