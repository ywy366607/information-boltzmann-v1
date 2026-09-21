"""Branch 1 with Batch=1 (Strict Single-Stream, No Accumulation).
Proves whether Never-Reset underperformance was caused by Batch slot jitter / cross-slot interference.
Configuration:
- Base: Branch 1 (Never-Reset + Port Injection + Corrective Flow).
- Batch size: 1 (Strictly single physical field [1, 9, 9, 256]).
- Gradient Accumulation: 1 (True online update: opt.step() on every single puzzle!).
- Sample count alignment: 16,000 steps * 1 = 16,000 puzzles seen (EXACTLY matching 500 steps * 32).
- Sequential stream: Puzzles stream in sequentially (puzzle 0 -> puzzle 1 -> ...), exactly like Language!
- Cosine LR schedule over 16,000 steps (3e-4 -> 1e-5).
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import math
import time
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


@torch.no_grad()
def evaluate_batch1_stream(model, test_in: np.ndarray, test_lbl: np.ndarray,
                           mature_state: torch.Tensor, horizons: List[int]) -> Dict[int, Dict[str, float]]:
    """Evaluates the 1,000 test puzzles in a strict single-stream (Batch=1) with Never-Reset carry!"""
    model.eval()
    results = {}
    device = mature_state.device
    num_test = len(test_in)

    for k in horizons:
        state = mature_state.detach().clone()  # [1, 9, 9, d]
        total_cells = 0
        correct_cells = 0
        exact_matches = 0
        losses = []

        for idx in range(num_test):
            inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device=device)
            lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device=device)

            clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))

            # Port Injection into single continuous field
            f_norm = torch.linalg.vector_norm(state.float(), dim=(1, 2, 3), keepdim=True)
            blended = state + clue_field
            curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

            for _ in range(k):
                f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
                f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
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

            flat = curr.view(1, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
            losses.append(loss)
            preds = torch.argmax(logits, dim=-1)
            correct_cells += (preds == lbl).sum().item()
            total_cells += 81
            exact_matches += ((preds == lbl).sum().item() == 81)

            # Never Reset: single stream carries forward!
            state = curr.detach()

        results[k] = {
            "cell_accuracy": correct_cells / total_cells,
            "exact_accuracy": exact_matches / num_test,
            "mean_loss": float(np.mean(losses))
        }

    return results


def main():
    total_steps = 16000  # 16,000 puzzles seen (aligns exactly with 500 * 32)
    d_channels = 256
    lr_base = 3e-4
    lr_min = 1e-5

    out_dir = Path("results/ablation_branch1_batch1_16k")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("   BRANCH 1: STRICT SINGLE STREAM (BATCH=1, GRAD_ACCUM=1, 16,000 STEPS)")
    print("   Alignment: 16,000 steps * 1 = 16,000 puzzles seen (EXACT match to 500 * 32)")
    print("   Core Hypothesis: Eliminates Batch slot jitter & random puzzle collision interference!")
    print("=" * 105)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=d_channels, ponder_steps=16, arm="corrective_flow", alpha_max=0.25).to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    optimizer = AdamATan2(model.parameters(), lr=lr_base, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state: single world [1, 9, 9, d]
    init_sample = torch.as_tensor(train_in[0:1], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.model.clue_proj(model.model.embed_tokens(init_sample).view(1, 9, 9, d_channels))

    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)
    rng = np.random.default_rng(42)
    start_time = time.perf_counter()

    for step in range(1, total_steps + 1):
        model.train()
        progress = step / total_steps
        current_lr = lr_min + 0.5 * (lr_base - lr_min) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        k_step = int(rng.choice(horizons, p=probs))

        # Strictly sequential stream (puzzle 0 -> puzzle 1 -> ...)
        puzzle_idx = (step - 1) % num_train
        inp = torch.as_tensor(train_in[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(train_lbl[puzzle_idx:puzzle_idx+1], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))

            # Port injection into single persistent field
            f_norm = torch.linalg.vector_norm(persistent_state.float(), dim=(1, 2, 3), keepdim=True)
            blended = persistent_state + clue_field
            curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

            for _ in range(k_step):
                f_5d = curr.view(1, 9, 9, model.model.n_v, model.model.d_c)
                f_tr = model.model.transport(f_5d).view(1, 9, 9, model.model.d)
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

            flat = curr.view(1, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))

        # True online update: NO accumulation!
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        with torch.no_grad():
            persistent_state.copy_(curr.detach())

        if step % 1000 == 0 or step == 1 or step == 500:
            elapsed = time.perf_counter() - start_time
            fps = step / elapsed
            print(f"[Batch=1] Step {step:5d}/{total_steps} ({step/total_steps*100:4.1f}%) | K: {k_step:2d} | Loss: {loss.item():6.3f} | LR: {current_lr:.1e} | {fps:5.1f} puz/s", flush=True)

    print(f"\n[Batch=1] Training complete in {(time.perf_counter()-start_time)/60:.2f} min! Running Single-Stream Never-Reset Evaluation...")
    val_results = evaluate_batch1_stream(model, test_in, test_lbl, persistent_state, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    print("\n" + "=" * 80)
    print("   BRANCH 1 (BATCH=1 TRUE SINGLE STREAM) FINAL RESULTS")
    print("=" * 80)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        m = val_results[k]
        print(f"  K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)


if __name__ == "__main__":
    main()
