"""Branch 2: Never-Reset + Port Injection + Pure TC (No Corrective Flow).
500-step controlled ablation on Sudoku-Extreme (d=256, Cosine LR).
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
def evaluate_branch2(model, test_in: np.ndarray, test_lbl: np.ndarray,
                     mature_state: torch.Tensor, horizons: List[int], batch_size: int = 32) -> Dict[int, Dict[str, float]]:
    model.eval()
    results = {}
    device = mature_state.device

    for k in horizons:
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

            # Port Injection
            f_norm = torch.linalg.vector_norm(sb.float(), dim=(1, 2, 3), keepdim=True)
            blended = sb + clue_field
            curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

            # Pure TC microsteps (Problem-Conditioned Collision, NO corrective net!)
            for _ in range(k):
                f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
                f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
                curr = model.model.collision(f_tr, cond=clue_field)

            flat = curr.view(b, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl_b.view(-1)).item()
            losses.append(loss)
            preds = torch.argmax(logits, dim=-1)
            correct_cells += (preds == lbl_b).sum().item()
            total_cells += b * 81
            exact_matches += ((preds == lbl_b).sum(dim=-1) == 81).sum().item()

            state[:b] = curr.detach()

        results[k] = {
            "cell_accuracy": correct_cells / total_cells,
            "exact_accuracy": exact_matches / len(test_in),
            "mean_loss": float(np.mean(losses))
        }

    return results


def main():
    steps = 500
    d_channels = 256
    micro_bs = 16
    accum = 2
    lr_base = 3e-4
    lr_min = 1e-5

    out_dir = Path("results/ablation_branch2_never_reset_pure_tc")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("   BRANCH 2: NEVER RESET + PORT INJECTION + PURE TC (NO CORRECTIVE FLOW, d=256, 500 STEPS)")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    # Use arm='tc_baseline' (Pure Transport + Collision, No Corrective Net)
    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=d_channels, ponder_steps=16, arm="tc_baseline").to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    optimizer = AdamATan2(model.parameters(), lr=lr_base, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state
    init_sample = torch.as_tensor(train_in[:micro_bs], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.model.clue_proj(model.model.embed_tokens(init_sample).view(micro_bs, 9, 9, d_channels))

    horizons = [2, 4, 8, 16, 32, 64]
    probs = [1.0 / len(horizons)] * len(horizons)
    rng = np.random.default_rng(42)
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        progress = step / steps
        current_lr = lr_min + 0.5 * (lr_base - lr_min) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        k_step = int(rng.choice(horizons, p=probs))
        accum_loss = 0.0

        for _ in range(accum):
            idx = rng.integers(0, num_train, size=micro_bs)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
            b = len(inp)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(b, 9, 9, model.model.d))

                # Port injection
                f_norm = torch.linalg.vector_norm(persistent_state.float(), dim=(1, 2, 3), keepdim=True)
                blended = persistent_state + clue_field
                curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

                # Pure TC: Problem-Conditioned Collision, NO corrective net
                for _ in range(k_step):
                    f_5d = curr.view(b, 9, 9, model.model.n_v, model.model.d_c)
                    f_tr = model.model.transport(f_5d).view(b, 9, 9, model.model.d)
                    curr = model.model.collision(f_tr, cond=clue_field)

                flat = curr.view(b, 81, model.model.d)
                logits = model.model.readout_mlp(flat)
                loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1))
                loss = loss / accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * accum

            with torch.no_grad():
                persistent_state.copy_(curr.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 50 == 0 or step == 1:
            print(f"[Branch 2] Step {step:3d}/{steps} | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e}", flush=True)

    print(f"\n[Branch 2] Training complete in {(time.perf_counter()-start_time)/60:.2f} min! Running Never-Reset Evaluation...")
    val_results = evaluate_branch2(model, test_in, test_lbl, persistent_state, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size=32)
    print("\n" + "=" * 80)
    print("   BRANCH 2 (NEVER RESET + INJECTION + PURE TC) FINAL RESULTS")
    print("=" * 80)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        m = val_results[k]
        print(f"  K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)


if __name__ == "__main__":
    main()
