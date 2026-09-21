"""Train CBIM-Sudoku with Full Unified Dissipative Open-System Architecture (500 steps, d=256, Cosine LR).
Aligns 1:1 with Language MaleCNS-Unified V6:
1. 2-Port State-and-Problem Aware Boundary Scattering:
   theta(F, P) actively detects conflict between existing field F and new clues P,
   dissipating obsolete memory via cos(theta)*F and injecting new clue energy via sin(theta)*P!
2. Unitary Cayley Transport on 9x9 Torus.
3. Problem-Conditioned Lie Algebra Collision in nullspace.
4. Quadratic Passive Radiation Cooling: J_bath = 2 * kappa * E^2 / R^2 * F.
5. Never-Reset persistent state carry across all training batches & evaluation!
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

from scripts.ib_local.cbim_sudoku_unified_dissipative import UnifiedDissipativeCBIMSudokuModel
from external.TinyRecursiveModels.adam_atan2 import AdamATan2


@torch.no_grad()
def evaluate_stream(model, test_in: np.ndarray, test_lbl: np.ndarray,
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

            curr, logits, _ = model.forward_stream_step(state[:b], inp_b, k_step=k)
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

    out_dir = Path("results/ablation_unified_dissipative_500")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("   TRAINING CBIM-SUDOKU UNIFIED DISSIPATIVE ARCHITECTURE (500 STEPS, d=256)")
    print("   Aligned 1:1 with Language MaleCNS-Unified V6: 2-Port State-Aware Scattering + Quadratic Bath")
    print("=" * 105)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/sudoku-extreme-1k-aug-100")
    train_in = np.load(data_path / "train" / "all__inputs.npy", mmap_mode="r")
    train_lbl = np.load(data_path / "train" / "all__labels.npy", mmap_mode="r")
    test_in = np.load(data_path / "test" / "all__inputs.npy")
    test_lbl = np.load(data_path / "test" / "all__labels.npy")
    num_train = len(train_in)

    model = UnifiedDissipativeCBIMSudokuModel(vocab_size=11, d_channels=d_channels, ponder_steps=16).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,} ({num_params/1e6:.3f}M)\n", flush=True)

    optimizer = AdamATan2(model.parameters(), lr=lr_base, weight_decay=1.0)
    scaler = torch.amp.GradScaler("cuda")

    # Initial persistent state
    init_sample = torch.as_tensor(train_in[:micro_bs], dtype=torch.long, device=device)
    with torch.no_grad():
        persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(micro_bs, 9, 9, d_channels))

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

            with torch.amp.autocast("cuda", dtype=torch.float16):
                curr, logits, diag = model.forward_stream_step(persistent_state, inp, k_step=k_step)
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
            theta_mean = diag.get("theta_mean", 0.0)
            cool_pwr = diag.get("cooling_power", 0.0)
            print(f"[Unified Dissip] Step {step:3d}/{steps} | K: {k_step:2d} | Loss: {accum_loss:6.3f} | LR: {current_lr:.1e} | theta: {theta_mean:.4f} | cool: {cool_pwr:.4e}", flush=True)

    print(f"\n[Unified Dissip] Training complete in {(time.perf_counter()-start_time)/60:.2f} min! Running Never-Reset Evaluation...")
    val_results = evaluate_stream(model, test_in, test_lbl, persistent_state, horizons=[1, 2, 4, 8, 16, 32, 64, 128, 256], batch_size=32)
    print("\n" + "=" * 80)
    print("   UNIFIED DISSIPATIVE ARCHITECTURE FINAL RESULTS (K in [1..256])")
    print("=" * 80)
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        m = val_results[k]
        print(f"  K = {k:3d} | Cell Acc: {m['cell_accuracy']*100:5.2f}% | Loss: {m['mean_loss']:6.3f}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)


if __name__ == "__main__":
    main()
