"""Direct Comparative Measurement: Branch A (Closed) vs Branch B (Conditioned).
Measures the exact same metrics across k = 0 .. 128:
Total E | DC % | Low % | Mid % | High % | S_spec | Rank | FTLE | Loss
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
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.ablate_problem_conditioning import ConditionedCBIMSudokuModel
from models.losses import ACTLossHead
from adam_atan2 import AdamATan2


@torch.no_grad()
def run_spectral_analysis(model, test_in: np.ndarray, test_lbl: np.ndarray, max_k: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.as_tensor(test_in, dtype=torch.long, device=device)
    lbl = torch.as_tensor(test_lbl, dtype=torch.long, device=device)
    B = len(inp)

    inner = model.model
    inner.eval()

    # Initial clue embedding P_t
    emb = inner.embed_tokens(inp).view(B, 9, 9, inner.d)
    clue_field = inner.clue_proj(emb)

    # Precompute 2D spatial wavenumber grid on 9x9 torus
    kx = torch.fft.fftfreq(9, d=1.0 / 9) * 2.0 * math.pi
    ky = torch.fft.fftfreq(9, d=1.0 / 9) * 2.0 * math.pi
    grid_kx, grid_ky = torch.meshgrid(kx, ky, indexing="ij")
    q_mag = torch.sqrt(grid_kx**2 + grid_ky**2).to(device)

    mask_dc = (q_mag == 0.0)
    mask_low = (q_mag > 0.0) & (q_mag <= 1.5 * math.pi)
    mask_mid = (q_mag > 1.5 * math.pi) & (q_mag <= 3.0 * math.pi)
    mask_high = (q_mag > 3.0 * math.pi)

    curr = clue_field.clone()
    eps = 1e-5
    curr_pert = curr + eps * torch.randn_like(curr)

    check_steps = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    check_set = set(check_steps)
    table = []

    for k in range(0, max_k + 1):
        if k in check_set:
            # 2D FFT over spatial dimensions
            f_fft = torch.fft.fft2(curr.to(torch.complex64), dim=(1, 2))
            p_q = f_fft.abs().square().sum(dim=-1).mean(dim=0)
            total_e = float(p_q.sum().item())

            e_dc = float(p_q[mask_dc].sum().item() / max(total_e, 1e-8))
            e_low = float(p_q[mask_low].sum().item() / max(total_e, 1e-8))
            e_mid = float(p_q[mask_mid].sum().item() / max(total_e, 1e-8))
            e_high = float(p_q[mask_high].sum().item() / max(total_e, 1e-8))

            # Spectral Entropy S_spec = -sum p log p
            norm_pq = (p_q / max(total_e, 1e-8)).flatten()
            norm_pq = norm_pq[norm_pq > 1e-12]
            s_spec = float(-(norm_pq * norm_pq.log()).sum().item())

            # Effective State Rank
            node_energy = curr.square().sum(dim=-1)
            node_prob = node_energy / node_energy.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            eff_rank = float(1.0 / (node_prob**2).sum(dim=(1, 2)).mean().item())

            # FTLE
            diff = (curr_pert - curr).norm().item()
            ftle = math.log(max(diff / eps, 1e-8)) / max(k, 1) if k > 0 else 0.0

            # Instantaneous loss & accuracy
            flat_curr = curr.view(B, 81, inner.d)
            logits_k = inner.readout_mlp(flat_curr)
            loss_k = float(F.cross_entropy(logits_k.view(-1, 11), lbl.view(-1), ignore_index=0).item())
            pred_k = torch.argmax(logits_k, dim=-1)
            cell_acc = float((pred_k == lbl).float().mean().item())

            table.append({
                "k": k, "total_e": total_e,
                "e_dc": e_dc, "e_low": e_low, "e_mid": e_mid, "e_high": e_high,
                "s_spec": s_spec, "rank": eff_rank, "ftle": ftle,
                "loss": loss_k, "cell_acc": cell_acc
            })

        if k < max_k:
            # Advance 1 step
            B_curr, H, W, D = curr.shape
            f_5d = curr.view(B_curr, H, W, inner.n_v, inner.d_c)
            f_tr = inner.transport(f_5d).view(B_curr, H, W, D)
            cond = clue_field if inner.conditioned else None
            curr = inner.collision(f_tr, cond=cond)

            # Advance perturbed state for FTLE
            f_5d_p = curr_pert.view(B_curr, H, W, inner.n_v, inner.d_c)
            f_tr_p = inner.transport(f_5d_p).view(B_curr, H, W, D)
            curr_pert = inner.collision(f_tr_p, cond=cond)
            diff_vec = curr_pert - curr
            curr_pert = curr + diff_vec / (diff_vec.norm().clamp_min(1e-12)) * eps

    return table


def train_and_measure():
    print("=" * 95)
    print("   QUANTITATIVE SPECTRAL & DYNAMICAL COMPARISON: BRANCH A vs BRANCH B")
    print("=" * 95)

    data_dir = "data/sudoku-extreme-1k-aug-100"
    train_in = np.load(os.path.join(data_dir, "train", "all__inputs.npy"), mmap_mode="r")
    train_lbl = np.load(os.path.join(data_dir, "train", "all__labels.npy"), mmap_mode="r")
    test_in = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")[:64]
    test_lbl = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")[:64]
    num_train = len(train_in)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for branch_name, is_cond in [("Branch_A_Closed", False), ("Branch_B_Conditioned", True)]:
        print(f"\n--- Training {branch_name} for 300 steps ---")
        inner = ConditionedCBIMSudokuModel(vocab_size=11, d_channels=128, n_velocities=8,
                                          ponder_steps=16, conditioned=is_cond).to(device)
        model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
        optim = AdamATan2(model.parameters(), lr=3e-4, weight_decay=1.0)
        scaler = torch.amp.GradScaler("cuda")

        carry = None
        rng = np.random.default_rng(42)
        for step in range(1, 301):
            model.train()
            idx = rng.integers(0, num_train, size=32)
            inp = torch.as_tensor(train_in[idx], dtype=torch.long, device=device)
            lbl = torch.as_tensor(train_lbl[idx], dtype=torch.long, device=device)
            batch = {"inputs": inp, "labels": lbl, "puzzle_identifiers": torch.zeros(32, dtype=torch.long, device=device)}
            if carry is None:
                carry = model.model.initial_carry(batch)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                carry, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        print(f"Running spectral measurement on {branch_name} across k = 0..128...")
        results[branch_name] = run_spectral_analysis(model, test_in, test_lbl, max_k=128)

    # Print Table 1: Branch A (Closed)
    print("\n" + "=" * 95)
    print("   TABLE 1: BRANCH A (ORIGINAL CLOSED PONDER - FREE EVOLUTION)")
    print("=" * 95)
    print(f"{'k':>4s} | {'Total E':>10s} | {'DC %':>7s} | {'Low %':>7s} | {'Mid %':>7s} | {'High %':>7s} | {'S_spec':>7s} | {'Rank':>6s} | {'FTLE':>8s} | {'Loss':>7s} | {'CellAcc':>7s}")
    print("-" * 105)
    for r in results["Branch_A_Closed"]:
        print(f"{r['k']:4d} | {r['total_e']:10.2f} | {r['e_dc']*100:6.2f}% | {r['e_low']*100:6.2f}% | {r['e_mid']*100:6.2f}% | {r['e_high']*100:6.2f}% | {r['s_spec']:7.3f} | {r['rank']:6.1f} | {r['ftle']:8.4f} | {r['loss']:7.3f} | {r['cell_acc']*100:6.2f}%")

    # Print Table 2: Branch B (Conditioned Ponder)
    print("\n" + "=" * 95)
    print("   TABLE 2: BRANCH B (CONDITIONED PONDER - ZERO ENERGY INJECTION)")
    print("=" * 95)
    print(f"{'k':>4s} | {'Total E':>10s} | {'DC %':>7s} | {'Low %':>7s} | {'Mid %':>7s} | {'High %':>7s} | {'S_spec':>7s} | {'Rank':>6s} | {'FTLE':>8s} | {'Loss':>7s} | {'CellAcc':>7s}")
    print("-" * 105)
    for r in results["Branch_B_Conditioned"]:
        print(f"{r['k']:4d} | {r['total_e']:10.2f} | {r['e_dc']*100:6.2f}% | {r['e_low']*100:6.2f}% | {r['e_mid']*100:6.2f}% | {r['e_high']*100:6.2f}% | {r['s_spec']:7.3f} | {r['rank']:6.1f} | {r['ftle']:8.4f} | {r['loss']:7.3f} | {r['cell_acc']*100:6.2f}%")

    # Print Side-by-Side Comparison
    print("\n" + "=" * 95)
    print("   DIRECT DELTA COMPARISON: BRANCH A vs BRANCH B")
    print("=" * 95)
    print(f"{'k':>4s} | {'Loss A':>8s} | {'Loss B':>8s} | {'Delta Loss':>11s} | {'Acc A':>7s} | {'Acc B':>7s} | {'Rank A':>7s} | {'Rank B':>7s} | {'S_spec A':>8s} | {'S_spec B':>8s}")
    print("-" * 105)
    for ra, rb in zip(results["Branch_A_Closed"], results["Branch_B_Conditioned"]):
        k = ra["k"]
        delta_loss = rb["loss"] - ra["loss"]
        print(f"{k:4d} | {ra['loss']:8.3f} | {rb['loss']:8.3f} | {delta_loss:+10.3f} | {ra['cell_acc']*100:6.2f}% | {rb['cell_acc']*100:6.2f}% | {ra['rank']:7.1f} | {rb['rank']:7.1f} | {ra['s_spec']:8.3f} | {rb['s_spec']:8.3f}")

    out_file = Path("results/spectral_comparison_branch_a_vs_b.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparative JSON results saved to {out_file}")


if __name__ == "__main__":
    train_and_measure()
