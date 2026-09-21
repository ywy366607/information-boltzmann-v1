"""Diagnostic Experiment:
1. Compare Mode 1 (Reset), Mode 2 (Naive Never-Reset), and Mode 4 (State-Aware Dissipation) across small K (1, 2, 4, 8) and deep K (16, 32, 64, 128, 256, 512).
2. Test the user's two physical hypotheses:
   - Hypothesis A: At small K (K=1, 2, 4), does Mode 2 suffer from lack of microstep dissipation, allowing Mode 4 (Active Dissipation) to win?
   - Hypothesis B: At deep K (K=64..512), does Never-Reset still converge to a stable unique attractor plateau, or does it drift?
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku_corrective import CorrectiveCBIMSudokuModel
from models.losses import ACTLossHead


@torch.no_grad()
def evaluate_mode1(model, test_in, test_lbl, k_val, batch_size=32):
    """Mode 1: Standard Reset Baseline."""
    model.eval()
    num_samples = len(test_in)
    total_correct = 0
    total_cells = 0
    losses = []

    for s in range(0, num_samples, batch_size):
        e = min(s + batch_size, num_samples)
        b = e - s
        inp = torch.as_tensor(test_in[s:e], dtype=torch.long, device="cuda")
        lbl = torch.as_tensor(test_lbl[s:e], dtype=torch.long, device="cuda")

        clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(b, 9, 9, model.model.d))
        curr = clue_field.clone()

        for _ in range(k_val):
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
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        flat = curr.view(b, 81, model.model.d)
        logits = model.model.readout_mlp(flat)
        loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
        losses.append(loss)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == lbl).sum().item()
        total_cells += b * 81

    return total_correct / total_cells, float(np.mean(losses))


@torch.no_grad()
def evaluate_mode2(model, test_in, test_lbl, k_val):
    """Mode 2: Naive Never-Reset (Single-stream carry, simple addition)."""
    model.eval()
    num_samples = len(test_in)
    total_correct = 0
    total_cells = 0
    losses = []

    inp_0 = torch.as_tensor(test_in[0:1], dtype=torch.long, device="cuda")
    state = model.model.clue_proj(model.model.embed_tokens(inp_0).view(1, 9, 9, model.model.d))

    for idx in range(num_samples):
        inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device="cuda")
        lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device="cuda")

        clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))

        f_norm = torch.linalg.vector_norm(state.float(), dim=(1,2,3), keepdim=True)
        blended = state + clue_field
        curr = blended * (f_norm / (torch.linalg.vector_norm(blended.float(), dim=(1,2,3), keepdim=True) + 1e-8))

        for _ in range(k_val):
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
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        flat = curr.view(1, 81, model.model.d)
        logits = model.model.readout_mlp(flat)
        loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
        losses.append(loss)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == lbl).sum().item()
        total_cells += 81

        state = curr.detach()

    return total_correct / total_cells, float(np.mean(losses))


@torch.no_grad()
def evaluate_mode4(model, test_in, test_lbl, k_val):
    """Mode 4: Never-Reset with State-and-Problem-Aware Dissipation (theta(F,P) conflict erasure)."""
    model.eval()
    num_samples = len(test_in)
    total_correct = 0
    total_cells = 0
    losses = []

    inp_0 = torch.as_tensor(test_in[0:1], dtype=torch.long, device="cuda")
    state = model.model.clue_proj(model.model.embed_tokens(inp_0).view(1, 9, 9, model.model.d))

    for idx in range(num_samples):
        inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device="cuda")
        lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device="cuda")

        clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))
        clue_mask = (inp.view(1, 9, 9, 1) > 1).float()

        # Measure alignment at clue positions
        dot_sc = (state * clue_field).sum(dim=-1, keepdim=True) / (
            torch.linalg.vector_norm(state, dim=-1, keepdim=True) *
            torch.linalg.vector_norm(clue_field, dim=-1, keepdim=True) + 1e-8
        )
        conflict = 0.5 * (1.0 - dot_sc) * clue_mask  # [1, 9, 9, 1] in [0, 1]

        # 2-Port Unitary Scattering Dissipation
        theta_diss = 0.5 * math.pi * conflict
        cos_diss = torch.cos(theta_diss)
        sin_diss = torch.sin(theta_diss)

        f_purified = cos_diss * state + sin_diss * clue_field
        f_norm = torch.linalg.vector_norm(state.float(), dim=(1,2,3), keepdim=True)
        curr = f_purified * (f_norm / (torch.linalg.vector_norm(f_purified.float(), dim=(1,2,3), keepdim=True) + 1e-8))

        for _ in range(k_val):
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
            f_norm = torch.linalg.vector_norm(f_star.float(), dim=(1,2,3), keepdim=True)
            curr = torch.cos(alpha_k)*f_star + f_norm*torch.sin(alpha_k)*u_hat.to(curr.dtype)

        flat = curr.view(1, 81, model.model.d)
        logits = model.model.readout_mlp(flat)
        loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
        losses.append(loss)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == lbl).sum().item()
        total_cells += 81

        state = curr.detach()

    return total_correct / total_cells, float(np.mean(losses))


def main():
    print("=" * 105)
    print("   TESTING HYPOTHESES: SMALL-K DISSIPATION SENSITIVITY & DEEP-K ATTRACTOR STABILITY")
    print("   Horizons: K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]")
    print("   Model: Best_Arm_C_Champion_d256.pt (50.50% Master Checkpoint)")
    print("=" * 105)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=64, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")[:300]  # 300 test puzzles for fast & dense scanning
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")[:300]
    print(f"Testing on {len(test_in)} sequential puzzles across 10 ponder horizons...\n", flush=True)

    horizons = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print(f"{'Horizon K':^10} | {'Mode 1 (Reset)':^26} | {'Mode 2 (Never-Reset)':^26} | {'Mode 4 (State Dissip)':^26}")
    print(f"{'':^10} | {'Acc (%)':^12} {'Loss':^12} | {'Acc (%)':^12} {'Loss':^12} | {'Acc (%)':^12} {'Loss':^12}")
    print("-" * 100)

    for k in horizons:
        a1, l1 = evaluate_mode1(model, test_in, test_lbl, k_val=k, batch_size=32)
        a2, l2 = evaluate_mode2(model, test_in, test_lbl, k_val=k)
        a4, l4 = evaluate_mode4(model, test_in, test_lbl, k_val=k)
        print(f" K = {k:4d}    |  {a1*100:6.2f}%     {l1:6.4f}   |  {a2*100:6.2f}%     {l2:6.4f}   |  {a4*100:6.2f}%     {l4:6.4f}", flush=True)


if __name__ == "__main__":
    main()
