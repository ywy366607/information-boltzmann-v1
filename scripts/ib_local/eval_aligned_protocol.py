"""Evaluate Champion Model under Language-Aligned Evaluation Protocols:
1. Mode 1: Standard Reset (Original Champion baseline: 50.50% at K=512)
2. Mode 2: Naive Never-Reset (Direct sequential state carry across 1,000 test puzzles)
3. Mode 3: Language-Aligned Never-Reset with Burn-in (First 100 puzzles as NESS burn-in, evaluate on the remaining 900)
4. Mode 4: Never-Reset with State-and-Problem-Aware Impedance Gating:
   theta(F, P) detects conflict between existing field F and new clues P, dissipates old state via cos(theta)*F!
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
def evaluate_mode1_standard_reset(model, test_in, test_lbl, k_val=64, batch_size=32):
    """Mode 1: Standard Reset per batch (Original Champion)."""
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
def evaluate_mode2_naive_never_reset(model, test_in, test_lbl, k_val=64):
    """Mode 2: Naive Never-Reset (Single stream, no burn-in)."""
    model.eval()
    num_samples = len(test_in)
    total_correct = 0
    total_cells = 0
    losses = []

    # Init state
    inp_0 = torch.as_tensor(test_in[0:1], dtype=torch.long, device="cuda")
    state = model.model.clue_proj(model.model.embed_tokens(inp_0).view(1, 9, 9, model.model.d))

    for idx in range(num_samples):
        inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device="cuda")
        lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device="cuda")

        clue_field = model.model.clue_proj(model.model.embed_tokens(inp).view(1, 9, 9, model.model.d))

        # Port addition
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
def evaluate_mode3_burnin_never_reset(model, test_in, test_lbl, k_val=64, burnin_count=100):
    """Mode 3: Language-aligned Never-Reset with Burn-in (First 100 puzzles as NESS warm-up)."""
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

        state = curr.detach()

        # Language Rule: Exclude the Burn-in window!
        if idx >= burnin_count:
            flat = curr.view(1, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
            losses.append(loss)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == lbl).sum().item()
            total_cells += 81

    return total_correct / total_cells, float(np.mean(losses))


@torch.no_grad()
def evaluate_mode4_state_aware_dissipation(model, test_in, test_lbl, k_val=64, burnin_count=100):
    """Mode 4: Never-Reset with State-and-Problem-Aware Impedance Gating:
    theta(F, P) detects conflict between existing field F and new clues P, dissipates old state via cos(theta)*F!
    """
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

        # Conflict Detection: Where clue exists, measure cosine alignment between state and clue
        # Clue mask: cells where input > 1
        clue_mask = (inp.view(1, 9, 9, 1) > 1).float()

        # State-Aware Impedance Angle:
        # If state at a clue position aligns with clue, keep it (theta ~ 0 -> cos theta ~ 1)
        # If state at a clue position conflicts, theta -> pi/2 (cos theta -> 0, dissipative erasure!)
        dot_sc = (state * clue_field).sum(dim=-1, keepdim=True) / (
            torch.linalg.vector_norm(state, dim=-1, keepdim=True) *
            torch.linalg.vector_norm(clue_field, dim=-1, keepdim=True) + 1e-8
        )
        # Cosine similarity in [-1, 1]. Conflict = (1 - dot_sc) / 2 in [0, 1]
        conflict = 0.5 * (1.0 - dot_sc) * clue_mask  # [1, 9, 9, 1]

        # 2-Port Unitary Scattering Dissipation:
        # theta = (pi/2) * conflict
        theta_diss = 0.5 * math.pi * conflict
        cos_diss = torch.cos(theta_diss)
        sin_diss = torch.sin(theta_diss)

        # Dissipate obsolete state & inject new packet
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

        state = curr.detach()

        if idx >= burnin_count:
            flat = curr.view(1, 81, model.model.d)
            logits = model.model.readout_mlp(flat)
            loss = F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item()
            losses.append(loss)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == lbl).sum().item()
            total_cells += 81

    return total_correct / total_cells, float(np.mean(losses))


def main():
    print("=" * 95)
    print("   EVALUATION PROTOCOL DIAGNOSIS ON CHAMPION CHECKPOINT (K=64)")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    inner = CorrectiveCBIMSudokuModel(vocab_size=11, d_channels=256, ponder_steps=64, arm="corrective_flow").to(device)
    head = ACTLossHead(inner, loss_type="stablemax_cross_entropy").to(device)
    head.load_state_dict(ckpt["model"])
    model = head
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    print("\nRunning Mode 1: Standard Reset Baseline (Original Champion)...", flush=True)
    acc1, loss1 = evaluate_mode1_standard_reset(model, test_in, test_lbl, k_val=64)
    print(f"  Mode 1 (Standard Reset):       Acc: {acc1*100:5.2f}% | Loss: {loss1:.4f}")

    print("\nRunning Mode 2: Naive Never-Reset (Single-stream carry, no burn-in)...", flush=True)
    acc2, loss2 = evaluate_mode2_naive_never_reset(model, test_in, test_lbl, k_val=64)
    print(f"  Mode 2 (Naive Never-Reset):     Acc: {acc2*100:5.2f}% | Loss: {loss2:.4f}")

    print("\nRunning Mode 3: Language-Aligned Burn-in (First 100 as warm-up)...", flush=True)
    acc3, loss3 = evaluate_mode3_burnin_never_reset(model, test_in, test_lbl, k_val=64, burnin_count=100)
    print(f"  Mode 3 (Burn-in Warm-up):       Acc: {acc3*100:5.2f}% | Loss: {loss3:.4f}")

    print("\nRunning Mode 4: State-Aware Dissipation (theta(F,P) Conflict Erasure)...", flush=True)
    acc4, loss4 = evaluate_mode4_state_aware_dissipation(model, test_in, test_lbl, k_val=64, burnin_count=100)
    print(f"  Mode 4 (State-Aware Dissip):    Acc: {acc4*100:5.2f}% | Loss: {loss4:.4f}")


if __name__ == "__main__":
    main()
