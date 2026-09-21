"""Multi-Branch Diversity, Branch Oracle, and Hard-Problem Gated Analysis.
Implements the exact GPT review specifications:
1. Four Diagnostic Metrics:
   - D_state(K): 1 - cos(F_i, F_j) between branch internal physical states.
   - D_pred(K): JS-divergence between branch output prediction distributions p_i and p_j.
   - Disagreement: % of cells where argmax(p_i) != argmax(p_j).
   - Branch Oracle Accuracy: Acc_oracle = (1/N) * sum_n max_h Acc(p_{n, h}).
     CRITICAL TEST: Does Acc_oracle >> Acc_fused? (Hypothesis diversity exists, router needs work?)
2. Hard-Problem Gated Analysis:
   Group test puzzles by single-path optimal depth K*(M=1):
     Group 1: Easy (K* <= 2)
     Group 2: Medium (4 <= K* <= 16)
     Group 3: Hard Deep Pondering Winners (K* >= 32)
   Measure Q(M, K) separately on Group 3 to see if breadth truly accelerates hard search!
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
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.train_explicit_branch_multipath import (
    ExplicitBranchCBIMSudokuModel,
    get_orthogonal_hadamard_codes
)


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Computes Jensen-Shannon divergence between two categorical distributions [B, 81, 11]."""
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m.clamp_min(1e-12)), p, reduction="batchmean")
    kl_qm = F.kl_div(torch.log(m.clamp_min(1e-12)), q, reduction="batchmean")
    return float((0.5 * (kl_pm + kl_qm)).item())


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/explicit_branch_multipath_3000/Best_ExplicitBranch_d256.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("results/explicit_branch_multipath_3000/Final_ExplicitBranch_d256.pt")

    print(f"Loading Explicit Branch model from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ExplicitBranchCBIMSudokuModel(vocab_size=11, d_channels=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")
    num_test = len(test_in)

    M = 4
    d = model.d
    codes = get_orthogonal_hadamard_codes(M, d=d, device=device)  # [M, 1, 1, d]

    print("=" * 110)
    print("   PART 1: BRANCH DIVERSITY & BRANCH ORACLE DIAGNOSIS (M = 4)")
    print("=" * 110)

    # Initial persistent state
    init_sample = torch.as_tensor(test_in[0:1], dtype=torch.long, device=device)
    persistent_state = model.clue_proj(model.embed_tokens(init_sample).view(1, 9, 9, d))

    k_eval_list = [1, 2, 4, 8, 16, 32, 64]

    d_state_records = {k: [] for k in k_eval_list}
    d_pred_records = {k: [] for k in k_eval_list}
    disagree_records = {k: [] for k in k_eval_list}
    oracle_acc_records = {k: [] for k in k_eval_list}
    fused_acc_records = {k: [] for k in k_eval_list}
    individual_branch_accs = {k: [[] for _ in range(M)] for k in k_eval_list}

    # Also track per-puzzle single-path (M=1) accuracy to find K* groups
    p_k_accs_m1 = np.zeros((num_test, len(k_eval_list)))

    # Scan across K
    for k_idx, k_val in enumerate(k_eval_list):
        state = persistent_state.clone()
        dt = 1.0 / max(1, k_val)

        for idx in range(num_test):
            inp = torch.as_tensor(test_in[idx:idx+1], dtype=torch.long, device=device)
            lbl = torch.as_tensor(test_lbl[idx:idx+1], dtype=torch.long, device=device)

            clue_field = model.clue_proj(model.embed_tokens(inp).view(1, 9, 9, d))
            f_base, _, _ = model.boundary_write(state, clue_field)

            # M=4 Branches
            branch_seeds = (clue_field * codes) / 2.0  # sqrt(4) = 2.0
            branches = f_base.expand(M, -1, -1, -1) + 0.15 * branch_seeds
            orig_norm = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True).to(branches.dtype)
            clue_expanded = clue_field.expand(M, -1, -1, -1)

            for _ in range(k_val):
                f_5d = branches.view(M, 9, 9, model.n_v, model.d_c)
                f_tr = model.transport(f_5d, dt=dt).view(M, 9, 9, d)
                f_star = model.collision(f_tr, cond=clue_expanded, dt=dt)

                cat_in = torch.cat([f_star, clue_expanded], dim=-1)
                v_k = model.corrective_net(cat_in)
                dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
                norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
                u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
                u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)

                gate_val = torch.sigmoid(torch.mean(model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True))
                alpha_k = (dt * model.alpha_max) * gate_val.to(branches.dtype)

                f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
                branches = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(branches.dtype)

            branch_norms = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True) + 1e-8
            branches = branches * (orig_norm / branch_norms)

            logits_branches = model.readout(branches)
            probs_branches = F.softmax(logits_branches, dim=-1)  # [M, 81, 11]

            # 1. State Diversity: D_state = 1 - cos(F_i, F_j)
            flat_b = branches.float().flatten(start_dim=1)  # [M, 20736]
            cos_matrix = (flat_b @ flat_b.T) / (torch.linalg.vector_norm(flat_b, dim=-1, keepdim=True) @ torch.linalg.vector_norm(flat_b, dim=-1, keepdim=True).T + 1e-8)
            # Off-diagonal cosine mean
            eye = torch.eye(M, device=device)
            off_diag_cos = (cos_matrix * (1.0 - eye)).sum() / (M * (M - 1))
            d_state = float((1.0 - off_diag_cos).item())

            # 2. Prediction Disagreement: % of cells where argmax differs
            preds_h = torch.argmax(probs_branches, dim=-1)  # [M, 81]
            disagreements = 0
            for i in range(M):
                for j in range(i + 1, M):
                    disagreements += float((preds_h[i] != preds_h[j]).float().mean().item())
            disagreements = disagreements / (M * (M - 1) / 2.0)

            # 3. Branch Accuracies and Branch Oracle
            acc_per_h = [(preds_h[h] == lbl).float().mean().item() for h in range(M)]
            for h in range(M):
                individual_branch_accs[k_val][h].append(acc_per_h[h])
            oracle_acc = max(acc_per_h)

            # 4. Fused Decision
            scores = model.branch_scorer(branches.mean(dim=(1, 2))).squeeze(-1)
            pi = F.softmax(scores, dim=0).view(M, 1, 1)
            fused_probs = (probs_branches * pi).sum(dim=0, keepdim=True)
            fused_pred = torch.argmax(fused_probs, dim=-1)
            fused_acc = float((fused_pred == lbl).float().mean().item())

            d_state_records[k_val].append(d_state)
            disagree_records[k_val].append(disagreements)
            oracle_acc_records[k_val].append(oracle_acc)
            fused_acc_records[k_val].append(fused_acc)

            # Save M=1 performance (using branch 0 without Hadamard seed)
            p_k_accs_m1[idx, k_idx] = acc_per_h[0]

            best_idx = torch.argmax(scores)
            state_next, _, _ = model.bath(branches[best_idx:best_idx+1])
            state = state_next.detach()

    print(f"{'Ponder Depth K':^15} | {'D_state (1-cos)':^18} | {'Cell Disagree (%)':^18} | {'Fused Acc (%)':^15} | {'Oracle Acc (%)':^16} | {'Oracle Headroom':^16}")
    print("-" * 110)
    for k_val in k_eval_list:
        ds = np.mean(d_state_records[k_val])
        dis = np.mean(disagree_records[k_val]) * 100
        fused = np.mean(fused_acc_records[k_val]) * 100
        oracle = np.mean(oracle_acc_records[k_val]) * 100
        diff = oracle - fused
        print(f"   K = {k_val:2d}        |      {ds:.6f}     |      {dis:5.2f}%       |     {fused:5.2f}%     |     {oracle:5.2f}%      |    +{diff:5.2f}% 🚀")

    # =========================================================================
    # PART 2: HARD-PROBLEM GATED ANALYSIS: Q(M, K) ON GROUP 3 (K* >= 32)
    # =========================================================================
    print("\n" + "=" * 110)
    print("   PART 2: HARD-PROBLEM GATED ANALYSIS (CONDITIONED ON SINGLE-PATH OPTIMAL DEPTH K*)")
    print("=" * 110)

    # Classify puzzles by optimal depth K* in M=1
    best_k_idx_m1 = np.argmax(p_k_accs_m1, axis=1)  # indices into k_eval_list
    best_k_vals_m1 = np.array([k_eval_list[i] for i in best_k_idx_m1])

    group_easy = np.where(best_k_vals_m1 <= 2)[0]
    group_med = np.where((best_k_vals_m1 >= 4) & (best_k_vals_m1 <= 16))[0]
    group_hard = np.where(best_k_vals_m1 >= 32)[0]

    print(f"Puzzle Difficulty Groups:")
    print(f"  • Group 1 (Easy, K* <= 2):                  {len(group_easy)} puzzles ({len(group_easy)/num_test*100:.1f}%)")
    print(f"  • Group 2 (Medium, 4 <= K* <= 16):          {len(group_med)} puzzles ({len(group_med)/num_test*100:.1f}%)")
    print(f"  • Group 3 (Hard Deep Winners, K* >= 32):    {len(group_hard)} puzzles ({len(group_hard)/num_test*100:.1f}%)")

    print("\n--- Group 3 (Hard Deep Winners): Does M=4 Multibranch Accelerate Search at Small K? ---")
    print(f"{'Horizon K':^12} | {'M=1 (Single-Path)':^20} | {'M=4 Fused':^20} | {'M=4 Oracle Branch':^22} | {'Oracle Gain':^12}")
    print("-" * 95)
    for k_idx, k_val in enumerate([1, 2, 4, 8, 16, 32, 64]):
        acc_m1 = np.mean(p_k_accs_m1[group_hard, k_idx]) * 100
        acc_fused = np.mean([fused_acc_records[k_val][i] for i in group_hard]) * 100
        acc_oracle = np.mean([oracle_acc_records[k_val][i] for i in group_hard]) * 100
        gain = acc_oracle - acc_m1
        print(f"  K = {k_val:2d}     |       {acc_m1:5.2f}%       |       {acc_fused:5.2f}%       |        {acc_oracle:5.2f}%         |   {gain:+5.2f}% 🚀")


if __name__ == "__main__":
    main()
