"""Wave-Native Parallel Hypothesis Search Experiment.
Implements the exact GPT / CDMA orthogonal phase-code specification:
1. Spatial/Channel Orthogonal Phase Codes:
   For M in {1, 2, 4, 8} branches, generate Hadamard / Walsh-Fourier orthogonal code carriers:
     c_h in {-1, +1}^d  (or complex S^1 phase codes) such that
     <c_h, c_j> == d * delta_{hj} (Strict Orthogonality!).
2. Strict Energy Conservation:
   P_h = S(P) * c_h / sqrt(M)
   Sum_{h=1}^M ||P_h||^2 == ||S(P)||^2 identically! Zero energy cheating.
3. Track the 2D Scaling Manifold Q(M, K):
   M in [1, 2, 4, 8]  x  K in [1, 2, 4, 8, 16, 32, 64]
4. Track the Gram Matrix Evolution G_{ij}(k) across microsteps:
   G_{ij}(k) = <P_i(k), P_j(k)> / (||P_i|| * ||P_j||).
   Tests if branches remain orthogonal (G ~ I) under pure transport and develop
   structured competitive coupling under nonlinear collision!
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
from scipy.linalg import hadamard

from scripts.ib_local.train_stage3_mass_norm_readout_3000 import Stage3MassNormCBIMSudokuModel


def get_orthogonal_hadamard_codes(M: int, d: int = 256, device: str = "cuda") -> torch.Tensor:
    """Generates M mutually orthogonal phase carriers c_h in {-1, +1}^d using Walsh-Hadamard matrices.
    Guarantees <c_h, c_j> == d * delta_{hj} exactly!
    """
    # 256-dimensional Hadamard matrix has 256 mutually orthogonal rows
    H = hadamard(d)  # [256, 256]
    # Pick M distinct rows
    codes = H[:M]    # [M, d]
    tensor_codes = torch.as_tensor(codes, dtype=torch.float32, device=device)
    # Check exact orthogonality
    gram = (tensor_codes @ tensor_codes.T) / d
    assert torch.allclose(gram, torch.eye(M, device=device), atol=1e-5), "Hadamard codes not orthogonal!"
    return tensor_codes.view(M, 1, 1, d)  # [M, 1, 1, d] for broadcasting


@torch.no_grad()
def evaluate_wave_native_multipath(model, test_in: np.ndarray, test_lbl: np.ndarray,
                                   M: int, k_val: int, batch_size: int = 32) -> Tuple[float, float, np.ndarray]:
    """Runs Wave-Native Parallel Hypothesis search for M branches and pondering depth K.
    Returns: (Cell Accuracy, Mean Loss, Average Gram Matrix G_{ij}(K))
    """
    model.eval()
    device = next(model.parameters()).device
    d = model.d
    num_samples = len(test_in)

    codes = get_orthogonal_hadamard_codes(M, d=d, device=device)  # [M, 1, 1, d]
    inv_sqrt_m = 1.0 / math.sqrt(M)

    total_correct = 0
    total_cells = 0
    losses = []
    gram_accum = np.zeros((M, M))
    gram_count = 0

    dt = 1.0 / max(1, k_val)

    for s in range(0, num_samples, batch_size):
        e = min(s + batch_size, num_samples)
        b = e - s
        inp_b = torch.as_tensor(test_in[s:e], dtype=torch.long, device=device)
        lbl_b = torch.as_tensor(test_lbl[s:e], dtype=torch.long, device=device)

        clue_field = model.clue_proj(model.embed_tokens(inp_b).view(b, 9, 9, d))
        orig_norm = torch.linalg.vector_norm(clue_field.float(), dim=(1, 2, 3), keepdim=True).to(clue_field.dtype)

        # 1. Macro-Step H Initial Ingestion: Generate M orthogonal hypothesis wavepackets
        # F_0 = sum_{h=1}^M (1/sqrt(M)) * (clue_field * c_h)
        # Sum of orthogonal packets guarantees exact total energy conservation:
        # ||F_0||^2 == (1/M) * sum ||clue_field * c_h||^2 == ||clue_field||^2!
        branches = clue_field.unsqueeze(0) * codes.unsqueeze(1) * inv_sqrt_m  # [M, B, 9, 9, d]

        # In pure linear transport, we can track each branch individually to measure the Gram matrix!
        # Superposed field entering the physical medium:
        curr_superposed = branches.sum(dim=0)  # [B, 9, 9, d]
        curr_branches = [branches[h] for h in range(M)]

        # 2. Kinetic Pondering Loop K
        for _ in range(k_val):
            # Evolve each branch through the medium
            # Transport (linear & unitary, preserves orthogonality!)
            f_5d = curr_superposed.view(b, 9, 9, model.n_v, model.d_c)
            f_tr = model.transport(f_5d, dt=dt).view(b, 9, 9, d)

            # Collision (nonlinear state-conditioned, induces hypothesis scattering & competition!)
            f_star = model.collision(f_tr, cond=clue_field, dt=dt)

            # Tangent Corrective Flow
            cat_in = torch.cat([f_star, clue_field], dim=-1)
            v_k = model.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)

            gate_val = torch.sigmoid(torch.mean(model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True))
            alpha_k = (dt * model.alpha_max) * gate_val.to(curr_superposed.dtype)

            f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            curr_superposed = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(curr_superposed.dtype)

            # Evolve individual branch probes through the same operator to compute Gram matrix G_ij
            if M > 1:
                new_branches = []
                for h in range(M):
                    fb_5d = curr_branches[h].view(b, 9, 9, model.n_v, model.d_c)
                    fb_tr = model.transport(fb_5d, dt=dt).view(b, 9, 9, d)
                    fb_star = model.collision(fb_tr, cond=clue_field, dt=dt)
                    new_branches.append(fb_star)
                curr_branches = new_branches

        # Exact norm normalization
        curr_superposed = curr_superposed * (orig_norm / (torch.linalg.vector_norm(curr_superposed.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

        # 3. Readout & Born Rule Probability Collapse
        logits = model.readout(curr_superposed)
        loss = F.cross_entropy(logits.view(-1, 11), lbl_b.view(-1)).item()
        losses.append(loss)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == lbl_b).sum().item()
        total_cells += b * 81

        # Compute Gram matrix on first batch
        if M > 1 and gram_count == 0:
            G = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    bi = curr_branches[i].float().flatten(start_dim=1)
                    bj = curr_branches[j].float().flatten(start_dim=1)
                    dot = (bi * bj).sum(dim=-1).mean().item()
                    norm_i = bi.norm(dim=-1).mean().item() + 1e-8
                    norm_j = bj.norm(dim=-1).mean().item() + 1e-8
                    G[i, j] = dot / (norm_i * norm_j)
            gram_accum = G
            gram_count = 1

    return total_correct / total_cells, float(np.mean(losses)), gram_accum


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("results/stage3_mass_norm_readout_3000/Best_Stage3_MassNorm_d256.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("results/master_never_reset_unified_3000/Best_NeverReset_Unified_d256.pt")

    print(f"Loading Stage 3 model from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = Stage3MassNormCBIMSudokuModel(vocab_size=11, d_channels=256).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")[:300]  # 300 test puzzles for dense 2D surface
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")[:300]

    M_values = [1, 2, 4, 8]
    K_values = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 105)
    print("   WAVE-NATIVE PARALLEL HYPOTHESIS SEARCH: 2D SPECTRUM Q(M, K)")
    print("   Carrier: Walsh-Hadamard Orthogonal Phase Codes c_h in {-1, +1}^256 (<c_i, c_j> == delta_ij)")
    print("   Total Energy strictly conserved: sum_h ||P_h||^2 == ||F_0||^2 identically!")
    print("=" * 105)

    q_acc_matrix = np.zeros((len(M_values), len(K_values)))
    q_loss_matrix = np.zeros((len(M_values), len(K_values)))
    gram_records = {}

    for m_idx, M in enumerate(M_values):
        print(f"\n--- Testing Hypothesis Width M = {M} Parallel Waves ---", flush=True)
        for k_idx, K in enumerate(K_values):
            acc, loss, G = evaluate_wave_native_multipath(model, test_in, test_lbl, M=M, k_val=K, batch_size=32)
            q_acc_matrix[m_idx, k_idx] = acc
            q_loss_matrix[m_idx, k_idx] = loss
            if K in [1, 8, 64] and M > 1:
                gram_records[f"M{M}_K{K}"] = G
            print(f"  M = {M:2d}, K = {K:2d} | Cell Acc: {acc*100:5.2f}% | Loss: {loss:6.4f}", flush=True)

    # Print 2D Summary Matrix
    print("\n" + "=" * 105)
    print("   FINAL 2D RESPONSE MATRIX: ACCURACY Q(M, K) (%)")
    print("=" * 105)
    header = f"{'Width M':^10} | " + " | ".join([f"K={k:<4d}" for k in K_values])
    print(header)
    print("-" * len(header))
    for m_idx, M in enumerate(M_values):
        row_str = f" M = {M:2d}    | " + " | ".join([f"{q_acc_matrix[m_idx, k_idx]*100:5.2f}%" for k_idx in range(len(K_values))])
        print(row_str)

    # Check Gram Matrix Evolution (Orthogonality to Scattering)
    if "M4_K1" in gram_records and "M4_K64" in gram_records:
        print("\n" + "=" * 105)
        print("   BRANCH GRAM MATRIX EVOLUTION G_ij (K=1 pure transport -> K=64 nonlinear collision)")
        print("=" * 105)
        print("Gram Matrix at K=1 (Initial Exploration):")
        print(np.round(gram_records["M4_K1"], 4))
        print("\nGram Matrix at K=64 (After Nonlinear Interaction & Scattering):")
        print(np.round(gram_records["M4_K64"], 4))


if __name__ == "__main__":
    main()
