"""Coherent Multi-Path Feynman Path-Integral Diagnostic Experiment.
Comparing Single-Path Classical Evolution vs Coherent Multi-Path Quantum-Like Wave Interference.

Core Concept:
1. Single-Path Classical:
   The initial field F_0 is a single fixed wavepacket derived from known clues.
   Unassigned cells start at neutral baseline.
2. Coherent Multi-Path (Feynman Superposition):
   For unassigned cells, generate orthogonal phase-modulated superposition packets:
     F_0(x) = F_clues(x) + sum_{cand d} (1/sqrt(M)) * clue_proj(d) * exp(i * phi_d)
   All hypothetical solution branches co-exist simultaneously in the SAME [1, 9, 9, 256] field!
   As the wave travels across the 9x9 Torus under Exact Exponential Advection exp(-i*dt*omega)
   and interacts in the Lie collision operator:
     - Conflicting hypothesis paths destructively interfere (phase cancelation to 0).
     - Constraint-satisfying hypothesis paths constructively interfere (phase resonance).
3. Evaluated on the Hardest Extreme Test Puzzles (including Puzzle 122, 898, 72, 863).
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

from scripts.ib_local.train_stage3_mass_norm_readout_3000 import Stage3MassNormCBIMSudokuModel


@torch.no_grad()
def generate_feynman_superposition_packet(model, inp: torch.Tensor, d_channels: int = 256) -> torch.Tensor:
    """Constructs a coherent multi-path phase-superposition field in a single tensor [1, 9, 9, d]."""
    b = inp.shape[0]
    device = inp.device
    clues_mask = (inp > 1).view(b, 9, 9, 1)  # Known clue cells

    # Base clue field
    clue_field = model.clue_proj(model.embed_tokens(inp).view(b, 9, 9, d_channels))

    # All candidate digits: tokens 2..10 represent digits 1..9
    cand_tokens = torch.arange(2, 11, device=device)
    cand_embs = model.clue_proj(model.embed_tokens(cand_tokens).view(9, 1, 1, d_channels))  # [9, 1, 1, d]

    # Orthogonal harmonic phase angles for 9 digits: phi_d = 2 * pi * d / 9
    angles = 2.0 * math.pi * torch.arange(9, device=device).float() / 9.0  # [9]
    # Channels 0..127 get cos(phi), channels 128..255 get sin(phi)
    half_d = d_channels // 2
    cos_phase = torch.cos(angles).view(9, 1, 1, 1).expand(-1, -1, -1, half_d)
    sin_phase = torch.sin(angles).view(9, 1, 1, 1).expand(-1, -1, -1, half_d)
    phase_tensor = torch.cat([cos_phase, sin_phase], dim=-1)  # [9, 1, 1, d]

    # Phase-modulated candidate packets: (1/sqrt(9)) * cand_emb * phase
    superposed_candidates = (cand_embs * phase_tensor).sum(dim=0, keepdim=True) / 3.0  # [1, 1, 1, d]
    superposed_field = superposed_candidates.expand(b, 9, 9, -1)

    # Blend: Known clues are pure; unassigned cells start as full coherent multi-path superposition!
    f_feynman = torch.where(clues_mask, clue_field, superposed_field)
    orig_norm = torch.linalg.vector_norm(clue_field.float(), dim=(1, 2, 3), keepdim=True).to(f_feynman.dtype)
    f_feynman = f_feynman * (orig_norm / (torch.linalg.vector_norm(f_feynman.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

    return f_feynman


@torch.no_grad()
def run_comparison_on_puzzles(model, test_in: np.ndarray, test_lbl: np.ndarray,
                              puzzle_indices: List[int], horizons: List[int]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    d_channels = model.d

    results = {}

    for p_idx in puzzle_indices:
        inp = torch.as_tensor(test_in[p_idx:p_idx+1], dtype=torch.long, device=device)
        lbl = torch.as_tensor(test_lbl[p_idx:p_idx+1], dtype=torch.long, device=device)
        clue_count = int((inp > 1).sum().item())

        clue_field = model.clue_proj(model.embed_tokens(inp).view(1, 9, 9, d_channels))

        p_res = {
            "index": p_idx,
            "clues": clue_count,
            "classical_single_path": {},
            "feynman_multi_path": {}
        }

        # 1. Classical Single-Path Evolution
        for k in horizons:
            dt = 1.0 / max(1, k)
            curr = clue_field.clone()
            orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

            for _ in range(k):
                f_5d = curr.view(1, 9, 9, model.n_v, model.d_c)
                f_tr = model.transport(f_5d, dt=dt).view(1, 9, 9, d_channels)
                f_star = model.collision(f_tr, cond=clue_field, dt=dt)

                cat_in = torch.cat([f_star, clue_field], dim=-1)
                v_k = model.corrective_net(cat_in)
                dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
                norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
                u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
                u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
                alpha_k = (dt * model.alpha_max) * torch.sigmoid(torch.mean(model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
                f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
                curr = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(curr.dtype)

            curr = curr * (orig_norm / (torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))
            logits = model.readout(curr)
            preds = torch.argmax(logits, dim=-1)
            acc = float((preds == lbl).float().mean().item())
            loss = float(F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item())
            p_res["classical_single_path"][k] = {"acc": acc, "loss": loss}

        # 2. Coherent Multi-Path Feynman Wave Superposition Evolution
        for k in horizons:
            dt = 1.0 / max(1, k)
            curr = generate_feynman_superposition_packet(model, inp, d_channels=d_channels)
            orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

            for _ in range(k):
                f_5d = curr.view(1, 9, 9, model.n_v, model.d_c)
                f_tr = model.transport(f_5d, dt=dt).view(1, 9, 9, d_channels)
                f_star = model.collision(f_tr, cond=clue_field, dt=dt)

                cat_in = torch.cat([f_star, clue_field], dim=-1)
                v_k = model.corrective_net(cat_in)
                dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
                norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
                u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
                u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)
                alpha_k = (dt * model.alpha_max) * torch.sigmoid(torch.mean(model.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True)).to(curr.dtype)
                f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
                curr = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(curr.dtype)

            curr = curr * (orig_norm / (torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))
            logits = model.readout(curr)
            preds = torch.argmax(logits, dim=-1)
            acc = float((preds == lbl).float().mean().item())
            loss = float(F.cross_entropy(logits.view(-1, 11), lbl.view(-1)).item())
            p_res["feynman_multi_path"][k] = {"acc": acc, "loss": loss}

        results[p_idx] = p_res

    return results


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

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")
    test_lbl = np.load("data/sudoku-extreme-1k-aug-100/test/all__labels.npy")

    # Focus on hard and classic puzzles
    target_puzzles = [122, 72, 898, 151, 909]  # 122=Extreme 17-clue, 72=24-clue, 898=26-clue, 151=deep winner, 909=easy
    horizons = [1, 2, 4, 8, 16, 32, 64]

    print(f"Testing Feynman Multi-Path Phase Interference across {len(target_puzzles)} benchmark puzzles...", flush=True)
    res = run_comparison_on_puzzles(model, test_in, test_lbl, target_puzzles, horizons)

    print("\n" + "=" * 105)
    print("   COHERENT MULTI-PATH (FEYNMAN SUPERPOSITION) vs CLASSICAL SINGLE-PATH ON HARD PUZZLES")
    print("=" * 105)

    names = {
        122: "Puzzle 122 (Hard 17-Clue Extreme)",
         72: "Puzzle  72 (Hard 24-Clue Deadlock)",
        898: "Puzzle 898 (Hard 26-Clue Winner)",
        151: "Puzzle 151 (Continuous Monotonic Winner)",
        909: "Puzzle 909 (Easy 35-Clue Baseline)"
    }

    for p_idx in target_puzzles:
        info = res[p_idx]
        print(f"\n--- {names[p_idx]} | Clues: {info['clues']} ---")
        print(f"{'Ponder Depth K':^16} | {'Classical Single-Path Acc':^28} | {'Feynman Multi-Path Acc':^28} | {'Delta':^10}")
        print("-" * 88)
        for k in horizons:
            c_acc = info["classical_single_path"][k]["acc"] * 100
            f_acc = info["feynman_multi_path"][k]["acc"] * 100
            diff = f_acc - c_acc
            marker = " 🚀" if diff > 0.5 else (" ⚠️" if diff < -0.5 else "  ")
            print(f"  K = {k:3d}          |           {c_acc:5.1f}%            |           {f_acc:5.1f}%            | {diff:+5.1f}%{marker}")


if __name__ == "__main__":
    main()
