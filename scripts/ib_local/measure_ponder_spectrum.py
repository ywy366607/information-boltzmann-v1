"""Quantitative Measurement of Phase-Space Spectrum & Problem Conditioning during Pondering.
Measures:
1. Fourier spectral shell energies: E_DC(k), E_low(k), E_mid(k), E_high(k) for k in 0..128.
2. Spectral Entropy S_spec(k) = -sum p_q log p_q.
3. Effective state rank and channel-average cancellation vs true energy.
4. Finite-Time Lyapunov Exponent (FTLE) along internal microsteps k.
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

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ib_local.cbim_sudoku import CBIMSudokuModel, UnitaryCayleyTransport2D, GivensCollision2D
from models.losses import ACTLossHead


@torch.no_grad()
def measure_ponder_spectrum(checkpoint_path: str, data_dir: str = "data/sudoku-extreme-1k-aug-100",
                            max_k: int = 128, num_test_samples: int = 64):
    print("=" * 85)
    print(f"   MEASURING INTERNAL PONDERING SPECTRAL DYNAMICS (k = 0 .. {max_k})")
    print(f"   Checkpoint: {checkpoint_path}")
    print("=" * 85)

    test_in = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")[:num_test_samples]
    test_lbl = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")[:num_test_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.as_tensor(test_in, dtype=torch.long, device=device)
    lbl = torch.as_tensor(test_lbl, dtype=torch.long, device=device)
    B = len(inp)

    # Initialize model
    inner = CBIMSudokuModel(vocab_size=11, d_channels=128, n_velocities=8, ponder_steps=max_k).to(device)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")

    if os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(saved["model"])
        print(f"Loaded checkpoint from {checkpoint_path} (step {saved.get('step', 'unknown')})")
    else:
        print("Warning: Checkpoint not found, using initialized weights for diagnostic.")

    model.eval()

    # Embed initial clues
    emb = inner.embed_tokens(inp).view(B, 9, 9, inner.d)
    clue_field = inner.clue_proj(emb)  # [B, 9, 9, 128]

    # Precompute 2D spatial wavenumber grid on 9x9 torus
    # k_x, k_y in {0, 1, 2, 3, 4, -4, -3, -2, -1}
    kx = torch.fft.fftfreq(9, d=1.0 / 9) * 2.0 * math.pi
    ky = torch.fft.fftfreq(9, d=1.0 / 9) * 2.0 * math.pi
    grid_kx, grid_ky = torch.meshgrid(kx, ky, indexing="ij")
    q_mag = torch.sqrt(grid_kx**2 + grid_ky**2).to(device)  # [9, 9]

    # Fourier shell definitions
    mask_dc = (q_mag == 0.0)
    mask_low = (q_mag > 0.0) & (q_mag <= 1.5 * math.pi)
    mask_mid = (q_mag > 1.5 * math.pi) & (q_mag <= 3.0 * math.pi)
    mask_high = (q_mag > 3.0 * math.pi)

    print("Wavenumber Shells (9x9 Torus):")
    print(f"  DC   (q = 0):              {mask_dc.sum().item()} modes")
    print(f"  Low  (0 < q <= 1.5 pi):    {mask_low.sum().item()} modes")
    print(f"  Mid  (1.5 pi < q <= 3 pi): {mask_mid.sum().item()} modes")
    print(f"  High (q > 3 pi):           {mask_high.sum().item()} modes")

    # Tracking trajectories
    curr = clue_field.clone()
    curr_pert = curr + 1e-5 * torch.randn_like(curr)  # For FTLE (Lyapunov exponent)
    eps = 1e-5

    results_table = []

    print("\n" + "-" * 95)
    print(f"{'k':>4s} | {'Total E':>10s} | {'DC %':>7s} | {'Low %':>7s} | {'Mid %':>7s} | {'High %':>7s} | {'S_spec':>7s} | {'Rank':>6s} | {'FTLE':>8s} | {'Loss':>7s}")
    print("-" * 95)

    check_steps = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    check_set = set(check_steps)

    for k in range(0, max_k + 1):
        if k in check_set:
            # 1. 2D FFT over spatial dimensions [B, 9, 9, 128] -> [B, 9, 9, 128]
            f_fft = torch.fft.fft2(curr.to(torch.complex64), dim=(1, 2))  # [B, 9, 9, 128]
            p_q = f_fft.abs().square().sum(dim=-1).mean(dim=0)  # [9, 9] power spectrum across batch and channels
            total_e = p_q.sum().item()

            e_dc = p_q[mask_dc].sum().item() / max(total_e, 1e-8)
            e_low = p_q[mask_low].sum().item() / max(total_e, 1e-8)
            e_mid = p_q[mask_mid].sum().item() / max(total_e, 1e-8)
            e_high = p_q[mask_high].sum().item() / max(total_e, 1e-8)

            # Spectral Entropy S_spec = -sum p log p
            norm_pq = (p_q / max(total_e, 1e-8)).flatten()
            norm_pq = norm_pq[norm_pq > 1e-12]
            s_spec = float(-(norm_pq * norm_pq.log()).sum().item())

            # Effective State Rank (Participation Ratio of spatial nodes)
            node_energy = curr.square().sum(dim=-1)  # [B, 9, 9]
            node_prob = node_energy / node_energy.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            eff_rank = float(1.0 / (node_prob**2).sum(dim=(1, 2)).mean().item())

            # FTLE: log(||pert|| / eps) / max(k, 1)
            diff = (curr_pert - curr).norm().item()
            ftle = math.log(max(diff / eps, 1e-8)) / max(k, 1) if k > 0 else 0.0

            # Instantaneous loss & accuracy
            flat_curr = curr.view(B, 81, inner.d)
            logits_k = inner.readout_mlp(flat_curr)
            loss_k = float(F.cross_entropy(logits_k.view(-1, 11), lbl.view(-1), ignore_index=0).item())

            print(f"{k:4d} | {total_e:10.2f} | {e_dc*100:6.2f}% | {e_low*100:6.2f}% | {e_mid*100:6.2f}% | {e_high*100:6.2f}% | {s_spec:7.3f} | {eff_rank:6.1f} | {ftle:8.4f} | {loss_k:7.3f}")

            results_table.append({
                "k": k, "total_energy": total_e,
                "e_dc": e_dc, "e_low": e_low, "e_mid": e_mid, "e_high": e_high,
                "s_spec": s_spec, "effective_rank": eff_rank,
                "ftle": ftle, "loss": loss_k
            })

        if k < max_k:
            # Advance 1 step: Transport + Collision
            B, H, W, D = curr.shape
            f_5d = curr.view(B, H, W, inner.n_v, inner.d_c)
            f_tr = inner.transport(f_5d).view(B, H, W, D)
            curr = inner.collision(f_tr)

            # Advance perturbed state for FTLE
            f_5d_p = curr_pert.view(B, H, W, inner.n_v, inner.d_c)
            f_tr_p = inner.transport(f_5d_p).view(B, H, W, D)
            curr_pert = inner.collision(f_tr_p)
            # Re-normalize perturbation magnitude
            diff_vec = curr_pert - curr
            curr_pert = curr + diff_vec / (diff_vec.norm().clamp_min(1e-12)) * eps

    out_file = Path("results/ponder_spectral_diagnostic.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print("-" * 95)
    print(f"Diagnostic results saved to {out_file}")
    return results_table


if __name__ == "__main__":
    ckpt = "results/cbim_sudoku_3000/Best_CBIM_Sudoku_3000.pt"
    measure_ponder_spectrum(ckpt)
