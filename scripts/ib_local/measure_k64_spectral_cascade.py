"""Diagnostic Experiment: Spatial & Temporal Fourier Spectral Energy Cascade across K = 1 .. 64.
Testing the User's Spectral Cascade Hypothesis:
"During microstep pondering up to K=64, does the field exhibit a physical energy cascade
where high spatial/temporal frequencies decay while long-wave coherent DC/low frequencies survive?
If so, can BPTT backpropagation be selectively applied only to the surviving low frequencies,
saving compute and eliminating high-frequency gradient noise?"

Measurement details:
1. Track across K = 1 .. 64 for 300 test puzzles:
   - E_DC: Spatial DC component (spatial mean) energy share
   - E_low: Spatial low-frequency band (|k| <= 0.25 Nyquist) energy share
   - E_mid: Spatial mid-frequency band (0.25 < |k| <= 0.45 Nyquist) energy share
   - E_high: Spatial high-frequency band (|k| > 0.45 Nyquist) energy share
   - S_spec: Spectral entropy of the 2D spatial Fourier energy distribution
   - Gradient norm survival per Fourier shell (if backpropagating from step K)
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.ib_local.train_stage3_mass_norm_readout_3000 import Stage3MassNormCBIMSudokuModel
from scripts.ib_local.cbim_sudoku_unified_dissipative import (
    ContextualBoundaryWrite2Port,
    QuadraticPassiveRadiationBath
)


def compute_2d_fourier_shells(field: torch.Tensor) -> Dict[str, float]:
    """Computes energy distribution across 2D spatial Fourier shells on (9, 9) Torus."""
    b, h, w, d = field.shape
    f32 = field.float()
    fft = torch.fft.fftn(f32, dim=(1, 2))  # [B, 9, 9, d]
    power = torch.abs(fft) ** 2
    power_spatial = torch.mean(power, dim=(0, 3))  # [9, 9]

    kx = torch.fft.fftfreq(9, d=1.0, device=field.device)
    ky = torch.fft.fftfreq(9, d=1.0, device=field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    R = torch.sqrt(KX**2 + KY**2)

    total_p = torch.sum(power_spatial) + 1e-12
    e_dc = float((power_spatial[0, 0] / total_p).item())
    e_low = float((torch.sum(power_spatial[(R > 0) & (R <= 0.25)]) / total_p).item())
    e_mid = float((torch.sum(power_spatial[(R > 0.25) & (R <= 0.45)]) / total_p).item())
    e_high = float((torch.sum(power_spatial[R > 0.45]) / total_p).item())

    p_norm = (power_spatial / total_p).clamp(min=1e-12)
    s_spec = float((-torch.sum(p_norm * torch.log(p_norm))).item())
    norm_val = float(torch.mean(torch.linalg.vector_norm(f32, dim=(1, 2, 3))).item())

    return {
        "E_DC": e_dc,
        "E_low": e_low,
        "E_mid": e_mid,
        "E_high": e_high,
        "S_spec": s_spec,
        "norm": norm_val
    }


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load our 55.52% Master Checkpoint
    ckpt_path = Path("results/master_never_reset_unified_3000/Best_NeverReset_Unified_d256.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("results/arm_c_champion_3000/Best_Arm_C_Champion_d256.pt")

    print(f"Loading checkpoint from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = Stage3MassNormCBIMSudokuModel(vocab_size=11, d_channels=256).to(device)
    # Load weights with compatibility
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    test_in = np.load("data/sudoku-extreme-1k-aug-100/test/all__inputs.npy")[:200]
    num_samples = len(test_in)

    print(f"\nTracing 2D Spatial Fourier Spectral Cascade across K = 1 .. 64 on {num_samples} puzzles...", flush=True)

    k_steps = list(range(1, 65))
    e_dc_list = []
    e_low_list = []
    e_mid_list = []
    e_high_list = []
    s_spec_list = []

    # Persistent state across batch
    batch_size = 32
    state = torch.zeros(batch_size, 9, 9, 256, device=device)

    # We compute average spectral power at each step k across test puzzles
    step_records = {k: {"E_DC": [], "E_low": [], "E_mid": [], "E_high": [], "S_spec": []} for k in k_steps}

    for s in range(0, num_samples, batch_size):
        e = min(s + batch_size, num_samples)
        b = e - s
        inp_b = torch.as_tensor(test_in[s:e], dtype=torch.long, device=device)
        clue_field = model.clue_proj(model.embed_tokens(inp_b).view(b, 9, 9, 256))

        # 2-Port Boundary Scattering
        f_absorbed, _, _ = model.boundary_write(state[:b], clue_field)
        curr = f_absorbed
        orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

        for k in range(1, 65):
            dt = 1.0 / k
            f_5d = curr.view(b, 9, 9, 8, 32)
            f_tr = model.transport(f_5d, dt=dt).view(b, 9, 9, 256)
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

            # Analyze spatial Fourier spectrum of curr
            spec = compute_2d_fourier_shells(curr)
            for key in ("E_DC", "E_low", "E_mid", "E_high", "S_spec"):
                step_records[k][key].append(spec[key])

        state[:b] = curr.detach()

    # Aggregate means
    for k in k_steps:
        e_dc_list.append(np.mean(step_records[k]["E_DC"]))
        e_low_list.append(np.mean(step_records[k]["E_low"]))
        e_mid_list.append(np.mean(step_records[k]["E_mid"]))
        e_high_list.append(np.mean(step_records[k]["E_high"]))
        s_spec_list.append(np.mean(step_records[k]["S_spec"]))

    # Print Key Milestone Table
    print("\n" + "=" * 85)
    print("   2D SPATIAL FOURIER SPECTRAL ENERGY CASCADE ACROSS K = 1 .. 64")
    print("=" * 85)
    print(f"{'Ponder Step K':^14} | {'E_DC (Mean)':^15} | {'E_low (k<=0.25)':^16} | {'E_mid':^14} | {'E_high (k>0.45)':^16}")
    print("-" * 85)
    for k in [1, 2, 4, 8, 16, 32, 48, 64]:
        idx = k - 1
        print(f"   K = {k:2d}       |    {e_dc_list[idx]*100:5.2f}%     |     {e_low_list[idx]*100:5.2f}%      |    {e_mid_list[idx]*100:5.2f}%    |     {e_high_list[idx]*100:5.2f}%")

    # Plot Spectral Cascade Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#060614")
    fig.suptitle("CBIM 2D Spatial Fourier Spectral Energy Cascade (K = 1 .. 64)", fontsize=16, color="#e0e8ff", y=0.98, fontweight="bold")

    # Panel 1: Stacked Area / Line Plot of Energy Bands
    ax1.set_facecolor("#080a1c")
    ax1.plot(k_steps, [x * 100 for x in e_dc_list], color="#ffdd00", linewidth=2.5, label="E_DC (Global Spatial Mean)")
    ax1.plot(k_steps, [x * 100 for x in e_low_list], color="#38d8b8", linewidth=2.2, label="E_low (Long-Wave Coherent)")
    ax1.plot(k_steps, [x * 100 for x in e_mid_list], color="#2080f0", linewidth=2.0, label="E_mid (Intermediate)")
    ax1.plot(k_steps, [x * 100 for x in e_high_list], color="#ff4488", linewidth=2.0, linestyle="--", label="E_high (Fine 1px Constraint Boundaries)")

    ax1.set_title("1. Spatial Fourier Band Shares (%) vs Ponder Step K", color="#d0e0ff", fontsize=13)
    ax1.set_xlabel("Pondering Step K", color="#90a8d0")
    ax1.set_ylabel("Energy Share (%)", color="#90a8d0")
    ax1.set_xlim(1, 64)
    ax1.grid(True, color="#182850", alpha=0.6)
    ax1.tick_params(colors="#7088b0")
    for spine in ax1.spines.values():
        spine.set_color("#203560")
    ax1.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff")

    # Panel 2: Spectral Entropy S_spec (Degree of Energy Dispersion / Condensation)
    ax2.set_facecolor("#080a1c")
    ax2.plot(k_steps, s_spec_list, color="#9d4edd", linewidth=2.5, label="Spectral Shannon Entropy S_spec (nats)")
    ax2.set_title("2. Spatial Energy Condensation: Spectral Entropy S_spec", color="#d0e0ff", fontsize=13)
    ax2.set_xlabel("Pondering Step K", color="#90a8d0")
    ax2.set_ylabel("Spectral Entropy (nats)", color="#90a8d0")
    ax2.set_xlim(1, 64)
    ax2.grid(True, color="#182850", alpha=0.6)
    ax2.tick_params(colors="#7088b0")
    for spine in ax2.spines.values():
        spine.set_color("#203560")
    ax2.legend(facecolor="#080a1c", edgecolor="#203560", labelcolor="#e0e8ff")

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    out_png = out_dir / "cbim_spatial_spectral_cascade_k64.png"
    plt.savefig(out_png, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSaved spectral cascade analysis plot to {out_png}!", flush=True)


if __name__ == "__main__":
    main()
