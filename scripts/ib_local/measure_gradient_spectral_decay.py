"""Measure multi-band spatial Fourier spectral decay of gradients across H=0..160 tokens.

This directly determines the optimal boundary between:
1. Local BPTT (handling fast/high-frequency modes)
2. Forward Eligibility Traces (handling slow/DC near-neutral modes)

Spectral Bands on T^3 (8, 8, 4):
- Band 0 (DC / Global Conserved Charge): |k| = 0
- Band 1 (Long-Wave Macro-Modes): 0 < |k| <= 1.5
- Band 2 (Mid-Frequency Mesoscale): 1.5 < |k| <= 3.0
- Band 3 (High-Frequency Micro-Fluctuations): |k| > 3.0
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_bptt128_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_gradient_spectral_bands_decay.json"))
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--h-max", type=int, default=160)
    parser.add_argument("--num-sites", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading champion model from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        v2_coordinate_components=True,
        readout_type=cfg.get("readout_type", "kernel_r1"),
        write_type=cfg.get("write_type", "w2_impedance"),
        micro_steps=3,
        adaptive_clock=True,
        continuous_velocities=True,
        dissipation_type="unified",
        dissipation_rank=4,
        three_clock=True,
        tau_mem=3.0,
        nu_s_init=0.020,
        decouple_source_feedback=True
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()

    H_max = args.h_max
    window_offsets = [8192 + i * 4096 for i in range(args.num_sites)]

    # Precompute 3D spatial Fourier wavenumber grid on (8, 8, 4)
    kx = torch.fft.fftfreq(8, d=1.0, device="cuda") * 8.0
    ky = torch.fft.fftfreq(8, d=1.0, device="cuda") * 8.0
    kz = torch.fft.fftfreq(4, d=1.0, device="cuda") * 4.0
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = torch.sqrt(Kx.square() + Ky.square() + Kz.square())  # [8, 8, 4]

    # Spectral band masks
    band_masks = {
        "Band 0 (DC, |k|=0)": (k_mag == 0.0),
        "Band 1 (Long, 0<|k|<=1.5)": (k_mag > 0.0) & (k_mag <= 1.5),
        "Band 2 (Mid, 1.5<|k|<=3.0)": (k_mag > 1.5) & (k_mag <= 3.0),
        "Band 3 (High, |k|>3.0)": (k_mag > 3.0),
    }

    print(f"\nMeasuring multi-band spectral gradient decay over H = 0 .. {H_max} across {args.num_sites} sites...", flush=True)

    all_site_band_energies = {b: [] for b in band_masks}
    all_site_total_energies = []

    for site_idx, offset in enumerate(window_offsets):
        state = mature_state_base.clone()
        with torch.no_grad():
            for t in range(args.warmup):
                inp = torch.as_tensor([val_data[offset + t]], dtype=torch.long, device="cuda")
                _, state, _ = model.step(state, inp, micro_steps=3)

        state_list = []
        curr_state = state.clone()
        start_t = offset + args.warmup

        for step_i in range(H_max + 1):
            curr_state = curr_state.clone()
            curr_state.requires_grad_(True)
            curr_state.retain_grad()
            state_list.append(curr_state)

            inp = torch.as_tensor([val_data[start_t + step_i]], dtype=torch.long, device="cuda")
            logits, next_state, _ = model.step(curr_state, inp, micro_steps=3)
            curr_state = next_state

        tgt = torch.as_tensor([val_data[start_t + H_max + 1]], dtype=torch.long, device="cuda")
        loss = F.cross_entropy(logits, tgt)
        loss.backward()

        site_band_e = {b: [] for b in band_masks}
        site_tot_e = []

        for t_idx in range(H_max + 1):
            H = H_max - t_idx
            grad = state_list[t_idx].grad  # [1, 8, 8, 4, 128]
            grad_freq = torch.fft.fftn(grad, dim=(1, 2, 3), norm="ortho")
            # Energy per (kx, ky, kz) mode
            e_k = grad_freq.abs().square().sum(dim=(0, -1))  # [8, 8, 4]
            total_e = float(e_k.sum().item())
            site_tot_e.append((H, total_e))

            for b_name, mask in band_masks.items():
                band_e = float(e_k[mask].sum().item())
                site_band_e[b_name].append((H, band_e))

        # Sort by H ascending
        site_tot_e.sort(key=lambda x: x[0])
        all_site_total_energies.append([v for _, v in site_tot_e])

        for b_name in band_masks:
            site_band_e[b_name].sort(key=lambda x: x[0])
            all_site_band_energies[b_name].append([v for _, v in site_band_e[b_name]])

        print(f"  Site {site_idx + 1}/{args.num_sites} evaluated.", flush=True)

    # Average across sites
    mean_total_e = np.mean(all_site_total_energies, axis=0)  # [H_max + 1]
    mean_band_e = {b: np.mean(all_site_band_energies[b], axis=0) for b in band_masks}

    # Relative survival curves S_b(H) = sqrt(E_b(H) / E_b(0))
    # Fraction curves P_b(H) = E_b(H) / E_total(H)
    survivals = {}
    fractions = {}
    for b in band_masks:
        e0 = mean_band_e[b][0]
        survivals[b] = np.sqrt(mean_band_e[b] / (e0 + 1e-12))
        fractions[b] = mean_band_e[b] / (mean_total_e + 1e-12)

    H_eval_points = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160]

    print("\n" + "=" * 115)
    print("      CBIM MULTI-BAND SPATIAL FOURIER GRADIENT SPECTRAL DECAY (H = 0 .. 160 TOKENS)")
    print("=" * 115)
    print(f"{'H (Tokens)':<10} | {'Total Norm':<12} | {'Band 0 (DC) S_0':<16} | {'Band 1 (Low) S_1':<17} | {'Band 2 (Mid) S_2':<17} | {'Band 3 (High) S_3':<18} | {'Slow Mode Share (0+1)'}")
    print("-" * 115)

    for H in H_eval_points:
        tot_norm = math.sqrt(mean_total_e[H])
        s0 = survivals["Band 0 (DC, |k|=0)"][H] * 100
        s1 = survivals["Band 1 (Long, 0<|k|<=1.5)"][H] * 100
        s2 = survivals["Band 2 (Mid, 1.5<|k|<=3.0)"][H] * 100
        s3 = survivals["Band 3 (High, |k|>3.0)"][H] * 100
        slow_share = (fractions["Band 0 (DC, |k|=0)"][H] + fractions["Band 1 (Long, 0<|k|<=1.5)"][H]) * 100
        print(f"{H:<10d} | {tot_norm:<12.4f} | {s0:<15.2f}% | {s1:<16.2f}% | {s2:<16.2f}% | {s3:<17.2f}% | {slow_share:.2f}%")

    print("=" * 115)

    # Find key transition horizons:
    # 1. H where Band 3 (High) drops below 10%, 5%, 1%
    h_high_10 = next((int(h) for h, s in enumerate(survivals["Band 3 (High, |k|>3.0)"]) if s < 0.10), None)
    h_high_5 = next((int(h) for h, s in enumerate(survivals["Band 3 (High, |k|>3.0)"]) if s < 0.05), None)
    h_high_1 = next((int(h) for h, s in enumerate(survivals["Band 3 (High, |k|>3.0)"]) if s < 0.01), None)

    # 2. H where Band 2 (Mid) drops below 10%, 5%, 1%
    h_mid_10 = next((int(h) for h, s in enumerate(survivals["Band 2 (Mid, 1.5<|k|<=3.0)"]) if s < 0.10), None)
    h_mid_5 = next((int(h) for h, s in enumerate(survivals["Band 2 (Mid, 1.5<|k|<=3.0)"]) if s < 0.05), None)
    h_mid_1 = next((int(h) for h, s in enumerate(survivals["Band 2 (Mid, 1.5<|k|<=3.0)"]) if s < 0.01), None)

    # 3. H where Slow Modes (0+1) dominate >= 80%, 90%, 95% of total gradient
    slow_fractions = fractions["Band 0 (DC, |k|=0)"] + fractions["Band 1 (Long, 0<|k|<=1.5)"]
    h_dom_80 = next((int(h) for h, f in enumerate(slow_fractions) if f >= 0.80), None)
    h_dom_90 = next((int(h) for h, f in enumerate(slow_fractions) if f >= 0.90), None)

    print("\n>>> CRITICAL PHYSICAL BOUNDARIES FOR LOCAL BPTT vs ELIGIBILITY TRACES:")
    print(f"  * High-Frequency Ripple (Band 3) Decays to < 10%:  H = {h_high_10} tokens")
    print(f"  * High-Frequency Ripple (Band 3) Decays to < 5%:   H = {h_high_5} tokens")
    print(f"  * High-Frequency Ripple (Band 3) Decays to < 1%:   H = {h_high_1} tokens")
    print(f"  * Mid-Frequency Ripple (Band 2)  Decays to < 10%:  H = {h_mid_10} tokens")
    print(f"  * Mid-Frequency Ripple (Band 2)  Decays to < 5%:   H = {h_mid_5} tokens")
    print(f"  * Mid-Frequency Ripple (Band 2)  Decays to < 1%:   H = {h_mid_1} tokens")
    print(f"  * Horizon where Slow Modes (DC+Long) reach >= 80% Energy: H = {h_dom_80} tokens")
    print(f"  * Horizon where Slow Modes (DC+Long) reach >= 90% Energy: H = {h_dom_90} tokens")
    print("=" * 115)

    report = {
        "checkpoint": str(args.checkpoint),
        "H_max": H_max,
        "critical_horizons": {
            "h_high_freq_below_10pct": h_high_10,
            "h_high_freq_below_5pct": h_high_5,
            "h_high_freq_below_1pct": h_high_1,
            "h_mid_freq_below_10pct": h_mid_10,
            "h_mid_freq_below_5pct": h_mid_5,
            "h_mid_freq_below_1pct": h_mid_1,
            "h_slow_modes_reach_80pct": h_dom_80,
            "h_slow_modes_reach_90pct": h_dom_90,
        },
        "curves": [
            {
                "H": int(H),
                "total_norm": float(math.sqrt(mean_total_e[H])),
                "survival_band0_dc": float(survivals["Band 0 (DC, |k|=0)"][H]),
                "survival_band1_long": float(survivals["Band 1 (Long, 0<|k|<=1.5)"][H]),
                "survival_band2_mid": float(survivals["Band 2 (Mid, 1.5<|k|<=3.0)"][H]),
                "survival_band3_high": float(survivals["Band 3 (High, |k|>3.0)"][H]),
                "fraction_slow_modes": float(slow_fractions[H]),
            }
            for H in H_eval_points
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved multi-band spectral decay report to {args.output}")


if __name__ == "__main__":
    main()
