"""Measure gradient survival curve g(H) = ||dL_T / dF_{T-H}|| and modal spectrum.

Protocol:
- Model: CBIM Three-Clock v1 (Run B Best Checkpoint, step 3000)
- Field: T^3 (8, 8, 4), 256 nodes, 128 channels
- State: Warm persistent state from OWT validation sequence
- Computes exact vector-Jacobian backpropagation across H = 0 .. 140 tokens
- Analyzes both:
  1. Raw coupled boundary write (recurrent feedback in write operator)
  2. Physically decoupled boundary write (external boundary packet driving)
- Decomposes gradient energy into spatial Fourier wavenumbers (DC, low-k, high-k)
- Fits two-component decay: r(H) = A_fast * exp(-H/tau_fast) + A_slow * exp(-H/tau_slow)
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
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def fit_two_component_exponential(H_vals: np.ndarray, r_vals: np.ndarray):
    """Fit r(H) = A_fast * exp(-H / tau_fast) + A_slow * exp(-H / tau_slow) using PyTorch."""
    H_t = torch.tensor(H_vals, dtype=torch.float32)
    r_t = torch.tensor(r_vals, dtype=torch.float32)

    log_a_fast = torch.tensor(math.log(0.6), requires_grad=True)
    log_tau_fast = torch.tensor(math.log(15.0), requires_grad=True)
    log_a_slow = torch.tensor(math.log(0.4), requires_grad=True)
    log_tau_slow = torch.tensor(math.log(80.0), requires_grad=True)

    optimizer = torch.optim.Adam([log_a_fast, log_tau_fast, log_a_slow, log_tau_slow], lr=0.03)
    for _ in range(2000):
        optimizer.zero_grad()
        a_f = torch.exp(log_a_fast)
        t_f = torch.exp(log_tau_fast)
        a_s = torch.exp(log_a_slow)
        t_s = torch.exp(log_tau_slow)

        pred = a_f * torch.exp(-H_t / t_f.clamp_min(1.0)) + a_s * torch.exp(-H_t / t_s.clamp_min(5.0))
        loss = F.mse_loss(pred, r_t) + 0.1 * F.mse_loss(torch.log(pred.clamp_min(1e-6)), torch.log(r_t.clamp_min(1e-6)))
        loss.backward()
        optimizer.step()

    a_f = float(torch.exp(log_a_fast).detach())
    t_f = float(torch.exp(log_tau_fast).detach())
    a_s = float(torch.exp(log_a_slow).detach())
    t_s = float(torch.exp(log_tau_slow).detach())

    tot = a_f + a_s
    a_f /= tot
    a_s /= tot

    return {
        "A_fast": a_f,
        "tau_fast": t_f,
        "A_slow": a_s,
        "tau_slow": t_s,
    }


def patch_clean_boundary_write(model):
    orig_forward = model.source.forward

    def clean_forward(field, token_ids):
        token = model.source.embedding(token_ids)
        anchor = torch.zeros(field.shape[0], 3, device=field.device, dtype=field.dtype)
        center_value = torch.sigmoid(model.source.address(token))
        nu_s = F.softplus(model.source.nu_s_param).clamp_min(1e-5)
        width_val = F.softplus(model.source.width(token))
        sigma_eff_sq = width_val.square() + 2.0 * nu_s * 1.0
        exponent = -0.5 * torch.einsum("bj,xyzj->bxyz", sigma_eff_sq, model.source.wave_sq.to(dtype=field.dtype))
        b_eff = torch.exp(exponent)
        phase = -torch.einsum("xyzj,bj->bxyz", model.source.wave.to(dtype=field.dtype), center_value)
        s_k = b_eff * torch.complex(phase.cos(), phase.sin())
        s_x = torch.fft.ifftn(s_k, dim=(1, 2, 3), norm="ortho").real
        spatial = (s_x / s_x.amax((1, 2, 3), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)

        # In physics, boundary packet is driven by the token embedding, not recurrent state feedback
        neighbors = model.source.neighborhood(field.detach())
        local = (model.source.content(token)[:, None, None, None]
                 + model.source.state_content(field.detach())
                 + model.source.neighbor_content(neighbors))
        local = F.normalize(local, dim=-1)
        packet = spatial[..., None] * (model.source.channel_scale * local)

        axes = (1, 2, 3)
        spatial_weight = spatial[..., None]
        spatial_sum = spatial.sum(axes)[..., None].clamp_min(1e-8)
        context = ((spatial_weight * field).sum(axes) / spatial_sum)
        packet_local = (spatial_weight * packet).sum(axes) / spatial_sum
        local_energy = 0.5 * (spatial_weight * field.square()).sum(axes).sum(-1, keepdim=True) / spatial_sum
        f_norm = context.norm(dim=-1, keepdim=True)
        p_norm = packet_local.norm(dim=-1, keepdim=True)
        f_hat = context / f_norm.clamp_min(1e-6)
        p_hat = packet_local / p_norm.clamp_min(1e-6)
        cos_fp = (f_hat * p_hat).sum(dim=-1, keepdim=True)
        phys = torch.cat([local_energy, cos_fp, f_norm, p_norm], dim=-1)
        angle = model.source.max_angle * torch.sigmoid(
            model.source.angle(torch.cat((token, context.detach(), phys.detach()), -1)))
        theta = spatial[..., None] * angle[:, None, None, None]
        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * packet
        reflected = -sine * field + cosine * packet
        return field_next, reflected, {
            "incident_energy": model.source.energy(packet).detach(),
            "reflected_energy": model.source.energy(reflected).detach(),
            "accepted_energy": (model.source.energy(field_next) - model.source.energy(field)).detach(),
            "delta_e_field": (model.source.energy(field_next) - model.source.energy(field)).detach(),
            "accepted_fraction": 1.0 - (model.source.energy(reflected) / model.source.energy(packet).clamp_min(1e-8)).detach(),
            "t_packet": 1.0 - (model.source.energy(reflected) / model.source.energy(packet).clamp_min(1e-8)).detach(),
            "cross_interference": (torch.sin(2.0 * theta) * field * packet).sum(dim=-1).mean().detach(),
            "write_to_f_ratio": (field_next - field).norm(dim=-1).mean().detach() / (field.norm(dim=-1).mean().detach() + 1e-6),
            "write_angle_abs_mean": theta.detach().abs().mean(),
            "write_angle_peak_mean": angle.detach().abs().mean(),
            "write_spatial_support": (spatial.detach() > 0.1).float().mean(),
            "write_balance_residual": (model.source.energy(field_next) + model.source.energy(reflected) - model.source.energy(field) - model.source.energy(packet)).abs().detach(),
            "source_center": center_value[:, None, None, None, :].detach().reshape(field.shape[0], 3),
            "source_anchor": anchor.detach(),
            "source_displacement": center_value.detach(),
            "write_width_mean": width_val.detach().mean(),
            "source_nu_s": nu_s.detach(),
        }

    model.source.forward = clean_forward


def measure_curve(model, val_data, mature_state_base, window_offsets, H_max, warmup):
    all_g_curves = []
    all_dc_shares = []
    all_lowk_shares = []

    kx = torch.fft.fftfreq(8, d=1.0, device="cuda") * 8.0
    ky = torch.fft.fftfreq(8, d=1.0, device="cuda") * 8.0
    kz = torch.fft.fftfreq(4, d=1.0, device="cuda") * 4.0
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = torch.sqrt(Kx.square() + Ky.square() + Kz.square())
    dc_mask = (k_mag == 0.0)
    lowk_mask = (k_mag <= 1.5)

    for w_idx, offset in enumerate(window_offsets):
        state = mature_state_base.clone()
        with torch.no_grad():
            for t in range(warmup):
                inp = torch.as_tensor([val_data[offset + t]], dtype=torch.long, device="cuda")
                _, state, _ = model.step(state, inp, micro_steps=3)

        state_list = []
        curr_state = state.clone()
        start_t = offset + warmup

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

        g_h = []
        dc_h = []
        lowk_h = []

        for t_idx in range(H_max + 1):
            H = H_max - t_idx
            grad = state_list[t_idx].grad
            grad_norm = float(grad.norm(p=2).item())
            g_h.append((H, grad_norm))

            grad_freq = torch.fft.fftn(grad, dim=(1, 2, 3), norm="ortho")
            grad_energy_k = grad_freq.abs().square().sum(dim=(0, -1))
            total_e = float(grad_energy_k.sum().item())

            if total_e > 1e-12:
                dc_share = float(grad_energy_k[dc_mask].sum().item()) / total_e
                lowk_share = float(grad_energy_k[lowk_mask].sum().item()) / total_e
            else:
                dc_share, lowk_share = 0.0, 0.0

            dc_h.append((H, dc_share))
            lowk_h.append((H, lowk_share))

        g_h.sort(key=lambda x: x[0])
        dc_h.sort(key=lambda x: x[0])
        lowk_h.sort(key=lambda x: x[0])

        all_g_curves.append([val for _, val in g_h])
        all_dc_shares.append([val for _, val in dc_h])
        all_lowk_shares.append([val for _, val in lowk_h])

    mean_g = np.mean(all_g_curves, axis=0)
    r_h = mean_g / mean_g[0]
    mean_dc = np.mean(all_dc_shares, axis=0)
    mean_lowk = np.mean(all_lowk_shares, axis=0)
    return mean_g, r_h, mean_dc, mean_lowk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/cbim_three_clock_w2_8x8x4_k3_3000/BBest.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2/validation.npy"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/published/cbim_gradient_survival_run_b.json"))
    parser.add_argument("--h-max", type=int, default=140)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--num-windows", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...", flush=True)
    saved = torch.load(args.checkpoint, map_location="cuda")
    cfg = saved["config"]

    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]),
        velocities=cfg["velocities"],
        content_dim=cfg["content_dim"],
        v2_coordinate_components=True,
        readout_type=cfg.get("readout_type", "kernel_r1"),
        write_type=cfg.get("write_type", "w2_impedance"),
        micro_steps=cfg.get("micro_steps", 3),
        adaptive_clock=cfg.get("adaptive_clock", True),
        continuous_velocities=cfg.get("continuous_velocities", True),
        dissipation_type="unified",
        dissipation_rank=cfg.get("dissipation_rank", 4),
        three_clock=cfg.get("three_clock", True),
        tau_mem=cfg.get("tau_mem", 3.0),
        nu_s_init=cfg.get("nu_s_init", 0.020)
    ).cuda()
    model.load_state_dict(saved["model"])
    model.eval()

    # Apply physical boundary decoupling
    patch_clean_boundary_write(model)

    val_data = np.load(args.data, mmap_mode="r")
    mature_state_base = saved["state"].detach().cuda()

    H_max = args.h_max
    window_offsets = [8192, 16384, 24576, 32768][:args.num_windows]

    print(f"\nMeasuring stabilized gradient survival curve over H = 0 .. {H_max} on {len(window_offsets)} distinct OWT sites...", flush=True)
    mean_g, r_h, mean_dc, mean_lowk = measure_curve(model, val_data, mature_state_base, window_offsets, H_max, args.warmup)

    H_arr = np.arange(H_max + 1)
    H_1pct = None
    H_01pct = None
    for H, r in zip(H_arr, r_h):
        if r < 0.01 and H_1pct is None:
            H_1pct = int(H)
        if r < 0.001 and H_01pct is None:
            H_01pct = int(H)

    fit_params = fit_two_component_exponential(H_arr, r_h)

    print("\n" + "=" * 95)
    print("      CBIM THREE-CLOCK V1 (PHYSICAL BOUNDARY): GRADIENT SURVIVAL SPECTRUM g(H) = ||dL_T / dF_{T-H}||")
    print("=" * 95)
    print(f"  H (Tokens) | g(H) (Norm)   | r(H) = g(H)/g(0) | DC Mode Share (|k|=0) | Low-k Share (|k|<=1.5)")
    print("-" * 95)
    checkpoints_H = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 140]
    for H in checkpoints_H:
        if H <= H_max:
            print(f"  {H:<10d} | {mean_g[H]:<13.5e} | {r_h[H]:<16.4e} | {mean_dc[H] * 100:<20.2f}% | {mean_lowk[H] * 100:.2f}%")
    print("-" * 95)
    print(f"\n>>> Two-Component Spectrum Fit:")
    print(f"    r(H) = A_fast * exp(-H / tau_fast) + A_slow * exp(-H / tau_slow)")
    print(f"    Fast Mode: Amplitude = {fit_params['A_fast']*100:.2f}%, Half-life / tau = {fit_params['tau_fast']:.2f} tokens")
    print(f"    Slow Mode: Amplitude = {fit_params['A_slow']*100:.2f}%, Half-life / tau = {fit_params['tau_slow']:.2f} tokens")
    print(f"\n>>> Effective Gradient Horizons:")
    print(f"    H_1%   (99.0% gradient dissipation horizon): {H_1pct if H_1pct else '> 140'} tokens")
    print(f"    H_0.1% (99.9% gradient dissipation horizon): {H_01pct if H_01pct else '> 140'} tokens")
    print(f"    Gradient survival at H=32 (old TBPTT chunk):  {r_h[32]*100:.2f}%")
    print(f"    Gradient survival at H=64:                   {r_h[64]*100:.2f}%")
    print(f"    Gradient survival at H=128 (full BPTT):       {r_h[128]*100:.2f}%")
    print("=" * 95)

    report = {
        "checkpoint": str(args.checkpoint),
        "H_max": H_max,
        "warmup": args.warmup,
        "num_windows": args.num_windows,
        "effective_horizons": {
            "H_1pct": H_1pct,
            "H_01pct": H_01pct,
            "survival_at_H32": float(r_h[32]),
            "survival_at_H64": float(r_h[64]),
            "survival_at_H128": float(r_h[128]),
        },
        "two_component_fit": fit_params,
        "curve_samples": [
            {
                "H": int(H),
                "g_norm": float(mean_g[H]),
                "r_ratio": float(r_h[H]),
                "dc_share": float(mean_dc[H]),
                "lowk_share": float(mean_lowk[H]),
            }
            for H in checkpoints_H if H <= H_max
        ]
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved gradient survival report to {args.output}")


if __name__ == "__main__":
    main()
