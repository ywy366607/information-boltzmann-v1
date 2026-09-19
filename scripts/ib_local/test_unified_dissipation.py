"""Verify UnifiedTorusDissipation: mathematical correctness, backward compatibility,
CUDAGraph capture, and E(q, t) impulse response spectrum.
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
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D, UnifiedTorusDissipation, QuadraticTorusBath


def test_instantiation():
    print("Testing instantiation...")
    # Unified
    m_uni = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4,
        continuous_velocities=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True
    ).cuda()
    assert isinstance(m_uni.bath, UnifiedTorusDissipation)
    print("  Unified model instantiated successfully:", m_uni.architecture)

    # Legacy quadratic
    m_quad = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        dissipation_type="quadratic",
        continuous_velocities=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True
    ).cuda()
    assert isinstance(m_quad.bath, QuadraticTorusBath)
    print("  Legacy quadratic model instantiated successfully:", m_quad.architecture)


def test_forward_backward():
    print("\nTesting forward & backward pass...")
    m = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4,
        continuous_velocities=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True
    ).cuda()

    ids = torch.randint(0, 50257, (1, 16), device="cuda")
    targets = torch.randint(0, 50257, (1, 16), device="cuda")
    loss, next_state, diag = m(ids, targets)
    loss.backward()
    assert torch.isfinite(loss).item(), "Loss must be finite"
    assert next_state.shape == (1, 8, 8, 4, 128)
    assert "dissipation_gamma0" in diag
    assert "dissipation_nu" in diag
    assert "dissipation_lambda_mean" in diag
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Diagnostics: gamma0={diag['dissipation_gamma0']:.4f}, nu={diag['dissipation_nu']:.4f}, lambda_mean={diag['dissipation_lambda_mean']:.4f}")
    print("  Forward and backward pass succeeded!")


def test_cudagraph():
    print("\nTesting TruncatedInternalTimeGraphTrainer CUDAGraph capture...")
    from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer
    m = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4,
        continuous_velocities=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=3, adaptive_clock=True
    ).cuda()

    runner = TruncatedInternalTimeGraphTrainer(m, tokens=32, chunk_tokens=32)
    ids = torch.randint(0, 50257, (1, 32), device="cuda")
    targets = torch.randint(0, 50257, (1, 32), device="cuda")
    step_loss, state, diag = runner.step(ids, targets)
    print(f"  Graph replay step loss: {step_loss.item():.4f}")
    print("  CUDAGraph capture and replay succeeded!")


def test_impulse_spectrum_comparison():
    print("\nTesting 40-step impulse response spectrum E(q, t)...")
    shape = (8, 8, 4)
    velocities = 8
    content_dim = 16
    d = velocities * content_dim

    # Compare Quadratic Bath vs Unified Dissipation under identical internal dynamics
    m_quad = CBIMTorus3D(
        shape=shape, velocities=velocities, content_dim=content_dim,
        dissipation_type="quadratic", continuous_velocities=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True
    ).cuda().eval()

    m_uni = CBIMTorus3D(
        shape=shape, velocities=velocities, content_dim=content_dim,
        dissipation_type="unified", continuous_velocities=True,
        readout_type="kernel_r1", write_type="w2_impedance",
        micro_steps=3, adaptive_clock=True
    ).cuda().eval()

    # Precompute wavevector norms |q| on (8, 8, 4)
    wave_axes = [torch.fft.fftfreq(n, device="cuda") * 2.0 * math.pi for n in shape]
    wave = torch.stack(torch.meshgrid(*wave_axes, indexing="ij"), -1)
    q_norm = wave.norm(dim=-1) # shape: (8, 8, 4)

    # Initial localized impulse perturbation at center (4, 4, 2)
    torch.manual_seed(42)
    init_field = torch.zeros(1, *shape, d, device="cuda")
    init_field[0, 4, 4, 2, :] = torch.randn(d, device="cuda") * 5.0
    init_energy = 0.5 * init_field.square().sum().item()

    # Define high-q (> 2.0) vs low-q (<= 1.0) masks
    low_q_mask = (q_norm <= 1.0) & (q_norm > 0.0)
    high_q_mask = q_norm >= 2.0

    print(f"  Initial impulse total energy: {init_energy:.4f}")

    def run_impulse(model):
        field = init_field.clone()
        energies, low_q_energies, high_q_energies = [], [], []
        dummy_tok = torch.zeros(1, d, device="cuda")

        for step in range(40):
            # Internal autonomous kinetic evolution without new external token injection
            for k in range(model.micro_steps):
                alpha_k = model.clock(field, dummy_tok) if model.adaptive_clock else 1.0
                dt_k = alpha_k * model.tau_0_tensor
                dir_k = model.direction_controller(field, dummy_tok) if model.continuous_velocities else None
                mult, _ = model.transport.multiplier(dt_k, direction=dir_k)
                field = model.transport.apply_multiplier(field, mult)
                field, _ = model.collision(field, dt_k)
                if isinstance(model.bath, UnifiedTorusDissipation):
                    field, _ = model.bath(field, dt_k, tok_embed=None)
                else:
                    field, _ = model.bath(field, dt_k)

            # Measure energy spectrum
            tot_e = 0.5 * field.square().sum().item()
            freq = torch.fft.fftn(field, dim=(1, 2, 3), norm="ortho")
            spec_power = 0.5 * freq.abs().square().sum(dim=-1)[0] # (8, 8, 4)

            low_e = spec_power[low_q_mask].sum().item()
            high_e = spec_power[high_q_mask].sum().item()

            energies.append(tot_e)
            low_q_energies.append(low_e)
            high_q_energies.append(high_e)

        return energies, low_q_energies, high_q_energies

    e_quad, low_quad, high_quad = run_impulse(m_quad)
    e_uni, low_uni, high_uni = run_impulse(m_uni)

    print(f"  Legacy Quadratic Bath: Step 0={e_quad[0]:.2f}, Step 10={e_quad[9]:.2f}, Step 40={e_quad[39]:.2f} (Residual: {e_quad[39]/init_energy*100:.1f}%)")
    print(f"  Unified Dissipation:   Step 0={e_uni[0]:.2f}, Step 10={e_uni[9]:.2f}, Step 40={e_uni[39]:.2f} (Residual: {e_uni[39]/init_energy*100:.1f}%)")

    print(f"  High-q Ringing Suppression:")
    print(f"    Legacy High-q at Step 40:  {high_quad[39]:.4f}")
    print(f"    Unified High-q at Step 40: {high_uni[39]:.4f} (Suppression ratio: {high_quad[39] / max(high_uni[39], 1e-12):.1f}x)")

    print(f"  Low-q Macro Wave Preservation:")
    print(f"    Unified Low-q fraction at Step 40: {low_uni[39] / max(e_uni[39], 1e-12) * 100:.1f}%")

    # Assertions
    assert e_uni[39] < e_uni[0], "Unified dissipation must monotonically attenuate overall energy in autonomous relaxation"
    assert high_uni[39] < high_quad[39] * 0.1, "Unified dissipation must suppress high-q ringing by at least 10x"
    print("  ALL SPECTRUM AND DISSIPATION CHECKS PASSED!")


if __name__ == "__main__":
    test_instantiation()
    test_forward_backward()
    test_cudagraph()
    test_impulse_spectrum_comparison()
