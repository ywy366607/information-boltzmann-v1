"""Unit tests for PhaseSpaceBoltzmann (Pieces 1 + 2 + 3) matching Lean 4 theorems."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.ib_local.phase_space_boltzmann import PhaseSpaceBoltzmann


def test_reference_measure_identities_theorem2():
    pb = PhaseSpaceBoltzmann(16, kappa_lang_init=0.0)
    pb.live_z_kin.fill_(0.7)
    g = torch.randn(1, 8, 256)
    z = g + torch.randn(1, 8, 256)
    # At kappa_lang = 0 -> beta = 1 -> exact identity recovery of z
    out = pb.modulate_language_logits(z, g)
    assert torch.allclose(out, z, atol=1e-6)
    # High kinetic surge with kappa_lang > 0 softens neural residual toward g
    pb_soft = PhaseSpaceBoltzmann(16, kappa_lang_init=0.25)
    pb_soft.live_z_kin.fill_(0.8)
    out_soft = pb_soft.modulate_language_logits(z, g)
    assert float((out_soft - g).abs().mean()) < float((z - g).abs().mean())


def test_phase_space_alignment_breaks_position_tie_theorem1():
    pb = PhaseSpaceBoltzmann(8, alpha_init=0.20)
    pb.live_momentum_dir.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    vel_keys = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # aligned with P
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # opposite to P
    ])
    bonus = pb.memory_velocity_bonus(vel_keys)
    assert bonus.shape == (1, 2)
    assert float(bonus[0, 0]) > float(bonus[0, 1])
    assert float(bonus[0, 0]) == pytest.approx(float(pb.alpha_p()), abs=1e-6)


def test_equipartition_monotonicity_and_envelope_theorem3():
    pb = PhaseSpaceBoltzmann(8, kappa_max=0.35, kappa_lang_init=0.20, kappa_mem_init=0.20, kappa_act_init=0.20)
    pb.live_z_kin.fill_(-0.6)
    b_cold = float(pb.lang_inv_temp_scale())
    pb.live_z_kin.fill_(0.6)
    b_hot = float(pb.lang_inv_temp_scale())
    assert b_hot < b_cold
    lo, hi = math.exp(-0.35), math.exp(0.35)
    for z in (-1.0, 0.0, 1.0):
        pb.live_z_kin.fill_(z)
        for fn in (pb.lang_inv_temp_scale, pb.mem_inv_temp_scale, pb.act_inv_temp_scale):
            val = float(fn())
            assert lo - 1e-6 <= val <= hi + 1e-6


def test_thermostat_ness_contraction_and_amplification_theorems4_5_6():
    pb = PhaseSpaceBoltzmann(8, eta_free_energy=0.12, gamma_diss=0.16, enable_momentum_read=False)
    momenta = [torch.ones(1, 2, 4, 4)]
    model = SimpleNamespace(_momentum_current=momenta)
    engine = SimpleNamespace(bfss_momenta=momenta, model=model)

    # Initialize baselines cleanly
    pb.refresh_live_phase_state(engine, update_ema=True)
    assert float(pb.kinetic_ema) == pytest.approx(1.0, rel=1e-5)

    # Seed low-entropy baseline, then feed high-entropy (uniform) logits -> positive pressure amplifies P
    sharp_logits = torch.zeros(1, 16, 256)
    sharp_logits[..., 0] = 10.0
    pb.apply_free_energy_thermostat(engine, sharp_logits)
    before_norm = float(momenta[0].norm())
    flat_logits = torch.zeros(1, 16, 256)
    info = pb.apply_free_energy_thermostat(engine, flat_logits)
    assert info['pressure'] > 0.0
    assert info['thermostat_scale'] > 1.0
    assert float(momenta[0].norm()) > before_norm

    # Inject huge kinetic spike above NESS ceiling -> thermostat strictly contracts P!
    momenta[0].mul_(100.0)
    huge_before = float(momenta[0].norm())
    info_contract = pb.apply_free_energy_thermostat(engine, flat_logits)
    assert info_contract['thermostat_scale'] < 1.0
    assert float(momenta[0].norm()) < huge_before


def test_end_to_end_gradients_into_all_four_couplings():
    pb = PhaseSpaceBoltzmann(8, alpha_init=0.10, kappa_lang_init=0.05, kappa_mem_init=0.05, kappa_act_init=0.05)
    pb.live_z_kin.fill_(0.5)
    pb.live_momentum_dir.copy_(torch.nn.functional.normalize(torch.randn(8), dim=-1))

    vel_keys = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    bonus = pb.memory_velocity_bonus(vel_keys)
    mem_logits = bonus * pb.mem_inv_temp_scale()
    g = torch.randn(1, 4, 256)
    z = g + torch.randn(1, 4, 256)
    lang_logits = pb.modulate_language_logits(z, g)
    act_logits = pb.act_inv_temp_scale() * torch.tensor([0.2, -0.1, 0.5])

    loss = (mem_logits.softmax(-1)[:, :3].sum()
            + lang_logits.log_softmax(-1)[..., 0].mean()
            + act_logits.log_softmax(-1)[0])
    loss.backward()
    assert pb.theta.grad is not None
    assert torch.isfinite(pb.theta.grad).all()
    assert all(abs(float(pb.theta.grad[i])) > 1e-8 for i in range(4))


def test_fixed_buffer_addresses_for_cuda_graph_safety():
    pb = PhaseSpaceBoltzmann(16, enable_momentum_read=False)
    ptr_dir = pb.live_momentum_dir.data_ptr()
    ptr_z = pb.live_z_kin.data_ptr()
    momenta = [torch.randn(1, 2, 4, 4)]
    engine = SimpleNamespace(bfss_momenta=momenta, model=SimpleNamespace(_momentum_current=momenta))
    pb.refresh_live_phase_state(engine, update_ema=True)
    pb.apply_free_energy_thermostat(engine, torch.randn(1, 8, 256))
    assert pb.live_momentum_dir.data_ptr() == ptr_dir
    assert pb.live_z_kin.data_ptr() == ptr_z
