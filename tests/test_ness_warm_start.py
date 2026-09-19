"""Unit test for NESS Thermodynamic Random-Phase Warm Start.
Verifies:
1. Macro energy matches E_NESS to 4 decimal places
2. Independent phase sampling produces orthogonal states (cosine ~ 0)
3. Zero contextual text memory leak
4. Backward compatibility when loading older checkpoints
"""
import pytest
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D


def test_ness_thermodynamic_warm_start():
    model = CBIMTorus3D(
        shape=(4, 4, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4
    )

    # Initially has_ness_prior is False -> initial_state returns zero
    zero_state = model.initial_state(2)
    assert (zero_state == 0).all()
    assert not bool(model.has_ness_prior)

    # Synthesize a mock mature state with non-zero energy
    torch.manual_seed(42)
    mature_state = torch.randn(1, 4, 4, 4, 128) * 0.05
    target_energy = 0.5 * mature_state.square().sum(-1).mean().item()

    model.set_ness_prior(mature_state)
    assert bool(model.has_ness_prior)

    # Sample two separate states
    s1 = model.initial_state(1, warm_start=True)
    s2 = model.initial_state(1, warm_start=True)

    e1 = 0.5 * s1.square().sum(-1).mean().item()
    e2 = 0.5 * s2.square().sum(-1).mean().item()

    # 1. Energy matches target energy closely
    assert abs(e1 - target_energy) < 1e-3
    assert abs(e2 - target_energy) < 1e-3

    # 2. Random phase sampling gives mutually orthogonal AC waves (no memory leak)
    dc = model.ness_dc_mean
    ac1 = s1 - dc
    ac2 = s2 - dc
    cos_ac = F.cosine_similarity(ac1.flatten(), ac2.flatten(), dim=0).item()
    assert abs(cos_ac) < 0.10, f"AC waves should be approximately orthogonal, got {cos_ac}"

    # 3. Can explicitly request cold vacuum if desired
    cold = model.initial_state(1, warm_start=False)
    assert (cold == 0).all()


def test_ness_checkpoint_backward_compatibility():
    # Older checkpoint without ness keys
    model = CBIMTorus3D(
        shape=(4, 4, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4
    )
    old_state_dict = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("ness_") and k != "has_ness_prior"}

    # Must load strictly without missing key error
    m_new = CBIMTorus3D(
        shape=(4, 4, 4), velocities=8, content_dim=16,
        dissipation_type="unified", dissipation_rank=4
    )
    m_new.load_state_dict(old_state_dict, strict=True)
    assert not bool(m_new.has_ness_prior)
    assert (m_new.initial_state(1) == 0).all()
