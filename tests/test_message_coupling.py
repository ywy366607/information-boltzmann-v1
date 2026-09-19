"""Unit tests for ParticleMessageCoupling."""
import pytest
import torch
from torch.nn import functional as F

from scripts.ib_local.message_coupling import ParticleMessageCoupling


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_particle_message_coupling_exact_identity_at_init():
    """Verify that zero-initialized v_head produces strictly zero Delta v and identity states."""
    n_particles = 256
    pmc = ParticleMessageCoupling(dim=4, hidden_dim=128, n_particles=n_particles)

    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    x_new, v_new, gamma, diag = pmc(x, v, u, diagnostics=True)

    assert torch.equal(x_new, x), "x must be identical to input"
    assert torch.equal(v_new, v), "v must be identical to input at initialization"
    assert diag['delta_v_norm_mean'] == 0.0


def test_particle_message_coupling_timescale_initialization():
    """Verify that multi-timescale gamma initialization spans [32, 1024] tokens."""
    n_particles = 256
    tau_min, tau_max = 32.0, 1024.0
    pmc = ParticleMessageCoupling(dim=4, hidden_dim=128, n_particles=n_particles, tau_min=tau_min, tau_max=tau_max)

    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    _, _, gamma, _ = pmc(x, v, u)
    gamma = gamma.squeeze(-1)
    taus = 1.0 / gamma

    assert torch.isclose(taus.min(), torch.tensor(tau_min), atol=1e-4)
    assert torch.isclose(taus.max(), torch.tensor(tau_max), atol=1e-4)


def test_particle_message_coupling_local_interaction_support():
    """Verify that persistent particles far from message particles receive strictly zero impulse."""
    n_particles = 10
    pmc = ParticleMessageCoupling(dim=4, hidden_dim=128, n_particles=n_particles, width=1.0)

    # Randomize weights so v_head is non-zero
    for p in pmc.parameters():
        p.data.normal_(0.0, 0.1)

    # Put persistent particles far away at x = 10.0
    x = torch.full((n_particles, 4), 10.0)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    x_new, v_new, _, _ = pmc(x, v, u)

    # Because message particles are bounded in [-1.5, 1.5] and width=1.0, distance is > 8.0
    # Kernel A is strictly 0.0, so delta_v must be strictly 0.0
    assert torch.equal(v_new, v), "Particles far outside compact support must receive strictly zero impulse"


def test_particle_message_coupling_gradient_flow():
    """Verify that gradients propagate back to token embedding, message generator, and persistent state."""
    n_particles = 32
    pmc = ParticleMessageCoupling(dim=4, hidden_dim=128, n_particles=n_particles)

    # Non-zero weights
    for p in pmc.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, 4, requires_grad=True)
    v = torch.randn(n_particles, 4, requires_grad=True)
    u = torch.randn(128, requires_grad=True)

    x_new, v_new, gamma, _ = pmc(x, v, u)
    loss = v_new.sum() + gamma.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert pmc.msg_pos_proj.weight.grad is not None
    assert pmc.msg_feat_proj.weight.grad is not None
    assert pmc.v_head.weight.grad is not None
    assert pmc.gamma_head.weight.grad is not None


def test_particle_message_coupling_permutation_equivariance():
    """Verify permutation equivariance across persistent particles with permuted timescale attributes."""
    n_particles = 32
    pmc = ParticleMessageCoupling(dim=4, hidden_dim=128, n_particles=n_particles)

    for p in pmc.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    perm = torch.randperm(n_particles)
    # Permute particle state and its intrinsic timescale bias attribute
    bias_perm = pmc.gamma_bias[perm]
    x_new_p, v_new_p, gamma_p, _ = pmc(x[perm], v[perm], u, particle_bias=bias_perm)
    x_new, v_new, gamma, _ = pmc(x, v, u)

    assert torch.allclose(x_new[perm], x_new_p, atol=1e-5)
    assert torch.allclose(v_new[perm], v_new_p, atol=1e-5)
    assert torch.allclose(gamma[perm], gamma_p, atol=1e-5)
