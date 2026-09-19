"""Comprehensive unit tests for M03 AdaptiveMemoryController and LocalEnvironmentField."""
import pytest
import torch
from torch.nn import functional as F

from scripts.ib_local.adaptive_memory import AdaptiveMemoryController, LocalEnvironmentField


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_multi_timescale_initialization():
    """Verify that gamma_bias produces multi-timescale timescales spanning tau in [2, 50] tokens."""
    n_particles = 256
    tau_min, tau_max = 2.0, 50.0
    ctrl = AdaptiveMemoryController(n_particles=n_particles, tau_min=tau_min, tau_max=tau_max)

    # Initial gammas before any adaptation
    init_gammas = F.softplus(ctrl.gamma_bias).squeeze(-1)

    # Convert to theoretical timescales tau = 2.0 / gamma
    taus = 2.0 / init_gammas

    assert torch.isclose(taus.min(), torch.tensor(tau_min), atol=1e-4)
    assert torch.isclose(taus.max(), torch.tensor(tau_max), atol=1e-4)
    # Check monotonic spacing across particles
    assert (taus[1:] >= taus[:-1]).all(), "Timescales must be monotonically increasing across particle index"


def test_exact_identity_state_at_initialization():
    """Verify that zero-initialized write_head produces strictly zero Delta z and identity state."""
    n_particles = 256
    ctrl = AdaptiveMemoryController(dim=4, hidden_dim=128, n_particles=n_particles)

    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    x_new, v_new, gamma, diag = ctrl(x, v, u, diagnostics=True)

    assert torch.equal(x_new, x), "x must be exactly identical to input at initialization"
    assert torch.equal(v_new, v), "v must be exactly identical to input at initialization"
    assert diag['write_norm_mean'] == 0.0


def test_local_environment_field_properties_and_batch_isolation():
    """Verify environment properties: rho >= 1, T_kin >= 0, and zero cross-sample mixing."""
    env = LocalEnvironmentField(dim=4, d_space=8, width=1.0)

    # Single sample
    x = torch.randn(64, 4)
    v = torch.randn(64, 4)
    e = env(x, v)

    assert e.shape == (64, 14)  # 1(rho) + 4(u) + 1(T_kin) + 8(h_x) = 14
    rho = e[:, 0]
    u = e[:, 1:5]
    t_kin = e[:, 5]
    h_x = e[:, 6:]

    # Density includes self-interaction (W_ii = 1), so rho >= 1
    assert (rho >= 1.0).all()
    # Kinetic temperature must be non-negative
    assert (t_kin >= 0.0).all()
    assert torch.isfinite(e).all()

    # Verify batch isolation: sample A must not affect sample B
    x2 = torch.randn(64, 4)
    v2 = torch.randn(64, 4)
    e2 = env(x2, v2)

    e_batched = env(torch.stack([x, x2]), torch.stack([v, v2]))

    diff1 = (e - e_batched[0]).abs().max().item()
    diff2 = (e2 - e_batched[1]).abs().max().item()
    assert diff1 < 1e-6, f"Cross-sample leakage on sample 0: {diff1:.2e}"
    assert diff2 < 1e-6, f"Cross-sample leakage on sample 1: {diff2:.2e}"


def test_gradient_reachability_all_parameters():
    """Verify that gradients propagate to all encoder, head, and environment parameters."""
    n_particles = 64
    ctrl = AdaptiveMemoryController(dim=4, hidden_dim=128, n_particles=n_particles)

    x = torch.randn(n_particles, 4, requires_grad=True)
    v = torch.randn(n_particles, 4, requires_grad=True)
    u = torch.randn(128, requires_grad=True)

    x_new, v_new, gamma, _ = ctrl(x, v, u)
    loss = x_new.sum() + v_new.sum() + gamma.sum()
    loss.backward()

    # Check inputs receive finite gradient
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()

    # Check all model parameters receive finite gradient
    for name, p in ctrl.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"


def test_decoupled_write_and_forget_parameterization():
    """Verify that write gate w_i and damping gamma_i are independently modulated."""
    n_particles = 64
    ctrl = AdaptiveMemoryController(dim=4, hidden_dim=128, n_particles=n_particles)

    # Randomize heads away from zero
    for p in ctrl.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    _, _, gamma, _ = ctrl(x, v, u)
    c = ctrl.encoder(torch.cat([u.unsqueeze(0).expand(n_particles, -1), torch.cat([x, v], -1), ctrl.environment_field(x, v)], -1))
    w = torch.sigmoid(ctrl.gate_head(c))
    alpha = torch.exp(-gamma * 1.0)  # Delta t = 1.0

    # Ensure w_i is not trivially locked to 1 - alpha_i
    diff = (w - (1.0 - alpha)).abs().mean().item()
    assert diff > 0.05, "w_i and 1 - alpha_i must be independently parameterized"
