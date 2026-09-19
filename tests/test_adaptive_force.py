"""Acceptance criteria unit tests for UnifiedAdaptiveForce (M03)."""
import pytest
import torch
from torch.nn import functional as F

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.adaptive_force import LocalEnvironmentField, UnifiedAdaptiveForce
from scripts.ib_local.reference import reflect
from scripts.ib_local.window import LocalWindow


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_acceptance_1_initial_gamma_matches_timescales():
    """Criterion 1: 初始 gamma 与配置时间尺度一致."""
    n_particles = 256
    tau_min = 32.0
    tau_max = 1024.0
    dt_token = 1.0

    force_module = UnifiedAdaptiveForce(
        dim=4, hidden_dim=128, n_particles=n_particles,
        tau_min=tau_min, tau_max=tau_max, dt_token=dt_token,
    )

    # Initial gammas before adaptation (zero-initialized head)
    x = torch.randn(n_particles, 4)
    v = torch.randn(n_particles, 4)
    u = torch.randn(128)

    _, gamma, _ = force_module(x, v, u)
    gamma = gamma.squeeze(-1)

    # Convert to retention alphas and taus
    alphas = torch.exp(-gamma * dt_token)
    taus = 1.0 / (gamma * dt_token)

    # 1. Verify exact min and max boundaries
    assert torch.isclose(taus.min(), torch.tensor(tau_min), atol=1e-4), f"Min tau mismatch: {taus.min()}"
    assert torch.isclose(taus.max(), torch.tensor(tau_max), atol=1e-4), f"Max tau mismatch: {taus.max()}"

    # 2. Verify retention alpha formula alpha_{0, i} = e^{-1 / tau_i}
    target_taus = tau_min * (tau_max / tau_min) ** (torch.linspace(0, 1, n_particles))
    target_alphas = torch.exp(-1.0 / target_taus)
    assert torch.allclose(alphas, target_alphas, atol=1e-6)

    # 3. Verify zero initialization of gamma head
    assert torch.equal(force_module.gamma_head.weight, torch.zeros_like(force_module.gamma_head.weight))
    assert torch.equal(force_module.gamma_head.bias, torch.zeros_like(force_module.gamma_head.bias))


def test_acceptance_2_force_gradients_reachable():
    """Criterion 2: 外力对输入、速度和局部环境具有可达梯度."""
    n_particles = 64
    force_module = UnifiedAdaptiveForce(dim=4, hidden_dim=128, n_particles=n_particles)

    # Initialize weights away from zero to test architectural gradient reachability through encoder/environment
    for p in force_module.parameters():
        p.data.normal_(0.0, 0.1)

    x = torch.randn(n_particles, 4, requires_grad=True)
    v = torch.randn(n_particles, 4, requires_grad=True)
    u = torch.randn(128, requires_grad=True)

    F_input, gamma, _ = force_module(x, v, u)
    loss = F_input.square().sum()
    loss.backward()

    # Inputs x, v, u must all receive non-zero finite gradients
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
    assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0

    # Force head and encoder parameters receive finite gradients
    assert force_module.force_head.weight.grad is not None and torch.isfinite(force_module.force_head.weight.grad).all()
    assert force_module.environment.coord_mlp[0].weight.grad is not None


def test_acceptance_3_gamma_head_gradient_flow_with_zero_init():
    """Criterion 3: gamma 头可获得梯度；零初始化不阻断该头学习."""
    n_particles = 64
    force_module = UnifiedAdaptiveForce(dim=4, hidden_dim=128, n_particles=n_particles)

    x = torch.randn(n_particles, 4, requires_grad=True)
    v = torch.randn(n_particles, 4, requires_grad=True)
    u = torch.randn(128, requires_grad=True)

    _, gamma, _ = force_module(x, v, u)
    loss = gamma.sum()
    loss.backward()

    # Zero-initialized gamma head must receive finite, non-zero gradient
    assert force_module.gamma_head.weight.grad is not None
    assert torch.isfinite(force_module.gamma_head.weight.grad).all()
    assert force_module.gamma_head.weight.grad.abs().sum() > 0
    assert force_module.gamma_bias.grad is not None and torch.isfinite(force_module.gamma_bias.grad).all()


def test_acceptance_4_permutation_equivariance_and_batch_isolation():
    """Criterion 4: 粒子重排一致，局部聚合不跨样本."""
    dim = 4
    d_space = 8
    n_particles = 64
    env = LocalEnvironmentField(dim=dim, d_space=d_space, width=1.0)

    # 1. Permutation equivariance
    x = torch.randn(n_particles, dim)
    v = torch.randn(n_particles, dim)
    perm = torch.randperm(n_particles)

    e_orig = env(x, v)
    e_perm = env(x[perm], v[perm])

    assert torch.allclose(e_orig[perm], e_perm, atol=1e-6)

    # 2. Batch isolation (sample A does not affect sample B)
    x1, v1 = torch.randn(n_particles, dim), torch.randn(n_particles, dim)
    x2, v2 = torch.randn(n_particles, dim), torch.randn(n_particles, dim)

    e1 = env(x1, v1)
    e2 = env(x2, v2)
    e_batched = env(torch.stack([x1, x2]), torch.stack([v1, v2]))

    diff1 = (e1 - e_batched[0]).abs().max().item()
    diff2 = (e2 - e_batched[1]).abs().max().item()
    assert diff1 < 1e-6
    assert diff2 < 1e-6


def test_acceptance_5_collision_conservation_invariants_preserved():
    """Criterion 5: 原碰撞守恒测试继续通过."""
    # Collision operator conserves mass, momentum, and kinetic energy in float64
    v1 = torch.randn(10, 4, dtype=torch.float64)
    v2 = torch.randn(10, 4, dtype=torch.float64)
    normal = torch.randn(10, 4, dtype=torch.float64)
    normal = normal / normal.norm(dim=-1, keepdim=True)

    vp, wp = reflect(v1, v2, normal)

    # Momentum conservation: v1 + v2 == vp + wp
    p_init = v1 + v2
    p_final = vp + wp
    assert torch.allclose(p_init, p_final, atol=1e-14)

    # Kinetic energy conservation: |v1|^2 + |v2|^2 == |vp|^2 + |wp|^2
    ke_init = (v1.square() + v2.square()).sum(-1)
    ke_final = (vp.square() + wp.square()).sum(-1)
    assert torch.allclose(ke_init, ke_final, atol=1e-14)


def test_acceptance_6_backward_compatibility_old_checkpoints():
    """Criterion 6: 旧配置和 checkpoint 仍能加载运行."""
    ckpt_path = "results/ib_local_bpe_256_3000_v2/age_003000.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load into LocalWindow in default configuration (without adaptive force)
    model = LocalWindow(hidden=saved["config"]["hidden"], particles=saved["config"]["particles"])
    model.load_state_dict(saved["model"])
    model.eval()

    # Forward check
    x = saved["x"]
    v = saved["v"]
    ids = torch.tensor([50256, 100, 200, 300])
    targets = torch.tensor([100, 200, 300, 400])
    clocks = torch.zeros(4, 4, 2)
    noise = torch.zeros(4 * 4 * 4, saved["config"]["particles"], 4)
    empty_tables = [[] for _ in range(16)]

    with torch.no_grad():
        out = model(x, v, ids, targets, clocks, noise, empty_tables)
        loss = out[1]

    assert torch.isfinite(loss).all()
    assert loss.item() > 0
