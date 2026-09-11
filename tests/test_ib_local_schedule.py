"""Tests for T02: Dependency scheduling and layer-batched local collision operator."""
import math
import pytest
import torch

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.batched import collision_batched
from scripts.ib_local.reference import collision_serial, sample_candidates
from scripts.ib_local.schedule import schedule_dependencies
from scripts.ib_local.types import CandidateTable, FrozenContext


@pytest.fixture(autouse=True)
def set_deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(11)


def test_schedule_dependencies_chain():
    # Chain: (0, 1) -> (1, 2) -> (2, 3) -> (3, 4)
    i = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    j = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    levels, perm, offsets = schedule_dependencies(i, j)

    assert levels.tolist() == [0, 1, 2, 3]
    assert perm.tolist() == [0, 1, 2, 3]
    assert offsets.tolist() == [0, 1, 2, 3, 4]


def test_schedule_dependencies_star():
    # Star: all pairs share particle 0: (0, 1), (0, 2), (0, 3)
    i = torch.tensor([0, 0, 0], dtype=torch.int64)
    j = torch.tensor([1, 2, 3], dtype=torch.int64)
    levels, perm, offsets = schedule_dependencies(i, j)

    assert levels.tolist() == [0, 1, 2]
    assert len(offsets) == 4
    assert offsets.tolist() == [0, 1, 2, 3]


def test_schedule_dependencies_disjoint():
    # Mutually disjoint pairs: (0, 1), (2, 3), (4, 5)
    i = torch.tensor([0, 2, 4], dtype=torch.int64)
    j = torch.tensor([1, 3, 5], dtype=torch.int64)
    levels, perm, offsets = schedule_dependencies(i, j)

    # All execute in parallel at level 0
    assert levels.tolist() == [0, 0, 0]
    assert len(offsets) == 2
    assert offsets.tolist() == [0, 3]


def test_schedule_dependencies_duplicate():
    # Duplicate pairs: (0, 1), (0, 1), (0, 1)
    i = torch.tensor([0, 0, 0], dtype=torch.int64)
    j = torch.tensor([1, 1, 1], dtype=torch.int64)
    levels, perm, offsets = schedule_dependencies(i, j)

    assert levels.tolist() == [0, 1, 2]
    assert offsets.tolist() == [0, 1, 2, 3]


def test_schedule_dependencies_empty():
    i = torch.zeros(0, dtype=torch.int64)
    j = torch.zeros(0, dtype=torch.int64)
    levels, perm, offsets = schedule_dependencies(i, j)

    assert len(levels) == 0
    assert len(perm) == 0
    assert offsets.tolist() == [0]


def test_schedule_dependencies_validation():
    # Self-collision
    with pytest.raises(ValueError, match="Self-collision"):
        schedule_dependencies(torch.tensor([1]), torch.tensor([1]))

    # Negative index
    with pytest.raises(ValueError, match="non-negative"):
        schedule_dependencies(torch.tensor([-1]), torch.tensor([2]))

    # Index exceeds n_particles
    with pytest.raises(IndexError, match="exceeds n_particles"):
        schedule_dependencies(torch.tensor([0, 1]), torch.tensor([1, 4]), n_particles=4)


def _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1", width=2.0, max_rate=5.0):
    """Helper to verify exact numerical match for outputs and full autograd gradients."""
    x_s = x.clone().detach().requires_grad_(True)
    v_s = v.clone().detach().requires_grad_(True)
    ctx_x_s = ctx.x.clone().detach().requires_grad_(True)
    ctx_v_s = ctx.v.clone().detach().requires_grad_(True)
    ctx_s = FrozenContext(ctx_x_s, ctx_v_s)

    x_b = x.clone().detach().requires_grad_(True)
    v_b = v.clone().detach().requires_grad_(True)
    ctx_x_b = ctx.x.clone().detach().requires_grad_(True)
    ctx_v_b = ctx.v.clone().detach().requires_grad_(True)
    ctx_b = FrozenContext(ctx_x_b, ctx_v_b)

    res_s = collision_serial(x_s, v_s, ctx_s, table, kernel, mode=mode, width=width, max_rate=max_rate)
    res_b = collision_batched(x_b, v_b, ctx_b, table, kernel, mode=mode, width=width, max_rate=max_rate)

    # 1. Acceptance decisions must match identically
    assert torch.equal(res_s.accepted, res_b.accepted), (
        f"Accepted mismatch: serial={res_s.accepted.tolist()}, batched={res_b.accepted.tolist()}"
    )

    # 2. Output velocities must match within FP64 numerical tolerance (atol 1e-10)
    assert torch.allclose(res_s.v, res_b.v, atol=1e-10, rtol=1e-8), (
        f"Max velocity diff: {(res_s.v - res_b.v).abs().max().item()}"
    )

    # 3. Log probabilities must match
    assert torch.allclose(res_s.log_prob, res_b.log_prob, atol=1e-10, rtol=1e-8), (
        f"Max log_prob diff: {(res_s.log_prob - res_b.log_prob).abs().max().item()}"
    )

    # 4. L2 endpoint differentiable loss for full gradient audit
    loss_s = res_s.v.square().sum() + res_s.log_prob
    loss_b = res_b.v.square().sum() + res_b.log_prob

    params = list(kernel.parameters())
    grads_s = torch.autograd.grad(loss_s, [x_s, v_s, ctx_x_s, ctx_v_s] + params, retain_graph=True)
    grads_b = torch.autograd.grad(loss_b, [x_b, v_b, ctx_x_b, ctx_v_b] + params)

    param_names = ['x', 'v', 'ctx_x', 'ctx_v'] + [f'param_{i}' for i in range(len(params))]
    for name, gs, gb in zip(param_names, grads_s, grads_b):
        diff = (gs - gb).abs().max().item()
        assert diff < 1e-10, f"Gradient mismatch for {name}: max diff = {diff:.2e}"


def test_batched_vs_serial_chain():
    # Chain topology: (0, 1), (1, 2), (2, 3)
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=2.0, max_rate=5.0).double()
    n = 4
    x = torch.zeros(n, dim, dtype=torch.float64)
    v = torch.randn(n, dim, dtype=torch.float64)
    ctx = FrozenContext(torch.randn(6, dim, dtype=torch.float64), torch.randn(6, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 1, 2], dtype=torch.int64),
        j=torch.tensor([1, 2, 3], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1")


def test_batched_vs_serial_star():
    # Star topology: (0, 1), (0, 2), (0, 3)
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=2.0, max_rate=5.0).double()
    n = 4
    x = torch.zeros(n, dim, dtype=torch.float64)
    v = torch.randn(n, dim, dtype=torch.float64)
    ctx = FrozenContext(torch.randn(6, dim, dtype=torch.float64), torch.randn(6, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 0, 0], dtype=torch.int64),
        j=torch.tensor([1, 2, 3], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1")


def test_batched_vs_serial_disjoint():
    # Disjoint topology: (0, 1), (2, 3), (4, 5) - single parallel layer
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=2.0, max_rate=5.0).double()
    n = 6
    x = torch.zeros(n, dim, dtype=torch.float64)
    v = torch.randn(n, dim, dtype=torch.float64)
    ctx = FrozenContext(torch.randn(6, dim, dtype=torch.float64), torch.randn(6, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 2, 4], dtype=torch.int64),
        j=torch.tensor([1, 3, 5], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.8, -0.6]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1")


def test_batched_vs_serial_duplicate():
    # Duplicate topology: same pair (0, 1) repeats 3 times
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=2.0, max_rate=5.0).double()
    n = 2
    x = torch.zeros(n, dim, dtype=torch.float64)
    v = torch.randn(n, dim, dtype=torch.float64)
    ctx = FrozenContext(torch.randn(4, dim, dtype=torch.float64), torch.randn(4, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 0, 0], dtype=torch.int64),
        j=torch.tensor([1, 1, 1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1")


def test_batched_vs_serial_mixed_inactive():
    # Mixed active and inactive pairs
    dim, hidden = 2, 8
    width = 1.0
    kernel = CollisionKernel(dim, hidden, width=width, max_rate=5.0).double()
    # Particles 0 and 1 are close (active); particle 2 is far away (inactive when paired with 0 or 1)
    x = torch.tensor([[0.0, 0.0], [0.2, -0.3], [5.0, 5.0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 0.5], [-1.0, -0.5], [0.0, 2.0]], dtype=torch.float64)
    ctx = FrozenContext(torch.randn(4, dim, dtype=torch.float64), torch.randn(4, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 0, 1], dtype=torch.int64),
        j=torch.tensor([1, 2, 2], dtype=torch.int64),  # (0, 1) active, (0, 2) inactive, (1, 2) inactive
        normal=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.5, 0.5], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="strict_local_v1", width=width)


def test_batched_vs_serial_legacy_eps():
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=2.0, max_rate=5.0).double()
    n = 4
    x = torch.zeros(n, dim, dtype=torch.float64)
    v = torch.randn(n, dim, dtype=torch.float64)
    ctx = FrozenContext(torch.randn(6, dim, dtype=torch.float64), torch.randn(6, dim, dtype=torch.float64))

    table = CandidateTable(
        i=torch.tensor([0, 2], dtype=torch.int64),
        j=torch.tensor([1, 3], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64),
        uniform=torch.tensor([0.001, 0.001], dtype=torch.float64),
    )

    _check_serial_vs_batched(x, v, ctx, table, kernel, mode="legacy_eps", width=2.0)


def test_batched_empty_and_all_inactive():
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden, width=1.0, max_rate=5.0).double()
    x = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    ctx = FrozenContext(torch.randn(4, dim, dtype=torch.float64), torch.randn(4, dim, dtype=torch.float64))

    # 1. Empty table
    empty_table = CandidateTable(
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros((0, dim), dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
    )
    res_b_empty = collision_batched(x, v, ctx, empty_table, kernel)
    res_s_empty = collision_serial(x, v, ctx, empty_table, kernel)
    assert torch.equal(res_b_empty.v, res_s_empty.v)
    assert torch.equal(res_b_empty.accepted, res_s_empty.accepted)
    assert torch.equal(res_b_empty.log_prob, res_s_empty.log_prob)

    # 2. All inactive table
    inactive_table = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.5], dtype=torch.float64),
    )
    res_b_inact = collision_batched(x, v, ctx, inactive_table, kernel, width=1.0)
    res_s_inact = collision_serial(x, v, ctx, inactive_table, kernel, width=1.0)
    assert torch.equal(res_b_inact.v, res_s_inact.v)
    assert torch.equal(res_b_inact.accepted, res_s_inact.accepted)
    assert torch.equal(res_b_inact.log_prob, res_s_inact.log_prob)
