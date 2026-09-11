"""Tests for T01: Local candidate table and serial reference collision operator."""
import math
import pytest
import torch
from torch import nn

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.reference import (
    collision_serial,
    compute_rate,
    reflect,
    sample_candidates,
    spatial_kernel,
)
from scripts.ib_local.types import CandidateTable, CollisionResult, FrozenContext


@pytest.fixture(autouse=True)
def set_deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(11)


def test_candidate_table_dataclass_and_validation():
    # Valid construction
    i = torch.tensor([0, 1], dtype=torch.int64)
    j = torch.tensor([2, 3], dtype=torch.int64)
    normal = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    uniform = torch.tensor([0.2, 0.7], dtype=torch.float64)
    table = CandidateTable(i, j, normal, uniform)
    assert len(table) == 2

    # Mismatched lengths
    with pytest.raises(ValueError, match="length mismatch"):
        CandidateTable(i[:1], j, normal, uniform)

    # Wrong dtype for indices
    with pytest.raises(TypeError, match="must be torch.int64"):
        CandidateTable(i.float(), j, normal, uniform)

    # Self collision i == j
    with pytest.raises(ValueError, match="self-collisions"):
        CandidateTable(torch.tensor([1, 2]), torch.tensor([1, 3]), normal, uniform)

    # to() casting
    casted = table.to(dtype=torch.float32)
    assert casted.normal.dtype == torch.float32
    assert casted.uniform.dtype == torch.float32
    assert casted.i.dtype == torch.int64


def test_frozen_context_dataclass():
    x = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(x, v)
    assert ctx.x.shape == (8, 2)
    assert ctx.x.requires_grad
    assert ctx.v.requires_grad

    # Mismatched shape
    with pytest.raises(ValueError, match="shape mismatch"):
        FrozenContext(x, v[:4])


def test_collision_result_dataclass():
    v = torch.randn(4, 2)
    lp = torch.tensor(0.5)
    accepted = torch.tensor([True, False])
    res = CollisionResult(v, lp, accepted, {"candidates": 2})
    assert res.v.shape == (4, 2)
    assert res.accepted.shape == (2,)

    # Invalid accepted dtype
    with pytest.raises(ValueError):
        CollisionResult(v, lp, torch.tensor([1, 0]), {})


def test_reflection_invariants():
    v = torch.randn(10, 4, dtype=torch.float64)
    w = torch.randn(10, 4, dtype=torch.float64)
    n = torch.randn(10, 4, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    vp, wp = reflect(v, w, n)

    # Exact involution: reflect twice recovers original velocities
    v_rec, w_rec = reflect(vp, wp, n)
    assert torch.allclose(v, v_rec, atol=1e-12)
    assert torch.allclose(w, w_rec, atol=1e-12)

    # Momentum conservation: v' + w' == v + w
    assert torch.allclose(vp + wp, v + w, atol=1e-12)

    # Kinetic energy conservation: |v'|^2 + |w'|^2 == |v|^2 + |w|^2
    ke_before = v.square().sum(-1) + w.square().sum(-1)
    ke_after = vp.square().sum(-1) + wp.square().sum(-1)
    assert torch.allclose(ke_before, ke_after, atol=1e-12)

    # Normal sign symmetry: normal n and -n produce identical reflections
    vp_neg, wp_neg = reflect(v, w, -n)
    assert torch.allclose(vp, vp_neg, atol=1e-12)
    assert torch.allclose(wp, wp_neg, atol=1e-12)


def test_spatial_kernel_locality():
    width = 1.5
    zero = torch.zeros(2, dtype=torch.float64)
    assert abs(float(spatial_kernel(zero, width)) - 1.0 / (width ** 2)) < 1e-12

    # Within support
    r_inside = torch.tensor([1.0, -0.5], dtype=torch.float64)
    assert spatial_kernel(r_inside, width) > 0.0

    # Outside support (one coordinate >= width)
    r_outside = torch.tensor([1.6, 0.2], dtype=torch.float64)
    assert float(spatial_kernel(r_outside, width)) == 0.0

    r_boundary = torch.tensor([1.5, 0.0], dtype=torch.float64)
    assert float(spatial_kernel(r_boundary, width)) == 0.0


def test_sample_candidates():
    n, dim, width, max_rate, duration = 16, 2, 1.0, 5.0, 0.5
    gen = torch.Generator().manual_seed(42)
    table = sample_candidates(n, dim, duration, width, max_rate, generator=gen, dtype=torch.float64)

    assert len(table) > 0
    assert (table.i >= 0).all() and (table.i < n).all()
    assert (table.j >= 0).all() and (table.j < n).all()
    assert (table.i != table.j).all()
    assert torch.allclose(table.normal.norm(dim=-1), torch.ones(len(table), dtype=torch.float64), atol=1e-12)
    assert ((table.uniform >= 0.0) & (table.uniform < 1.0)).all()

    # Zero duration returns empty table
    empty = sample_candidates(n, dim, 0.0, width, max_rate, generator=gen, dtype=torch.float64)
    assert len(empty) == 0
    assert empty.normal.shape == (0, dim)


def test_rate_symmetry_and_microreversibility():
    dim, hidden = 2, 8
    kernel = CollisionKernel(dim, hidden).double()
    context_x = torch.randn(8, dim, dtype=torch.float64)
    context_v = torch.randn(8, dim, dtype=torch.float64)
    ctx = FrozenContext(context_x, context_v)

    center = torch.zeros(dim, dtype=torch.float64)
    v = torch.randn(dim, dtype=torch.float64)
    w = torch.randn(dim, dtype=torch.float64)
    n = torch.tensor([0.6, 0.8], dtype=torch.float64)
    vp, wp = reflect(v, w, n)

    for mode in ("legacy_eps", "strict_local_v1"):
        rate_base = compute_rate(center, v, w, n, ctx, kernel, mode=mode)
        # Exchange pair
        rate_ex = compute_rate(center, w, v, n, ctx, kernel, mode=mode)
        assert torch.allclose(rate_base, rate_ex, atol=1e-12)
        # Post-collision reflection
        rate_rev = compute_rate(center, vp, wp, n, ctx, kernel, mode=mode)
        assert torch.allclose(rate_base, rate_rev, atol=1e-12)
        # Normal sign reversal
        rate_norm = compute_rate(center, v, w, -n, ctx, kernel, mode=mode)
        assert torch.allclose(rate_base, rate_norm, atol=1e-12)


def test_rate_consistency_with_legacy_collision_kernel():
    dim, hidden = 2, 16
    kernel = CollisionKernel(dim, hidden, width=1.2, max_rate=2.5).double()
    context_x = torch.randn(12, dim, dtype=torch.float64)
    context_v = torch.randn(12, dim, dtype=torch.float64)
    ctx = FrozenContext(context_x, context_v)

    for _ in range(5):
        center = torch.randn(dim, dtype=torch.float64)
        v = torch.randn(dim, dtype=torch.float64)
        w = torch.randn(dim, dtype=torch.float64)
        n = torch.randn(dim, dtype=torch.float64)
        n = n / n.norm()

        # Legacy rate from CollisionKernel
        legacy_rate = kernel.rate(center, v, w, n, context_x, context_v)
        # Reference rate with legacy_eps mode
        ref_rate = compute_rate(center, v, w, n, ctx, kernel, mode="legacy_eps", width=1.2, max_rate=2.5)

        # Exact match within floating point tolerance
        assert torch.allclose(legacy_rate, ref_rate, atol=1e-14, rtol=1e-12)


def test_strict_locality_zero_gradient_outside_support():
    dim, hidden = 2, 8
    width = 1.0
    kernel = CollisionKernel(dim, hidden, width=width).double()

    center = torch.zeros(dim, dtype=torch.float64)
    v = torch.tensor([0.5, -0.2], dtype=torch.float64)
    w = torch.tensor([-0.3, 0.4], dtype=torch.float64)
    n = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # Particle 0: inside support (distance 0.3 < width=1.0)
    # Particle 1: far outside support (distance 3.0 > width=1.0)
    context_x = torch.tensor([[0.3, -0.2], [3.0, 2.5]], dtype=torch.float64, requires_grad=True)
    context_v = torch.tensor([[0.1, 0.2], [0.8, -0.9]], dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(context_x, context_v)

    rate_strict = compute_rate(center, v, w, n, ctx, kernel, mode="strict_local_v1", width=width)
    rate_strict.backward()

    # Distant context particle 1 MUST receive strictly zero gradient
    assert float(context_x.grad[1].abs().max()) == 0.0
    assert float(context_v.grad[1].abs().max()) == 0.0

    # Near context particle 0 receives non-zero gradient
    assert float(context_x.grad[0].abs().max()) > 0.0
    assert float(context_v.grad[0].abs().max()) > 0.0


def test_strict_locality_single_axis_out_of_bounds_no_nan():
    dim, hidden = 2, 8
    width = 1.0
    kernel = CollisionKernel(dim, hidden, width=width).double()

    center = torch.zeros(dim, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([0.5, -0.2], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([-0.3, 0.4], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # Particle 0: inside support [0.3, 0.2]
    # Particle 1: single-axis out of bounds [3.0, 0.2] (x_0=3.0 >= 1.0, but x_1=0.2 < 1.0)
    context_x = torch.tensor([[0.3, 0.2], [3.0, 0.2]], dtype=torch.float64, requires_grad=True)
    context_v = torch.tensor([[0.1, 0.2], [0.8, -0.9]], dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(context_x, context_v)

    rate_strict = compute_rate(center, v, w, n, ctx, kernel, mode="strict_local_v1", width=width)
    rate_strict.backward()

    # All gradients must be strictly finite - NO NaNs allowed anywhere
    assert torch.isfinite(center.grad).all(), f"center.grad contains NaN: {center.grad}"
    assert torch.isfinite(v.grad).all(), f"v.grad contains NaN: {v.grad}"
    assert torch.isfinite(w.grad).all(), f"w.grad contains NaN: {w.grad}"
    assert torch.isfinite(context_x.grad).all(), f"context_x.grad contains NaN: {context_x.grad}"
    assert torch.isfinite(context_v.grad).all(), f"context_v.grad contains NaN: {context_v.grad}"

    # Distant context particle 1 MUST receive strictly zero gradient (no leak)
    assert float(context_x.grad[1].abs().max()) == 0.0
    assert float(context_v.grad[1].abs().max()) == 0.0

    # Near context particle 0 receives non-zero finite gradient
    assert float(context_x.grad[0].abs().max()) > 0.0
    assert float(context_v.grad[0].abs().max()) > 0.0


def test_strict_local_log_domain_probabilities_and_rejection():
    # Test extreme probabilities, uniform=0, and stable rejection
    kernel = CollisionKernel(2, 8, width=2.0, max_rate=5.0).double()
    x = torch.zeros(2, 2, dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(x, v)

    # 1. uniform = 0.0 must be accepted stably without NaN
    table_u0 = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.0], dtype=torch.float64),
    )
    res_u0 = collision_serial(x, v, ctx, table_u0, kernel, mode="strict_local_v1", width=2.0, max_rate=5.0)
    assert res_u0.accepted[0]
    assert torch.isfinite(res_u0.log_prob)
    res_u0.log_prob.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # 2. Rejection branch with uniform near 1.0 (uniform = 0.99999)
    v2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    table_rej = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.99999], dtype=torch.float64),
    )
    res_rej = collision_serial(x, v2, ctx, table_rej, kernel, mode="strict_local_v1", width=2.0, max_rate=5.0)
    assert not res_rej.accepted[0]
    assert torch.isfinite(res_rej.log_prob)
    assert float(res_rej.log_prob.detach()) < 0.0  # valid negative log-prob
    res_rej.log_prob.backward()
    assert v2.grad is not None and torch.isfinite(v2.grad).all()

    # 3. Microscopic probability near boundary (|delta_x| = 1.999999, width=2.0)
    x_micro = torch.tensor([[0.0, 0.0], [1.999999, 0.0]], dtype=torch.float64)
    v3 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    table_micro = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.5], dtype=torch.float64),
    )
    res_micro = collision_serial(x_micro, v3, FrozenContext(x_micro, v3), table_micro, kernel, mode="strict_local_v1", width=2.0, max_rate=5.0)
    assert not res_micro.accepted[0]
    assert torch.isfinite(res_micro.log_prob)
    res_micro.log_prob.backward()
    assert v3.grad is not None and torch.isfinite(v3.grad).all()


def test_candidate_table_strict_input_validation():
    i = torch.tensor([0], dtype=torch.int64)
    j = torch.tensor([1], dtype=torch.int64)

    # 1. Non-unit normal [2.0, 0.0] MUST be rejected
    with pytest.raises(ValueError, match="unit vectors"):
        CandidateTable(i, j, torch.tensor([[2.0, 0.0]], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))

    # 2. Non-finite normal (NaN, Inf) MUST be rejected
    with pytest.raises(ValueError, match="finite"):
        CandidateTable(i, j, torch.tensor([[float('nan'), 0.0]], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        CandidateTable(i, j, torch.tensor([[float('inf'), 0.0]], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))

    # 3. Illegal uniform (< 0 or >= 1) MUST be rejected
    with pytest.raises(ValueError, match="uniforms must be in half-open interval"):
        CandidateTable(i, j, torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([-0.1], dtype=torch.float64))
    with pytest.raises(ValueError, match="uniforms must be in half-open interval"):
        CandidateTable(i, j, torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        CandidateTable(i, j, torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([float('nan')], dtype=torch.float64))

    # 4. Negative index MUST be rejected
    with pytest.raises(ValueError, match="non-negative"):
        CandidateTable(torch.tensor([-1], dtype=torch.int64), j, torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))


def test_strict_locality_empty_context_query_only_fallback():
    dim, hidden = 2, 8
    width = 1.0
    kernel = CollisionKernel(dim, hidden, width=width).double()

    center = torch.zeros(dim, dtype=torch.float64)
    v = torch.tensor([0.5, -0.2], dtype=torch.float64)
    w = torch.tensor([-0.3, 0.4], dtype=torch.float64)
    n = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # All particles outside support
    context_x = torch.tensor([[5.0, 5.0], [-6.0, 7.0]], dtype=torch.float64)
    context_v = torch.tensor([[0.1, 0.2], [0.8, -0.9]], dtype=torch.float64)
    ctx = FrozenContext(context_x, context_v)

    # Fallback to query-only branch
    rate = compute_rate(center, v, w, n, ctx, kernel, mode="strict_local_v1", width=width)
    assert rate > 0.0
    assert torch.isfinite(rate)


def test_collision_serial_empty_table():
    kernel = CollisionKernel(2, 8).double()
    x = torch.randn(6, 2, dtype=torch.float64)
    v = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(x, v)
    empty_table = CandidateTable(
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros((0, 2), dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
    )

    res = collision_serial(x, v, ctx, empty_table, kernel, mode="strict_local_v1")
    assert torch.equal(res.v, v)
    assert len(res.accepted) == 0
    assert res.log_prob == 0.0
    assert res.log_prob.requires_grad  # preserves autograd graph
    assert res.stats["accepted"] == 0
    assert res.stats["candidates"] == 0


def test_collision_serial_all_inactive_pairs():
    kernel = CollisionKernel(2, 8, width=1.0).double()
    # All particles far apart (> width)
    x = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    ctx = FrozenContext(x, v)

    table = CandidateTable(
        i=torch.tensor([0, 1], dtype=torch.int64),
        j=torch.tensor([1, 2], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        uniform=torch.tensor([0.1, 0.1], dtype=torch.float64),
    )

    res = collision_serial(x, v, ctx, table, kernel, mode="strict_local_v1", width=1.0)
    assert torch.equal(res.v, v)
    assert res.stats["accepted"] == 0
    assert not res.accepted.any()
    assert res.log_prob == 0.0


def test_collision_serial_active_reflection_and_conservation():
    kernel = CollisionKernel(2, 8, width=2.0, max_rate=10.0).double()
    # Particles at exact same position (local weight = 1/width^2 > 0)
    x = torch.zeros(4, 2, dtype=torch.float64)
    v = torch.tensor([[1.0, 0.5], [-1.0, -0.5], [0.2, 0.8], [-0.2, -0.8]], dtype=torch.float64)
    ctx = FrozenContext(x, v)

    table = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[0.6, 0.8]], dtype=torch.float64),
        uniform=torch.tensor([0.001], dtype=torch.float64),  # very small uniform -> accept
    )

    res = collision_serial(x, v, ctx, table, kernel, mode="strict_local_v1", width=2.0, max_rate=10.0)
    assert res.accepted[0]
    assert res.stats["accepted"] == 1
    assert len(res.stats["pairs"]) == 1

    # Invariants
    assert torch.allclose(res.v.sum(0), v.sum(0), atol=1e-12)
    assert torch.allclose(res.v.square().sum(), v.square().sum(), atol=1e-12)
    assert torch.isfinite(res.log_prob)
    assert abs(res.stats["cross_moment_change"]) < 1e-12


def test_collision_serial_no_aliasing_or_inplace_mutation():
    kernel = CollisionKernel(2, 8, width=2.0, max_rate=10.0).double()
    x = torch.zeros(2, 2, dtype=torch.float64)
    v_orig = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    v = v_orig.clone()
    ctx = FrozenContext(x.clone(), v_orig.clone())

    table = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.0001], dtype=torch.float64),
    )

    res = collision_serial(x, v, ctx, table, kernel, mode="strict_local_v1", width=2.0, max_rate=10.0)
    assert res.accepted[0]

    # Context v and input v must NOT be mutated in-place
    assert torch.equal(ctx.v, v_orig)
    assert torch.equal(v, v_orig)
    assert not torch.equal(res.v, v_orig)


def test_collision_serial_autograd_flow():
    kernel = CollisionKernel(2, 8, width=2.0, max_rate=10.0).double()
    x = torch.zeros(2, 2, dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    ctx_x = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    ctx_v = torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=torch.float64, requires_grad=True)
    ctx = FrozenContext(ctx_x, ctx_v)

    table = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.001], dtype=torch.float64),
    )

    res = collision_serial(x, v, ctx, table, kernel, mode="strict_local_v1", width=2.0, max_rate=10.0)
    loss = res.v.square().sum() + res.log_prob
    loss.backward()

    # Gradients flow to kernel parameters, velocities, and context
    assert kernel.output.weight.grad is not None and kernel.output.weight.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
    assert ctx_x.grad is not None and ctx_x.grad.abs().sum() > 0


def test_collision_serial_and_sample_candidates_strict_validation():
    kernel = CollisionKernel(2, 8, width=1.0, max_rate=1.0).double()
    x = torch.zeros(4, 2, dtype=torch.float64)
    v = torch.zeros(4, 2, dtype=torch.float64)
    ctx = FrozenContext(x, v)
    table = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.5], dtype=torch.float64),
    )

    # 1. User reproduction 1: x.shape=(2,1) vs v.shape=(2,2) silent broadcasting must be rejected
    x_broadcast = torch.zeros(2, 1, dtype=torch.float64)
    v_target = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="identical shapes"):
        collision_serial(x_broadcast, v_target, FrozenContext(v_target, v_target), table, kernel)

    # 2. User reproduction 2: max_rate < 0 must be rejected
    with pytest.raises(ValueError, match="max_rate must be non-negative"):
        collision_serial(x, v, ctx, table, kernel, max_rate=-1.0)
    with pytest.raises(ValueError, match="max_rate must be non-negative"):
        collision_serial(x, v, ctx, table, kernel, max_rate=float("nan"))

    # 3. width <= 0 or non-finite must be rejected
    with pytest.raises(ValueError, match="width must be positive"):
        collision_serial(x, v, ctx, table, kernel, width=0.0)
    with pytest.raises(ValueError, match="width must be positive"):
        collision_serial(x, v, ctx, table, kernel, width=-1.0)
    with pytest.raises(ValueError, match="width must be positive"):
        collision_serial(x, v, ctx, table, kernel, width=float("inf"))

    # 4. table normal dimension mismatch with state (dim 3 vs dim 2)
    table_dim3 = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([1], dtype=torch.int64),
        normal=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.5], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="dimension"):
        collision_serial(x, v, ctx, table_dim3, kernel)

    # 5. context dimension mismatch with state (context dim 3 vs state dim 2)
    ctx_dim3 = FrozenContext(torch.zeros(4, 3, dtype=torch.float64), torch.zeros(4, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="Context dimension"):
        collision_serial(x, v, ctx_dim3, table, kernel)

    # 6. Candidate index out of bounds (N=4, index=5)
    table_oob = CandidateTable(
        i=torch.tensor([0], dtype=torch.int64),
        j=torch.tensor([5], dtype=torch.int64),  # 5 >= 4
        normal=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        uniform=torch.tensor([0.5], dtype=torch.float64),
    )
    with pytest.raises(IndexError, match="out of bounds"):
        collision_serial(x, v, ctx, table_oob, kernel)

    # 7. x vs v dtype mismatch (float32 vs float64)
    with pytest.raises(TypeError, match="identical dtype"):
        collision_serial(x.float(), v, ctx, table, kernel)

    # 8. context vs state dtype mismatch
    ctx_float = FrozenContext(x.float(), v.float())
    with pytest.raises(TypeError, match="Context dtype must match"):
        collision_serial(x, v, ctx_float, table, kernel)

    # 9. table normal vs state dtype mismatch
    table_float = table.to(dtype=torch.float32)
    with pytest.raises(TypeError, match="Candidate normal and uniform dtype"):
        collision_serial(x, v, ctx, table_float, kernel)

    # 10. sample_candidates input validation
    with pytest.raises(ValueError, match="n must be an integer >= 2"):
        sample_candidates(1, 2, 0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="dim must be an integer >= 1"):
        sample_candidates(4, 0, 0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="duration must be non-negative"):
        sample_candidates(4, 2, -0.1, 1.0, 1.0)
    with pytest.raises(ValueError, match="width must be positive"):
        sample_candidates(4, 2, 0.5, 0.0, 1.0)
    with pytest.raises(ValueError, match="max_rate must be non-negative"):
        sample_candidates(4, 2, 0.5, 1.0, -1.0)
