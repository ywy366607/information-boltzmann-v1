"""Arithmetic, smooth-model and full-state continuation checks; no capability task."""
import copy
from fractions import Fraction
from itertools import product
import math

import pytest
import torch

from information_boltzmann.conservative_cycle_optimizer import (
    ConservativeCycle, ResourceCycle, _cycle_chunk, _filtered_cycle_chunk, _resource_observer_chunk,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_filtered_reference_carries_signed_history_and_keeps_scaled_step_safe(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mean = torch.tensor([[1., -2., 3., -4.]], dtype=torch.float64, device=device)
    scale = torch.tensor([[1., 2., .5, 4.]], dtype=mean.dtype, device=device)
    code = torch.tensor([552], dtype=torch.int16, device=device)
    zero = torch.zeros_like(mean)
    gain = torch.tensor(.25, dtype=mean.dtype, device=device)
    next_code, next_mean, update = _filtered_cycle_chunk(code, mean, zero, gain, scale, .125)
    reverse_code, reverse_mean, reverse_update = _filtered_cycle_chunk(
        code, -mean, zero, gain, scale, .125)
    assert torch.equal(next_code, reverse_code)
    assert torch.equal(next_mean, -reverse_mean)
    assert torch.equal(update, -reverse_update) and bool((update != 0).any())
    assert torch.equal(code, torch.tensor([552], dtype=torch.int16, device=device))
    # Exact diagonal quadratic: its gradient is the filtered estimate, and
    # its curvature is scale^2. Test the physical parameter step, not a proxy.
    theta = next_mean / scale.square()
    before = .5 * (scale.square() * theta.square()).sum()
    after = .5 * (scale.square() * (theta - update).square()).sum()
    promised = .125 / 256 * (next_mean / scale).square().sum()
    assert bool(after + promised <= before)


def test_filter_error_envelope_pays_for_zero_gain_drift_and_changing_scale():
    mean = torch.zeros((1, 4), dtype=torch.float64)
    actual = torch.full_like(mean, .25)
    radius = .25
    code = torch.tensor([552], dtype=torch.int16)
    for step, gain in enumerate([0., .25, .5, 1.]):
        actual += .125
        observed = actual + torch.tensor([[.0625, -.0625, .0625, -.0625]])
        scale = torch.full_like(mean, 2. ** (step - 1))
        code, mean, _ = _filtered_cycle_chunk(code, mean, observed, torch.tensor(gain), scale, .125)
        radius = (1 - gain) * (radius + .125) + gain * .0625
        assert bool(((mean - actual).abs() <= radius).all())
    assert radius == .0625


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_preconditioning_changes_the_allocation_evidence_before_the_update(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    observed = torch.ones((1, 4), dtype=torch.float64, device=device)
    code = torch.tensor([552], dtype=torch.int16, device=device)
    gain = torch.ones((), dtype=observed.dtype, device=device)
    uniform, _, _ = _filtered_cycle_chunk(code, observed * 0, observed, gain, observed, .125)
    anisotropic, _, update = _filtered_cycle_chunk(
        code, observed * 0, observed, gain,
        torch.tensor([[1., 16., 16., 1.]], dtype=observed.dtype, device=device), .125)
    assert int(uniform[0] % 65 - 32) == 0
    assert int(anisotropic[0] % 65 - 32) == 32
    torch.testing.assert_close(update, torch.tensor([[.125, 0., 0., .125]],
                              dtype=observed.dtype, device=device), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_resource_observer_normalizes_startup_and_bounds_the_actual_update(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    code = torch.full((2, 16), 552, dtype=torch.int16, device=device)
    mean = torch.zeros((2, 16, 4), dtype=dtype, device=device)
    rms = torch.zeros((2, 1, 1), dtype=dtype, device=device)
    mass = torch.zeros_like(rms)
    gradient = torch.tensor([1., 2., -3., -4.], dtype=dtype, device=device).expand_as(mean)
    for step in range(4):
        old_mean, old_rms, old_mass, old_code = [t.clone() for t in (mean, rms, mass, code)]
        code, mean, rms, mass, update = _resource_observer_chunk(
            code, mean, gradient / (step + 1), rms, mass, .125, .125)
        if step == 0:
            torch.testing.assert_close(mean, gradient, rtol=0, atol=0)
        assert bool(((mass > 0) & (mass <= 1)).all())
        assert bool(((mean / rms).square().mean((1, 2)) <= 1 + 1e-6).all())
        assert bool((update.square().sum((1, 2)) <= .125**2 * 64 + 1e-6).all())
        assert bool(((code >= 0) & (code < 585)).all())
        # Return-value state is complete; rerunning the same old state is identical.
        again = _resource_observer_chunk(old_code, old_mean, gradient / (step + 1),
                                          old_rms, old_mass, .125, .125)
        for actual, expected in zip(again, (code, mean, rms, mass, update)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if dtype == torch.float32:
        storage = sum(t.numel() * t.element_size() for t in (code, mean, rms, mass))
        assert storage == mean.numel() * 4.625  # layout cost, not a training peak


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resource_observer_has_defined_zero_energy_and_extreme_scale_behavior(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    code = torch.tensor([[552, 552]], dtype=torch.int16, device=device)
    mean = torch.zeros((1, 2, 4), device=device)
    rms = torch.zeros((1, 1, 1), device=device)
    mass = torch.zeros_like(rms)
    zero = _resource_observer_chunk(code, mean, mean, rms, mass, .125, .125)
    assert torch.equal(zero[1], mean) and torch.equal(zero[2], rms)
    assert torch.equal(zero[4], mean) and bool(torch.isfinite(zero[4]).all())
    base = torch.tensor([[[1., 2., -3., -4.], [2., 1., -4., -3.]]], device=device)
    reference = _resource_observer_chunk(code, mean, base, rms, mass, .125, .125)
    for factor in (1e-30, 1e30):
        scaled = _resource_observer_chunk(code, mean, base * factor, rms, mass, .125, .125)
        assert all(bool(torch.isfinite(t).all()) for t in scaled)
        assert torch.equal(scaled[0], reference[0])
        torch.testing.assert_close(scaled[4], reference[4], rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("feedback", [True, False])
def test_resource_optimizer_preserves_representation_and_full_resume(device, dtype, feedback):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.linspace(-9.125, 7.875, 131, dtype=dtype, device=device).requires_grad_()
    initial = p.detach().clone()
    first = ResourceCycle([p], .001, chunk_compartments=1, feedback=feedback)
    assert not first.state and torch.equal(p, initial)
    p.grad = torch.linspace(-.125, .25, 131, dtype=dtype, device=device)
    first.step()
    assert p.dtype == dtype and not torch.equal(p, initial)
    assert first.state[p]["mean"].dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    restored = p.detach().clone().requires_grad_()
    second = ResourceCycle([restored], .03, chunk_compartments=7)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    assert second.param_groups[0]["feedback"] is feedback
    # Changing workspace blocking must preserve the same joint recursion.
    second.param_groups[0]["chunk_compartments"] = 2
    for g in (None, torch.zeros_like(p), p.detach() / 32):
        p.grad = g
        restored.grad = None if g is None else g.clone()
        first.step(); second.step()
        torch.testing.assert_close(p, restored, rtol=0, atol=0)
        for name in ("phase", "row_reserve", "col_reserve", "mean", "rms", "mass"):
            torch.testing.assert_close(first.state[p][name], second.state[restored][name], rtol=0, atol=0)
    assert all(t.is_floating_point() for t in first.state[p].values())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_feedback_intervention_uses_equal_live_state_and_exact_integer_law(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.witness.validate_optimizer_cycle_integer import encode
    cases = list(product(range(-32, 33), range(8, 17), range(-64, 65)))
    code = torch.tensor([encode(k, r) for k, r, _ in cases], dtype=torch.int16, device=device)
    gradient = torch.tensor([[math.sqrt(64+c), math.sqrt(64-c), 0, 0] for _, _, c in cases],
                            dtype=torch.float32, device=device)
    actual, _ = _cycle_chunk(code, gradient, .125, feedback=False)
    expected = torch.tensor([encode(max(-32, min(32, (8*k+r*c+8)//16)),
                                   r+(19-r)//4-r*abs(k)//128) for k, r, c in cases],
                            dtype=torch.int16, device=device)
    assert torch.equal(actual, expected)
    layouts = []
    for feedback in (True, False):
        p = torch.zeros(128, device=device, requires_grad=True)
        optimizer = ResourceCycle([p], .001, feedback=feedback)
        p.grad = torch.linspace(-1., 1., 128, device=device)
        optimizer.step()
        state = optimizer.state[p]
        layouts.append({k: (tuple(v.shape), v.dtype, v.numel()*v.element_size()) for k,v in state.items()})
        assert set(state) == {"phase", "row_reserve", "col_reserve", "resource_pool", "mean", "rms", "mass"}
    assert layouts[0] == layouts[1]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resource_optimizer_corrects_padding_and_separates_missing_observations(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.tensor([3.14159], device=device, requires_grad=True)
    optimizer = ResourceCycle([p], .01)
    optimizer.step()
    assert not optimizer.state
    p.grad = torch.tensor([2.], device=device)
    old = p.detach().clone()
    stale_loss = p.square().sum()
    optimizer.step()
    torch.testing.assert_close(p, old-.01, rtol=0, atol=1e-6)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        stale_loss.backward()
    state = copy.deepcopy(optimizer.state_dict())
    old = p.detach().clone()
    p.grad = None
    optimizer.step()
    assert torch.equal(p, old)
    for key, value in state["state"][0].items():
        assert torch.equal(optimizer.state[p][key], value)
    # Zero is an explicit observation: old direction decays and still has a
    # signed effect. None has no such meaning and never manufactures zeros.
    p.grad = torch.zeros_like(p)
    optimizer.step()
    assert float(optimizer.state[p]["mean"][0, 0]) < 2
    assert float(p) < float(old)
    broken = copy.deepcopy(optimizer.state_dict())
    broken["state"][0]["rms"].zero_()
    with pytest.raises(ValueError, match="zero RMS"):
        optimizer.load_state_dict(broken)


def test_resource_optimizer_validates_all_gradients_before_any_mutation():
    p = torch.tensor([1.], requires_grad=True)
    q = torch.tensor([2.], requires_grad=True)
    opt = ResourceCycle([p, q])
    p.grad, q.grad = torch.tensor([1.]), torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="nonfinite gradient"):
        opt.step()
    assert not opt.state and float(p) == 1 and float(q) == 2
    for kwargs in ({"observer_rate": 0}, {"observer_rate": 2}, {"chunk_compartments": 0}):
        with pytest.raises(ValueError):
            ResourceCycle([p], **kwargs)


def test_outcome_modulation_uses_fresh_scalar_and_preserves_control_state():
    p = torch.zeros(4, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    modulated = ResourceCycle([p], lr=.01, adam_base=True,
                              outcome_modulation=True, outcome_deadband=1e-6)
    control = ResourceCycle([q], lr=.01, adam_base=True,
                            outcome_modulation=False, outcome_deadband=1e-6)
    p.grad = torch.ones_like(p)
    q.grad = torch.ones_like(q)
    modulated.step()
    control.step()
    predicted = modulated.param_groups[0]['last_pred_decrease']
    assert predicted > 0
    modulated.observe_outcome(predicted + 1.0)
    control.observe_outcome(predicted + 1.0)
    assert modulated.param_groups[0]['rho'] > 1.0
    assert control.param_groups[0]['rho'] == 1.0
    assert modulated.param_groups[0]['last_outcome_evidence'] == pytest.approx(1.0)
    restored = torch.zeros(4, requires_grad=True)
    other = ResourceCycle([restored], lr=.5, adam_base=True)
    other.load_state_dict(copy.deepcopy(modulated.state_dict()))
    assert other.param_groups[0]['outcome_modulation'] is True
    assert other.param_groups[0]['rho'] == modulated.param_groups[0]['rho']


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resource_optimizer_keeps_independent_coordinate_scales(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.zeros(4, device=device, requires_grad=True)
    optimizer = ResourceCycle([p], .001)
    p.grad = torch.tensor([1., 2., 4., 8.], device=device)
    optimizer.step()
    state = optimizer.state[p]
    torch.testing.assert_close(state["rms"].view(-1), p.grad, rtol=0, atol=0)
    assert all(value.is_floating_point() for value in state.values())
    assert sum(value.numel() * value.element_size() for value in state.values()) == 76


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resource_optimizer_allocates_actual_matrix_budgets(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.zeros((3, 5), dtype=torch.float64, device=device, requires_grad=True)
    p.grad = torch.arange(1, 16, dtype=p.dtype, device=device).reshape_as(p)
    optimizer = ResourceCycle([p], lr=.01, sinkhorn_steps=32)
    optimizer.step()
    allocation = -p.detach()/.01  # first-step direction is +1 and old reserves are 1
    torch.testing.assert_close(allocation.mean(dim=1), torch.ones(3, device=device, dtype=p.dtype), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(allocation.mean(dim=0), torch.ones(5, device=device, dtype=p.dtype), rtol=1e-6, atol=1e-6)
    assert optimizer.state[p]['phase'].shape == p.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matrix_saturated_phase_yields_to_sustained_contrary_evidence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.zeros((2, 2), device=device, requires_grad=True)
    optimizer = ResourceCycle([p], lr=.001)
    p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.state[p]['phase'][0, 0] = 1.
    optimizer.step()
    assert float(optimizer.state[p]['phase'][0, 0]) == 1.
    p.grad = torch.tensor([[0., 2.], [2., 2.]], device=device)
    optimizer.step()
    assert float(optimizer.state[p]['phase'][0, 0]) < 1.
    for _ in range(3):
        optimizer.step()
    assert float(optimizer.state[p]['phase'][0, 0]) <= 0.


@pytest.mark.parametrize("fused", [False, True])
def test_arbitrary_fp32_parameters_remain_unrestricted_and_backward_safe(fused):
    device = "cuda" if fused else "cpu"
    if fused and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Outside the previous material range and away from its seven-level grid.
    p = torch.tensor([-9.12345, .012345, .123456, 3.14159, 7.89123],
                     device=device, requires_grad=True)
    initial = p.detach().clone()
    opt = ConservativeCycle([p], .001, fused=fused)
    assert torch.equal(p, initial) and not opt.state
    stale_loss = p.square().sum()
    p.grad = torch.tensor([.13, -.17, .19, -.23, .29], device=device)
    opt.step()
    assert not torch.equal(p, initial)
    assert int(torch.unique(p.detach()).numel()) == p.numel()
    assert float(p.min()) < -9 and float(p.max()) > 7
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        stale_loss.backward()
    opt.zero_grad(set_to_none=False)
    p.square().sum().backward()
    torch.testing.assert_close(p.grad, 2*p.detach(), rtol=0, atol=0)


def _reference(phase, reserve, gradient, lr):
    g = [Fraction(float(value)) for value in gradient]
    diagonal, cross = g[0] ** 2 + g[3] ** 2, g[1] ** 2 + g[2] ** 2
    cue = 2 * (diagonal - cross) / (diagonal + cross) if diagonal + cross else Fraction(0)
    code = (32 * cue + Fraction(1, 2)).__floor__()
    new_phase = max(-32, min(32, (2 * (4 + reserve) * phase + reserve * code + 8) // 16))
    new_reserve = reserve + (19 - reserve) // 4 - reserve * abs(phase) // 128
    shares = [(32 + new_phase) / 64, (32 - new_phase) / 64] * 2
    shares[2], shares[3] = shares[3], shares[2]
    update = [float(Fraction(lr) * reserve / 16 * Fraction(share) * value)
              for share, value in zip(shares, g)]
    return new_phase + 32 + 65 * (new_reserve - 8), update


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kernel_matches_exact_rational_reference_and_final_smooth_decrease(dtype):
    gradients = [(0, 0, 0, 0), (1, 0, 0, 1), (0, -1, 1, 0),
                 (1, 2, -3, -4), (0.125, -0.25, 0.5, -1)]
    inputs = list(product([-32, -8, 0, 8, 32], [8, 12, 16], gradients))
    code = torch.tensor([k + 32 + 65 * (r - 8) for k, r, _ in inputs], dtype=torch.int16)
    g = torch.tensor([gradient for _, _, gradient in inputs], dtype=dtype)
    new_code, update = _cycle_chunk(code, g, 0.125)
    expected = [_reference(k, r, gradient, 0.125) for k, r, gradient in inputs]
    assert torch.equal(new_code, torch.tensor([e[0] for e in expected], dtype=torch.int16))
    torch.testing.assert_close(update, torch.tensor([e[1] for e in expected], dtype=dtype),
                               rtol=0, atol=0)
    decrease = (g * update - 0.5 * update.square()).sum(dim=1)
    assert bool((decrease >= (0.125 / 64) * g.square().sum(dim=1)).all())


def test_chunk_partition_padding_and_gradient_immutability():
    values = torch.linspace(-1, 1, 31, dtype=torch.float64)
    gradient = torch.linspace(0.125, 2, 31, dtype=torch.float64)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    a = ConservativeCycle([first], 0.125, chunk_cycles=1)
    b = ConservativeCycle([second], 0.125, chunk_cycles=100)
    first.grad, second.grad = gradient.clone(), gradient.clone()
    a.step()
    b.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(a.state[first]["code"], b.state[second]["code"])
    assert torch.equal(first.grad, gradient)
    assert set(a.state[first]) == {"code"}
    assert a.state[first]["code"].numel() * a.state[first]["code"].element_size() == 16


def test_absent_gradient_advances_regenerative_state_without_weight_write():
    parameter = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    optimizer = ConservativeCycle([parameter], 0.125)
    optimizer.state[parameter]["code"] = torch.tensor([8 + 32 + 65 * 8], dtype=torch.int16)
    old_parameter = parameter.detach().clone()
    optimizer.step()
    code = int(optimizer.state[parameter]["code"][0])
    assert code % 65 - 32 > 8
    assert code // 65 + 8 < 16
    assert torch.equal(parameter, old_parameter)


def test_cue_encoding_is_stable_for_tiny_and_large_finite_gradients():
    code = torch.tensor([552], dtype=torch.int16)
    g = torch.tensor([[1.0, 2.0, -3.0, -4.0]])
    expected, update = _cycle_chunk(code, g, 0.125)
    for scale in [1e-30, 1e30]:
        actual, scaled_update = _cycle_chunk(code, g * scale, 0.125)
        assert torch.equal(actual, expected)
        assert bool(torch.isfinite(scaled_update).all())
        torch.testing.assert_close(scaled_update / scale, update, rtol=2e-6, atol=0)


def test_full_parameter_trace_obeys_declared_stationary_quadratic_bound():
    # Analytic numerical contract, not a capability/convergence experiment.
    parameter = torch.tensor([1.0, -0.5, 0.25, -1.0, 2.0, -0.25, 0.5],
                             dtype=torch.float64, requires_grad=True)
    optimizer = ConservativeCycle([parameter], 0.125, chunk_cycles=1)
    initial_error = float(parameter.detach().square().sum() / 2)
    for step in range(1, 4):
        parameter.grad = parameter.detach().clone()
        optimizer.step()
        error = float(parameter.detach().square().sum() / 2)
        assert error <= (1 - 0.125 / 32) ** step * initial_error
        code = optimizer.state[parameter]["code"]
        assert bool(((code >= 0) & (code < 585)).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_restores_packed_bits_and_identical_future(dtype):
    parameter = torch.zeros(19, dtype=dtype, requires_grad=True)
    optimizer = ConservativeCycle([parameter], 0.125, chunk_cycles=2)
    parameter.grad = torch.arange(19, dtype=dtype) / 32
    optimizer.step()
    # Legal codes above 256 are not all exactly representable in BF16.
    optimizer.state[parameter]["code"].copy_(torch.tensor([257, 319, 383, 447, 583], dtype=torch.int16))
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored = parameter.detach().clone().requires_grad_()
    other = ConservativeCycle([restored], 0.5)
    other.load_state_dict(checkpoint)
    assert other.state[restored]["code"].dtype == torch.int16
    assert torch.equal(optimizer.state[parameter]["code"], other.state[restored]["code"])
    for sign in [1, -1, 0]:
        gradient = sign * torch.arange(19, dtype=dtype) / 16
        parameter.grad, restored.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        other.step()
        assert torch.equal(parameter, restored)
        assert torch.equal(optimizer.state[parameter]["code"], other.state[restored]["code"])


def test_invalid_gradient_rejected_before_any_mutation():
    first = torch.zeros(4, requires_grad=True)
    second = torch.ones(4, requires_grad=True)
    optimizer = ConservativeCycle([first, second], 0.125)
    first.grad = torch.ones_like(first)
    second.grad = torch.full_like(second, float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        optimizer.step()
    assert torch.equal(first, torch.zeros_like(first))
    assert not optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_integer_kernel_and_complete_continuation_match_cpu():
    cpu = torch.linspace(-0.5, 0.5, 35).requires_grad_()
    gpu = cpu.detach().cuda().requires_grad_()
    first, second = ConservativeCycle([cpu], 0.125, chunk_cycles=2), ConservativeCycle([gpu], 0.125, chunk_cycles=3)
    for scale in [1.0, -0.5, 0.0]:
        gradient = torch.arange(35, dtype=torch.float32) * (scale / 64)
        cpu.grad, gpu.grad = gradient.clone(), gradient.cuda()
        first.step()
        second.step()
        assert torch.equal(first.state[cpu]["code"], second.state[gpu]["code"].cpu())
        torch.testing.assert_close(cpu, gpu.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_cuda_matches_chunked_reference_on_nondefault_stream(dtype):
    pytest.importorskip("numba")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        first = torch.linspace(-1, 1, 519, device="cuda", dtype=dtype).requires_grad_()
        second = first.detach().clone().requires_grad_()
        a, b = ConservativeCycle([first], 0.125), ConservativeCycle([second], 0.125, fused=True)
        for scale in [1.0, -0.5, 0.0, None]:
            gradient = None if scale is None else torch.arange(519, device="cuda", dtype=dtype) * (scale / 1024)
            first.grad = gradient
            second.grad = gradient
            a.step()
            b.step()
        stream.synchronize()
        assert torch.equal(a.state[first]["code"], b.state[second]["code"])
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_integer_transition_exhausts_the_lean_reference_domain():
    pytest.importorskip("numba")
    from scripts.witness.validate_optimizer_cycle_integer import encode, integer_step

    # Center each cue bin, so this tests CUDA integer state execution rather
    # than ambiguous floating rounding at an encoder boundary.
    inputs = list(product(range(-32, 33), range(8, 17), range(-64, 65)))
    parameter = torch.zeros(4 * len(inputs), device="cuda", requires_grad=True)
    parameter.grad = torch.tensor(
        [[math.sqrt(64 + c), math.sqrt(64 - c), 0, 0] for _, _, c in inputs],
        dtype=torch.float32, device="cuda").view(-1)
    optimizer = ConservativeCycle([parameter], 0.125, fused=True)
    optimizer.state[parameter]["code"] = torch.tensor(
        [encode(k, r) for k, r, _ in inputs], dtype=torch.int16, device="cuda")
    expected = torch.tensor([encode(*integer_step(k, r, c)) for k, r, c in inputs],
                            dtype=torch.int16, device="cuda")
    optimizer.step()
    assert torch.equal(optimizer.state[parameter]["code"], expected)


def test_resource_cycle_auto_step_expands_on_consistent_gradients():
    p = torch.zeros(4, requires_grad=True)
    opt = ResourceCycle([p], lr=0.01, auto_step=True)
    initial_rho = opt.param_groups[0]["rho"]
    assert initial_rho == 1.0
    for _ in range(5):
        p.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        opt.step()
    # Consistent gradients along descent direction should expand trust region rho
    assert opt.param_groups[0]["rho"] > initial_rho


def test_resource_cycle_auto_step_deadband_prevents_zero_gradient_collapse():
    p = torch.zeros(4, requires_grad=True)
    opt = ResourceCycle([p], lr=0.01, auto_step=True)
    # Step once with gradient to establish initial prediction
    p.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
    opt.step()
    rho_before = opt.param_groups[0]["rho"]
    # Subsequent steps with zero gradients must trigger deadband without collapsing rho to zero
    for _ in range(20):
        p.grad = torch.zeros(4)
        opt.step()
    assert opt.param_groups[0]["rho"] == rho_before
    assert opt.param_groups[0]["rho"] >= opt.param_groups[0]["rho_min"] > 0


def test_resource_cycle_auto_step_strictly_respects_drift_floor():
    p = torch.zeros(4, requires_grad=True)
    opt = ResourceCycle([p], lr=0.01, auto_step=True, rho_min=1e-3, gamma_down=0.5)
    # Repeatedly simulate severe gradient reversals to force contraction
    for step in range(30):
        # Alternate signs drastically
        sign = 1.0 if step % 2 == 0 else -10.0
        p.grad = torch.full((4,), sign)
        opt.step()
    # Must remain strictly bounded at rho_min, never vanishing to 0
    assert opt.param_groups[0]["rho"] >= 1e-3


def test_resource_cycle_auto_step_preserves_full_state_continuation():
    p = torch.linspace(-2.0, 2.0, 16, requires_grad=True)
    opt1 = ResourceCycle([p], lr=0.01, auto_step=True)
    for _ in range(3):
        p.grad = torch.randn(16)
        opt1.step()

    # Checkpoint
    checkpoint = copy.deepcopy(opt1.state_dict())

    p2 = p.detach().clone().requires_grad_()
    opt2 = ResourceCycle([p2], lr=0.05, auto_step=True)
    opt2.load_state_dict(checkpoint)

    assert opt1.param_groups[0]["rho"] == opt2.param_groups[0]["rho"]
    assert opt1.param_groups[0]["last_pred_decrease"] == opt2.param_groups[0]["last_pred_decrease"]

    # Continued steps must be bitwise identical
    for _ in range(3):
        g = torch.randn(16)
        p.grad = g.clone()
        p2.grad = g.clone()
        opt1.step()
        opt2.step()
        torch.testing.assert_close(p, p2, rtol=0, atol=0)
        assert opt1.param_groups[0]["rho"] == opt2.param_groups[0]["rho"]


def test_resource_cycle_auto_step_contracts_on_overshoot():
    # In a valley, overshooting reverses the fresh gradient sign
    p = torch.tensor([5.0], requires_grad=True)
    opt = ResourceCycle([p], lr=0.1, auto_step=True)
    # Step 1: positive gradient (5.0) -> steps towards negative
    p.grad = torch.tensor([5.0])
    opt.step()
    assert opt.param_groups[0]["rho"] == 1.0
    # Step 2: simulate that the step was an overshoot so fresh gradient is reversed in sign
    p.grad = torch.tensor([-5.0])
    opt.step()
    # Trust region must contract
    assert opt.param_groups[0]["rho"] < 1.0


def test_auto_step_uses_allocated_previous_update_not_unweighted_mean():
    p = torch.zeros((2, 2), requires_grad=True)
    opt = ResourceCycle([p], lr=0.01, auto_step=True, deadband_eps=0)
    p.grad = torch.ones_like(p)
    opt.step()
    state = opt.state[p]
    # A feasible matrix allocation can reverse the aggregate feedback sign.
    state['mean'].fill_(1)
    state['rms'].fill_(1)
    state['phase'].copy_(torch.tensor([[1., -1.], [-1., 1.]]))
    p.grad = torch.tensor([[-1., 2.], [0., 0.]])
    opt.step()
    assert opt.param_groups[0]['rho'] < 1


def test_missing_gradient_clears_old_feedback_and_old_checkpoint_stays_fixed():
    p = torch.zeros(2, requires_grad=True)
    opt = ResourceCycle([p], auto_step=True)
    p.grad = torch.ones_like(p)
    opt.step()
    p.grad = None
    opt.step()
    assert 'previous_update' not in opt.state[p]
    checkpoint = copy.deepcopy(opt.state_dict())
    del checkpoint['param_groups'][0]['auto_step']
    restored = ResourceCycle([p], auto_step=True)
    restored.load_state_dict(checkpoint)
    assert not restored.param_groups[0]['auto_step']


@pytest.mark.parametrize('shape', [(4, 7), (9,)])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_feedback_reconstructs_previous_update_from_reserves(shape, device):
    from information_boltzmann.conservative_cycle_optimizer import _previous_resource_update
    p = torch.zeros(shape, device=device, requires_grad=True)
    opt = ResourceCycle([p], lr=.01, auto_step=True, sinkhorn_steps=7)
    p.grad = torch.linspace(-2, 3, p.numel(), device=device).reshape(shape)
    opt.step()
    state, group = opt.state[p], opt.param_groups[0]
    reconstructed = _previous_resource_update(state, p, group['previous_eff_lr'],
                                               group['previous_sinkhorn_steps'])
    torch.testing.assert_close(reconstructed, -p.detach(), rtol=0, atol=0)
    assert 'previous_update' not in state
    assert state['previous_row_reserve'].numel() + state['previous_col_reserve'].numel() == (
        (shape[0] + shape[1]) if len(shape) == 2 else (1 + shape[0]))


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_adam_base_without_competition_matches_torch_adam_and_resume(device):
    torch.manual_seed(37)
    p = torch.randn(4, 7, device=device, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    opt = ResourceCycle([p], lr=.001, adam_base=True, competition_strength=0)
    ref = torch.optim.Adam([q], lr=.001, weight_decay=0, foreach=False)
    for t in range(30):
        g = torch.randn_like(p) if t % 4 else torch.zeros_like(p)
        p.grad, q.grad = g.clone(), g.clone()
        opt.step(); ref.step()
        torch.testing.assert_close(p, q, rtol=1e-6, atol=2e-7)
    saved = copy.deepcopy(opt.state_dict())
    restored_p = p.detach().clone().requires_grad_()
    restored = ResourceCycle([restored_p], adam_base=True, competition_strength=0)
    restored.load_state_dict(saved)
    g = torch.randn_like(p)
    p.grad, restored_p.grad = g.clone(), g.clone()
    opt.step(); restored.step()
    torch.testing.assert_close(p, restored_p, rtol=0, atol=0)


def test_adam_base_matrix_competition_and_control_share_state():
    p = torch.zeros(3, 3, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    opt = ResourceCycle([p], adam_base=True, competition_strength=1, sinkhorn_steps=32)
    control = ResourceCycle([q], adam_base=True, competition_strength=0, sinkhorn_steps=32)
    for _ in range(3):
        g = torch.tensor([[3., 1., 2.], [1., 3., 2.], [2., 1., 3.]])
        p.grad, q.grad = g.clone(), g.clone()
        opt.step(); control.step()
    assert set(opt.state[p]) == set(control.state[q])
    assert not torch.equal(p, q)
    assert float(opt.last_budget_residual) < 1e-5


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_resource_latch_restores_perturbation_and_tracks_persistent_change(device):
    p = torch.zeros((2, 2), device=device, requires_grad=True)
    opt = ResourceCycle([p], lr=.01, latch_state=True, latch_strength=10)
    p.grad = torch.zeros_like(p)
    opt.step()
    opt.state[p]['phase'].fill_(1)
    with torch.no_grad():
        p[0, 0] = .1
    opt.step()
    assert 0 < float(p[0, 0]) < .1
    before = opt.state[p]['anchor'][0, 0].item()
    for _ in range(30):
        p.grad = torch.tensor([[-1., 0.], [0., 0.]], device=device)
        opt.step()
    assert opt.state[p]['anchor'][0, 0].item() > before
    saved = copy.deepcopy(opt.state_dict())
    q = p.detach().clone().requires_grad_()
    other = ResourceCycle([q], lr=.01, latch_state=True, latch_strength=10)
    other.load_state_dict(saved)
    p.grad = torch.ones_like(p); q.grad = p.grad.clone()
    opt.step(); other.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_resource_fatigue_control_depletes_and_recovers_same_state(device):
    p = torch.zeros((2, 3), device=device, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    on = ResourceCycle([p], lr=.001, adam_base=True, feedback=False, resource_fatigue=True)
    off = ResourceCycle([q], lr=.001, adam_base=True, feedback=False, resource_fatigue=False)
    for _ in range(10):
        p.grad = torch.tensor([[1., 1., 1.], [0., 0., 0.]], device=device)
        q.grad = p.grad.clone()
        on.step(); off.step()
    assert set(on.state[p]) == set(off.state[q])
    assert float(on.state[p]['row_reserve'][0]) < 1
    assert float(on.state[p]['resource_pool']) > 0
    assert torch.equal(off.state[q]['row_reserve'], torch.ones_like(off.state[q]['row_reserve']))
    total = on.state[p]['row_reserve'].sum() + on.state[p]['col_reserve'].sum() + on.state[p]['resource_pool']
    assert float(total) == pytest.approx(5.0, abs=2e-6)
    low = on.state[p]['row_reserve'].clone()
    on.state[p]['mean'].zero_(); on.state[p]['rms'].zero_()
    for _ in range(20):
        p.grad = torch.zeros_like(p)
        on.step()
    assert float(on.state[p]['row_reserve'][0]) > float(low[0])


def test_calibrated_outcome_scores_are_centered_and_bounded():
    p = torch.zeros(4, requires_grad=True)
    opt = ResourceCycle([p], lr=.01, adam_base=True,
                        outcome_modulation=True, calibrated_outcome=True,
                        outcome_ema=.25, calibrated_outcome_rate=.2,
                        outcome_deadband=1e-8)
    values = [1.0, -1.0, 1.0, -1.0] * 8
    for value in values:
        p.grad = torch.ones_like(p)
        opt.step()
        opt.observe_outcome(value)
        group = opt.param_groups[0]
        assert group['rho_min'] <= group['rho'] <= group['rho_max']
        assert 0 <= group['outcome_probability'] <= 1
    group = opt.param_groups[0]
    assert group['outcome_score_count'] == len(values)
    assert group['outcome_probability'] == pytest.approx(.5, abs=.08)
    assert group['rho'] > group['rho_min']
    saved = copy.deepcopy(opt.state_dict())
    q = p.detach().clone().requires_grad_()
    restored = ResourceCycle([q], lr=.01, adam_base=True,
                             outcome_modulation=True, calibrated_outcome=True)
    restored.load_state_dict(saved)
    assert restored.param_groups[0]['outcome_score_count'] == len(values)
    assert restored.param_groups[0]['outcome_probability'] == pytest.approx(
        group['outcome_probability'])


def test_calibrated_outcome_gates_existing_phase_without_extra_parameter_state():
    p = torch.zeros((2, 2), requires_grad=True)
    q = p.detach().clone().requires_grad_()
    gated = ResourceCycle([p], lr=.01, adam_base=True,
                          outcome_modulation=True, calibrated_outcome=True)
    full = ResourceCycle([q], lr=.01, adam_base=True,
                         outcome_modulation=True, calibrated_outcome=True)
    gated.param_groups[0]['outcome_gate'] = 0.0
    full.param_groups[0]['outcome_gate'] = 1.0
    p.grad = torch.tensor([[2., 0.], [0., 0.]])
    q.grad = p.grad.clone()
    gated.step(); full.step()
    assert torch.equal(gated.state[p]['phase'], torch.zeros_like(gated.state[p]['phase']))
    assert not torch.equal(full.state[q]['phase'], torch.zeros_like(full.state[q]['phase']))
    assert set(gated.state[p]) == set(full.state[q])


def test_calibrated_latch_holds_phase_then_releases_after_contrary_run():
    p = torch.zeros((2, 2), requires_grad=True)
    opt = ResourceCycle([p], lr=.01, adam_base=True,
                        outcome_modulation=True, calibrated_outcome=True,
                        latch_state=True, latch_window=2,
                        outcome_ema=.1, calibrated_outcome_rate=0.0)
    group = opt.param_groups[0]
    # Make two clearly positive calibrated outcomes arm the latch.
    group['outcome_probability'] = .9
    group['outcome_score_ema'] = -1.0
    group['outcome_score_count'] = 1
    opt.observe_outcome(1.0)
    opt.observe_outcome(1.0)
    assert group['latch_active'] is True
    p.grad = torch.tensor([[2., 0.], [0., 0.]])
    opt.step()
    held = opt.state[p]['phase'].clone()
    p.grad = torch.tensor([[0., 2.], [0., 0.]])
    opt.step()
    torch.testing.assert_close(opt.state[p]['phase'], held, rtol=0, atol=0)
    # Two strong contrary outcomes release the latch while sensing continues.
    group['outcome_score_ema'] = 1.0
    opt.observe_outcome(0.0)
    opt.observe_outcome(0.0)
    assert group['latch_active'] is False
    p.grad = torch.tensor([[0., 2.], [0., 0.]])
    opt.step()
    assert not torch.equal(opt.state[p]['phase'], held)


def test_calibrated_latch_release_with_reachable_score_baseline():
    p = torch.zeros((2, 2), requires_grad=True)
    opt = ResourceCycle([p], adam_base=True, calibrated_outcome=True,
                        latch_state=True, latch_window=2, outcome_ema=.125)
    group = opt.param_groups[0]
    for _ in range(30):
        opt.observe_outcome(1.0)
    assert group['latch_active']
    opt.observe_outcome(-1.0)
    assert group['latch_active'] and group['latch_contrary_run'] == 1
    opt.observe_outcome(-1.0)
    assert not group['latch_active']
    # A moving score baseline can falsely re-arm on consistently bad outcomes.
    for _ in range(5):
        opt.observe_outcome(-1.0)
    assert group['latch_active']


@pytest.mark.parametrize('anchor,phase_latch,rho_control', [
    (False, False, False), (True, False, False),
    (False, True, False), (False, False, True)])
def test_calibrated_controls_are_independent_and_resume(anchor, phase_latch, rho_control):
    p = torch.zeros((2, 2), requires_grad=True)
    kwargs = dict(adam_base=True, calibrated_outcome=True, latch_state=False,
                  anchor_enabled=anchor, phase_latch_enabled=phase_latch,
                  outcome_rho_enabled=rho_control, latch_window=2)
    opt = ResourceCycle([p], **kwargs)
    p.grad = torch.tensor([[1., 0.], [0., 0.]])
    opt.step()
    assert ('anchor' in opt.state[p]) == anchor
    group = opt.param_groups[0]
    assert group['last_outcome_prediction_error'] == 0
    opt.observe_outcome(1.0)
    assert group['last_outcome_prediction_error'] == pytest.approx(.5)
    for _ in range(10):
        opt.observe_outcome(1.0)
    assert group['latch_active'] == phase_latch
    assert (group['rho'] != 1.0) == rho_control
    state = copy.deepcopy(opt.state_dict())
    q = p.detach().clone().requires_grad_()
    restored = ResourceCycle([q], **kwargs)
    restored.load_state_dict(state)
    p.grad = torch.ones_like(p); q.grad = p.grad.clone()
    opt.step(); restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
