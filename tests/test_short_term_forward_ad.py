"""Exact STP forward-mode path, with CUDA checks explicitly opt-in."""
import os

import pytest
import torch
from torch.autograd import forward_ad

from information_boltzmann.core.short_term_plasticity import LocalShortTermPlasticity


def setup(device='cpu', dtype=torch.float32):
    torch.manual_seed(819)
    rule = LocalShortTermPlasticity(2).to(device=device, dtype=dtype)
    field = torch.randn(1, 2, 2, 2, 4, device=device, dtype=dtype)
    flux = tuple(torch.randn_like(field) for _ in range(3))
    state = .2 + .6 * torch.rand(1, 2, 2, 2, 3, 2, device=device, dtype=dtype)
    duration = field.new_tensor(.037)
    rates = .3 + torch.rand(2, 2, 2, 3, 3, device=device, dtype=dtype)
    baseline = .2 + .6 * torch.rand(2, 2, 2, 3, device=device, dtype=dtype)
    return rule, (state, field, *flux, duration, rates, baseline)


def apply(function, values):
    return function(values[0], values[1], values[2:5], values[5], values[6:8])


@pytest.mark.parametrize('dual_index', range(8))
@pytest.mark.parametrize('zero_tangent', [False, True])
def test_every_dual_input_bypasses_cuda_gate_without_gpu_allocation(monkeypatch, dual_index, zero_tangent):
    rule, values = setup()
    compiled_calls = []
    def cached(*arguments):
        compiled_calls.append(True)
        return rule.native_step(*arguments)
    rule._compiled_step = cached
    # Exercise the CUDA dispatch predicate on real CPU tensors; this test never
    # constructs a CUDA tensor or compiles a GPU graph.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda unused: True))
    apply(rule, values)
    assert len(compiled_calls) == 1  # ordinary primal retains the fused route
    with forward_ad.dual_level():
        inputs = list(values)
        tangent = (torch.zeros_like(inputs[dual_index]) if zero_tangent
                   else torch.randn_like(inputs[dual_index]))
        inputs[dual_index] = forward_ad.make_dual(inputs[dual_index], tangent)
        expected = forward_ad.unpack_dual(apply(rule.native_step, inputs))
        actual = forward_ad.unpack_dual(apply(rule, inputs))
        torch.testing.assert_close(actual.primal, expected.primal, rtol=0, atol=0)
        torch.testing.assert_close(actual.tangent, expected.tangent, rtol=0, atol=0)
        assert torch.isfinite(actual.tangent).all()
    assert len(compiled_calls) == 1


def test_complete_stp_jvp_matches_fp64_finite_difference():
    rule, values = setup(dtype=torch.float64)
    tangents = tuple(.1 * torch.randn_like(value) for value in values)
    with forward_ad.dual_level():
        dual = [forward_ad.make_dual(value, tangent) for value, tangent in zip(values, tangents)]
        actual = forward_ad.unpack_dual(apply(rule, dual))
        primal, tangent = actual.primal.clone(), actual.tangent.clone()
    epsilon = 1e-6
    positive = apply(rule.native_step, [value + epsilon * tangent for value, tangent in zip(values, tangents)])
    negative = apply(rule.native_step, [value - epsilon * tangent for value, tangent in zip(values, tangents)])
    torch.testing.assert_close(primal, apply(rule.native_step, values), rtol=0, atol=0)
    torch.testing.assert_close(tangent, (positive - negative) / (2 * epsilon), rtol=2e-7, atol=1e-10)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation; resource calibration owns GPU')
@pytest.mark.parametrize('dual_index', range(8))
def test_cuda_dual_matches_exact_native_and_bypasses_warm_aot_cache(dual_index):
    rule, values = setup('cuda')
    def forbidden(*unused):
        raise AssertionError('AOT path does not support the forward-mode contract')
    rule._compiled_step = forbidden
    with forward_ad.dual_level():
        dual = list(values)
        dual[dual_index] = forward_ad.make_dual(dual[dual_index], torch.randn_like(dual[dual_index]))
        expected = forward_ad.unpack_dual(apply(rule.native_step, dual))
        actual = forward_ad.unpack_dual(apply(rule, dual))
        torch.testing.assert_close(actual.primal, expected.primal, rtol=0, atol=0)
        torch.testing.assert_close(actual.tangent, expected.tangent, rtol=0, atol=0)
