"""Numerical clipping checks, independent of learned model capability."""

import pytest
import torch

from information_boltzmann.core.gradient_norms import (
    stable_grad_norm, stable_clip_grad_norm_,
)


def parameter_with_grad(values):
    parameter = torch.nn.Parameter(torch.zeros_like(values))
    parameter.grad = values.clone()
    return parameter


def test_finite_entries_with_overflowing_fp32_norm_keep_direction():
    values = torch.tensor([3e20, -4e20])
    parameter = parameter_with_grad(values)
    assert torch.isfinite(parameter.grad).all()
    assert torch.isinf(parameter.grad.norm())
    norm = stable_clip_grad_norm_([parameter], 1.0, chunk_elements=1)
    assert float(norm) == pytest.approx(5e20, rel=1e-6)
    torch.testing.assert_close(parameter.grad, torch.tensor([.6, -.8]))


def test_extreme_finite_entries_do_not_zero_clipped_direction():
    values = torch.full((1024,), torch.finfo(torch.float32).max)
    parameter = parameter_with_grad(values)
    norm = stable_clip_grad_norm_([parameter], 1.0, chunk_elements=17)
    assert torch.isfinite(norm)
    torch.testing.assert_close(parameter.grad, torch.full_like(values, 1 / 32))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_entries_fail_without_mutating_other_gradients(value):
    healthy = parameter_with_grad(torch.tensor([3., 4.]))
    broken = parameter_with_grad(torch.tensor([value]))
    before = healthy.grad.clone()
    with pytest.raises(FloatingPointError):
        stable_clip_grad_norm_([healthy, broken], 1.)
    torch.testing.assert_close(healthy.grad, before)


def test_ordinary_clipping_matches_pytorch_and_chunk_size_is_numerical_only():
    torch.manual_seed(14)
    parameters = [parameter_with_grad(torch.randn(37)), parameter_with_grad(torch.randn(53))]
    reference = [parameter_with_grad(p.grad) for p in parameters]
    n1 = stable_grad_norm(parameters, chunk_elements=7)
    n2 = stable_grad_norm(parameters, chunk_elements=1024)
    torch.testing.assert_close(n1, n2)
    stable_clip_grad_norm_(parameters, .3, chunk_elements=7)
    torch.nn.utils.clip_grad_norm_(reference, .3)
    for actual, expected in zip(parameters, reference):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_empty_zero_and_unclipped_gradients():
    assert stable_grad_norm([]) == 0
    parameter = parameter_with_grad(torch.zeros(7))
    assert stable_clip_grad_norm_([parameter], 1.) == 0
    parameter.grad.fill_(.01)
    before = parameter.grad.clone()
    stable_clip_grad_norm_([parameter], 1.)
    assert torch.equal(parameter.grad, before)


def test_tiny_gradients_and_missing_gradients_are_preserved():
    parameter = parameter_with_grad(torch.full((7,), 1e-38))
    unused = torch.nn.Parameter(torch.ones(3))
    expected = torch.linalg.vector_norm(parameter.grad.double())
    torch.testing.assert_close(stable_grad_norm([unused, parameter]), expected,
                               atol=0, rtol=1e-12)
    before = parameter.grad.clone()
    stable_clip_grad_norm_([unused, parameter], 1.)
    assert torch.equal(parameter.grad, before)
    assert unused.grad is None


def test_sparse_gradients_are_explicitly_rejected():
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0, 2]]),
                                             torch.tensor([1., 2.]), (3,))
    with pytest.raises(ValueError, match="dense real"):
        stable_clip_grad_norm_([parameter], 1.)
