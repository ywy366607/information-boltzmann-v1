"""First-order numerical/replay contracts, not training capability evidence."""
from dataclasses import dataclass

import pytest
import torch

from information_boltzmann.core.state_checkpoint import checkpoint_state_vjp


@dataclass(frozen=True)
class NumericalState:
    field: torch.Tensor
    history: torch.Tensor
    elapsed: torch.Tensor
    unused: torch.Tensor
    optional: torch.Tensor | None = None


def state():
    return NumericalState(
        torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True),
        torch.tensor([.1 + .2j, -.2 + .3j], dtype=torch.complex128, requires_grad=True),
        torch.tensor([123.], dtype=torch.float64, requires_grad=True),
        torch.tensor([.7], dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('scale', [.37, -1.25])
def test_32_events_full_state_complex_clock_and_all_parameter_vjps(scale):
    direct, checked = state(), state()
    p = torch.nn.Parameter(torch.tensor([.8, .6], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    unused_copy = torch.nn.Parameter(unused.detach().clone())
    left_dt = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    right_dt = left_dt.detach().clone().requires_grad_(True)
    # Shared nonleaf graphs must be explicit arguments, not callable captures.
    prepared, prepared_copy = p.square(), q.square()

    def event(current, duration, material, parameter):
        field = (current.field * material + parameter * duration).sin()
        history = .7 * current.history + torch.complex(field, field * duration)
        out = NumericalState(field, history, current.elapsed + duration,
                             current.unused, current.optional)
        return out, {'score': field.square().sum(), 'audit': field.detach(),
                     'flag': torch.tensor(True), 'optional': None}

    left, right = direct, checked
    left_scores, right_scores = [], []
    # Parameters are read as actual module leaves, not detached normal args.
    left_event = lambda current, dt, material: event(current, dt, material, p)
    right_event = lambda current, dt, material: event(current, dt, material, q)
    for _ in range(32):
        left, info = left_event(left, left_dt, prepared)
        left_scores.append(info['score'])
        right, info = checkpoint_state_vjp(right_event, right, right_dt, prepared_copy,
                                           parameters=(q, unused_copy))
        right_scores.append(info['score'])
        assert not info['flag'].requires_grad and info['optional'] is None
    for name in ('field', 'history', 'elapsed', 'unused'):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), atol=0, rtol=0)
    assert right.elapsed.dtype == torch.float64 and right.history.dtype == torch.complex128
    left_loss = torch.stack(left_scores).mean() + left.history.abs().square().sum() + left.elapsed.sum()
    right_loss = torch.stack(right_scores).mean() + right.history.abs().square().sum() + right.elapsed.sum()
    left_targets = (direct.field, direct.history, direct.elapsed, direct.unused, left_dt, p, unused)
    right_targets = (checked.field, checked.history, checked.elapsed, checked.unused, right_dt, q, unused_copy)
    expected = torch.autograd.grad(scale * left_loss, left_targets, allow_unused=True)
    actual = torch.autograd.grad(scale * right_loss, right_targets, allow_unused=True)
    for a, b in zip(expected, actual):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)


def test_duplicate_parameter_input_and_outputs_count_each_path_once():
    p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def event(current, duplicate):
        value = current * duplicate + p.square()
        return {'value': value, 'duplicate': (value, value)}

    result = checkpoint_state_vjp(event, p, p, parameters=(p, p))
    loss = result['value'] + 2 * result['duplicate'][0] + 3 * result['duplicate'][1]
    gradient = torch.autograd.grad(loss, p)[0]
    torch.testing.assert_close(gradient, torch.tensor(48., dtype=torch.float64), atol=0, rtol=0)


def test_unused_output_grad_is_none_and_detached_float_output_has_no_vjp():
    p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def event(current):
        return current * p, p / 0, p.detach() * 3

    used, unused, detached = checkpoint_state_vjp(event, torch.tensor(3.), parameters=(p,))
    torch.testing.assert_close(torch.autograd.grad(used, p, retain_graph=True)[0], p.new_tensor(3.))
    assert torch.autograd.grad(detached, p, allow_unused=True)[0] is None
    assert torch.isinf(unused)


def test_saved_tensor_hooks_do_not_replace_module_parameter_targets():
    p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    x = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda value: value.detach().clone(), lambda value: value):
        output = checkpoint_state_vjp(lambda current: current * p.square(), x, parameters=(p,))
    dx, dp = torch.autograd.grad(output, (x, p))
    torch.testing.assert_close(dx, torch.tensor(4., dtype=torch.float64))
    torch.testing.assert_close(dp, torch.tensor(12., dtype=torch.float64))


def test_changed_output_shape_and_parameter_version_are_rejected():
    p = torch.nn.Parameter(torch.tensor(2.))
    calls = []

    def changes_shape(current):
        calls.append(None)
        return (current * p).expand(len(calls))

    result = checkpoint_state_vjp(changes_shape, torch.tensor(3.), parameters=(p,))
    with pytest.raises(RuntimeError, match='output structure changed'):
        result.sum().backward()
    result = checkpoint_state_vjp(lambda current: current * p, torch.tensor(3.), parameters=(p,))
    with torch.no_grad():
        p.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation|parameter changed'):
        result.backward()


def test_original_leaf_parameters_and_first_order_contract():
    p = torch.nn.Parameter(torch.tensor(2.))
    with pytest.raises(TypeError, match='original module leaf'):
        checkpoint_state_vjp(lambda current: current, p, parameters=(p.square(),))
    result = checkpoint_state_vjp(lambda current: current * p.square(), torch.tensor(3.), parameters=(p,))
    derivative = torch.autograd.grad(result, p, create_graph=True)[0]
    assert not derivative.requires_grad
    with torch.no_grad():
        direct = checkpoint_state_vjp(lambda current: {'result': current + 1}, p, parameters=(p,))
    assert not direct['result'].requires_grad


def test_shared_prepared_graph_receives_all_event_cotangents_once():
    p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    prepared = p.square()
    x = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    event = lambda current, coefficient: current * coefficient
    result = checkpoint_state_vjp(event, x, prepared, parameters=(p,))
    result = checkpoint_state_vjp(event, result, prepared, parameters=(p,))
    dx, dp = torch.autograd.grad(result, (x, p))
    torch.testing.assert_close(dx, torch.tensor(16., dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(dp, torch.tensor(96., dtype=torch.float64), atol=0, rtol=0)


def test_parameter_gradient_hooks_cannot_silently_run_twice():
    parameter = torch.nn.Parameter(torch.tensor(2.))
    handle = parameter.register_hook(lambda value: value * .5)
    try:
        with pytest.raises(ValueError, match='parameter gradient hooks'):
            checkpoint_state_vjp(lambda current: current * parameter,
                                 torch.tensor(3.), parameters=(parameter,))
    finally:
        handle.remove()
