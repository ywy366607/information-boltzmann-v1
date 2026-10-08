"""Numerical gradients and storage lifetime of complete nested physical state."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from information_boltzmann.core.plastic_medium import MediumState
from information_boltzmann.core.plastic_ports import PlasticBelief
from information_boltzmann.core.state_checkpoint import checkpoint_state
from information_boltzmann.core.temporal_probes import TemporalProbeState
from information_boltzmann.runtime.training import belief_tensors


def state(optional=True):
    shape = (1, 2, 2, 2, 3)
    field = torch.linspace(-.3, .4, 24, dtype=torch.float64).reshape(shape)

    def leaf(value):
        return value.clone().requires_grad_(True)

    medium = MediumState(leaf(field), tuple(leaf(field * (i + 1) / 4) for i in range(3)),
        leaf(torch.tensor([123.], dtype=torch.float64)),
        leaf(torch.full((1, 2, 2, 2, 3), .1, dtype=torch.float64)) if optional else None,
        leaf(torch.full((1, 2, 2, 2, 2, 3), .2, dtype=torch.float64)) if optional else None,
        leaf(torch.full((1, 2, 2, 2, 3, 2), .3, dtype=torch.float64)) if optional else None)
    history = TemporalProbeState(leaf(torch.complex(field[:, 0], field[:, 1])),
        leaf(torch.tensor([123.], dtype=torch.float64))) if optional else None
    return PlasticBelief(medium, leaf(torch.ones(1, 3, dtype=torch.float64)), history)


def evolve(current, duration, update_optional=True):
    def update(value):
        return None if value is None else .97 * value.sin() + .01 * duration

    medium = current.medium
    outgoing = replace(medium, field=update(medium.field),
        flux=tuple(update(value) for value in medium.flux),
        elapsed=medium.elapsed + duration,
        conduction=update(medium.conduction) if update_optional else medium.conduction,
        receptors=update(medium.receptors) if update_optional else medium.receptors,
        transmission=update(medium.transmission) if update_optional else medium.transmission)
    history = None if current.temporal is None else TemporalProbeState(
        update(current.temporal.value), current.temporal.elapsed + duration)
    return PlasticBelief(outgoing, update(current.precision), history)


@pytest.mark.parametrize('optional', [False, True])
def test_full_state_nested_checkpoints_preserve_all_values_and_leaf_gradients(optional):
    direct, checked = state(optional), state(optional)
    left_duration = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    right_duration = left_duration.detach().clone().requires_grad_(True)

    def event(current, duration):
        for _ in range(5):
            current = checkpoint_state(evolve, current, duration, True)
        return current

    expected, actual = direct, checked
    for _ in range(4):
        for _ in range(5):
            expected = evolve(expected, left_duration)
        actual = checkpoint_state(event, actual, right_duration)
    for a, b in zip(belief_tensors(expected), belief_tensors(actual)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    expected_loss = sum(value.abs().square().sum() for value in belief_tensors(expected))
    actual_loss = sum(value.abs().square().sum() for value in belief_tensors(actual))
    expected_gradients = torch.autograd.grad(expected_loss, (*belief_tensors(direct), left_duration))
    actual_gradients = torch.autograd.grad(actual_loss, (*belief_tensors(checked), right_duration))
    for a, b in zip(expected_gradients, actual_gradients):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def nested_storage_lifetimes(flattened):
    initial = state()
    duration = .03
    references, externally_saved = [], []

    def pack(value):
        externally_saved.append((value.dtype, tuple(value.shape), weakref.ref(value)))
        return value

    def run(function, current, *args):
        if flattened:
            return checkpoint_state(function, current, *args)
        return checkpoint(function, current, *args, use_reentrant=False, preserve_rng_state=False)

    def event(current):
        for _ in range(8):
            references.append(weakref.ref(current.medium.field))
            current = run(evolve, current, duration)
        return current

    output = initial
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        for _ in range(4):
            output = run(event, output)
    gc.collect()
    retained = len({id(reference()) for reference in references if reference() is not None})
    output_value = output.medium.field.detach().clone()
    output.medium.field.square().sum().backward()
    return retained, externally_saved, output_value, initial.medium.field.grad


def test_tensor_hooks_replace_nested_observer_storage_with_event_boundaries():
    old_count, _, old_value, old_gradient = nested_storage_lifetimes(False)
    new_count, saved, new_value, new_gradient = nested_storage_lifetimes(True)
    # Non-tensor dataclass arguments capture every observer input strongly.
    # Flattening lets the outer hook replace inner input saves with holders.
    assert old_count == 32
    assert new_count == 4
    torch.testing.assert_close(old_value, new_value, atol=0, rtol=0)
    torch.testing.assert_close(old_gradient, new_gradient, atol=0, rtol=0)
    # Each outer event saves every real state tensor via hooks. Depending on
    # checkpoint context, Torch also exposes a storage-free zero-sized dummy.
    real_saved = [entry for entry in saved if entry[1] != (0,)]
    assert len(real_saved) == 4 * len(belief_tensors(state()))
    assert any(dtype == torch.complex128 for dtype, _, _ in saved)
    assert sum(dtype == torch.float64 and shape == (1,) for dtype, shape, _ in saved) == 8


def test_unsupported_opaque_state_cannot_hide_tensor_references():
    class Opaque:
        def __init__(self):
            self.tensor = torch.ones(3, requires_grad=True)

    with pytest.raises(TypeError, match='Unsupported checkpoint-state component'):
        checkpoint_state(lambda current: current.tensor.sin(), Opaque())
