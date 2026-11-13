"""Per-layer policy isolation and shared inference boundary checks."""
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.native_state_policy import apply_native_boundary, configure_native_policy


def body():
    layers = [SimpleNamespace(recurrence='decay_then_delta', conv_kernel=4,
                              share_size=0, matrix_size=6) for _ in range(4)]
    return SimpleNamespace(model=SimpleNamespace(base=SimpleNamespace(layers=layers)))


def test_alternating_layers_clear_only_S_preserve_conv_and_other_layers():
    engine = body()
    state = torch.arange(1., 41.).reshape(4, 1, 10).requires_grad_()
    original = state.detach().clone()
    configure_native_policy(engine, (0, 2))
    result = apply_native_boundary(engine, state)
    assert torch.equal(state.detach(), original)
    assert torch.count_nonzero(result[[0, 2], :, :6]) == 0
    assert torch.equal(result[[0, 2], :, 6:], state[[0, 2], :, 6:])
    assert torch.equal(result[[1, 3]], state[[1, 3]])
    result.sum().backward()
    assert torch.count_nonzero(state.grad[[0, 2], :, :6]) == 0
    assert torch.all(state.grad[[1, 3]] == 1)
    assert torch.all(state.grad[[0, 2], :, 6:] == 1)


def test_all_carry_is_identity_and_default_has_no_effect():
    engine = body()
    state = torch.ones(4, 1, 10)
    assert apply_native_boundary(engine, state) is state
    configure_native_policy(engine)
    assert apply_native_boundary(engine, state) is state


def test_mismatched_training_and_inference_period_fails():
    engine = body()
    configure_native_policy(engine, (0, 2))
    with pytest.raises(ValueError, match='periods differ'):
        apply_native_boundary(engine, torch.ones(4, 1, 10), window=64)


def test_invalid_layer_or_exchange_state_is_rejected():
    engine = body()
    with pytest.raises(ValueError):
        configure_native_policy(engine, (4,))
    configure_native_policy(engine, (0,))
    engine.model.base.layers[0].recurrence = 'contact_exchange'
    with pytest.raises(ValueError, match='exchange-state'):
        apply_native_boundary(engine, torch.ones(4, 1, 10))
