"""Deterministic full-window numerical execution tests, not capability tasks."""
import copy

import pytest
import torch

from information_boltzmann.runtime import training
from test_structural_auxiliary_replay import candidate, initial, same_graph_reference


@pytest.mark.parametrize('tokens,frozen', [(4, False), (32, True)])
def test_shared_primal_preserves_full_window_gradients_and_writer_scope(tokens, frozen, monkeypatch):
    torch.set_num_threads(1)
    reference = candidate()
    reference.source.embedding.weight.requires_grad_(not frozen)
    actual = copy.deepcopy(reference)
    actual.shared_credit_execution = True
    ids = (torch.arange(tokens) % 12 + 1)[None]
    targets = ((torch.arange(tokens) + 2) % 12 + 1)[None]
    start_a, start_b = initial(reference), initial(actual)
    start_a.medium.field.requires_grad_(True)
    start_b.medium.field.requires_grad_(True)
    expected = same_graph_reference(reference, ids, targets, start_a)
    def forbidden(*args, **kwargs):
        raise AssertionError('Separate writer replay was not eliminated')
    monkeypatch.setattr(training, '_writer_auxiliary_gradients', forbidden)
    count = 0
    original = training.training_event
    def measured(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'training_event', measured)
    rng = torch.random.get_rng_state().clone()
    result = training.quiet_training_chunk(actual, ids, targets, start_b,
        event_duration=.005, activation_checkpointing=True,
        checkpoint_granularity='event', return_token_nll=True)
    assert count == tokens
    for a, b in zip(expected[:1] + expected[2:], result[:1] + result[2:]):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    for a, b in zip(training.belief_tensors(expected[1]), training.belief_tensors(result[1])):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for net in (reference, actual):
        for p in net.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, .0003)
    (expected[0] * .37).backward()
    (result[0] * .37).backward()
    assert count == 2 * tokens
    torch.testing.assert_close(start_a.medium.field.grad, start_b.medium.field.grad,
                               atol=3e-11, rtol=3e-9)
    for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-11, rtol=3e-9, msg=name)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_shared_primal_live_external_prefix_preserves_fallback():
    torch.set_num_threads(1)
    reference = candidate()
    actual = copy.deepcopy(reference)
    actual.shared_credit_execution = True
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    expected = same_graph_reference(reference, ids, targets, initial(reference, live_prefix=True))
    result = training.quiet_training_chunk(actual, ids, targets, initial(actual, live_prefix=True),
        event_duration=.005, activation_checkpointing=True, checkpoint_granularity='event')
    expected[0].backward()
    result[0].backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-11, rtol=3e-9, msg=name)
