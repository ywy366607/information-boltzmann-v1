"""Checkpoint granularity changes storage/replay, not the learning objective."""
import copy

import pytest
import torch

from information_boltzmann.runtime.training import quiet_training_chunk, belief_tensors
from test_medium_structural_integration import candidate


def test_event_only_replay_matches_nested_and_full_storage_all_gradients():
    torch.set_num_threads(1)
    original = candidate()
    noise = torch.randn_like(original.medium.structural_posterior.mean)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    results = []
    for enabled, granularity in ((False, 'nested'), (True, 'nested'), (True, 'event')):
        model = copy.deepcopy(original)
        model.medium.structural_posterior.begin_window(noise)
        initial = model.initial_belief()
        initial.medium.field.requires_grad_(True)
        output = quiet_training_chunk(model, ids, targets, initial,
            event_duration=.023, activation_checkpointing=enabled,
            checkpoint_granularity=granularity)
        output[0].backward()
        results.append((output[0].detach(),
            [v.detach() for v in belief_tensors(output[1])],
            {n: None if p.grad is None else p.grad.detach().clone()
             for n, p in model.named_parameters()}, initial.medium.field.grad))
    reference = results[0]
    for actual in results[1:]:
        torch.testing.assert_close(actual[0], reference[0], atol=1e-11, rtol=1e-10)
        for a, b in zip(actual[1], reference[1]):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
        for name, gradient in actual[2].items():
            expected = reference[2][name]
            assert (gradient is None) == (expected is None), name
            if gradient is not None:
                torch.testing.assert_close(gradient, expected, atol=2e-10, rtol=2e-9, msg=name)
        torch.testing.assert_close(actual[3], reference[3], atol=2e-10, rtol=2e-9)


def test_unknown_checkpoint_granularity_is_rejected():
    model = candidate()
    with pytest.raises(ValueError, match='granularity'):
        quiet_training_chunk(model, torch.tensor([[1]]), torch.tensor([[2]]),
            model.initial_belief(), event_duration=.023, checkpoint_granularity='truncate')
