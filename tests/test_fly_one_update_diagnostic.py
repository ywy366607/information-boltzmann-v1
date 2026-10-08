"""Numerical diagnostic identities, not synthetic capability experiments."""
import copy

import torch

from test_fly_bptt_learning import make_model, physical
from scripts.ib import diagnose_fly_one_update as audit
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner


def test_checkpointed_event_preserves_forward_state_and_all_surrogate_gradients(tmp_path, monkeypatch):
    torch.manual_seed(23)
    model = make_model(tmp_path)
    state = physical(model)
    state.h.copy_(torch.linspace(.01, .08, model.n_neurons)[None])
    state.ring = tuple(torch.ones_like(state.h)*.4 for _ in range(4))
    native = FlyBPTTLearner(model, state, learn_stp=True, settle_ticks=1)
    replay = FlyBPTTLearner(copy.deepcopy(model), copy.deepcopy(state), learn_stp=True, settle_ticks=1)
    ids, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    native_scores, native_state, _ = native.forward_window(ids, targets)
    native_scores.mean().backward()
    monkeypatch.setattr(audit.learning, 'advance_fly_input_event', audit.checkpoint_event)
    replay_scores, replay_state, _ = replay.forward_window(ids, targets)
    replay_scores.mean().backward()
    torch.testing.assert_close(replay_scores, native_scores, rtol=0, atol=0)
    for a, b in zip(audit.flatten_state(native_state), audit.flatten_state(replay_state)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (name, a), (_, b) in zip(native.model.named_parameters(), replay.model.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-6, atol=2e-8, msg=name)
    assert torch.equal(native.model.topographic_writer.a_adapt, replay.model.topographic_writer.a_adapt)


def test_actual_adam_displacement_mean_covariance_identity():
    torch.manual_seed(4)
    weight = torch.nn.Parameter(torch.randn(7, 3, dtype=torch.float64))
    optimizer = torch.optim.AdamW([weight], lr=.002, weight_decay=.01)
    # A genuine preceding update creates nonzero Adam history.
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    a, h = torch.randn(5, 7, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    gradient = a.T@h/5
    before = weight.detach().clone()
    weight.grad = gradient.clone()
    optimizer.step()
    parts = audit.linear_gradient_parts(a, h)
    measured = audit.displacement_accounting(before, weight, gradient, parts)
    common = a.mean(0)[:, None]*h.mean(0)[None]
    covariance = gradient-common
    delta = weight.detach()-before
    assert parts['identity_relative_error'] < 1e-13
    torch.testing.assert_close(torch.tensor(measured['common_gradient_dot_update']), (common*delta).sum().float())
    torch.testing.assert_close(torch.tensor(measured['covariance_gradient_dot_update']), (covariance*delta).sum().float())
    assert abs(measured['gradient_dot_actual_update']-float((gradient*delta).sum())) < 1e-14


def test_common_head_decomposition_is_only_a_descriptive_output_split():
    torch.manual_seed(5)
    h = torch.randn(4, 3)
    weights = {'output_read.weight': torch.randn(2, 3), 'read_norm.weight': torch.ones(2),
               'decoder.weight': torch.randn(5, 2), 'decoder.bias': torch.randn(5)}
    targets = torch.tensor([0, 1, 2, 3])
    description, _, logits = audit.head_scores(h, weights, targets)
    expected = torch.nn.functional.cross_entropy(logits.mean(0, keepdim=True).expand_as(logits), targets)
    assert abs(description['own_window_common_nll']-float(expected)) < 1e-7
