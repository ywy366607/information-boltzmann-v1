"""Algebraic deferred-gradient and full recurrent update contracts."""
import copy

import pytest
import torch

from information_boltzmann.core.deferred_linear_credit import DeferredWriterCredit
from information_boltzmann.runtime import training
from test_structural_auxiliary_replay import candidate, initial, same_graph_reference


@pytest.mark.parametrize('tokens', [4, 32])
def test_deferred_gemm_preserves_joint_credit_pending_gradients_and_boundary_vjp(tokens):
    torch.set_num_threads(1)
    reference = candidate()
    reference.source.embedding.weight.requires_grad_(False)
    actual = copy.deepcopy(reference)
    actual.shared_credit_execution = actual.deferred_writer_credit = True
    actual.deferred_credit = DeferredWriterCredit(actual.write_agent, actual.readout)
    ids = (torch.arange(tokens) % 12 + 1)[None]
    targets = ((torch.arange(tokens) + 2) % 12 + 1)[None]
    a, b = initial(reference), initial(actual)
    a.medium.field.requires_grad_(True)
    b.medium.field.requires_grad_(True)
    expected = same_graph_reference(reference, ids, targets, a)
    result = training.quiet_training_chunk(actual, ids, targets, b,
        event_duration=.005, activation_checkpointing=True, checkpoint_granularity='event')
    for net in (reference, actual):
        for p in net.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, .0003)
    (expected[0] * .37).backward()
    (result[0] * .37).backward()
    assert actual.deferred_credit.factors
    with pytest.raises(RuntimeError):
        actual.deferred_credit.assert_flushed()
    actual.flush_deferred_credit()
    actual.deferred_credit.assert_flushed()
    torch.testing.assert_close(a.medium.field.grad, b.medium.field.grad, atol=3e-11, rtol=3e-9)
    for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-11, rtol=3e-9, msg=name)
    assert actual.deferred_credit.factor_rows > tokens


def test_batched_spatial_linear_uses_standard_gradient_and_unused_none_is_preserved():
    torch.set_num_threads(1)
    linear = torch.nn.Sequential(torch.nn.Linear(7, 5)).double()
    reference = copy.deepcopy(linear)
    collector = DeferredWriterCredit(linear)
    inputs = torch.linspace(-1., 1., 21, dtype=torch.float64).reshape(3, 7)
    reference(inputs).square().sum().backward()
    linear(inputs).square().sum().backward()
    assert not collector.factors
    collector.flush()
    for p, q in zip(reference.parameters(), linear.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_partial_window_phase_bridge_and_resume_flush_exact_credit_before_adam():
    import numpy as np
    from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
    from information_boltzmann.runtime.optimization import make_medium_optimizer
    torch.set_num_threads(1)
    def learner(deferred):
        model = candidate()
        model.source.embedding.weight.requires_grad_(False)
        model.medium.structural_posterior.window_active.fill_(False)
        model.shared_credit_execution = True
        model.deferred_writer_credit = deferred
        if deferred:
            model.deferred_credit = DeferredWriterCredit(model.write_agent, model.readout)
        return ActiveMediumTrainer(model, make_medium_optimizer(model, lr=2e-4),
            initial(model), carry_token=1, event_duration=.005, chunk_tokens=2,
            tokens_per_update=4, activation_checkpointing=True, checkpoint_granularity='event')
    reference, actual = learner(False), learner(True)
    prior = np.zeros(13)
    for net in (reference, actual):
        torch.manual_seed(63)
        net.consume(torch.tensor([2, 3, 4]), phase='train_first_pass', prior_nll=prior)
    assert actual.pending == 3 and actual.optimizer_updates == 0
    actual.model.deferred_credit.assert_flushed()
    restored = learner(True)
    restored.model.load_state_dict(copy.deepcopy(actual.model.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(actual.optimizer.state_dict()))
    restored.belief = copy.deepcopy(actual.belief)
    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
    for net in (reference, actual, restored):
        net.consume(torch.tensor([5]), phase='revisit_bridge', prior_nll=prior)
        assert net.pending == 0 and net.optimizer_updates == 1
    for p, q, r in zip(reference.model.parameters(), actual.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(p, q, atol=3e-11, rtol=3e-9)
        torch.testing.assert_close(q, r, atol=0, rtol=0)


@pytest.mark.parametrize('input_credit', [False, True])
def test_aot_compiled_linear_retains_deferred_side_effect_and_exact_weight_credit(input_credit):
    torch.set_num_threads(1)
    original = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.Tanh(),
                                   torch.nn.Linear(5, 3)).double()
    actual = copy.deepcopy(original)
    collector = DeferredWriterCredit(actual)
    operation = torch.compile(actual, backend='aot_eager', fullgraph=True)
    left = torch.linspace(-.3, .5, 7, dtype=torch.float64).requires_grad_(input_credit)
    right = left.detach().clone().requires_grad_(input_credit)
    original(left).square().sum().backward()
    operation(right).square().sum().backward()
    assert len(collector.factors) == 2
    collector.flush()
    if input_credit:
        torch.testing.assert_close(left.grad, right.grad, atol=3e-12, rtol=3e-12)
    for p, q in zip(original.parameters(), actual.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-12, rtol=3e-12)
