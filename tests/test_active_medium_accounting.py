"""Numerical update/accounting invariants, not synthetic capability studies."""
import copy

import numpy as np
import pytest
import torch

from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer


class NumericalBelief:
    def detach(self):
        return self


class NumericalParameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.active = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor([3., 4., 5.], dtype=torch.float64),
                                         requires_grad=False)
        self.disconnected = torch.nn.Parameter(torch.tensor([6.], dtype=torch.float64))
        self.unoptimized = torch.nn.Parameter(torch.tensor([7.], dtype=torch.float64))


def numerical_learner():
    model = NumericalParameters()
    optimizer = torch.optim.AdamW([model.active], lr=.01, weight_decay=.1)
    return ActiveMediumTrainer(model, optimizer, NumericalBelief(), carry_token=1,
                               event_duration=.005, chunk_tokens=2, tokens_per_update=4)


def full_parameter_norm(model):
    return torch.stack([torch.linalg.vector_norm(p).double().square()
                        for p in model.parameters()]).sum().sqrt()


def test_in_place_restore_retains_none_gradient_adamw_semantics():
    learner = numerical_learner()
    saved = copy.deepcopy(learner.state_dict())
    learner.model.active.grad = torch.ones_like(learner.model.active)
    learner.load_state_dict(saved)
    assert learner.model.active.grad is None
    before = learner.model.active.detach().clone()
    learner.optimizer.step()
    torch.testing.assert_close(learner.model.active, before, atol=0, rtol=0)


def test_health_snapshot_excludes_immutable_parameters_and_keeps_full_model_norm(monkeypatch):
    learner = numerical_learner()
    model = learner.model
    model.active.grad = torch.tensor([.3, -.4], dtype=torch.float64)
    model.unoptimized.grad = torch.ones_like(model.unoptimized)
    excluded = {p.data_ptr() for p in (model.frozen, model.disconnected, model.unoptimized)}
    original_clone = torch.Tensor.clone

    def checked_clone(value, *args, **kwargs):
        assert value.data_ptr() not in excluded
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'clone', checked_clone)
    before = learner._health_update_snapshot()
    assert len(before) == 1 and before[0][0] is model.active
    learner.optimizer.step()
    update_norm, parameter_norm = learner._health_update_norms(before)
    torch.testing.assert_close(update_norm, torch.linalg.vector_norm(model.active - before[0][1]),
                               atol=0, rtol=0)
    torch.testing.assert_close(parameter_norm, full_parameter_norm(model), atol=0, rtol=0)
    assert parameter_norm > torch.linalg.vector_norm(model.active)


def test_health_cache_preserves_zero_gradient_momentum_and_invalidates_external_changes(monkeypatch):
    learner = numerical_learner()
    model = learner.model
    original_norm = torch.linalg.vector_norm
    frozen_reads = []

    def counted_norm(value, *args, **kwargs):
        if value.data_ptr() == model.frozen.data_ptr():
            frozen_reads.append(value._version)
        return original_norm(value, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'vector_norm', counted_norm)
    model.active.grad = torch.tensor([.3, -.4], dtype=torch.float64)
    before = learner._health_update_snapshot()
    learner.optimizer.step()
    learner._health_update_norms(before)
    assert len(frozen_reads) == 1
    model.active.grad.zero_()
    before = learner._health_update_snapshot()
    assert len(before) == 1
    learner.optimizer.step()
    update_norm, _ = learner._health_update_norms(before)
    assert update_norm > 0  # Adam momentum and decay still change zero-grad weights.
    assert len(frozen_reads) == 1
    with torch.no_grad():
        model.frozen.mul_(2)
    _, parameter_norm = learner._health_update_norms([])
    assert len(frozen_reads) == 2
    torch.testing.assert_close(parameter_norm, full_parameter_norm(model), atol=0, rtol=0)


def install_numerical_chunk(monkeypatch, records):
    import information_boltzmann.runtime.active_medium_training as execution

    def numerical_chunk(model, observed, targets, state, *, return_loss_components=False, **unused):
        token_nll = model.active.square().mean() + targets.double()
        nll = token_nll.mean()
        port_nll = observed.double().mean()
        policy_kl = nll.new_tensor(.125)
        port_objective = port_nll + policy_kl
        joint = nll + port_objective
        components = dict(task_nll=nll.detach(), port_nll=port_nll.detach(),
                          policy_kl=policy_kl, port_objective=port_objective.detach(),
                          joint_objective=joint.detach())
        records.append((targets.numel(), {k: float(v) for k, v in components.items()}))
        result = joint, state, nll, token_nll
        return (*result, components) if return_loss_components else result

    monkeypatch.setattr(execution, 'quiet_training_chunk', numerical_chunk)


def test_component_ledger_weights_partial_chunks_and_resumes_pending_credit(monkeypatch):
    records = []
    install_numerical_chunk(monkeypatch, records)
    learner = numerical_learner()
    prior = np.zeros(12)
    learner.consume(torch.tensor([2, 3, 4]), phase='numerical_contract', prior_nll=prior)
    report = learner.summary()['loss_components']
    for name in records[0][1]:
        expected = sum(count * row[name] for count, row in records) / 3
        assert report[name] == pytest.approx(expected)
        assert report['component_events'][name] == 3
    assert learner.pending == 3
    restored = numerical_learner()
    restored.model.load_state_dict(copy.deepcopy(learner.model.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(learner.optimizer.state_dict()))
    restored.load_state_dict(copy.deepcopy(learner.state_dict()))
    assert restored.summary() == learner.summary()
    expected = learner.consume(torch.tensor([5, 6]), phase='numerical_contract', prior_nll=prior)
    actual = restored.consume(torch.tensor([5, 6]), phase='numerical_contract', prior_nll=prior)
    assert actual == expected
    assert restored.summary() == learner.summary()
    torch.testing.assert_close(restored.model.active, learner.model.active, atol=0, rtol=0)


def test_legacy_learner_starts_component_coverage_at_first_new_chunk(monkeypatch):
    records = []
    install_numerical_chunk(monkeypatch, records)
    learner = numerical_learner()
    learner.consume(torch.tensor([2, 3]), phase='numerical_contract', prior_nll=np.zeros(12))
    saved = copy.deepcopy(learner.state_dict())
    for key in ('loss_component_sums', 'loss_component_events', 'last_loss_components'):
        saved.pop(key)
    restored = numerical_learner()
    restored.load_state_dict(saved)
    assert restored.events == 2
    assert restored.summary()['loss_components'] == {'component_events': {}}
    restored.consume(torch.tensor([4]), phase='numerical_contract', prior_nll=np.zeros(12))
    assert restored.events == 3
    assert set(restored.summary()['loss_components']['component_events'].values()) == {1}


def test_optional_components_have_their_actual_coverage_and_do_not_retain_graphs():
    learner = numerical_learner()
    loss = learner.model.active.square().mean()
    learner._record_loss_components(loss, loss / 2, 2)
    learner._record_loss_components(loss, loss / 2, 1, {'policy_kl': loss / 4})
    report = learner.summary()['loss_components']
    assert report['component_events']['task_nll'] == 3
    assert report['component_events']['policy_kl'] == 1
    assert report['policy_kl'] == pytest.approx(float((loss / 4).detach()))
    assert all(isinstance(value, float) for value in learner.last_loss_components.values())


def test_captured_interface_and_eager_record_same_components_without_cuda(monkeypatch):
    records = []
    install_numerical_chunk(monkeypatch, records)
    import information_boltzmann.runtime.active_medium_training as execution

    eager, captured = numerical_learner(), numerical_learner()

    class NumericalCapture:
        def backward(self, observed, targets, belief):
            loss, output, nll, self.token_nll, self.loss_components = execution.quiet_training_chunk(
                captured.model, observed, targets, belief, return_loss_components=True)
            (loss * (targets.numel() / captured.tokens_per_update)).backward()
            return loss.detach(), output, nll.detach()

    captured.captured = NumericalCapture()
    targets, prior = torch.tensor([2, 3, 4, 5]), np.zeros(12)
    assert captured.consume(targets, phase='numerical_contract', prior_nll=prior) == eager.consume(
        targets, phase='numerical_contract', prior_nll=prior)
    assert captured.summary() == eager.summary()
    torch.testing.assert_close(captured.model.active, eager.model.active, atol=0, rtol=0)
