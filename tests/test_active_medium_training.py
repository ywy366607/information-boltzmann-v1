"""Causal scoring and full learning continuation, not capability experiments."""
import copy
import math
import os

import numpy as np
import torch
import pytest

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk
from information_boltzmann.runtime.training import CapturedPlasticChunk


def make_learner(health=False):
    torch.manual_seed(21)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=4,
                                hidden=8, bath_type='conductance', activity_adaptation=True,
                                short_term_plasticity=True).double()
    optimizer = make_medium_optimizer(model, lr=2e-4)
    from information_boltzmann.runtime.medium_health import MediumHealthAuditor
    auditor = MediumHealthAuditor(model, window_tokens=16, block_tokens=2) if health else None
    return ActiveMediumTrainer(model, optimizer, model.initial_belief(),
                               carry_token=1, event_duration=.005,
                               chunk_tokens=2, tokens_per_update=4, health=auditor)


def test_token_scores_match_one_forward_and_target_cannot_change_its_state():
    learner = make_learner()
    model = learner.model
    state = learner.belief
    observed = torch.tensor([[1, 2]])
    loss, evolved, nll, scores = quiet_training_chunk(
        model, observed, torch.tensor([[2, 3]]), state, event_duration=.005,
        return_token_nll=True)
    alternative = quiet_training_chunk(model, observed, torch.tensor([[2, 8]]),
                                      state, event_duration=.005, return_token_nll=True)
    torch.testing.assert_close(scores.mean(), nll, atol=0, rtol=0)
    torch.testing.assert_close(scores[:, :1], alternative[3][:, :1], atol=0, rtol=0)
    for left, right in zip(belief_tensors(evolved), belief_tensors(alternative[1])):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    result = learner.consume(torch.tensor([2, 3]), phase='train_first_pass',
                             prior_nll=np.full(17, math.log(17)))
    torch.testing.assert_close(torch.tensor([r[2] for r in result], dtype=scores.dtype),
                               scores.detach().flatten(), atol=0, rtol=0)
    assert learner.optimizer_updates == 0 and learner.pending == 2


def test_resume_releases_only_completed_zero_decoder_gradient():
    learner = make_learner()
    learner.model.decoder.weight.grad = torch.zeros_like(learner.model.decoder.weight)
    saved = copy.deepcopy(learner.state_dict())
    restored = make_learner()
    restored.load_state_dict(saved)
    assert restored.model.decoder.weight.grad is None
    saved['pending_gradients']['decoder.weight'].fill_(1)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.model.decoder.weight.grad,
                               torch.ones_like(restored.model.decoder.weight))
    saved['pending_gradients']['decoder.weight'].zero_()
    saved['pending'] = 1
    restored.load_state_dict(saved)
    assert restored.model.decoder.weight.grad is not None


def test_context_changes_preserve_pending_credit_cadence_and_full_belief():
    learner = make_learner()
    prior = np.full(17, math.log(17))
    a = learner.consume(torch.tensor([2, 3]), phase='train_first_pass', prior_nll=prior)
    before = float(learner.belief.medium.elapsed[0])
    b = learner.consume(torch.tensor([7, 8]), phase='fresh_B', prior_nll=prior)
    assert b[0][0] == a[-1][1] and learner.optimizer_updates == 1
    learner.consume(torch.tensor([1]), phase='revisit_bridge', prior_nll=prior)
    replay = learner.consume(torch.tensor([2, 3]), phase='revisit_A', prior_nll=prior)
    assert replay[0][:2] == a[0][:2]
    assert learner.pending == 3 and learner.events == 7
    assert float(learner.belief.medium.elapsed[0]) == before + .025
    assert all(torch.isfinite(value).all() for value in belief_tensors(learner.belief))
    assert learner.summary()['phases']['fresh_B']['events'] == 2


def test_checkpoint_resumes_pending_gradients_and_optimizer_exactly():
    learner = make_learner()
    prior = np.full(17, math.log(17))
    learner.consume(torch.tensor([2, 3, 4, 5, 6]), phase='train_first_pass', prior_nll=prior)
    assert learner.pending == 1 and learner.optimizer_updates == 1
    restored = make_learner()
    restored.model.load_state_dict(copy.deepcopy(learner.model.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(learner.optimizer.state_dict()))
    restored.belief = copy.deepcopy(learner.belief)
    restored.load_state_dict(copy.deepcopy(learner.state_dict()))
    left = learner.consume(torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior)
    right = restored.consume(torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior)
    assert left == right
    assert learner.summary() == restored.summary()
    for a, b in zip(learner.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(belief_tensors(learner.belief), belief_tensors(restored.belief)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_mixed_bridge_revisit_chunk_keeps_cadence_and_phase_accounting():
    learner = make_learner()
    result = learner.consume(torch.tensor([7, 2, 3, 4]),
        phase=['revisit_bridge', 'revisit_A', 'revisit_A', 'revisit_A'],
        prior_nll=np.full(17, math.log(17)))
    assert learner.pending == 0 and learner.optimizer_updates == 1
    assert [r[4] for r in result] == ['revisit_bridge'] + ['revisit_A'] * 3
    assert learner.summary()['phases']['revisit_bridge']['events'] == 1
    assert learner.summary()['phases']['revisit_A']['events'] == 3


def test_finite_large_medium_gradient_uses_stable_clip_without_losing_direction(monkeypatch):
    import information_boltzmann.runtime.active_medium_training as training
    class Belief:
        def detach(self):
            return self
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    learner = ActiveMediumTrainer(model, optimizer, Belief(), carry_token=0,
                                  event_duration=1., chunk_tokens=2, tokens_per_update=2)
    def numerical_chunk(model, observed, targets, state, **unused):
        loss = (model.weight * model.weight.new_tensor([[1e20, 2e20]])).sum()
        return loss, state, loss.detach(), torch.zeros_like(targets, dtype=loss.dtype)
    monkeypatch.setattr(training, 'quiet_training_chunk', numerical_chunk)
    learner.consume(torch.tensor([1, 2]), phase='numerical_contract', prior_nll=np.zeros(3))
    assert learner.optimizer_updates == 1
    assert math.isfinite(learner.last_gradient_norm) and learner.last_gradient_norm > 1e20
    torch.testing.assert_close(model.weight, torch.tensor([[-.1/math.sqrt(5), -.2/math.sqrt(5)]]))


def test_bounded_update_norm_workspace_measures_same_actual_parameter_change():
    class Belief:
        def detach(self):
            return self
    torch.set_num_threads(1)
    model = torch.nn.Linear(1048593, 2, bias=False)
    learner = ActiveMediumTrainer(model, torch.optim.SGD(model.parameters(), lr=.1),
        Belief(), carry_token=0, event_duration=1.)
    parameter = model.weight
    previous = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(torch.linspace(-.001, .002, parameter.numel()).reshape_as(parameter))
    change, magnitude = learner._health_update_norms([(parameter, previous)])
    expected = torch.linalg.vector_norm((parameter.detach() - previous).double())
    torch.testing.assert_close(change, expected, atol=1e-6, rtol=3e-6)
    torch.testing.assert_close(magnitude, torch.linalg.vector_norm(parameter).double(), atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation')
def test_capture_token_scores_gradients_and_adam_update_match_eager():
    torch.manual_seed(21)
    graph_model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=4,
                                      hidden=8, bath_type='conductance',
                                      activity_adaptation=True, short_term_plasticity=True).cuda()
    eager_model = copy.deepcopy(graph_model)
    ids, targets = torch.tensor([[1, 2]], device='cuda'), torch.tensor([[2, 3]], device='cuda')
    initial = graph_model.initial_belief()
    graph = CapturedPlasticChunk(graph_model, ids, targets, initial, event_duration=.005)
    _, actual, _ = graph.backward(ids, targets, initial)
    loss, expected, _, scores = quiet_training_chunk(
        eager_model, ids, targets, initial, event_duration=.005, return_token_nll=True)
    loss.backward()
    torch.testing.assert_close(graph.token_nll, scores, atol=3e-6, rtol=3e-5)
    for left, right in zip(graph_model.parameters(), eager_model.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-4)
    for left, right in zip(belief_tensors(actual), belief_tensors(expected)):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    for model in (graph_model, eager_model):
        optimizer = make_medium_optimizer(model, lr=2e-4)
        from information_boltzmann.core.gradient_norms import stable_clip_grad_norm_
        stable_clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    for left, right in zip(graph_model.parameters(), eager_model.parameters()):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)


def test_bptt_health_records_each_token_without_changing_learning_and_resumes():
    monitored, reference = make_learner(True), make_learner()
    prior = np.full(17, math.log(17))
    targets = torch.tensor([2, 3, 4, 5, 6])
    phases = ['train_first_pass'] * 2 + ['fresh_B'] * 2 + ['revisit_A']
    assert monitored.consume(targets, phase=phases, prior_nll=prior) == reference.consume(
        targets, phase=phases, prior_nll=prior)
    for p, q in zip(monitored.model.parameters(), reference.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    report = monitored.health.summary()
    assert report['events'] == 5
    assert report['energy']['max_abs_continuity_residual'] == 0
    assert report['energy']['max_abs_write_residual'] < 2e-12
    assert report['energy']['max_abs_evolution_residual'] < 2e-12
    assert len(report['recent_optimizer_updates']) == 1
    assert [row['phase'] for row in monitored.health.rows] == phases
    assert len({row['energy_after'] for row in monitored.health.rows}) > 1
    restored = make_learner(True)
    restored.model.load_state_dict(copy.deepcopy(monitored.model.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(monitored.optimizer.state_dict()))
    restored.belief = copy.deepcopy(monitored.belief)
    restored.load_state_dict(copy.deepcopy(monitored.state_dict()))
    assert restored.consume(torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior) == monitored.consume(
        torch.tensor([7, 8, 9]), phase='fresh_B', prior_nll=prior)
    assert restored.health.state_dict() == monitored.health.state_dict()


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation')
def test_chunk_health_cuda_replay_matches_eager_per_token_observations():
    from information_boltzmann.runtime.medium_health import ChunkHealthCapture
    torch.manual_seed(21)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=4,
        hidden=8, bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True).cuda()
    eager_model = copy.deepcopy(model)
    ids = torch.tensor([[1, 2]], device='cuda')
    targets = torch.tensor([[2, 3]], device='cuda')
    initial = model.initial_belief()
    graph_health, eager_health = ChunkHealthCapture(model, 2), ChunkHealthCapture(eager_model, 2)
    graph = CapturedPlasticChunk(model, ids, targets, initial, event_duration=.005,
                                 health_capture=graph_health)
    for current_ids in (ids, torch.tensor([[4, 5]], device='cuda')):
        graph.zero_grad()
        graph.backward(current_ids, targets, initial)
        eager_model.zero_grad(set_to_none=True)
        loss, _, _, _ = quiet_training_chunk(eager_model, current_ids, targets, initial,
            event_duration=.005, health_capture=eager_health, return_token_nll=True)
        loss.backward()
        for name in ('event_values', 'event_features', 'event_decode'):
            torch.testing.assert_close(getattr(graph_health, name), getattr(eager_health, name),
                                       atol=3e-6, rtol=3e-5)
        for left, right in zip(model.parameters(), eager_model.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-4)


def test_event_activation_recomputation_preserves_state_scores_and_gradients():
    plain, recomputed = make_learner(True), make_learner(True)
    recomputed.activation_checkpointing = True
    prior = np.full(17, math.log(17))
    targets = torch.tensor([2, 3, 4, 5])
    assert plain.consume(targets, phase='train_first_pass', prior_nll=prior) == recomputed.consume(
        targets, phase='train_first_pass', prior_nll=prior)
    for p, q in zip(plain.model.parameters(), recomputed.model.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    for p, q in zip(belief_tensors(plain.belief), belief_tensors(recomputed.belief)):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert plain.health.state_dict() == recomputed.health.state_dict()


def test_compiled_event_with_recomputation_matches_joint_gradients_and_health(monkeypatch):
    import information_boltzmann.runtime.training as execution
    compiled = torch.compile(execution.training_event, backend='aot_eager', fullgraph=True, dynamic=False)
    monkeypatch.setattr(execution, '_compiled_training_event', compiled)
    plain, optimized = make_learner(True), make_learner(True)
    optimized.compile_event = True
    plain.activation_checkpointing = optimized.activation_checkpointing = True
    prior = np.full(17, math.log(17))
    a = plain.consume(torch.tensor([2, 3, 4, 5]), phase='train_first_pass', prior_nll=prior)
    b = optimized.consume(torch.tensor([2, 3, 4, 5]), phase='train_first_pass', prior_nll=prior)
    np.testing.assert_allclose([r[2] for r in a], [r[2] for r in b], rtol=1e-10, atol=1e-11)
    for p, q in zip(plain.model.parameters(), optimized.model.parameters()):
        torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-9)
    for p, q in zip(belief_tensors(plain.belief), belief_tensors(optimized.belief)):
        torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(list(plain.health.energy_totals.values()),
                               list(optimized.health.energy_totals.values()), rtol=1e-10, atol=1e-11)
