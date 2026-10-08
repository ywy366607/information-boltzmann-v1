"""Numerical/accounting checks only; no synthetic capability experiments."""
import copy
import os

import numpy as np
import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.lifelong_evaluation import LiveOnlineEvaluation
from information_boltzmann.runtime.local_credit import LocalPlasticTrainer
from information_boltzmann.runtime.medium_health import (
    MediumHealthAuditor, conditional_state_response, representation_summary, risk_trend, spatial_energy)
from information_boltzmann.runtime.online_credit import credit_tensors, replace_credit
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk


def model_and_runner(health=True, device='cpu'):
    torch.manual_seed(53)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8, hidden=8,
                                bath_type='conductance', activity_adaptation=True,
                                short_term_plasticity=True).to(device)
    if device == 'cpu':
        model.double()
    learner = LocalPlasticTrainer(model, event_duration=.005)
    auditor = MediumHealthAuditor(model, window_tokens=12, block_tokens=2) if health else None
    if auditor is not None:
        learner.health_capture = auditor.capture
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    runner = LiveOnlineEvaluation(learner, optimizer, tokens_per_update=3,
                                 window_tokens=4, health=auditor)
    return model, runner


def test_observer_preserves_complete_state_joint_gradients_and_learning():
    model, monitored = model_and_runner()
    reference_model, reference = model_and_runner(False)
    for left, right in ((1, 2), (2, 3), (3, 4), (4, 5)):
        a = monitored.step(torch.tensor([left]), torch.tensor([right]))
        b = reference.step(torch.tensor([left]), torch.tensor([right]))
        torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
        for p, q in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for p, q in zip(belief_tensors(monitored.learner.belief), belief_tensors(reference.learner.belief)):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    report = monitored.health.summary()
    assert report['decoding']['feature_rms_before_norm']['latest'] > 0
    assert report['decoding']['feature_rms_after_norm']['latest'] > 0
    assert report['decoding']['logit_standard_deviation']['latest'] > 0
    assert 0 < report['decoding']['prediction_entropy_nats']['latest'] <= np.log(17)
    assert report['events'] == 4
    assert report['energy']['max_abs_write_residual'] < 2e-12
    assert report['energy']['max_abs_evolution_residual'] < 2e-12
    assert abs(report['energy']['cumulative_balance_residual']) < 2e-12
    assert report['energy']['max_abs_continuity_residual'] == 0
    assert report['recent_optimizer_updates'][0]['actual_update_norm'] > 0
    changes = report['structure']['window_operator_changes']
    assert abs(changes['transport_energy_change']) < 2e-12
    assert abs(changes['collision_energy_change']) < 2e-12


def test_spatial_dc_ac_uses_channel_mean_and_closes_quadrature():
    value = torch.arange(64, dtype=torch.float64).reshape(1, 2, 2, 2, 8)
    dc, ac = spatial_energy(value)
    torch.testing.assert_close(dc + ac, .5 * value.square().sum(-1).mean())
    assert spatial_energy(torch.ones_like(value))[1] == 0


def test_quadratic_bath_uses_released_energy_without_fictitious_source():
    torch.manual_seed(18)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
                                hidden=8, bath_type='quadratic').double()
    auditor = MediumHealthAuditor(model, window_tokens=4, block_tokens=1)
    loss, output, nll = quiet_training_chunk(
        model, torch.tensor([[1]]), torch.tensor([[2]]), model.initial_belief(),
        event_duration=.005, health_capture=auditor.capture)
    auditor.record_event(float(nll.detach()), novel=True, phase='stream')
    report = auditor.summary()['energy']
    assert report['response_source_work'] == 0
    assert report['dissipated_energy'] > 0
    assert abs(report['cumulative_balance_residual']) < 2e-12
    assert report['max_abs_write_residual'] < 2e-12


def test_forward_ad_reference_monitoring_preserves_random_credit_factors():
    from information_boltzmann.runtime.online_credit import OnlinePlasticTrainer
    model, _ = model_and_runner()
    reference = copy.deepcopy(model)
    watched = OnlinePlasticTrainer(model, event_duration=.005, seed=77)
    unobserved = OnlinePlasticTrainer(reference, event_duration=.005, seed=77)
    auditor = MediumHealthAuditor(model, window_tokens=4, block_tokens=1)
    watched.health_capture = auditor.capture
    a = watched.backward_event(torch.tensor([1]), torch.tensor([2]))
    b = unobserved.backward_event(torch.tensor([1]), torch.tensor([2]))
    torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for left, right in zip(watched.state_factors[0], unobserved.state_factors[0]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(watched.generator.get_state(), unobserved.generator.get_state())
    auditor.record_event(float(a['token_nll']), novel=True, phase='stream')
    assert auditor.summary()['energy']['max_abs_evolution_residual'] < 2e-12


def test_spectral_structure_reports_constant_and_rank_instead_of_health_verdict():
    constant = representation_summary(np.ones((8, 4)))
    assert constant['effective_rank'] == 0 and constant['spectral_entropy'] is None
    centered = np.arange(8)[:, None] * np.ones((1, 4))
    rank_one = representation_summary(centered)
    assert rank_one['effective_rank'] == pytest.approx(1)
    assert rank_one['rank_bound'] == 4
    assert 'snr_db' not in rank_one and 'noise_expelled' not in rank_one
    # Slow dynamics and rapidly alternating dynamics are both fully deterministic.
    slow = representation_summary(np.linspace(0, 1, 40)[:, None])
    fast = representation_summary(((-1.) ** np.arange(40))[:, None])
    assert slow['roughness_to_variance'] < fast['roughness_to_variance']


def test_loss_trend_keeps_replay_context_changes_and_partial_blocks_separate():
    rows = [{'phase': 'stream', 'novel': True, 'event': i + 1, 'nll': 8 - i / 10}
            for i in range(12)]
    rows += [{'phase': 'revisit_A', 'novel': False, 'event': 13 + i, 'nll': 2.}
             for i in range(5)]
    result = risk_trend(rows, 2)
    assert result['by_phase']['stream/first_pass']['slope_nats_per_event'] == pytest.approx(-.1)
    assert result['by_phase']['revisit_A/replay']['complete_blocks'] == 2
    assert result['by_phase']['revisit_A/replay']['slope_nats_per_event'] is None
    # A later stream segment is not regressed together with the earlier one.
    later = rows + [{'phase': 'stream', 'novel': True, 'event': 18 + i, 'nll': 12.}
                    for i in range(6)]
    split = risk_trend(later, 2)
    assert split['by_phase']['stream/first_pass']['slope_nats_per_event'] == pytest.approx(-.1)
    assert split['by_phase']['stream/first_pass/segment2']['slope_nats_per_event'] == pytest.approx(0, abs=1e-12)


def test_auditor_checkpoint_continues_energy_and_window_without_birth_reset():
    model, runner = model_and_runner()
    for left, right in ((1, 2), (2, 3)):
        runner.step(torch.tensor([left]), torch.tensor([right]))
    saved = (copy.deepcopy(model.state_dict()), runner.learner.state_dict(),
             copy.deepcopy(runner.optimizer.state_dict()), runner.state_dict())
    other_model, other = model_and_runner()
    other_model.load_state_dict(saved[0])
    other.learner.load_state_dict(saved[1])
    other.optimizer.load_state_dict(saved[2])
    other.load_state_dict(saved[3])
    runner.observe(torch.tensor([4]))
    other.observe(torch.tensor([4]))
    assert runner.health.summary() == other.health.summary()
    a = torch.tensor([2, 3, 4])
    scores = [row[2] for row in list(runner.recent_events)[-2:]]
    report = runner.revisit_after_change(a, scores, torch.tensor([5, 6, 7]), block_tokens=1, hold_blocks=1)
    assert report['health_before_change']['events'] == 3
    assert report['health_after_change']['events'] == 6
    assert report['health']['events'] == 9
    assert report['health']['energy']['max_abs_continuity_residual'] == 0
    assert 'revisit_A/replay' in report['health']['predictive_risk']['by_phase']


def test_temporal_response_preserves_history_and_matches_finite_difference():
    from dataclasses import replace
    torch.manual_seed(53)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
        hidden=8, bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True, read_mode='temporal',
        temporal_rates=[2., 5.], temporal_frequencies=[0., 3.]).double()
    belief = model.initial_belief()
    with torch.no_grad():
        for token in (1, 2, 3):
            belief, _ = model.assimilate(belief, torch.tensor([token]), diagnostics=True)
            belief, _ = model.advance(belief, .005, diagnostics=True)
    before = [x.clone() for x in belief_tensors(belief)]
    observed = torch.tensor([4])
    result = conditional_state_response(model, belief, observed,
                                        event_duration=.005, directions=1)
    inputs = (*credit_tensors(belief), belief.temporal.value.real, belief.temporal.value.imag)
    generator = torch.Generator().manual_seed(0)
    tangent = tuple(torch.randn(x.shape, dtype=x.dtype, generator=generator) for x in inputs)
    norm = sum(x.square().mean() for x in tangent).sqrt()
    tangent = tuple(x / norm for x in tangent)
    def transition(sign):
        values = tuple(x + sign * 1e-6 * u for x, u in zip(inputs, tangent))
        physical = replace_credit(belief, values[:-2])
        state = replace(physical, temporal=replace(belief.temporal,
                        value=torch.complex(values[-2], values[-1])))
        with torch.no_grad():
            written, _ = model.assimilate(state, observed, diagnostics=True, training_terms=False)
            outgoing, _ = model.advance(written, .005, diagnostics=True)
        return (*credit_tensors(outgoing), outgoing.temporal.value.real, outgoing.temporal.value.imag)
    plus, minus = transition(1), transition(-1)
    expected = sum(((x - y) / 2e-6).square().mean() for x, y in zip(plus, minus)).sqrt()
    assert result['gains'][0] == pytest.approx(float(expected), rel=2e-5)
    assert len(result['state_shapes']) == len(inputs)
    for x, y in zip(belief_tensors(belief), before):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_full_state_response_matches_finite_difference_and_preserves_life():
    model, runner = model_and_runner()
    for token in (1, 2, 3):
        runner.step(torch.tensor([token]), torch.tensor([token + 1]))
    learner_state = runner.learner.state_dict()
    parameters = copy.deepcopy(model.state_dict())
    before_rng = torch.get_rng_state().clone()
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    result = conditional_state_response(model, runner.learner.belief, torch.tensor([4]),
                                        event_duration=.005, directions=1)
    inputs = credit_tensors(runner.learner.belief)
    generator = torch.Generator().manual_seed(0)
    tangent = tuple(torch.randn(t.shape, dtype=t.dtype, generator=generator) for t in inputs)
    norm = sum(t.square().mean() for t in tangent).sqrt()
    tangent = tuple(t / norm for t in tangent)
    def transition(sign):
        perturbed = replace_credit(runner.learner.belief, tuple(x + sign * 1e-6 * u for x, u in zip(inputs, tangent)))
        with torch.no_grad():
            written, _ = model.assimilate(perturbed, torch.tensor([4]), diagnostics=False, training_terms=False)
            output, _ = model.advance(written, .005, diagnostics=False)
        return credit_tensors(output)
    positive, negative = transition(1), transition(-1)
    expected = sum(((p - m) / 2e-6).square().mean() for p, m in zip(positive, negative)).sqrt()
    assert result['gains'][0] == pytest.approx(float(expected), rel=2e-5)
    assert len(result['state_shapes']) == len(inputs) == 8
    assert torch.equal(before_rng, torch.get_rng_state())
    for key, value in parameters.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for p, g in zip(model.parameters(), grads):
        if g is not None:
            torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
    for a, b in zip(belief_tensors(runner.learner.belief), belief_tensors(learner_state['belief'])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(runner.learner.eligibility.trace, learner_state['trace'], rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='Optional CUDA graph check')
def test_captured_auditor_matches_eager_event_ledgers_and_updated_parameters():
    from information_boltzmann.runtime.local_credit import CapturedLocalEvent
    model, captured_runner = model_and_runner(device='cuda')
    reference_model, reference = model_and_runner(device='cuda')
    captured_runner.captured = CapturedLocalEvent(captured_runner.learner, loss_scale=1 / 3)
    for token in range(1, 5):
        observed, target = torch.tensor([token], device='cuda'), torch.tensor([token + 1], device='cuda')
        captured_runner.step(observed, target)
        reference.step(observed, target)
        torch.testing.assert_close(captured_runner.health.capture.values, reference.health.capture.values)
        torch.testing.assert_close(captured_runner.health.capture.feature, reference.health.capture.feature)
    assert captured_runner.health.events == reference.health.events == 4
    for p, q in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(p, q)
