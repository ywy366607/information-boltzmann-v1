"""Numerical categorical-evidence and physical-rollout interface contracts only."""
import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_native_generative import NativeCategoricalWorld
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
from test_fly_bptt_learning import make_model, physical


def make(tmp_path):
    from information_boltzmann.core.fly_active_inference import attach_motor_acts
    model = make_model(tmp_path)
    attach_motor_acts(model, brain_gradient=True)
    learner = SimpleNamespace(model=model, hold_input=True, writer_baseline_clock='physical',
                              stream_segment_ticks=3)
    reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                 if name in native_parameter_names(model)}
    return learner, NativeCategoricalWorld(learner, reference, {'checkpoint': 'explicit-identity-control'})


def test_categorical_information_and_bayes_update_are_exact(tmp_path):
    learner, world = make(tmp_path)
    lp = torch.tensor([[.9, .1], [.1, .9]], dtype=torch.float64).log()
    expected = .9 * math.log(1.8) + .1 * math.log(.2)
    assert float(world.information(lp)) == pytest.approx(expected)
    report = world.update(lp, 0, 'mother_reply')
    assert report['prequential_logp'] == pytest.approx(math.log(.5))
    assert report['realized_information'] == pytest.approx(expected)
    torch.testing.assert_close(world.weights, torch.tensor([.9, .1], dtype=torch.float64))
    resumed = NativeCategoricalWorld(learner)
    torch.testing.assert_close(resumed.weights, world.weights)


def test_identical_high_entropy_forecasts_have_no_epistemic_value(tmp_path):
    _, world = make(tmp_path)
    lp = torch.full((2, 100), -math.log(100))
    assert float(world.information(lp)) == pytest.approx(0, abs=1e-7)
    disjoint = torch.tensor([[0., -float('inf')], [-float('inf'), 0.]])
    assert float(world.information(disjoint)) == pytest.approx(math.log(2))


def test_finite_log_support_survives_probability_underflow_and_can_recover(tmp_path):
    _, world = make(tmp_path)
    world.state['log_weights'] = torch.tensor([-1000., 0.], dtype=torch.float64)
    lp = torch.tensor([[0., -1200.], [-1200., 0.]], dtype=torch.float64)
    report = world.update(lp, 0, 'mother_reply')
    assert math.isfinite(report['realized_information'])
    assert torch.isfinite(world.state['log_weights']).all()
    assert world.weights[0] > .999


def test_self_command_is_not_external_evidence_and_calls_are(tmp_path):
    _, world = make(tmp_path)
    lp = torch.tensor([[.8, .2], [.3, .7]]).log()
    before = world.weights.clone()
    assert not world.update(lp, 0, 'self_hearing')['updated']
    assert not world.update(lp, 0, 'infrastructure_error')['updated']
    torch.testing.assert_close(before, world.weights)
    assert world.state['external_updates'] == 0
    assert world.update(lp, 0, 'mother_call')['updated']
    assert world.state['external_updates'] == 1


def test_reference_must_be_explicit_and_forecasts_normalized(tmp_path):
    learner, world = make(tmp_path)
    del learner.native_generative_state
    with pytest.raises(ValueError, match='explicit'):
        NativeCategoricalWorld(learner)
    with pytest.raises(ValueError, match='normalized'):
        world.information(torch.zeros(2, 3))
    with pytest.raises(ValueError, match='parameter names'):
        NativeCategoricalWorld(learner, {})


def test_matched_rollout_restores_all_parameters_physiology_rng_and_housekeeping(tmp_path):
    learner, world = make(tmp_path)
    model = learner.model
    model.train()
    initial = physical(model)
    initial.h.add_(.03)
    model.homeo_spikes = torch.arange(model.n_neurons).float()
    model.homeo_ticks = 17
    model.step_grad_clips = 4
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    physical_before = copy.deepcopy(initial.state_dict())
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    result = world.predict(initial, 2, 5, quiet_ranges=((2, 4),), eye_gain=.7)
    assert result.shape == (2, 9)
    torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)
    assert float(world.information(result)) == pytest.approx(0, abs=1e-12)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, physical_before[key], rtol=0, atol=0)
    assert model.training and model.homeo_ticks == 17 and model.step_grad_clips == 4
    torch.testing.assert_close(model.homeo_spikes, torch.arange(model.n_neurons).float())
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    assert np_rng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
    assert learner.native_generative_state['external_updates'] == 0


def test_rollout_failure_restores_current_native_parameters_and_modes(tmp_path, monkeypatch):
    learner, world = make(tmp_path)
    model = learner.model
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    # An explicit different ancestral tensor proves the finally block restores
    # actual native parameters even after the second branch has been installed.
    world.state['reference_parameters'][world.names[0]].add_(.1)
    from information_boltzmann.core import fly_bptt_learning as fl
    calls = []
    real = fl.advance_fly_token_stream
    def fail_on_ancestral(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('diagnostic failure')
        return real(*args, **kwargs)
    monkeypatch.setattr(fl, 'advance_fly_token_stream', fail_on_ancestral)
    with pytest.raises(RuntimeError, match='diagnostic'):
        world.predict(physical(model), 1, 2)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert model.training


def test_native_reference_and_posterior_survive_durable_copy(tmp_path):
    learner, world = make(tmp_path)
    world.update(torch.tensor([[.6, .4], [.2, .8]]).log(), 0, 'external')
    saved = copy.deepcopy(learner.native_generative_state)
    learner.native_generative_state = saved
    restored = NativeCategoricalWorld(learner)
    torch.testing.assert_close(restored.weights, world.weights)
    assert restored.state['provenance']['checkpoint'] == 'explicit-identity-control'
    for name in world.names:
        torch.testing.assert_close(restored.state['reference_parameters'][name],
                                   world.state['reference_parameters'][name], rtol=0, atol=0)


def test_command_values_use_actual_known_words_and_silent_physical_branch(tmp_path):
    learner, world = make(tmp_path)
    values = world.command_values(physical(learner.model), [1, 3], ticks=2)
    assert values['candidate_ids'] == [1, 3]
    assert values['log_probs_words'].shape == (2, 2, 9)
    torch.testing.assert_close(values['words'], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    assert float(values['wait']) == pytest.approx(0, abs=1e-12)
    assert world.state['external_updates'] == 0
