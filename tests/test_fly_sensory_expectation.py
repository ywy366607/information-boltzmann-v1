"""Sensory event-clock/efference interfaces; no learning capability claims."""
import copy

import pytest
import torch

from information_boltzmann.core.fly_bptt_learning import (
    advance_fly_token_stream, attach_predictive_writer, expect_next,
    prepare_sensory_expectation, word_packets, write_input,
)
from information_boltzmann.core.fly_reservoir import ReceptorWriter
from test_fly_bptt_learning import make_model, physical


def model_and_state(tmp_path):
    torch.manual_seed(13)
    model = make_model(tmp_path)
    model.topographic_writer = ReceptorWriter(3, [0, 1, 2], [0, 1, 2], 3,
                                              E_E=1., jitter=0., gain_jitter=0.)
    attach_predictive_writer(model, beta=.4)
    with torch.no_grad():
        model.topographic_writer.proj.weight.copy_(torch.eye(3))
        model.embedding.weight[1].copy_(torch.tensor([.2, .4, .1]))
        model.embedding.weight[2].copy_(torch.tensor([.7, -.1, .3]))
    state = physical(model)
    state.baseline.copy_(torch.tensor([[.1, .05, .02]]))
    state.expected_input = torch.tensor([[-.2, .3, -.1]], requires_grad=True)
    return model, state


def assert_state_close(actual, expected):
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


def test_known_command_reuses_existing_predictive_gain_and_receptor_conductance(tmp_path):
    model, original = model_and_state(tmp_path)
    command = torch.tensor([1])
    state = prepare_sensory_expectation(model, original, command, 'self_hearing')
    writer = model.topographic_writer
    packet = writer.packet(model.embedding(command))
    beta = torch.sigmoid(model.writer_prediction_logit)
    expected_innovation = packet - ((1. - beta) * original.baseline + beta * packet)
    drive, baseline = write_input(model, state, command, packets=word_packets(model, state, command))
    expected_drive = writer.conductance(expected_innovation) * (
        writer.E_E - state.h[:, writer.injection_index]).clamp_min(0.)
    torch.testing.assert_close(drive[:, writer.injection_index], expected_drive)
    torch.testing.assert_close(baseline, writer.lambda_adapt * original.baseline +
                               (1. - writer.lambda_adapt) * packet)
    assert float(drive.detach().abs().sum()) > 0  # knowing a command does not blind perception
    assert state.expected_input is not original.expected_input
    torch.testing.assert_close(original.expected_input, torch.tensor([[-.2, .3, -.1]]))
    # This path changes no reward/voice or external-current annotation.
    assert not hasattr(model, 'reward_cells') and not hasattr(model, 'voice_cells')
    assert getattr(model, 'external_drive', None) is None


@pytest.mark.parametrize('source', ['mother_reply', 'mother_call', 'external', 'unknown'])
def test_external_input_keeps_prearrival_prior_independent_of_its_token(tmp_path, source):
    model, state = model_and_state(tmp_path)
    first = prepare_sensory_expectation(model, state, 1, source)
    second = prepare_sensory_expectation(model, state, 2, source)
    assert first.expected_input is state.expected_input
    assert second.expected_input is state.expected_input
    prior = torch.tensor([[.3, -.1, .2]], requires_grad=True)
    replacement = prepare_sensory_expectation(model, state, 2, source, prior_expected=prior)
    assert replacement.expected_input is prior
    assert state.expected_input is first.expected_input


@pytest.mark.parametrize('read_every', [False, True])
def test_held_input_expectation_and_physiology_do_not_depend_on_segmentation(tmp_path, read_every):
    model, initial = model_and_state(tmp_path)
    prepared = prepare_sensory_expectation(model, initial, 1, 'self_hearing')
    expected = prepared.expected_input
    full = advance_fly_token_stream(model, prepared, torch.tensor([1]), ticks=7,
                                   hold_input=True, read_every=read_every, update_expectation=False)
    first = advance_fly_token_stream(model, copy.deepcopy(prepared.detached()), torch.tensor([1]),
                                    ticks=3, hold_input=True, read_every=read_every, update_expectation=False)
    second = advance_fly_token_stream(model, first[0], torch.tensor([1]), ticks=4,
                                     fresh=False, hold_input=True, read_every=read_every,
                                     update_expectation=False)
    assert_state_close(full[0], second[0])
    torch.testing.assert_close(full[0].expected_input, expected, rtol=0, atol=0)
    torch.testing.assert_close(second[0].expected_input, expected, rtol=0, atol=0)
    torch.testing.assert_close(full[1], second[1], rtol=0, atol=0)
    if read_every:
        torch.testing.assert_close(full[2], torch.cat([first[2], second[2]]), rtol=0, atol=0)


def test_event_boundary_updates_once_and_legacy_default_is_unchanged(tmp_path):
    model, initial = model_and_state(tmp_path)
    old = advance_fly_token_stream(model, initial, torch.tensor([2]), ticks=4, hold_input=True)
    held = advance_fly_token_stream(model, initial, torch.tensor([2]), ticks=4,
                                   hold_input=True, update_expectation=False)
    assert_state_close(old[0], expect_next(model, held[0], held[1]))
    assert held[0].expected_input is initial.expected_input
    assert not torch.equal(old[0].expected_input, initial.expected_input)


@pytest.mark.parametrize('source', ['mother_reply', 'self_hearing'])
def test_physical_sensory_and_prediction_gradients_remain_live(tmp_path, source):
    model, state = model_and_state(tmp_path)
    prepared = prepare_sensory_expectation(model, state, 1, source)
    result = advance_fly_token_stream(model, prepared, torch.tensor([1]), ticks=3,
                                     hold_input=True, update_expectation=False)
    result[0].h.sum().backward()
    for parameter in (model.writer_prediction_logit, model.topographic_writer.proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0
    if source == 'mother_reply':
        assert state.expected_input.grad is not None
        assert float(state.expected_input.grad.abs().sum()) > 0
    else:
        assert state.expected_input.grad is None
        assert float(model.embedding.weight.grad[1].abs().sum()) > 0


def test_input_source_ownership_at_self_external_transition(tmp_path):
    model, initial = model_and_state(tmp_path)
    # The prior was formed before the next external token became known. The
    # current held command keeps its copy even if its NEXT target is Mama's word.
    self_input = prepare_sensory_expectation(model, initial, 1, 'self_hearing')
    out = advance_fly_token_stream(model, self_input, torch.tensor([1]), ticks=2,
                                  hold_input=True, update_expectation=False)
    next_prior = expect_next(model, out[0], out[1])
    external_input = prepare_sensory_expectation(model, next_prior, 2, 'mother_reply')
    assert external_input.expected_input is next_prior.expected_input
    assert not torch.equal(external_input.expected_input, model.embedding(torch.tensor([2])))
    # The next command receives its own copy only when it actually arrives.
    next_self = prepare_sensory_expectation(model, external_input, 1, 'self_hearing')
    torch.testing.assert_close(next_self.expected_input, model.embedding(torch.tensor([1])))


def test_unconfigured_predictive_writer_is_an_identity_interface(tmp_path):
    model = make_model(tmp_path)
    state = physical(model)
    assert prepare_sensory_expectation(model, state, 1, 'self_hearing') is state
