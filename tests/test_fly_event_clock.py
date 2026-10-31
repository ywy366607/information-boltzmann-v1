"""Causal event-clock contracts; no learning/capability claims."""
import copy

import numpy as np
import pytest
import torch

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_active_inference import attach_motor_acts, motor_rest
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, advance_fly_token_stream
from information_boltzmann.core.fly_mother_world import MotherWorld, live_window


def assert_physical_equal(a, b):
    for key, value in a.state_dict().items():
        expected = b.state_dict()[key]
        if isinstance(value, tuple):
            for left, right in zip(value, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_commit_boundary_matches_exact_duration_and_gradients(tmp_path):
    model = make_model(tmp_path)
    attach_motor_acts(model, brain_gradient=True)
    state = physical(model)
    g = motor_rest(model, state.h)
    outputs = []
    for stop, ticks in ((True, 9), (False, 3)):
        tape = [False, False, True]
        result = advance_fly_token_stream(
            model, copy.deepcopy(state), torch.tensor([1]), ticks=ticks,
            hold_input=True, read_every=True, ai_g=g.clone(), ai_tape=tape,
            stop_on_commit=stop)
        loss = model.decoder(model.read_norm(result[1])).square().sum()
        grad = torch.autograd.grad(loss, model.decoder.weight)[0]
        outputs.append((result, grad))
    left, right = outputs
    assert_physical_equal(left[0][0], right[0][0])
    assert len(left[0][2]) == len(left[0][-2]) == 3
    for i in (1, 2, -4, -3, -2, -1):
        torch.testing.assert_close(left[0][i], right[0][i], rtol=0, atol=0)
    torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)


class Tokens:
    def encode(self, text):
        return [1, 2]

    def decode(self, ids):
        return 'word' if ids else ''


class Mother:
    asynchronous = False

    def reply(self, history):
        return 'reply'


def test_native_credit_cannot_override_known_echo_exclusion(tmp_path, monkeypatch):
    from information_boltzmann.core import fly_consequence_belief as cb

    class Credit:
        def __init__(self, learner):
            self.state = {'source': 'self_hearing', 'in_child_turn': True}
            self.native_events = []

        def features(self, state):
            return state.h[0], None

        def input_ratio(self, source):
            return 0.

        def advance_ticks(self, ticks):
            self.native_events.append({'kind': 'clock', 'ticks': ticks})

        def observe(self, *args, **kwargs):
            pass

        def choose(self, *args):
            pass

        def say(self, *args):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cb, 'NativeCircuitConsequenceCredit', Credit)
    model = make_model(tmp_path)
    attach_motor_acts(model, brain_gradient=True)
    with torch.no_grad():
        model.ai_b[1] = 100.
    learner = FlyBPTTLearner(model, physical(model), stream_ticks=1, hold_input=True,
                            lambda_jepa=0, use_ctm_loss=False)
    learner.previous_token, learner.stream_pace_mean = 0, 12
    learner.speak_credit, learner.native_credit_enabled = 'consequence', True
    learner.consequence_native_enabled, learner.mother_action_clock = False, 'commit'
    w = MotherWorld(Mother(), Tokens(), opening=None, max_fly_words=4)
    w.turn = 'fly'
    _, forced, _ = live_window(learner, w, 1, np.random.default_rng(14), 12, 12, speak_act=True)
    assert forced['sources'] == ['self_hearing']
    assert forced['exclude_self_prediction'] is True
    assert sum(e['ticks'] for e in forced['consequence_native_events']) == 1


@pytest.mark.parametrize('mother_turn,interrupt', [(False, False), (True, False), (True, True)])
def test_live_command_is_immediate_but_external_input_is_not_delayed(tmp_path, monkeypatch, mother_turn, interrupt):
    from information_boltzmann.core import fly_bptt_learning as fl
    torch.manual_seed(13)
    model = make_model(tmp_path)
    attach_motor_acts(model, brain_gradient=True)
    with torch.no_grad():
        model.ai_b[1] = 100.
    learner = FlyBPTTLearner(model, physical(model), stream_ticks=1, hold_input=True,
                            lambda_jepa=0, use_ctm_loss=False)
    learner.previous_token = 0
    learner.stream_pace_mean = 12
    learner.stream_segment_ticks = 4
    learner.speak_credit = 'epistemic'
    learner.mother_action_clock = 'commit'
    world = MotherWorld(Mother(), Tokens(), opening='hello' if mother_turn else None,
                        max_fly_words=1)
    if not mother_turn:
        world.turn = 'fly'
    states = []
    advance = fl.advance_fly_token_stream

    def record(*args, **kwargs):
        out = advance(*args, **kwargs)
        states.append(out[0])
        return out

    monkeypatch.setattr(fl, 'advance_fly_token_stream', record)
    targets, forced, speakers = live_window(
        learner, world, 1, np.random.default_rng(14), 12, 12, speak_act=True, interrupt=interrupt)
    live_state = states[-1]
    assert forced['action_clock'] == 'commit'
    if mother_turn and not interrupt:
        assert speakers == ['mother'] and forced['intervals'][0] > 1
        assert not forced['decisions']
    else:
        assert speakers == ['fly'] and forced['intervals'] == [1]
        assert forced['decisions'] == [(0, 0, 1, 1)]
        assert forced['events'] == [{'tick': 1, 'interval_ticks': 1, 'source': 'self_hearing',
                                    'token': targets[0], 'kind': 'command_echo'}]
    learner.forced_world = forced
    with torch.no_grad():
        scores, replay_state, _ = learner.forward_window(torch.tensor([[0]]), torch.tensor([targets]))
    assert_physical_equal(live_state, replay_state)
    if mother_turn and not interrupt:
        torch.testing.assert_close(scores, learner.last_train_loss, rtol=0, atol=0)
    else:
        assert learner.last_train_loss.sum() == 0
    assert learner.last_adaptive_metrics['prediction_objective'] == 'external_arrival'
