"""Native actuation, evidence and score contracts; not developmental capability."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_fly_bptt_learning import make_model, physical
from test_fly_native_policy_credit import Model, learner as score_learner
from information_boltzmann.core.fly_active_inference import attach_motor_acts
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
from information_boltzmann.core.fly_mother_world import MotherWorld, live_window
from information_boltzmann.core.fly_native_generative import NativeCategoricalWorld
from information_boltzmann.core.fly_native_generative_control import NativeGenerativeControl
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names, native_policy_credit


def make(tmp_path):
    m = make_model(tmp_path)
    attach_motor_acts(m, brain_gradient=True)
    m.comp_logit_trace = torch.nn.Parameter(torch.tensor(2.))
    obj = FlyBPTTLearner(m, physical(m), stream_ticks=1, hold_input=True,
                          lambda_jepa=0, use_ctm_loss=False)
    obj.previous_token, obj.stream_pace_mean, obj.stream_segment_ticks = 0, 4, 3
    obj.speak_credit, obj.mother_action_clock = 'consequence', 'commit'
    obj.native_generative_enabled = True
    obj.native_generative_candidates, obj.native_generative_ticks = 2, 4
    NativeCategoricalWorld(obj, {name: p.detach().clone() for name, p in m.named_parameters()
                                if name in native_parameter_names(m)}, {'identity': True})
    return obj


def test_joint_timing_score_includes_word_partition_derivatives_and_raw_extremes():
    for raw in (-100., -.4):
        obj = score_learner()
        m = obj.model
        reads = torch.stack((m.comp_weight[0], m.comp_eta)).expand(2, 2)
        logits = (m.comp_eta + m.ai_w + raw).expand(2)
        values = torch.tensor([.2, -.7, .6])
        lp = F.log_softmax(m.decoder(reads).float() / .8, -1)
        z = logits + torch.logsumexp(lp + values, -1) - .1
        exact = (F.logsigmoid(-z[0]) + F.logsigmoid(z[1])) * 2.
        params = [m.comp_weight, m.comp_eta, m.ai_w, m.decoder.weight]
        expected = torch.autograd.grad(exact, params, retain_graph=True)
        events = [{'kind': 'clock', 'ticks': 2},
                  {'kind': 'joint_timing', 'start': 0, 'length': 2, 'first': 1,
                   'tail_ticks': 0, 'word_value': values, 'wait_value': .1, 'temperature': .8},
                  {'kind': 'response_begin'}, {'kind': 'response_end', 'advantage': 2.},
                  {'kind': 'trial_end'}]
        surrogate, diag = native_policy_credit(obj, events, logits, reads, [2],
                                               raw_logits=True, include_policy_heads=True)
        actual = torch.autograd.grad(surrogate, params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        assert not obj.consequence_state['native_credit']['E']
        assert diag['native_timing_connected'] == 1


def test_reference_identity_changes_no_native_policy_and_preserves_full_tail(tmp_path):
    obj = make(tmp_path)
    engine = NativeGenerativeControl(obj)
    engine.begin_child_turn()
    engine.prepare(None, obj.state)
    assert torch.count_nonzero(engine.word_bias) == 0
    assert engine.wait_value == 0
    assert engine.plans[0]['applied'] is True
    assert len(engine.word_bias) == obj.model.decoder.out_features
    assert engine.plans[0]['unassessed_tail'] == 'retained_native_mass_neutral_reference'
    assert engine.timing_bias(obj.state, None) == pytest.approx(0., abs=1e-6)


class Tokens:
    def encode(self, text):
        return [1, 2]

    def decode(self, ids):
        return 'word' if ids else ''


class Mother:
    asynchronous = False

    def reply(self, history):
        return 'reply'


def test_actual_native_control_is_used_and_replayed_with_no_count_or_habit_reward(tmp_path):
    obj = make(tmp_path)
    with torch.no_grad():
        obj.model.ai_b[1] = 100.
    w = MotherWorld(Mother(), Tokens(), opening=None, max_fly_words=1)
    w.turn = 'fly'
    targets, forced, _ = live_window(obj, w, 2, np.random.default_rng(4), 4, 4, speak_act=True)
    assert forced['native_generative_control'] is True
    assert forced['sources'][0] == 'self_hearing'
    assert any(e['kind'] == 'joint_timing' for e in forced['consequence_native_events'])
    assert len(forced['consequence_timing_bias']) == sum(forced['intervals'])
    assert obj.last_epistemic['policy_actuated'] is True
    assert obj.native_generative_state['external_updates'] == 1
    obj.forced_world = forced
    eye_gradients = []
    hook = obj.model.ai_b.register_hook(lambda gradient: eye_gradients.append(gradient[0].detach().clone()))
    try:
        scores, stats = obj.observe(targets)
    finally:
        hook.remove()
    assert stats['epistemic'] == 0.
    assert len(scores) == 2 and np.isfinite(scores).all()
    assert stats['native_policy_native_timing_connected'] > 0
    assert stats['native_policy_native_content_connected'] > 0
    # observe clears .grad after its actual update; inspect the real backward.
    assert eye_gradients and all(torch.isfinite(g) for g in eye_gradients)
    assert any(g != 0 for g in eye_gradients)


def test_unfinished_legacy_response_cannot_be_relabelled(tmp_path):
    obj = make(tmp_path)
    obj.consequence_state = {'format': 'kc-action-consequence-v1', 'response': {'old': True}}
    with pytest.raises(ValueError, match='Drain'):
        NativeGenerativeControl(obj)


def test_old_pending_head_credit_must_settle_before_migration(tmp_path):
    obj = make(tmp_path)
    obj.pending_speak_credit = {'trace': {'ai_b': torch.ones(2)}, 'decisions': [True]}
    with pytest.raises(ValueError, match='pending legacy head'):
        NativeGenerativeControl(obj)
    assert obj.pending_speak_credit['decisions'] == [True]


def test_evidence_replay_mismatch_cannot_update_the_subjective_weights(tmp_path, monkeypatch):
    obj = make(tmp_path)
    engine = NativeGenerativeControl(obj)
    engine.begin_interval(obj.state, 0, False, False)
    wrong = torch.arange(9).float().log_softmax(-1).expand(2, -1)
    monkeypatch.setattr(engine.world_model, 'predict', lambda *args, **kwargs: wrong)
    with pytest.raises(RuntimeError, match='differs from actual sensing'):
        engine.arrive(obj.state, 0, 'mother_call', 1, [], [1.], end=True)
    assert obj.native_generative_state['external_updates'] == 0
    torch.testing.assert_close(engine.world_model.weights, torch.tensor([.5, .5], dtype=torch.float64))


def test_noncausal_source_cannot_change_previous_reply_credit(tmp_path, monkeypatch):
    obj = make(tmp_path)
    engine = NativeGenerativeControl(obj)
    engine.begin_child_turn()
    prior = engine.world_model.state['log_weights'].clone()
    engine.state['commands'] = 1
    engine.state['response'] = {'prior_log_weights': prior, 'source': 'mother_reply'}
    lp = torch.full((2, 9), -torch.log(torch.tensor(9.)).item(), dtype=torch.float64)
    lp[0, 0] += .1
    lp = lp.log_softmax(-1)
    engine.world_model.update(lp, 0, 'mother_reply')
    after = engine.world_model.state['log_weights']
    expected = float((after.exp() * (after - prior)).sum())
    engine.begin_interval(obj.state, 0, False, False)
    other = torch.full((2, 9), -100., dtype=torch.float64)
    other[0, 1] = 0.
    other[1, 0] = 0.
    other = other.log_softmax(-1)
    from information_boltzmann.core.fly_bptt_learning import read_fly_motor
    other[0] = obj.model.decoder(obj.model.read_norm(read_fly_motor(
        obj.model, obj.state))).float()[0].log_softmax(-1).detach().double()
    monkeypatch.setattr(engine.world_model, 'predict', lambda *args, **kwargs: other)
    engine.arrive(obj.state, 0, 'mother_call', 1, [], [1.], end=True)
    settlements = [e for e in engine.native_events if e['kind'] == 'response_end']
    assert len(settlements) == 1 and settlements[0]['advantage'] == pytest.approx(expected)
