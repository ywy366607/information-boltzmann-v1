"""Ordered-action numerical/continuation contracts, not language capability tests."""
import copy

import pytest
import torch
from torch import nn

from test_fly_consequence_belief import learner, choose
from information_boltzmann.core.fly_action_model import OrderedConsequenceCredit, SCHEMA
from information_boltzmann.core.fly_action_model import external_prediction_loss
from information_boltzmann.core.fly_consequence_belief import NativeConsequenceCredit


def make():
    obj = learner()
    obj.model.embedding = nn.Embedding(4, 3).requires_grad_(False)
    return obj, OrderedConsequenceCredit(obj)


def append(engine, word):
    context, _ = engine.features(engine.learner.state)
    engine.decision_bounds = (0, 3)
    engine.choose(context, torch.ones(3, 4), 1)
    engine.say(torch.tensor([[.2, -.3, .8]]), word, .7)


def test_order_count_and_pure_wait_are_explicit_action_coordinates():
    obj, engine = make()
    c, _ = engine.features(obj.state)
    pure = engine.action_features(c, False)
    append(engine, 0); once = engine.action_features(c, False)
    append(engine, 1); ab = engine.action_features(c, False)
    engine.begin_child_turn(); append(engine, 1); append(engine, 0)
    ba = engine.action_features(c, False)
    assert not torch.equal(ab, ba)
    engine.begin_child_turn(); append(engine, 0); append(engine, 0)
    twice = engine.action_features(c, False)
    assert not torch.equal(once, twice)
    assert not torch.equal(pure, once)


@pytest.mark.parametrize('initialized', [False, True])
def test_fast_all_vocabulary_values_equal_direct_posterior_quadratic(initialized):
    obj, engine = make()
    c, y = engine.features(obj.state)
    if initialized:
        x = engine.action_features(c, False)
        engine.belief.initialize(x, y)
        z = torch.randn(len(x), len(x), dtype=torch.float64)
        engine.state['belief']['P'] = z @ z.T / len(x)
    for prefix in [[], [0], [0, 1]]:
        engine.begin_child_turn()
        for word in prefix:
            append(engine, word)
        values, wait = engine.policy_values(c)
        expected = torch.tensor([.5 * __import__('math').log1p(engine.belief.ratio(
            engine.action_features(c, True, candidate=e))) for e in engine.word_table])
        torch.testing.assert_close(values, expected, rtol=2e-5, atol=2e-6)
        assert wait >= 0


def test_word_coordinates_do_not_move_when_the_decoder_learns():
    obj, engine = make()
    old = engine.word_table.clone()
    with torch.no_grad():
        obj.model.decoder.weight.add_(100)
    newer = OrderedConsequenceCredit(obj)
    torch.testing.assert_close(old, newer.word_table, rtol=0, atol=0)


def test_native_familiarity_changes_conditional_action_values_without_rewarding_self_echo():
    obj, engine = make()
    c, _ = engine.features(obj.state)
    before, _ = engine.policy_values(c)
    obj.state.comp_memory.add_(.8)
    changed, _ = engine.features(obj.state)
    after, _ = engine.policy_values(changed)
    assert not torch.equal(before, after)
    append(engine, 0)
    engine.observe(obj.state, 'self_hearing', 1.)
    assert not engine.gains and not engine.state['belief']


def test_legacy_response_finishes_before_changing_feature_schema():
    obj = learner()
    obj.model.embedding = nn.Embedding(4, 3).requires_grad_(False)
    old = NativeConsequenceCredit(obj)
    choose(old); old.observe(obj.state, 'mother_reply', .8, end=False)
    saved_response = copy.deepcopy(obj.consequence_state['response'])
    newer = OrderedConsequenceCredit(obj)
    assert not newer.active
    torch.testing.assert_close(saved_response['action'], newer.state['response']['action'])
    newer.observe(obj.state, 'mother_reply', .8, end=True)
    assert newer.active and newer.state['action_schema'] == SCHEMA
    assert newer.state['legacy_v1']['belief']['n'] == 1
    assert newer.state['belief'] == {}
    assert newer.native_events[-1]['kind'] == 'response_end'
    assert newer.gradients  # preserve real old policy credit, not just the posterior


def test_pending_ordered_utterance_and_response_survive_copy_resume():
    obj, engine = make(); append(engine, 0); append(engine, 1)
    engine.observe(obj.state, 'mother_reply', .8, end=False)
    restored = copy.deepcopy(obj)
    resumed = OrderedConsequenceCredit(restored)
    engine.observe(obj.state, 'mother_reply', .8)
    resumed.observe(restored.state, 'mother_reply', .8)
    assert engine.gains == resumed.gains
    torch.testing.assert_close(engine.state['belief']['P'], resumed.state['belief']['P'])
    torch.testing.assert_close(engine.state['ordered_action']['weighted'], resumed.state['ordered_action']['weighted'])


def test_timing_bias_is_the_joint_word_policy_log_partition(monkeypatch):
    obj, engine = make(); c, _ = engine.features(obj.state)
    values = torch.tensor([0., .5, 1., 2.])
    monkeypatch.setattr(engine, 'policy_values', lambda *a: (values, .2))
    import information_boltzmann.core.fly_bptt_learning as fl
    monkeypatch.setattr(fl, 'read_fly_motor', lambda *a: obj.state.h[:, :3])
    engine.prepare(c, obj.state, temperature=.7)
    logits = obj.model.decoder(obj.model.read_norm(obj.state.h[:, :3])).float()[0] / .7
    expected = torch.logsumexp(torch.log_softmax(logits, -1) + values, 0) - .2
    assert engine.timing_bias == pytest.approx(float(expected.detach()))


def test_real_clock_events_include_all_ticks_and_no_extra_self_reward():
    obj, engine = make()
    engine.advance_ticks(3); append(engine, 0)
    before = engine.state['eligibility']['decoder.bias'].clone()
    engine.advance_ticks(7)
    torch.testing.assert_close(engine.state['eligibility']['decoder.bias'], before * engine.retention**7)
    assert sum(e['ticks'] for e in engine.native_events if e['kind'] == 'clock') == 10
    assert len(engine.timing_biases) == 10
    assert not engine.gains


def test_external_loss_keeps_actual_target_denominator_and_all_self_graph():
    raw = torch.tensor([2., 8., 4.], requires_grad=True)
    loss = external_prediction_loss(raw, ['mother_reply', 'self_hearing', 'mother_call']).mean()
    assert float(loss.detach()) == 3.
    loss.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([.5, 0., .5]))
    own = torch.ones(2, requires_grad=True)
    external_prediction_loss(own, ['self_hearing'] * 2).mean().backward()
    torch.testing.assert_close(own.grad, torch.zeros(2))


def test_efference_expectation_survives_resume_and_never_updates_external_belief():
    obj, engine = make(); append(engine, 2)
    restored = copy.deepcopy(obj)
    resumed = OrderedConsequenceCredit(restored)
    resumed.hear_token(2, 'self_hearing', .8)
    assert resumed.state['self_echo']['matched'] == 1
    assert resumed.state['self_echo']['exposure_sum'] == .8
    assert resumed.state['pending_self_echo'] == []
    assert resumed.state['belief'] == {}
