"""Shadow event ingestion and continuation contracts, not capability studies."""
import copy
import io

import pytest
import torch
from torch import nn

from test_fly_consequence_belief import learner
from information_boltzmann.core.fly_generative_observer import NativeGenerativeObserver


def make():
    obj = learner()
    obj.model.embedding = nn.Embedding(4, 3).requires_grad_(False)
    with torch.no_grad():
        obj.model.embedding.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                                       [0., 0., 1.], [1., 1., -1.]]))
    return obj, NativeGenerativeObserver(obj)


def assert_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def hear(obj, observer, ticks, token, source):
    observer.begin_interval(obj.state, .8)
    return observer.end_interval(ticks, token, source)


def test_shared_event_belief_generates_exact_joint_wait_and_word_values():
    obj, observer = make()
    hear(obj, observer, 4, 2, 'mother_reply')
    context = observer.features(obj.state)
    coordinates = torch.stack([observer.word_coordinates(i) for i in range(4)])
    values, wait = observer.candidate_values(context, .7, coordinates, horizon=8, batch_size=2)
    direct = torch.stack([observer.belief.predict(observer.action_features(context, .7, c), 8)
                          ['total_information'] for c in coordinates])
    torch.testing.assert_close(values, direct)
    torch.testing.assert_close(wait, observer.belief.predict(observer.action_features(context, .7), 8)
                               ['total_information'])
    p = observer.prospective_policy(context, [.7], torch.zeros(1), torch.tensor([-2.]),
                                    torch.zeros(1, 4), coordinates, horizon=8)
    torch.testing.assert_close(p.word_value[0].double(), direct)
    torch.testing.assert_close(p.log_wait.exp().sum() + p.log_words.exp().sum(), torch.tensor(1.))
    assert observer.state['intervals'] == 1 and observer.state['belief']['events'] == 1


def test_native_context_reused_without_reward_credit_or_consequence_mutation():
    obj, observer = make()
    assert not hasattr(obj, 'consequence_state')
    state_before = copy.deepcopy(obj.state.state_dict())
    weights = copy.deepcopy(obj.model.state_dict())
    c = observer.features(obj.state)
    obj.state.comp_memory.add_(.4)
    changed = observer.features(obj.state)
    assert not torch.equal(c, changed)
    obj.state.dopamine[0].add_(.2)
    assert not torch.equal(changed, observer.features(obj.state))
    assert observer.state['belief']['events'] == 0
    assert observer.state['intervals'] == 0
    assert not hasattr(obj, 'pending_speak_credit')
    for name, parameter in obj.model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(weights[name], parameter)
    obj.state.comp_memory.copy_(state_before['comp_memory'])
    obj.state.dopamine.copy_(state_before['dopamine'])
    hear(obj, observer, 3, 2, 'mother_reply')
    assert_equal(state_before, obj.state.state_dict())
    assert not hasattr(obj, 'consequence_state')


def test_snapshot_has_no_future_label_or_post_observation_state():
    a, first = make()
    b = copy.deepcopy(a)
    second = NativeGenerativeObserver(b)
    x = first.begin_interval(a.state, .3)
    torch.testing.assert_close(second.begin_interval(b.state, .3), x)
    before_first = first.belief.predict(x, 1)
    b.state.h.add_(100)
    one = first.end_interval(4, 0, 'mother_opening', actual_eye_gain=.1)
    two = second.end_interval(4, 3, 'mother_call', actual_eye_gain=.9)
    assert one['timing_log_likelihood'] == two['timing_log_likelihood']
    assert one['expected_information_before'] == two['expected_information_before']
    assert one['expected_information_before'] == float(before_first['total_information'])
    torch.testing.assert_close(first.state['belief']['mean'], second.state['belief']['mean'])
    assert not torch.equal(first.state['belief']['mark']['mean'], second.state['belief']['mark']['mean'])
    assert one['eye_gain'] == two['eye_gain'] == pytest.approx(.3)
    assert one['actual_eye_gain'] != two['actual_eye_gain']


def test_self_echo_is_survival_without_stochastic_mark_then_adds_command():
    obj, observer = make()
    x = observer.begin_interval(obj.state, .9)
    mark_before = copy.deepcopy(observer.state['belief']['mark'])
    expected = copy.deepcopy(observer.state['belief'])
    from information_boltzmann.core.fly_event_belief import ProbitEventBelief
    reference = ProbitEventBelief(expected)
    reference.resolve(False, quiet_ticks=5)
    report = observer.end_interval(5, 1, 'self_hearing')
    assert_equal(observer.state['belief'], expected)
    assert_equal(observer.state['belief']['mark'], mark_before)
    assert report['mark_log_likelihood'] == 0
    assert report['external_event'] is False
    assert observer.state['prefix']['n'] == 1
    assert not torch.equal(observer.action_features(observer.features(obj.state), .9), x)
    assert observer.state['clock_ticks'] == 5


def test_all_external_sources_fit_world_marks_and_quiet_has_no_token():
    obj, observer = make()
    for source in ('mother_opening', 'mother_call', 'mother_reply', 'external'):
        report = hear(obj, observer, 3, 2, source)
        assert report['external_event']
        assert report['log_likelihood'] == pytest.approx(
            report['timing_log_likelihood'] + report['mark_log_likelihood'])
    mark_before = copy.deepcopy(observer.state['belief']['mark'])
    quiet = hear(obj, observer, 7, None, 'quiet')
    assert quiet['token'] is None
    assert_equal(mark_before, observer.state['belief']['mark'])
    assert observer.state['belief']['events'] == 4
    assert observer.state['belief']['quiet_ticks'] == 4 * 2 + 7
    assert observer.state['clock_ticks'] == 4 * 3 + 7
    assert observer.state['source_counts'] == dict(mother_opening=1, mother_call=1,
                                                  mother_reply=1, external=1, quiet=1)


def test_prefix_order_and_explicit_turn_reset_survive_window_construction():
    a, first = make()
    b = copy.deepcopy(a)
    second = NativeGenerativeObserver(b)
    for obj, engine, order in ((a, first, (0, 1)), (b, second, (1, 0))):
        for token in order:
            hear(obj, engine, 2, token, 'self_hearing')
    context = first.features(a.state)
    ab, ba = first.action_features(context, .8), second.action_features(context, .8)
    assert not torch.equal(ab, ba)
    reconstructed = NativeGenerativeObserver(a)
    torch.testing.assert_close(reconstructed.action_features(context, .8), ab)
    reconstructed.begin_child_turn()
    assert reconstructed.state['prefix']['n'] == 0
    assert reconstructed.state['child_turns'] == 1


def test_save_resume_pending_interval_exact_and_old_state_untouched():
    obj, observer = make()
    obj.consequence_state = {'legacy': torch.tensor([3.])}
    old = copy.deepcopy(obj.consequence_state)
    hear(obj, observer, 2, 0, 'self_hearing')
    hear(obj, observer, 4, 2, 'mother_call')
    observer.begin_interval(obj.state, .6)
    stream = io.BytesIO()
    torch.save(obj.generative_event_state, stream)
    stream.seek(0)
    restored = copy.deepcopy(obj)
    restored.generative_event_state = torch.load(stream, weights_only=False)
    resumed = NativeGenerativeObserver(restored)
    assert observer.end_interval(6, 3, 'mother_reply') == resumed.end_interval(6, 3, 'mother_reply')
    assert_equal(obj.generative_event_state, restored.generative_event_state)
    assert_equal(obj.consequence_state, old)
    assert_equal(restored.consequence_state, old)


@pytest.mark.parametrize('ticks,token,source', [(0, 0, 'mother_call'), (2, 0, 'unknown'),
                                                (2, 0, 'quiet'), (2, None, 'mother_reply')])
def test_bad_events_fail_without_partial_mutation(ticks, token, source):
    obj, observer = make()
    observer.begin_interval(obj.state, .5)
    before = copy.deepcopy(observer.state)
    with pytest.raises(ValueError):
        observer.end_interval(ticks, token, source)
    assert_equal(before, observer.state)


def test_eye_is_fixed_initial_feature_not_inverse_observation_noise():
    obj, observer = make()
    x = observer.begin_interval(obj.state, [.2, .4])
    assert x[-1] == pytest.approx(.3)
    assert observer.state['belief']['trial']['x'][-1] == pytest.approx(.3)
    with pytest.raises(ValueError):
        observer.begin_child_turn()
    observer.end_interval(3, 1, 'mother_reply', actual_eye_gain=[0., 1.])
    assert observer.state['last_report']['actual_eye_gain'] == .5


def test_fixed_mark_rejects_moving_embedding():
    obj, _ = make()
    obj.model.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        NativeGenerativeObserver(obj)
