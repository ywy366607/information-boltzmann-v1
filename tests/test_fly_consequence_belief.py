"""Numerical, policy-gradient and live/replay contracts, not capability experiments."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from information_boltzmann.core.fly_consequence_belief import (
    GaussianConsequences, NativeConsequenceCredit, NativeCircuitConsequenceCredit, first_commit_log_prob,
)
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ai_mode = 'motor'
        self.ai_w = nn.Parameter(torch.zeros(2, 4))
        self.ai_b = nn.Parameter(torch.zeros(2))
        self.decoder = nn.Linear(3, 4)
        self.read_norm = nn.LayerNorm(3)
        self.comp_pre = torch.tensor([0, 1, 0, 1])
        self.comp_post = torch.tensor([2, 2, 3, 3])
        self.comp_logit_trace = nn.Parameter(torch.tensor(3.))

    def get_thresholds(self):
        return torch.ones(1, 4)


def learner():
    h = torch.tensor([[.1, .4, .3, .2]])
    state = FlyPhysicalState(h, (h.clone(),), h.clone(), h.clone(), h.clone(), h.clone(), h.clone(), h.clone(),
                             comp_memory=torch.tensor([[.1, -.3, -.2, .4]]), dopamine=torch.zeros(2, 4))
    return SimpleNamespace(model=Model(), state=state, stream_pace_mean=2., eye_binary=False)


def choose(engine, said=True):
    context, _ = engine.features(engine.learner.state)
    engine.choose(context, torch.ones(3, 4), 1 if said else None)
    if said:
        engine.say(torch.tensor([[.2, -.3, .8]]), 2, .7)


def test_covariance_update_matches_gaussian_information_and_stays_psd():
    state = {}
    b = GaussianConsequences(state)
    x, y = torch.tensor([1., .3], dtype=torch.float64), torch.tensor([.2], dtype=torch.float64)
    b.initialize(x, y)
    before = torch.linalg.slogdet(state['P']).logabsdet
    r = b.observe(x, y)
    after = torch.linalg.slogdet(state['P']).logabsdet
    assert r['gain'] == pytest.approx(float((before - after) / 2))
    assert torch.linalg.eigvalsh(state['P']).min() > 0
    gains = [r['gain']] + [b.observe(x, y)['gain'] for _ in range(32)]
    assert gains[-1] < gains[0] / 10


def test_noise_reduces_value_and_observation_cannot_set_its_own_scale():
    x, y = torch.ones(2), torch.zeros(1)
    low, high = GaussianConsequences({}), GaussianConsequences({})
    low.initialize(x, y); high.initialize(x, y)
    high.state['noise_sum'] = 100.
    assert high.observe(x, y)['gain'] < low.observe(x, y)['gain']
    a, b = GaussianConsequences({}), GaussianConsequences({})
    assert a.observe(x, y)['gain'] == b.observe(x, torch.ones(1) * 100)['gain']


def test_first_commit_credit_includes_exact_survival_and_ignores_later_ticks():
    m = Model()
    frames = torch.ones(4, 4)
    score = first_commit_log_prob(m, frames, 1)
    assert float(score.detach()) == pytest.approx(2 * torch.log(torch.tensor(.5)).item())
    torch.testing.assert_close(score, first_commit_log_prob(m, frames[:2], 1))
    quiet = first_commit_log_prob(m, frames, None)
    assert float(quiet.detach()) == pytest.approx(4 * torch.log(torch.tensor(.5)).item())


@pytest.mark.parametrize('source,exposure', [('self_hearing', 1.), ('mother_call', 1.),
                                           ('mother_opening', 1.), ('mother_reply', 0.)])
def test_self_echo_calls_and_unseen_words_cannot_collect_information(source, exposure):
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine)
    before = copy.deepcopy(obj.consequence_state)
    engine.observe(obj.state, source, exposure)
    assert engine.gains == []
    assert obj.consequence_state['belief'] == before['belief']
    assert not engine.gradients


def test_external_consequence_credits_sampled_content_and_wait_without_forced_speech():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine)
    assert not engine.gradients  # the word itself is not a reward
    engine.observe(obj.state, 'mother_reply', 1.)
    engine.finish()
    grads = obj.pending_speak_credit['trace']
    assert grads['decoder.bias'][2] > 0
    assert grads['decoder.bias'][0] < 0
    assert grads['read_norm.weight'].abs().sum() > 0
    quiet_obj = learner(); quiet = NativeConsequenceCredit(quiet_obj)
    choose(quiet, said=False)
    quiet.observe(quiet_obj.state, 'mother_reply', 1.)
    assert quiet.gradients['ai_b'][1] < 0  # positive information can favor waiting


def test_native_phi_and_dopamine_are_context_not_fake_precision():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    original_context, original_y = engine.features(obj.state)
    obj.state.comp_memory.add_(.2)
    changed_context, y = engine.features(obj.state)
    assert not torch.equal(original_context, changed_context)
    torch.testing.assert_close(original_y, y)  # no phi novelty injected into the KC outcome
    obj.state.dopamine[0] += 1
    newer, _ = engine.features(obj.state)
    assert not torch.equal(changed_context, newer)
    assert 'P' not in obj.consequence_state['belief']


def test_full_interaction_resume_keeps_belief_and_word_eligibility():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine); engine.observe(obj.state, 'mother_reply', .8)
    saved = copy.deepcopy(FlyBPTTLearner.interaction_state_dict(obj))
    restored = learner(); restored.model.load_state_dict(obj.model.state_dict())
    FlyBPTTLearner.load_interaction_state(restored, saved)
    engine.observe(obj.state, 'mother_reply', .8)
    split = NativeConsequenceCredit(restored)
    split.observe(restored.state, 'mother_reply', .8)
    assert split.gains[0] == engine.gains[-1]
    for key in ('P', 'W'):
        torch.testing.assert_close(restored.consequence_state['belief'][key], obj.consequence_state['belief'][key])
    for key in obj.consequence_state['eligibility']:
        torch.testing.assert_close(restored.consequence_state['eligibility'][key], obj.consequence_state['eligibility'][key])


def test_mode_migration_preserves_unapplied_legacy_credit_once():
    obj = learner(); old = torch.ones_like(obj.model.ai_b)
    obj.pending_speak_credit = {'trace': {'ai_b': old.clone()}, 'decisions': [False]}
    engine = NativeConsequenceCredit(obj)
    engine.finish()
    torch.testing.assert_close(obj.pending_speak_credit['trace']['ai_b'], old)


def test_correlated_reply_is_one_observation_and_partial_reply_survives_resume():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine)
    for _ in range(4):
        engine.observe(obj.state, 'mother_reply', .8, end=False)
    assert not engine.gains and 'P' not in obj.consequence_state['belief']
    saved = copy.deepcopy(FlyBPTTLearner.interaction_state_dict(obj))
    restored = learner(); restored.model.load_state_dict(obj.model.state_dict())
    FlyBPTTLearner.load_interaction_state(restored, saved)
    split = NativeConsequenceCredit(restored)
    engine.observe(obj.state, 'mother_reply', .8, end=True)
    split.observe(restored.state, 'mother_reply', .8, end=True)
    assert obj.consequence_state['belief']['n'] == 1
    assert engine.gains == split.gains
    for key in engine.gradients:
        torch.testing.assert_close(engine.gradients[key], split.gradients[key])


def test_yield_preserves_spoken_content_until_the_next_child_turn():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    engine.begin_child_turn(); choose(engine)
    content = obj.consequence_state['action'][-3:].clone()
    choose(engine, said=False)
    torch.testing.assert_close(obj.consequence_state['action'][-3:], content)
    engine.begin_child_turn(); choose(engine, said=False)
    assert not bool(obj.consequence_state['action'][-3:].any())


def test_unseen_last_word_still_settles_previously_heard_response():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine); engine.observe(obj.state, 'mother_reply', 1., end=False)
    engine.observe(obj.state, 'mother_reply', 0., end=True)
    assert obj.consequence_state['belief']['n'] == 1
    assert 'response' not in obj.consequence_state
    engine.begin_child_turn(); choose(engine)  # no hanging response can block speech


def test_history_clock_advances_per_word_while_direct_archive_is_unchanged():
    obj = learner(); engine = NativeConsequenceCredit(obj)
    choose(engine)
    before = obj.consequence_state['eligibility']['decoder.bias'].clone()
    for _ in range(3):
        engine.observe(obj.state, 'mother_reply', 1., end=False)
    torch.testing.assert_close(obj.consequence_state['eligibility']['decoder.bias'], before * engine.decay**3)
    torch.testing.assert_close(obj.consequence_state['response']['score']['decoder.bias'], before)
    engine.end_response(); engine.end_response()
    assert obj.consequence_state['belief']['n'] == 1


def test_harness_separates_new_information_protocol_from_old_nll_credit(tmp_path):
    import json
    from scripts.ib.fly_interface.ling_harness import _consequence_stats
    p = tmp_path / 'ai_trace.jsonl'
    rows = [{'consequence': {'source': 'old_learning_progress', 'consequence_observations': 99}},
            {'consequence': {'source': 'native_action_consequence_parameter_ig',
                             'consequence_observations': 2, 'consequence_gain_mean': .4,
                             'consequence_updates': 2, 'content_decisions': 1}}]
    p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    stats = _consequence_stats(p)
    assert stats['observed_responses'] == 2
    assert stats['gain_mean'] == .4
    assert stats['content_decisions'] == 1
    assert not stats['behavior_verified']


def test_many_updates_keep_covariance_positive_and_ratio_finite():
    belief = GaussianConsequences({})
    for i in range(1000):
        x = torch.tensor([1., (i % 7) / 7])
        belief.observe(x, torch.tensor([.3]))
    assert torch.linalg.eigvalsh(belief.state['P']).min() > 0
    assert 0 <= belief.ratio(x) < float('inf')


def test_reply_reward_cause_survives_world_checkpoint_and_excludes_pure_yield():
    from test_fly_mother_continuation import world
    w = world(opening=None)
    w.contingent = False
    w.turn = 'fly'; w.yield_turn()
    assert w.mother_source == 'mother_reply' and not w.reply_has_speech
    while w.turn == 'mother':
        w.next_word()
    w.speak(w.tok.encode('hello')[0]); w.yield_turn()
    assert w.reply_has_speech
    restored = world(opening=None); restored.load_state_dict(w.state_dict())
    assert restored.reply_has_speech


def test_name_call_is_fully_perceived_even_without_epistemic_reward():
    from test_fly_mother_continuation import world
    w = world(opening=None)
    w.call_prompt = 'Mengli?'
    w.call()
    expected = w.tok.encode(' Mengli?\n')
    assert list(w.queue) == expected
    heard = [w.next_word()[0] for _ in expected]
    assert heard == expected and w.mother_source == 'mother_call'
    obj = learner(); engine = NativeConsequenceCredit(obj); choose(engine)
    engine.observe(obj.state, 'mother_call', 1.)
    assert not engine.gains  # zero novelty reward is distinct from removing the perceived words


def test_live_window_credits_previous_external_input_once_without_replay_update(monkeypatch):
    """Exercise actual live_window orchestration with a deterministic numerical engine stub."""
    import collections
    import numpy as np
    from information_boltzmann.core import fly_bptt_learning as fl
    from information_boltzmann.core.fly_mother_world import live_window
    from test_fly_mother_continuation import world

    obj = learner(); choose(NativeConsequenceCredit(obj))
    obj.previous_token, obj.ai_g = 1, torch.zeros(1, 4)
    obj.speak_credit, obj.hold_input, obj.read_lag = 'consequence', True, 0
    obj.writer_baseline_clock = 'physical'
    obj.model.get_decay_rates = lambda: ()
    obj.model.get_conductance_gains = lambda: ()
    obj.model.get_alif_params = lambda: ()
    obj.model.get_stp_params = lambda: ()

    def advance(model, state, token, schedule, **kw):
        ticks = kw['ticks']
        reads = torch.tensor([[.2, -.3, .8]]).repeat(ticks, 1)
        frames = [torch.ones(1, 4) for _ in range(ticks)]
        kw['ai_g_tape'].extend(frames)
        kw['ai_tape'].extend([i == 1 for i in range(ticks)])
        return state, reads[-1:], reads, torch.ones(ticks), torch.ones(ticks, dtype=torch.bool), torch.full((ticks,), .5), frames[-1]

    monkeypatch.setattr(fl, 'advance_fly_token_stream', advance)
    w = world(opening=None); w.turn = 'mother'; w.queue = collections.deque([2])
    w.mother_source, w.reply_has_speech = 'mother_reply', True
    targets, forced, speakers = live_window(obj, w, 2, np.random.default_rng(1), 0, 3, speak_act=True)
    assert targets[0] == 2 and speakers[0] == 'mother'
    assert forced['consequence_ratios'][0] == 0  # unknown initial source is never guessed
    assert forced['consequence_ratios'][1] > 0
    assert obj.last_epistemic['consequence_updates'] == 1
    assert obj.last_epistemic['content_decisions'] == 1
    assert obj.pending_speak_credit['trace']['decoder.weight'].abs().sum() > 0


def test_native_credit_preserves_posterior_but_transports_head_score_on_commit_clock():
    original = learner()
    routed = copy.deepcopy(original)
    old, new = NativeConsequenceCredit(original), NativeCircuitConsequenceCredit(routed)
    new.advance_ticks(3)
    new.decision_bounds = (0, 3)
    for engine in (old, new):
        choose(engine)
        engine.observe(engine.learner.state, 'mother_reply', .8, end=False)
        engine.advance_ticks(7) if isinstance(engine, NativeCircuitConsequenceCredit) else None
        engine.observe(engine.learner.state, 'mother_reply', .8, end=True)
        engine.finish()
    assert old.gains == new.gains
    assert new.state.get('action_schema') is None
    assert not hasattr(new, 'word_bias') and not hasattr(new, 'timing_bias')
    for key in ('P', 'W'):
        torch.testing.assert_close(old.state['belief'][key], new.state['belief'][key], rtol=0, atol=0)
    for key, value in original.pending_speak_credit['trace'].items():
        # Commit at1 in a3-tick interval leaves one physical tick of transport.
        rho = torch.sigmoid(routed.model.comp_logit_trace.detach())
        torch.testing.assert_close(value * rho, routed.pending_speak_credit['trace'][key])
    assert [e['kind'] for e in new.native_events] == [
        'clock', 'timing', 'content', 'response_begin', 'clock', 'response_end']
    assert new.native_events[1]['first'] == 1
    assert new.native_events[2]['tick'] == 1
    assert new.native_events[-1]['advantage'] == new.gains[-1]


def test_head_eligibility_ages_through_quiet_and_call_and_resumes_same_clock():
    obj = learner()
    engine = NativeCircuitConsequenceCredit(obj)
    engine.advance_ticks(3)
    engine.decision_bounds = (0, 3)
    choose(engine, said=False)
    original = obj.consequence_state['eligibility']['ai_b'].clone()
    rho = torch.sigmoid(obj.model.comp_logit_trace.detach())
    engine.advance_ticks(11)  # Quiet physical evolution, no external word.
    engine.observe(obj.state, 'mother_call', 1., end=True)
    saved = copy.deepcopy(FlyBPTTLearner.interaction_state_dict(obj))
    restored = learner()
    restored.model.load_state_dict(obj.model.state_dict())
    FlyBPTTLearner.load_interaction_state(restored, saved)
    split = NativeCircuitConsequenceCredit(restored)
    for branch in (engine, split):
        branch.advance_ticks(29)
        branch.observe(branch.learner.state, 'self_hearing', 1., end=True)
    expected = original * rho ** 40
    torch.testing.assert_close(engine.state['eligibility']['ai_b'], expected)
    torch.testing.assert_close(split.state['eligibility']['ai_b'], expected)
    assert engine.state['head_clock_ticks'] == split.state['head_clock_ticks'] == 43
    assert not engine.gains and not split.gains  # Quiet/call did not earn reward.


def test_head_response_archive_survives_physical_aging_without_double_word_decay():
    obj = learner()
    engine = NativeCircuitConsequenceCredit(obj)
    engine.advance_ticks(3)
    engine.decision_bounds = (0, 3)
    choose(engine)
    score = {k: v.clone() for k, v in engine.state['eligibility'].items()}
    engine.observe(obj.state, 'mother_reply', 1., end=False)
    for k, v in score.items():
        torch.testing.assert_close(engine.state['eligibility'][k], v)
    engine.advance_ticks(400)
    engine.observe(obj.state, 'mother_reply', 1., end=True)
    for k, v in score.items():
        torch.testing.assert_close(engine.gradients[k], v * engine.gains[0])
    assert engine.state['head_clock_ticks'] == 403


def test_native_credit_only_preserves_pending_legacy_response_without_inventing_score():
    obj = learner()
    old = NativeConsequenceCredit(obj)
    choose(old)
    old.observe(obj.state, 'mother_reply', .8, end=False)
    archived = copy.deepcopy(obj.consequence_state)
    new = NativeCircuitConsequenceCredit(obj)
    torch.testing.assert_close(archived['response']['action'], new.state['response']['action'])
    new.observe(obj.state, 'mother_reply', .8, end=True)
    assert [e['kind'] for e in new.native_events] == ['response_end']
    assert new.state['belief']['n'] == 1


def test_native_credit_only_rejects_speech_without_commit_and_schema_downgrade():
    obj = learner()
    engine = NativeCircuitConsequenceCredit(obj)
    with pytest.raises(ValueError, match='first-commit'):
        engine.say(torch.ones(1, 3), 1, .7)
    obj.consequence_state['action_schema'] = 'ordered-native-actions-v2'
    with pytest.raises(ValueError, match='downgrade'):
        NativeCircuitConsequenceCredit(obj)
