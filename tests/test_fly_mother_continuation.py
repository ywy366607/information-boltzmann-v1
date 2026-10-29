"""Numerical and continuation contracts; no synthetic capability study."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tiktoken
import torch

from information_boltzmann.core.fly_mother_world import (
    MotherWorld, sensory_segments, accumulate_speaking_credit,
)


class Mother:
    asynchronous = False

    def reply(self, history):
        return 'A real reply follows the child.'


def world(opening='Hello, Mengli.'):
    return MotherWorld(Mother(), tiktoken.get_encoding('gpt2'), opening=opening,
                       contingent=True, end_of_turn=('\n',), pause_beats_before_reply=0)


def test_multiline_mother_note_is_queued_and_restored_as_external_input():
    direct = world(opening=None)
    note = 'Here is a note:\n\nDear Mengli,\nI am here.\nMama'
    direct._say_mother(note, source='mother_reply')
    assert direct.transcript[-1] == ('mother', note)
    assert list(direct.queue) == direct.tok.encode(' ' + note + '\n')
    direct.next_word()
    saved = json.loads(json.dumps(direct.state_dict()))
    restored = world(opening=None)
    restored.load_state_dict(saved)
    assert restored.state_dict() == direct.state_dict()
    assert 'Dear Mengli' in restored.tok.decode(list(restored.queue))
    assert restored.mother_source == 'mother_reply'


@pytest.mark.parametrize('phase', ['mother', 'fly', 'silent'])
def test_world_restores_unread_queue_partial_speech_and_silence(phase):
    direct = world()
    direct.next_word()
    if phase != 'mother':
        while direct.turn == 'mother':
            direct.next_word()
        if phase == 'fly':
            direct.speak(direct.tok.encode('hello')[0])
        else:
            direct.yield_turn()
    saved = json.loads(json.dumps(direct.state_dict()))
    restored = world(opening=None)
    restored.load_state_dict(saved)
    assert restored.state_dict() == direct.state_dict()
    if phase == 'mother':
        assert restored.next_word() == direct.next_word()
    else:
        restored.yield_turn()
        direct.yield_turn()
    assert restored.state_dict() == direct.state_dict()
    assert restored.rng.random() == direct.rng.random()


def test_world_tags_automatic_calls_separately_from_replies():
    w = world()
    assert w.mother_source == 'mother_opening'
    while w.turn == 'mother':
        w.next_word()
    w.speak(w.tok.encode('hello')[0])
    w.yield_turn()
    assert w.mother_source == 'mother_reply'
    while w.turn == 'mother':
        w.next_word()
    for _ in range(3):
        w.yield_turn()
    assert w.mother_source == 'mother_call'


@pytest.mark.parametrize('reply', ['Listen to this gentle rhyme.', ''])
def test_quiet_presence_can_continue_world_without_speech_reward(reply):
    w = world(opening=None)
    w.turn = 'fly'
    w.mother.plan = {'mother': {'spontaneous_when_quiet': True}}
    seen = []
    w.mother.reply = lambda history: seen.append(history) or reply
    for _ in range(w.silent_beats_threshold):
        w.yield_turn()
    assert len(seen) == 1 and seen[0][-1] == ('fly', '<quiet_turn/>')
    assert w.transcript[-1] == ('mother', reply or w.get_call_prompt())
    assert w.mother_source == 'mother_call' and not w.reply_has_speech
    assert w.stats['mother_spontaneous_calls'] == bool(reply)
    assert w.fly_turns == [[], [], []]  # No speech or synthetic tokens invented.
    restored = world(opening=None)
    restored.load_state_dict(json.loads(json.dumps(w.state_dict())))
    assert restored.mother_source == 'mother_call' and not restored.reply_has_speech


def test_short_motor_pause_preserves_utterance_and_can_resume():
    w = world(opening=None)
    w.turn = 'fly'
    hello, again = w.tok.encode('hello')[0], w.tok.encode(' again')[0]
    w.speak(hello)
    w.pause_speech(1)
    assert w.fly_words == [hello] and not w.transcript and not w.queue
    assert w.turn == 'fly' and w.fly_pause_ticks == 1
    w.speak(again)
    assert w.fly_words == [hello, again] and w.fly_pause_ticks == 0
    assert w.stats['fly_resumed_pauses'] == 1
    w.speak(w.tok.encode('\n')[0])
    assert w.transcript[0] == ('fly', 'hello again')
    assert w.stats['end_word'] == 1 and w.stats['end_pause'] == 0


def test_quiet_events_separate_adult_replies_across_world_restore():
    w = world(opening=None)
    w.turn = 'fly'
    w.mother.plan = {'mother': {'spontaneous_when_quiet': True}}
    seen = []
    w.mother.reply = lambda history: seen.append(history) or 'The next event follows.'
    for _ in range(w.silent_beats_threshold):
        w.yield_turn()
    while w.turn == 'mother':
        w.next_word()
    saved = json.loads(json.dumps(w.state_dict()))
    restored = world(opening=None)
    restored.mother.plan = w.mother.plan
    restored.mother.reply = w.mother.reply
    restored.load_state_dict(saved)
    for _ in range(restored.silent_beats_threshold):
        restored.yield_turn()
    assert seen == [[('fly', '<quiet_turn/>')],
                    [('fly', '<quiet_turn/>'), ('mother', 'The next event follows.'),
                     ('fly', '<quiet_turn/>')]]
    assert restored.fly_turns == [[], [], [], [], [], []]
    assert restored.stats['mother_spontaneous_calls'] == 2
    assert restored.mother_source == 'mother_call' and not restored.reply_has_speech
    assert len(restored.transcript) == 8  # Rendering did not erase lived events.


def test_speech_pause_uses_elapsed_ticks_and_restores_mid_pause():
    direct = world(opening=None)
    direct.turn = 'fly'
    direct.speak(direct.tok.encode('hello')[0])
    direct.pause_speech(1)
    direct.pause_speech(8)
    restored = world(opening=None)
    restored.load_state_dict(json.loads(json.dumps(direct.state_dict())))
    assert restored.state_dict() == direct.state_dict()
    for obj in (direct, restored):
        obj.pause_speech(obj.speech_pause_limit() - 10)
        assert obj.turn == 'fly' and obj.fly_words
        obj.pause_speech(1)
        assert obj.turn == 'mother' and not obj.fly_words
        assert obj.stats['end_pause'] == 1 and obj.stats['end_yield'] == 0
        assert obj.stats['fly_pause_ticks'] == obj.speech_pause_limit()
        assert obj.reply_has_speech
    assert restored.state_dict() == direct.state_dict()


def test_old_world_state_defaults_to_no_pending_speech_pause():
    w = world(opening=None)
    w.turn = 'fly'
    saved = w.state_dict()
    del saved['fly_pause_ticks']
    w.load_state_dict(saved)
    assert w.fly_pause_ticks == 0


@pytest.mark.parametrize('commit_on_wait', [False, True])
def test_live_motor_pause_preserves_real_ticks_and_decisions(monkeypatch, commit_on_wait):
    from information_boltzmann.core import fly_bptt_learning as fl
    from information_boltzmann.core.fly_mother_world import live_window
    from test_fly_consequence_belief import learner
    obj = learner()
    obj.previous_token, obj.ai_g = 1, torch.zeros(1, 4)
    obj.speak_credit, obj.hold_input, obj.read_lag = None, True, 0
    obj.writer_baseline_clock = 'physical'
    for name in ('get_decay_rates', 'get_conductance_gains', 'get_alif_params', 'get_stp_params'):
        setattr(obj.model, name, lambda: ())
    calls = []

    def advance(model, state, token, schedule, **kw):
        ticks = kw['ticks']
        calls.append((ticks, kw['sensory_on']))
        commit = commit_on_wait and len(calls) == 2
        kw['ai_tape'].extend([commit] + [False] * (ticks - 1))
        reads = torch.ones(ticks, 3)
        return state, reads[-1:], reads, torch.ones(ticks), torch.ones(ticks, dtype=torch.bool), torch.full((ticks,), .5), kw['ai_g']

    monkeypatch.setattr(fl, 'advance_fly_token_stream', advance)
    w = world(opening=None)
    w.turn = 'fly'
    hello, again = w.tok.encode('hello')[0], w.tok.encode(' again')[0]
    w.speak(hello)
    monkeypatch.setattr(torch, 'multinomial', lambda *a, **kw: torch.tensor([again]))
    targets, forced, speakers = live_window(obj, w, 1, np.random.default_rng(1), 0, 1, speak_act=True)
    assert calls[0] == (1, True)
    assert all(not sensed for _, sensed in calls[1:])
    assert sum(t for t, _ in calls) == forced['intervals'][0]
    assert sum(d[2] for d in forced['decisions']) == forced['intervals'][0]
    assert sum(b - a for a, b in forced['quiet_ranges'][0]) == sum(t for t, _ in calls[1:])
    assert w.stats['end_yield'] == 0
    if commit_on_wait:
        assert targets == [again] and speakers == ['fly']
        assert w.fly_words == [hello, again] and not w.transcript
        assert forced['decisions'] == [(0, 0, 1, 0), (0, 1, 8, 1)]
    else:
        assert speakers == ['mother'] and w.transcript[0] == ('fly', 'hello')
        assert forced['intervals'] == [w.speech_pause_limit()]
        assert [d[2] for d in forced['decisions']] == [1, 8, 8, 7]
        assert w.stats['end_pause'] == 1


def test_newline_only_is_a_real_boundary_action_without_a_fake_spoken_reply():
    w = world(opening=None)
    w.turn = 'fly'
    newline = w.tok.encode('\n')[0]
    w.speak(newline)
    assert w.turn == 'fly' and not w.queue
    assert w.silent_beats == 1
    assert w.fly_turns[-1] == [newline]  # Preserve the actual selected action.
    assert w.turn_observation(0) == {'turn_kind': 'boundary_only', 'token_ids': [newline], 'spoken_text': ''}
    assert not w._reply_has_speech
    assert w.cached_history() == [('fly', '<quiet_turn/>')]
    w.yield_turn()
    assert w.silent_beats == 2
    w.speak(newline)
    assert w.mother_source == 'mother_call'
    assert w.stats['mother_calls'] == 1 and w.stats['fly_silent_turns'] == 3


def test_literal_ellipsis_and_name_speech_are_preserved_as_actual_utterances():
    w = world(opening=None)
    w.turn = 'fly'
    ellipsis = w.tok.encode('...')
    for token in ellipsis:
        w.speak(token)
    w.yield_turn()
    assert w._reply_has_speech
    assert w.turn_observation(0)['turn_kind'] == 'utterance'
    assert w.turn_observation(0)['token_ids'] == ellipsis
    assert w.cached_history()[0] == ('fly', '...')


def test_historical_boundary_placeholder_is_not_sent_as_child_speech_after_restore():
    w = world(opening=None)
    w.turn = 'fly'
    w.speak(w.tok.encode('\n')[0])
    w.yield_turn()
    saved = json.loads(json.dumps(w.state_dict()))
    restored = world(opening=None)
    restored.load_state_dict(saved)
    assert restored.state_dict() == w.state_dict()
    assert restored.cached_history() == [('fly', '<quiet_turn/>')]
    assert len(restored.transcript) == 2  # Actual experience was not erased.
    assert restored.turn_observation(1)['turn_kind'] == 'quiet_yield'


def test_sensory_segments_preserve_every_tick_and_silence_boundaries():
    segments = list(sensory_segments(19, 8, [(5, 11), (16, 19)]))
    assert segments == [(0, 5, True), (5, 3, False), (8, 3, False),
                        (11, 5, True), (16, 3, False)]
    assert sum(n for _, n, _ in segments) == 19
    for ranges in ([(3, 20)], [(5, 11), (10, 12)], [(-1, 1)], [(3, 3)]):
        with pytest.raises(ValueError):
            list(sensory_segments(19, 8, ranges))


def record(intervals, decisions=(), sources=None):
    return {'g': torch.arange(sum(intervals) * 2).reshape(-1, 2).float(),
            'intervals': intervals, 'decisions': list(decisions),
            'speakers': ['mother'] * len(intervals),
            'sources': sources or ['mother_reply'] * len(intervals)}


def test_delayed_reply_credit_crosses_window_and_restore_without_name_reward():
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    learner = SimpleNamespace(speak_epistemic_baseline=0.0, state=SimpleNamespace(h=torch.zeros(1)))
    first = record([4], [(0, 0, 4, 1)], ['self_hearing'])
    metrics = accumulate_speaking_credit(learner, first, [10.0], horizon_ticks=8)
    assert metrics['credit_pending_decisions'] == 1
    saved = copy.deepcopy(FlyBPTTLearner.interaction_state_dict(learner))
    restored = SimpleNamespace(state=SimpleNamespace(h=torch.zeros(1)))
    FlyBPTTLearner.load_interaction_state(restored, saved)
    second = record([4, 4], sources=['mother_call', 'mother_reply'])
    for obj in (learner, restored):
        metrics = accumulate_speaking_credit(obj, second, [100.0, 2.0], horizon_ticks=8)
        assert metrics['speak_said_return'] == 2.0
        assert metrics['credit_pending_decisions'] == 0
        assert obj.pending_speak_credit[2] == [2.0]
    torch.testing.assert_close(learner.pending_speak_credit[0], restored.pending_speak_credit[0])


def test_input_rng_and_pending_policy_credit_restore_exactly():
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    learner = SimpleNamespace(state=SimpleNamespace(h=torch.zeros(1)), last_voice=True,
                              pace_rng=np.random.default_rng(14), sense_rng=np.random.default_rng(24),
                              pending_speak_credit=(torch.ones(3, 2), [(0, 0, 3, 0)], [0.4]))
    learner.pace_rng.geometric(0.2, 7)
    learner.sense_rng.random(5)
    saved = copy.deepcopy(FlyBPTTLearner.interaction_state_dict(learner))
    restored = SimpleNamespace(state=learner.state)
    FlyBPTTLearner.load_interaction_state(restored, saved)
    assert restored.last_voice is True
    assert restored.pace_rng.geometric(0.2, 8).tolist() == learner.pace_rng.geometric(0.2, 8).tolist()
    assert restored.sense_rng.random() == learner.sense_rng.random()
    assert restored.pending_speak_credit[1:] == learner.pending_speak_credit[1:]


def test_activity_audit_distinguishes_latching_from_graded_release():
    from information_boltzmann.core.fly_activity_audit import CellActivityAudit
    audit = CellActivityAudit(torch.zeros(1, 4), torch.tensor([False, False, False, True]))
    for release in ([1., 0., 1., .2], [1., 0., 0., .3], [1., 0., 1., .4]):
        audit.observe(torch.tensor(release))
    result = audit.report({'all': [0, 1, 2, 3]})['groups']['all']
    assert result['graded_cells'] == 1 and result['spiking_cells'] == 3
    assert result['always_spiking_fraction'] == pytest.approx(1 / 3)
    assert result['never_spiking_fraction'] == pytest.approx(1 / 3)
    assert result['longest_spike_run_max'] == 3
    assert result['graded_release_temporal_sd_mean'] > 0


def test_activity_audit_detects_saturation_at_refractory_ceiling():
    from information_boltzmann.core.fly_activity_audit import CellActivityAudit
    audit = CellActivityAudit(torch.zeros(1, 2), refractory_ticks=1)
    for release in ([1., 0.], [0., 0.], [1., 1.], [0., 0.]):
        audit.observe(torch.tensor(release))
    result = audit.report({'all': [0, 1]})['groups']['all']
    assert result['always_spiking_fraction'] == 0
    assert result['near_ceiling_fraction'] == 0.5


def test_natural_name_address_is_preserved_without_output_filter():
    w = world(opening=None)
    w._say_mother('Mengli, look at this boat.')
    assert w.transcript[-1] == ('mother', 'Mengli, look at this boat.')


def test_teacher_context_repair_is_once_only_and_preserves_child_history_and_queue():
    w = world()
    w.mother.plan = {'mother': {'context_revision': 'Ling-reviewed-repair'}}
    w.transcript.extend([('fly', 'Meng'), ('mother', 'A mistaken reply.'), ('fly', 'say')])
    history, queue = list(w.transcript), list(w.queue)
    assert w.cached_history() == [('fly', 'say')]
    assert w.transcript == history and list(w.queue) == queue
    w.transcript.extend([('mother', 'A fresh reply.'), ('fly', 'more')])
    saved = w.state_dict()
    restored = world(opening=None)
    restored.mother.plan = w.mother.plan
    restored.load_state_dict(saved)
    assert restored.cached_history() == [('fly', 'say'), ('mother', 'A fresh reply.'), ('fly', 'more')]
    assert restored.stats['caregiver_context_repairs'] == 1
    assert restored.transcript == w.transcript
