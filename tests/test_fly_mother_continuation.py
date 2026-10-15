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
