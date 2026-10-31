"""Numerical/interface acceptance of the read-only native circuit audit."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib.fly_interface import native_causal_audit as audit


def physical():
    return audit.fl.FlyPhysicalState(
        h=torch.ones(1, 3), ring=tuple(torch.ones(1, 3) for _ in range(4)),
        ge=torch.zeros(1, 3), gi=torch.zeros(1, 3), b=torch.zeros(1, 3),
        x=torch.ones(1, 3), u=torch.ones(1, 3), baseline=torch.zeros(1, 3),
        comp_memory=torch.tensor([[-0.2]]))


def test_native_restore_ast_stops_before_mutation_or_world():
    source = (audit.ROOT / 'scripts/ib/train_fly_bptt_stream.py').read_text(encoding='utf8')
    parser, resume = audit.trainer_restore_program(source)
    assert 'load_interaction_state' in resume.co_names
    assert 'ai_g' in resume.co_names
    for name in ('atomic_save', 'atomic_json', 'live_window', 'MotherWorld', 'mkdir'):
        assert name not in resume.co_names
    with pytest.raises((RuntimeError, StopIteration)):
        audit.trainer_restore_program(source.replace('Loading archived weights', 'CHANGED'))


def test_shared_mother_sensory_wiring_preserves_learned_gains_and_matches_flags(monkeypatch, tmp_path):
    from scripts.ib import train_fly_bptt_stream as trainer
    model = torch.nn.Module()
    model.voice_log_gain = torch.nn.Parameter(torch.tensor(.8))
    model.reward_log_gain = torch.nn.Parameter(torch.tensor(.2))
    voice, reward = model.voice_log_gain, model.reward_log_gain
    called = []
    def attach_reward(m, graph, annotations, **kw):
        called.append(('reward', kw))
        m.reward_ticks = kw['ticks']
        m.register_buffer('reward_cells', torch.tensor([1, 2]), persistent=False)
        return 2
    def attach_voice(m, graph, annotations, **kw):
        called.append(('voice', kw))
        m.register_buffer('voice_cells', torch.tensor([3]), persistent=False)
        return 1
    monkeypatch.setattr(trainer, 'attach_reply_reward', attach_reward)
    monkeypatch.setattr(trainer, 'attach_mother_voice', attach_voice)
    args = SimpleNamespace(mother_world='deepseek', mother_reward=True, mother_voice=True,
                           graph=tmp_path/'graph.npz', sense_gains_learnable=True,
                           mother_reward_level=1., mother_reward_ticks=4, mother_voice_level=1.5)
    assert trainer.attach_mother_senses(model, args) == {'reward_cells': 2, 'voice_cells': 1}
    assert model.voice_log_gain is voice and model.reward_log_gain is reward
    assert set(model.state_dict()) == {'voice_log_gain', 'reward_log_gain'}
    args.mother_world = None
    assert trainer.attach_mother_senses(model, args) == {} and len(called) == 2


def test_pending_clock_uses_saved_rng_without_mutating_checkpoint():
    rng = np.random.default_rng(42)
    saved = {'mother_world': {'turn': 'mother', 'queue': list(range(12)), 'mother_source': 'mother_reply',
                             'mother_turn_fresh': False},
             'mother_rng': copy.deepcopy(rng.bit_generator.state),
             'config': {'stream_pace_min': 12, 'stream_pace_mean': 16},
             'learner': {'previous_token': 99, 'interaction': {'last_voice': True, 'last_reward': False}}}
    untouched = copy.deepcopy(saved)
    chunks = audit.pending_queue_tape(saved, 10, 8)
    pace = np.exp(rng.uniform(np.log(12), np.log(16)))
    expected = [int(rng.geometric(1 / pace)) for _ in range(8)]
    assert chunks[0]['forced']['intervals'] == expected
    assert chunks[0]['previous_token'] == 99
    assert chunks[1]['previous_token'] == 7
    assert chunks[1]['targets'] == [8, 9]
    assert chunks[1]['last_voice'] is True
    assert saved == untouched
    saved['mother_world']['mother_turn_fresh'] = True
    with pytest.raises(ValueError, match='pause is not fully archived'):
        audit.pending_queue_tape(saved, 10, 8)


def test_hard_clamp_precedes_local_state_update_and_restores_hooks(monkeypatch):
    model = SimpleNamespace()
    def finish(context, drive, **kwargs):
        return (torch.ones(1, 3), torch.ones(1, 3), tuple(torch.ones(1, 3) for _ in range(4)))
    model.finish_coba_tick = finish
    def step(m, s, *args, **kwargs):
        h, spikes, ring = m.finish_coba_tick(None, None)
        # Local plasticity must see the clamped output already in this tick.
        phi = s.comp_memory + spikes[:, 0:1]
        return audit.fl.FlyPhysicalState(**{**s.state_dict(), 'h': h, 'ring': ring, 'comp_memory': phi})
    monkeypatch.setattr(audit.fl, 'step_fly_physical_tick', step)
    s = physical()
    groups = {'KC': torch.tensor([0]), 'DAN': torch.tensor([1])}
    with audit.circuit_intervention(model, s, groups, 'KC_off') as observed:
        out = audit.fl.step_fly_physical_tick(model, s)
        assert out.h[0, 0] == 0
        assert out.ring[0][0, 0] == 0
        assert out.comp_memory.item() == pytest.approx(-0.2)
        assert observed['clamped_target_release'][0] == 0
        assert observed['raw_target_release'][0] == 1
    assert model.finish_coba_tick is finish
    assert audit.fl.step_fly_physical_tick is step


@pytest.mark.parametrize('arm, expected', [('phi_zero', 0.0), ('phi_frozen', -0.2)])
def test_phi_intervention_controls_current_and_next_memory(monkeypatch, arm, expected):
    model = SimpleNamespace(finish_coba_tick=lambda *a, **k: None)
    seen = []
    def step(m, s):
        seen.append(s.comp_memory.item())
        return audit.fl.FlyPhysicalState(**{**s.state_dict(), 'comp_memory': s.comp_memory + 0.3})
    monkeypatch.setattr(audit.fl, 'step_fly_physical_tick', step)
    s = physical()
    with audit.circuit_intervention(model, s, {'KC': torch.tensor([0])}, arm):
        s = audit.fl.step_fly_physical_tick(model, s)
        s = audit.fl.step_fly_physical_tick(model, s)
    assert seen == pytest.approx([expected, expected])
    assert s.comp_memory.item() == pytest.approx(expected)


def test_clone_preserves_source_and_all_physical_fields():
    source = physical()
    cloned = audit.clone_state(source)
    cloned.h.zero_(); cloned.ring[0].zero_(); cloned.comp_memory.zero_()
    assert source.h.sum() == 3
    assert source.ring[0].sum() == 3
    assert source.comp_memory.item() == pytest.approx(-0.2)


def test_terminal_route_is_enabled_only_inside_explicit_intervention(monkeypatch):
    prepared = torch.tensor([0])
    model = SimpleNamespace(finish_coba_tick=lambda *a, **k: None, kcd_pre=None,
                            _audit_terminal_pre=prepared)
    seen = []
    def step(m, s):
        seen.append(m.kcd_pre)
        return s
    monkeypatch.setattr(audit.fl, 'step_fly_physical_tick', step)
    original_terminal = audit.fl.kenyon_terminal_release
    with audit.circuit_intervention(model, physical(), {'KC': torch.tensor([0])}, 'kc_terminal_on'):
        audit.fl.step_fly_physical_tick(model, physical())
    assert seen[0] is prepared and model.kcd_pre is None
    assert audit.fl.kenyon_terminal_release is original_terminal
    del model._audit_terminal_pre
    with pytest.raises(ValueError, match='explicitly prepared'):
        with audit.circuit_intervention(model, physical(), {}, 'kc_terminal_on'):
            pass


def test_terminal_off_preserves_the_restored_route_after_intervention(monkeypatch):
    prepared = torch.tensor([0])
    model = SimpleNamespace(finish_coba_tick=lambda *a, **k: None, kcd_pre=prepared)
    monkeypatch.setattr(audit.fl, 'step_fly_physical_tick', lambda m, s: s)
    with audit.circuit_intervention(model, physical(), {}, 'kc_terminal_off'):
        assert model.kcd_pre is None
    assert model.kcd_pre is prepared


def test_historical_clock_recovers_ids_and_rewards_without_transcript(tmp_path):
    rows = [
        {'bptt_train_tokens': 6, 'interval': [1, 1], 'quiet_ranges': [[], []],
         'token': [8, 9], 'pace': 14, 'speakers': ['mother', 'mother'],
         'sources': ['mother_reply', 'mother_reply']},
        {'bptt_train_tokens': 8, 'interval': [3, 2], 'quiet_ranges': [[[1, 2]], []],
         'token': [10, 10, 11, 11], 'pace': 14, 'speakers': ['mother', 'fly'],
         'sources': ['mother_reply', 'self_hearing']},
        {'bptt_train_tokens': 10, 'interval': [1], 'token': [12], 'pace': 13,
         'speakers': ['mother'], 'sources': ['mother_reply']},
    ]
    trace = tmp_path / 'ai_trace.jsonl'
    trace.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    (tmp_path / 'scores_c4.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'trained': 8, 'sources': rows[0]['sources'], 'reply_onsets': [False, False]},
        {'trained': 10, 'sources': rows[1]['sources'], 'reply_onsets': [True, False]}]), encoding='utf8')
    chunks = audit.historical_tape(trace, 2, 12)
    assert chunks[0]['previous_token'] == 10
    assert chunks[0]['targets'] == [11, 12]
    assert chunks[0]['forced']['quiet_ranges'] == [[[1, 2]], []]
    assert chunks[0]['forced']['reply_onsets'] == [True, False]
