"""Continuous-context numerical, causality and native gradient interfaces."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_fly_native_generative_control import make, Mother, Tokens
from information_boltzmann.core.fly_native_context import (
    NativeContextControl, compartment_chart, apply_context, native_context_teaching,
)
from information_boltzmann.core.fly_mother_world import MotherWorld, live_window


def context_learner(tmp_path):
    obj = make(tmp_path)
    obj.native_generative_enabled = False
    obj.native_context_enabled = True
    # An interface-only chart on the six-cell fixture; real learning is checked
    # separately on the retained complete individual, not inferred from this.
    obj.model.register_buffer('comp_post', torch.tensor([2, 2, 3]))
    obj.state = replace(obj.state, comp_memory=torch.tensor([[.1, .3, -.2]]))
    return obj


def test_fixed_chart_counterfactual_matches_requested_absolute_field(tmp_path):
    obj = context_learner(tmp_path)
    before = obj.state.comp_memory.clone()
    field = torch.tensor([1.2, -.7])
    changed = apply_context(obj.model, obj.state, field)
    torch.testing.assert_close(compartment_chart(obj.model, changed)[0], field)
    torch.testing.assert_close(obj.state.comp_memory, before)
    assert torch.all(changed.comp_memory > -1.)
    # Within-compartment efficacy ratios are preserved.
    torch.testing.assert_close((1 + changed.comp_memory[0, 0]) / (1 + changed.comp_memory[0, 1]),
                               (1 + before[0, 0]) / (1 + before[0, 1]))


def test_pulse_mode_requires_an_explicit_write_tick_exposure_model(tmp_path):
    obj = context_learner(tmp_path)
    obj.hold_input = False
    with pytest.raises(ValueError, match='held input'):
        NativeContextControl(obj)


def test_actor_cannot_use_unprocessed_teacher_target_or_covariance(tmp_path):
    obj = context_learner(tmp_path)
    engine = NativeContextControl(obj)
    world = engine.world_model
    direction = torch.tensor([1., 0.], dtype=torch.float64)
    expected = world.actor_nodes(obj.state, 4, direction)
    world.filter.state['mean'].fill_(12.)
    world.filter.state['covariance'].mul_(.001)
    world.state['pending_teaching'] = {'mean': torch.full((2,), 90.)}
    torch.testing.assert_close(world.actor_nodes(obj.state, 4, direction), expected)
    assert world.filter.state['clock_ticks'] == 0
    assert world.control['clock_ticks'] == 0


def test_recognition_job_publishes_once_after_matching_actual_sensation(tmp_path):
    obj = context_learner(tmp_path)
    engine = NativeContextControl(obj)
    engine.begin_interval(obj.state, 0, False, False)
    engine.advance_ticks(4)
    engine.arrive(obj.state, 2, 'mother_call', 4, [], [1.] * 4, end=False)
    job = copy.deepcopy(engine.world_model.state['pending_teaching'])
    assert engine.teaching == [None]
    assert engine.world_model.state['published_clock'] == 0
    engine.state['source'] = 'mother_call'
    engine.begin_interval(obj.state, 2, True, False)
    engine.advance_ticks(3)
    engine.arrive(obj.state, 1, 'self_hearing', 3, [], [1.] * 3, end=False)
    assert engine.world_model.state['published_clock'] == 7
    assert engine.world_model.state['teaching_jobs'] == 1
    assert engine.world_model.state['pending_teaching'] is None
    torch.testing.assert_close(engine.teaching[1]['target'], job['mean'])
    torch.testing.assert_close(engine.world_model.state['published_covariance'],
                               job['covariance'] + torch.eye(2) * job['process_variance'] * 3)
    assert engine.world_model.filter.state['updates'] == 1


def test_native_teacher_is_exact_vjp_and_excludes_language_ports(tmp_path):
    obj = context_learner(tmp_path)
    model = obj.model
    model.comp_eta = torch.nn.Parameter(torch.tensor(.2))
    model.embedding.weight.requires_grad_(True)
    model.decoder.weight.requires_grad_(True)
    field = (model.comp_eta + model.embedding.weight[0, 0] + model.decoder.weight[0, 0]).expand(2)
    state = apply_context(model, obj.state, field)
    job = {'target': torch.tensor([.9, -.2]), 'covariance': torch.eye(2)}
    delta = compartment_chart(model, state)[0] - job['target']
    expected = torch.autograd.grad(.5 * delta.square().sum(), model.comp_eta, retain_graph=True)[0]
    surrogate, diag = native_context_teaching(model, [state], [job])
    grads = torch.autograd.grad(surrogate, (model.comp_eta, model.embedding.weight, model.decoder.weight),
                                allow_unused=True)
    torch.testing.assert_close(grads[0], expected)
    assert grads[1:] == (None, None)
    assert diag['context_teacher_jobs'] == 1 and diag['context_teacher_ports_and_decoder'] is False
    with pytest.raises(ValueError, match='every actual'):
        native_context_teaching(model, [state], [])


def test_zero_exposure_cannot_publish_hidden_label_even_through_later_history(tmp_path, monkeypatch):
    obj = context_learner(tmp_path)
    engine = NativeContextControl(obj)
    engine.begin_child_turn()
    monkeypatch.setattr(engine.world_model, 'likelihood', lambda *a, **kw:
                        torch.tensor([[.9, .1], [.5, .5], [.1, .9]], dtype=torch.float64).log())
    engine.begin_interval(obj.state, 0, False, False)
    engine.advance_ticks(4)
    engine.arrive(obj.state, 0, 'mother_call', 4, [], [1.] * 4, end=True)
    hidden = copy.deepcopy(engine.world_model.state['pending_teaching'])
    assert hidden['information'] > 0 and hidden['deferred_credit'] is True
    end_event = next(e for e in engine.native_events if e['kind'] == 'response_end')
    assert end_event['advantage'] == 0. and end_event['defer_key'] == hidden['id']
    prior_covariance = engine.world_model.state['published_covariance'].clone()
    engine.state['source'] = 'mother_call'
    engine.begin_interval(obj.state, 0, True, False)
    engine.advance_ticks(3)
    engine.arrive(obj.state, 1, 'self_hearing', 3, [], [0.] * 3, end=False)
    assert engine.world_model.state['teaching_jobs'] == 0
    assert engine.world_model.filter.state['updates'] == 0
    assert engine.world_model.state['published_clock'] == 0
    torch.testing.assert_close(engine.world_model.state['published_covariance'], prior_covariance)
    torch.testing.assert_close(engine.world_model.filter.mean, hidden['unconditioned_filter']['mean'])
    assert engine.native_events[-1] == {'kind': 'deferred_response_end', 'key': hidden['id'], 'advantage': 0.}
    assert engine.teaching == [None, None]


@pytest.mark.parametrize('gain', [0., 2.])
def test_unprocessed_response_credit_is_quarantined_across_optimizer_windows(gain):
    from test_fly_native_policy_credit import learner as score_learner
    from information_boltzmann.core.fly_native_policy_credit import native_policy_credit
    obj = score_learner()
    m = obj.model
    hazards = torch.sigmoid((m.comp_eta + m.ai_w).expand(1))
    reads = torch.stack((m.comp_weight[0], m.comp_eta)).expand(1, 2)
    expected = torch.autograd.grad(hazards[0].log() * gain, m.comp_eta, retain_graph=True)[0]
    events = [{'kind': 'clock', 'ticks': 1},
              {'kind': 'timing', 'start': 0, 'length': 1, 'first': 0},
              {'kind': 'response_begin'},
              {'kind': 'response_end', 'advantage': 0., 'defer_key': 'Y'},
              {'kind': 'trial_end'}]
    first, _ = native_policy_credit(obj, events, hazards, reads, [1])
    assert float(first.detach()) == 0.
    assert 'Y' in obj.consequence_state['native_credit']['deferred_response_scores']
    archive = copy.deepcopy(obj.consequence_state)
    obj.consequence_state = archive
    second, _ = native_policy_credit(obj, [{'kind': 'clock', 'ticks': 1},
        {'kind': 'deferred_response_end', 'key': 'Y', 'advantage': gain}], hazards, reads, [1])
    torch.testing.assert_close(torch.autograd.grad(second, m.comp_eta)[0], expected)
    assert not obj.consequence_state['native_credit']['deferred_response_scores']


def test_real_event_replay_and_durable_pending_recognition(tmp_path):
    obj = context_learner(tmp_path)
    with torch.no_grad():
        obj.model.ai_b[1] = 100.
    world = MotherWorld(Mother(), Tokens(), opening=None, max_fly_words=1)
    world.turn = 'fly'
    targets, forced, _ = live_window(obj, world, 2, np.random.default_rng(4), 4, 4, speak_act=True)
    assert forced['native_context_control'] is True
    assert forced['input_sources'][1] == forced['sources'][0] == 'self_hearing'
    assert obj.native_context_state['pending_teaching']['token'] == targets[-1]
    archive = copy.deepcopy(obj.interaction_state_dict())
    obj.native_context_state = {}
    obj.load_interaction_state(archive)
    assert obj.native_context_state['pending_teaching']['token'] == targets[-1]
    obj.forced_world = forced
    scores, stats = obj.observe(targets)
    assert np.isfinite(scores).all()
    assert stats['epistemic'] == 0.
    assert 'context_teacher_jobs' in stats
    assert stats['native_policy_native_content_connected'] > 0
