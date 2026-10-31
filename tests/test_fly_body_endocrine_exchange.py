"""Executed-actuator and LK receptor numerical/continuation contracts."""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_bptt_learning import advance_fly_token_stream, FlyPhysicalState
from information_boltzmann.core.fly_body_loop import (
    attach_body_loop, attach_body_actuator, step_body_loop, step_body_actuator,
)
from information_boltzmann.core.fly_endocrine_exchange import attach_endocrine_exchange, step_endocrine_exchange


def setup(tmp_path):
    model = make_model(tmp_path)
    model.body_proxy_compatibility = True  # historical compatibility tests
    ann = tmp_path/'ann.feather'
    pd.DataFrame({'bodyId': np.arange(6), 'type': ['LK', 'IPC', 'ITP', 'SNpp', 'DN', 'AN'],
                  'class': ['', '', '', 'mechanosensory_proprioceptive', '', ''],
                  'superclass': ['vnc_endocrine', 'cb_endocrine', 'cb_endocrine',
                                 'vnc_sensory', 'descending_neuron', 'ascending_neuron']}).to_feather(ann)
    graph = tmp_path/'graph.npz'
    assert attach_body_loop(model, graph, ann) == (1, 1)
    assert attach_body_actuator(model, graph, ann) == 1
    assert attach_endocrine_exchange(model, graph, ann) == (1, 2)
    return model, physical(model)


def test_body_responds_only_to_executed_command_and_has_causal_decay(tmp_path):
    model, state = setup(tmp_path)
    model.body_proprioceptive_gain.data.fill_(.5)
    state = replace(state, ring=(torch.ones_like(state.h), *state.ring[1:]))
    # An active brain/descending population alone cannot manufacture execution.
    _, muscle0, _ = step_body_actuator(model, state, None, False)
    assert muscle0.item() == 0
    drive, muscle, feedback = step_body_actuator(model, state, None, True)
    assert muscle.item() > 0 and drive[0, 3] > 0
    assert torch.count_nonzero(drive).item() == 1
    state = replace(state, body_muscle=muscle, body_reafference=feedback)
    _, later, movement = step_body_actuator(model, state, None, False)
    assert 0 < later.item() < muscle.item()
    assert movement[0, 1] < 0
    source = torch.zeros_like(state.h)
    model.body_feedback.data.fill_(10.)
    old_return, _, _ = step_body_loop(model, state, source)
    assert torch.equal(old_return, source)  # no parallel descending shortcut


def test_lk_occupancy_is_bounded_and_targets_only_receptor_supported_cells(tmp_path):
    model, state = setup(tmp_path)
    model.endocrine_response.data.fill_(.5)
    state = replace(state, ring=(torch.ones_like(state.h), *state.ring[1:]))
    for _ in range(20):
        thresholds, calcium, ligand, occupancy = step_endocrine_exchange(model, state, torch.ones_like(state.h))
        state = replace(state, endocrine_calcium=calcium, endocrine_lk=ligand, endocrine_receptor=occupancy)
    assert 0 < occupancy.item() < 1 and ligand.item() > 0
    assert (thresholds[:, 1:3] > 1).all()
    assert torch.equal(thresholds[:, [0, 3, 4, 5]], torch.ones(1, 4))
    thresholds.sum().backward()
    for name in ('endocrine_response', 'endocrine_log_tau', 'endocrine_log_affinity'):
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0


def test_lk_zero_response_is_frozen_equivalent(tmp_path):
    model, state = setup(tmp_path)
    threshold = torch.randn(1, 6)
    out, *_ = step_endocrine_exchange(model, state, threshold)
    assert torch.equal(out, threshold)


def test_actuator_and_ligand_state_survive_segment_and_checkpoint_roundtrip(tmp_path):
    model, state = setup(tmp_path)
    model.body_proprioceptive_gain.data.fill_(.2)
    token = torch.tensor([1])
    kwargs = dict(hold_input=True, update_expectation=False, executed_action=True)
    full = advance_fly_token_stream(model, state, token, ticks=7, **kwargs)[0]
    first = advance_fly_token_stream(model, state, token, ticks=3, **kwargs)[0]
    path = tmp_path/'state.pt'
    torch.save(first.detached().state_dict(), path)
    restored = FlyPhysicalState(**torch.load(path, weights_only=False))
    segmented = advance_fly_token_stream(model, restored, token, ticks=4, fresh=False, **kwargs)[0]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, segmented.state_dict()[key], rtol=0, atol=0)
    full.h.sum().backward()
    assert model.body_proprioceptive_gain.grad.abs().sum() > 0


def test_live_and_gradient_replay_use_same_executed_source_tape(tmp_path, monkeypatch):
    from test_fly_native_context import context_learner
    from test_fly_native_generative_control import Mother, Tokens
    from information_boltzmann.core.fly_mother_world import MotherWorld, live_window
    obj = context_learner(tmp_path)
    model = obj.model
    model.body_proxy_compatibility = True  # historical compatibility tests
    model.body_loop_enabled = model.body_actuator_enabled = True
    model.register_buffer('body_descending_index', torch.tensor([4]), persistent=False)
    model.register_buffer('body_ascending_index', torch.tensor([5]), persistent=False)
    model.register_buffer('body_proprioceptive_index', torch.tensor([0]), persistent=False)
    model.body_logit_decay = torch.nn.Parameter(torch.tensor(3.))
    model.body_feedback = torch.nn.Parameter(torch.zeros(1))
    model.body_muscle_logit_decay = torch.nn.Parameter(torch.tensor(2.))
    model.body_proprioceptive_gain = torch.nn.Parameter(torch.full((2, 1), .02))
    # Keep this fixture's optimizer unchanged; compare physical live/replay
    # continuity. Full native-parameter migration is checked on real restore.
    model.body_muscle_logit_decay.requires_grad_(True)
    model.body_proprioceptive_gain.requires_grad_(True)
    model.ai_b.data[1] = 100.
    world = MotherWorld(Mother(), Tokens(), opening=None, max_fly_words=1)
    world.turn = 'fly'
    from information_boltzmann.core import fly_bptt_learning as fl
    original = fl.advance_fly_token_stream
    live_muscles = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        # Forecasts also call this function; only the live world samples every
        # physical tick. A hypothetical command must not count as execution.
        if kwargs.get('read_every', False):
            live_muscles.append(result[0].body_muscle.detach().clone())
        return result
    monkeypatch.setattr(fl, 'advance_fly_token_stream', capture)
    targets, forced, _ = live_window(obj, world, 2, np.random.default_rng(4), 4, 4, speak_act=True)
    assert forced['input_sources'][1] == 'self_hearing'
    assert live_muscles[-1].item() > 0
    monkeypatch.setattr(fl, 'advance_fly_token_stream', original)
    obj.forced_world = forced
    obj.observe(targets)
    assert obj.state.body_muscle.item() > 0
    torch.testing.assert_close(obj.state.body_muscle, live_muscles[-1], rtol=0, atol=0)


def test_native_execution_never_injects_scalar_actuator_proxy(tmp_path):
    model, state = setup(tmp_path)
    del model.body_proxy_compatibility
    model.body_proprioceptive_gain.data.fill_(100.)
    source = torch.randn_like(state.h)
    out, muscle, feedback = step_body_actuator(model, state, source, True)
    assert out is source
    assert muscle is state.body_muscle
    assert feedback is state.body_reafference
