"""Native terminal/peripheral causality and replay contracts, not capability studies."""
import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_native_motor_port import (
    attach_native_motor_port, read_action, action_logits, motor_availability,
    MOTOR_PORT_PARAMETER_NAMES,
    checkpoint_model_weights,
)
from information_boltzmann.core.fly_active_inference import attach_motor_acts, ai_step, motor_rest, motor_logits
from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, advance_fly_token_stream


def attach(model, tmp_path):
    ann = tmp_path/'motor-ann.feather'
    pd.DataFrame({'bodyId': np.arange(6),
                  'superclass': ['cb_sensory']*3 + ['cb_motor', 'descending_neuron', 'cb_motor']}).to_feather(ann)
    before = torch.get_rng_state().clone()
    assert attach_native_motor_port(model, tmp_path/'graph.npz', ann) == 2
    assert torch.equal(before, torch.get_rng_state())
    return ann


def setup(tmp_path):
    model = make_model(tmp_path)
    attach_motor_acts(model, brain_gradient=True)
    attach(model, tmp_path)
    return model, physical(model)


def test_individual_motor_packets_and_prediction_are_separate(tmp_path):
    model, state = setup(tmp_path)
    with torch.no_grad():
        model.ai_w.fill_(.3)
    first = replace(state, ring=(torch.tensor([[0., 0., 0., 1., 0., 0.]]), *state.ring[1:]))
    other = replace(state, ring=(torch.tensor([[0., 0., 0., 0., 0., 1.]]), *state.ring[1:]))
    noise = replace(first, h=torch.randn_like(state.h)*100,
                    ring=(first.ring[0] + torch.tensor([[8., 4., 2., 0., 900., 0.]]), *first.ring[1:]))
    torch.testing.assert_close(read_action(model, first), read_action(model, noise), rtol=0, atol=0)
    assert not torch.equal(read_action(model, first), read_action(model, other))
    a, b = torch.tensor([[.1, .2, .3]]), torch.tensor([[.1, 900., .3]])
    assert motor_logits(model, a)[0, 1] == motor_logits(model, b)[0, 1]
    assert motor_logits(model, a)[0, 0] != motor_logits(model, b)[0, 0]  # attention remains independent
    before = model.decoder(model.read_norm(torch.ones(1, 3))).clone()
    model.motor_port_decoder.bias.data.add_(100.)
    torch.testing.assert_close(model.decoder(model.read_norm(torch.ones(1, 3))), before, rtol=0, atol=0)
    assert model.motor_port_decoder.weight is not model.decoder.weight


def test_no_motor_release_cannot_execute_even_with_large_bias(tmp_path):
    model, state = setup(tmp_path)
    model.ai_b.data[1] = 100.
    noise = replace(state, ring=(torch.tensor([[1., 1., 1., 0., 100., 0.]]), *state.ring[1:]))
    h, _, committed = ai_step(model, noise, motor_rest(model, state.h), outcome_bias=1000.)
    assert h.item() == 0 and not committed
    with pytest.raises(ValueError, match='no native motor release'):
        ai_step(model, noise, motor_rest(model, state.h), taped=True)
    active = replace(state, ring=(torch.tensor([[0., 0., 0., 1., 0., 0.]]), *state.ring[1:]))
    h, _, committed = ai_step(model, active, motor_rest(model, state.h))
    assert h.item() == 1 and committed
    model.native_motor_terminal_enabled = False
    h, _, committed = ai_step(model, active, motor_rest(model, state.h))
    assert h.item() == 0 and not committed


def test_stream_return_keeps_prediction_and_actual_packet_clocks(tmp_path):
    model, state = setup(tmp_path)
    tape = []
    out = advance_fly_token_stream(model, state, torch.tensor([1]), ticks=5, hold_input=True,
        read_every=True, ai_g=motor_rest(model, state.h), ai_tape=tape)
    assert out[2].shape == out[3].shape == (5, 3)
    assert out[4].shape == (5,) and out[4].dtype == torch.bool
    assert len(tape) == 5
    torch.testing.assert_close(out[3][-1:], read_action(model, out[0]), rtol=0, atol=0)
    assert out[4][-1] == motor_availability(model, out[0])[0]


def test_native_joint_score_uses_action_decoder_and_actual_availability():
    from test_fly_native_policy_credit import learner
    from information_boltzmann.core.fly_native_policy_credit import native_policy_credit
    obj = learner()
    m = obj.model
    m.native_motor_port_enabled = True
    m.motor_port_norm = nn.Identity()
    m.motor_port_decoder = copy.deepcopy(m.decoder)
    reads = torch.stack((m.comp_weight[0], m.comp_eta)).expand(2, 2)
    z = (m.comp_eta + m.ai_w).expand(2)
    values = torch.zeros(3)
    events = [{'kind':'clock','ticks':2},
              {'kind':'joint_timing','start':0,'length':2,'first':1,'tail_ticks':0,
               'word_value':values,'wait_value':0.,'temperature':1.},
              {'kind':'content','tick':1,'word':1,'temperature':1.},
              {'kind':'response_begin'}, {'kind':'response_end','advantage':2.}, {'kind':'trial_end'}]
    j, _ = native_policy_credit(obj, events, z, reads, [2], raw_logits=True,
                               include_policy_heads=True, motor_availability=torch.tensor([False,True]))
    new, old, brain = torch.autograd.grad(j, (m.motor_port_decoder.weight, m.decoder.weight,m.comp_eta), allow_unused=True)
    assert new.abs().sum() > 0 and old is None and brain.abs() > 0
    broken = copy.deepcopy(events)
    broken[1]['first'] = 0
    with pytest.raises(ValueError, match='actual terminal release'):
        native_policy_credit(learner(m), broken, z, reads, [2], raw_logits=True,
                             include_policy_heads=True, motor_availability=torch.tensor([False,True]))


def test_action_parameters_enter_optimizer_and_roundtrip(tmp_path):
    model, state = setup(tmp_path)
    obj = FlyBPTTLearner(model, state, stream_ticks=1, hold_input=True, lambda_jepa=0, use_ctm_loss=False)
    names = {n for n,_ in model.named_parameters() if n in MOTOR_PORT_PARAMETER_NAMES}
    assert names <= set(obj.adam_names)
    state_before = {n:p.detach().clone() for n,p in model.named_parameters() if n in names}
    (action_logits(model, read_action(model, replace(state,ring=(torch.ones_like(state.h),*state.ring[1:])))).square().sum()).backward()
    obj.optimizer.step()
    assert any(not torch.equal(dict(model.named_parameters())[n], v) for n,v in state_before.items())
    saved = copy.deepcopy(checkpoint_model_weights(model))
    clone = make_model(tmp_path)
    attach_motor_acts(clone, brain_gradient=True)
    attach_native_motor_port(clone,tmp_path/'graph.npz',tmp_path/'motor-ann.feather',saved=saved)
    for n in names:
        torch.testing.assert_close(clone.state_dict()[n], saved[n], rtol=0, atol=0)
    broken = dict(saved)
    broken['native_motor_body_ids'] = saved['native_motor_body_ids'].flip(0)
    with pytest.raises(ValueError, match='identities changed'):
        attach_native_motor_port(clone,tmp_path/'graph.npz',tmp_path/'motor-ann.feather',saved=broken)
    missing = dict(saved)
    del missing['native_motor_body_ids']
    with pytest.raises(ValueError, match='lacks anatomical identities'):
        attach_native_motor_port(clone,tmp_path/'graph.npz',tmp_path/'motor-ann.feather',saved=missing)


def test_recomputation_partition_preserves_context_clock_and_learning(tmp_path):
    from test_fly_native_context import context_learner
    from test_fly_native_generative_control import Mother, Tokens
    from information_boltzmann.core.fly_mother_world import MotherWorld, live_window
    from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
    obj = context_learner(tmp_path)
    attach(obj.model, tmp_path)
    obj.use_checkpointing = True
    other = copy.deepcopy(obj)
    results = []
    rng = torch.get_rng_state().clone()
    for learner, segment in ((obj, 16), (other, 4)):
        torch.set_rng_state(rng)
        learner.stream_segment_ticks = segment
        world = MotherWorld(Mother(), Tokens(), opening='read')
        targets, forced, _ = live_window(learner, world, 2, np.random.default_rng(4), 24, 24,
                                         speak_act=True)
        learner.forced_world = forced
        scores, _ = learner.observe(targets)
        results.append((scores, forced['intervals'], world.state_dict(), learner.physical_ticks))
    assert results[0][1] == results[1][1] and max(results[0][1]) > 16
    assert state_digest(results[0][2]) == state_digest(results[1][2])
    assert results[0][3] == results[1][3]
    torch.testing.assert_close(torch.tensor(results[0][0]), torch.tensor(results[1][0]), atol=1e-6, rtol=1e-5)
    for name, value in obj.state.state_dict().items():
        right = getattr(other.state, name)
        for a,b in zip(value, right) if name == 'ring' else [(value,right)]:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for a,b in zip(obj.model.parameters(),other.model.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('checkpointing', [False, True])
def test_world_execution_self_hearing_and_credit_replay_match(tmp_path, checkpointing):
    from test_fly_native_context import context_learner
    from test_fly_native_generative_control import Mother, Tokens
    from information_boltzmann.core.fly_mother_world import MotherWorld, live_window
    obj = context_learner(tmp_path)
    m = obj.model
    with torch.no_grad():
        m.embedding.weight.fill_(1.)
        for name in ('proj_vis', 'proj_chemo', 'proj_mech'):
            getattr(m.topographic_writer, name).weight.fill_(1.)
        m.log_threshold.fill_(-6.)
        m.ai_b[1] = 100.
    attach(m, tmp_path)
    new = [(n,p) for n,p in m.named_parameters() if n in MOTOR_PORT_PARAMETER_NAMES]
    for _,p in new:
        p.requires_grad_(True)
    obj.adam_names += [n for n,_ in new]
    obj.optimizer.add_param_group({'params':[p for _,p in new], 'parameter_names':[n for n,_ in new],
                                  'lr':1e-4, 'weight_decay':0.})
    with torch.no_grad():
        m.decoder.bias.fill_(-100.)
        m.decoder.bias[8] = 100.  # prediction strongly disagrees with actual speech
        m.motor_port_decoder.bias.fill_(-100.)
        m.motor_port_decoder.bias[1] = 100.
    obj.use_checkpointing = checkpointing
    world = MotherWorld(Mother(), Tokens(), opening=None, max_fly_words=1)
    world.turn = 'fly'
    targets, forced, _ = live_window(obj, world, 3, np.random.default_rng(4), 4, 4, speak_act=True)
    assert forced['native_motor_commands'] and forced['native_motor_commands'][0]['word'] == 1
    assert set(forced['native_motor_commands'][0]['body_ids']) <= {3,5}
    assert 'self_hearing' in forced['sources'] and 'mother_reply' in forced['sources']
    assert len(forced['native_motor_availability']) == sum(forced['intervals'])
    obj.forced_world = forced
    scores, stats = obj.observe(targets)
    assert np.isfinite(scores).all()
    assert stats['native_policy_native_content_connected'] > 0
