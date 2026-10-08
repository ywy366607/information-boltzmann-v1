"""Deterministic repair contracts; no claims about learned capability."""
import argparse
import copy
import numpy as np
import pytest
import torch

from test_fly_bptt_learning import make_model, physical
from information_boltzmann.core.fly_bptt_learning import (
    FlyBPTTLearner, FlyBPTTGraph, FlyPhysicalState, advance_fly_input_event,
)
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.triton_synapse import (
    PyTorchDelayedSynapticTransmission, execute_delayed_synaptic_transmission,
)
from scripts.ib.train_fly_bptt_stream import resolve_learning_modes


@pytest.mark.parametrize('saved,explicit,expected', [
    ({}, None, False), ({'detach_reset': True}, None, True),
    ({'detach_reset': True}, False, False), ({}, True, True),
])
def test_reset_derivative_convention_inherits_or_explicitly_overrides(saved, explicit, expected):
    args = argparse.Namespace(read_centering=None, dan_plastic_lr=None,
                              detach_reset=explicit)
    resolve_learning_modes(args, saved, {})
    assert args.detach_reset is expected


@pytest.mark.parametrize('saved,explicit,expected', [
    ({}, None, 'absolute'), ({'surrogate_mode': 'threshold'}, None, 'threshold'),
    ({'surrogate_mode': 'threshold'}, 'absolute', 'absolute'),
])
def test_surrogate_width_convention_inherits_or_explicitly_overrides(saved, explicit, expected):
    args = argparse.Namespace(read_centering=None, dan_plastic_lr=None,
                              surrogate_mode=explicit)
    resolve_learning_modes(args, saved, {})
    assert args.surrogate_mode == expected


@pytest.mark.parametrize('saved,explicit,expected', [
    ({}, None, 'atomic'), ({'transmission_mode': 'incoming'}, None, 'incoming'),
    ({'transmission_mode': 'incoming'}, 'atomic', 'atomic'),
])
def test_summation_convention_inherits_or_explicitly_overrides(saved, explicit, expected):
    args = argparse.Namespace(read_centering=None, dan_plastic_lr=None,
                              transmission_mode=explicit)
    resolve_learning_modes(args, saved, {})
    assert args.transmission_mode == expected


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('splits', [(0,), (0, 0, 0, 0, 0), (0, 1)])
def test_empty_and_short_delay_tiers_have_four_ring_gradients(device, splits):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    n = splits[-1]
    ring = tuple(torch.randn(1, 3, device=device, requires_grad=True) for _ in range(4))
    pre = torch.zeros(n, device=device, dtype=torch.int64)
    post = torch.ones(n, device=device, dtype=torch.int64)
    w = torch.ones(n, device=device, requires_grad=True)
    out = execute_delayed_synaptic_transmission(ring, pre, post, w, splits)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ring)
    assert w.grad is not None and w.grad.shape == w.shape
    for p in ring[1:]:
        assert torch.count_nonzero(p.grad) == 0
    if n:
        torch.testing.assert_close(w.grad, ring[0].detach()[0, :1])


def add_dan(model):
    model.has_dopamine, model.dan_scale = True, 2.0
    model.dan_delay_splits = (0, 1, 2, 2, 2)
    model.dan_plastic_lr = .1
    dev = model.output_read.weight.device
    for name, value in (
        ('dan_edge_pre', torch.tensor([0, 1], dtype=torch.long)),
        ('dan_edge_post', torch.tensor([5, 4], dtype=torch.long)),
        ('dan_edge_weight', torch.tensor([.4, .8])),
    ):
        model.register_buffer(name, value.to(dev), persistent=False)
    return model


def test_dan_consumes_all_delayed_edges_filters_quiet_ticks_and_stays_frozen(tmp_path):
    model = add_dan(make_model(tmp_path))
    initial = physical(model)
    initial.ring = tuple(torch.full_like(initial.h, float(k+1)) for k in range(4))
    learner = FlyBPTTLearner(model, initial)
    assert 'dan_edge_weight' in model._buffers
    assert not model.dan_edge_weight.requires_grad
    assert all(p is not model.dan_edge_weight for p in learner.trainable)
    token = torch.tensor([1])
    rates = model.get_decay_rates()
    next_state = advance_fly_input_event(model, learner.state, token, base_rates=rates)
    arrival = torch.zeros_like(initial.h)
    arrival[0, 5], arrival[0, 4] = .4/2., .8*2./2.
    expected = (1-rates[1].detach())*arrival
    torch.testing.assert_close(next_state.dan_gate, expected)
    assert not next_state.dan_gate.requires_grad
    manual = next_state
    # The second tick consumes the preceding tick's ring, with zero drive.
    arrival2 = execute_delayed_synaptic_transmission(
        manual.ring, model.dan_edge_pre, model.dan_edge_post,
        model.dan_edge_weight, model.dan_delay_splits)/model.dan_scale
    both = advance_fly_input_event(model, learner.state, token, settle_ticks=1, base_rates=rates)
    torch.testing.assert_close(both.dan_gate, rates[1]*expected+(1-rates[1])*arrival2)
    learner.previous_token = 0
    learner.observe([1, 2, 3, 4])
    assert model.dan_edge_weight.grad is None


def test_local_delta_and_inhibitory_conductance_are_correct(tmp_path):
    model = add_dan(make_model(tmp_path))
    state = physical(model)
    state.h.fill_(.2)
    state.dan_gate = torch.ones_like(state.h)*.3
    learner = FlyBPTTLearner(model, state)
    before = {name: getattr(model, name).detach().clone() for name in learner.edge_names}
    learner._dan_update(state)
    for name in learner.edge_names:
        expected = before[name].clone()
        if name in learner.dan_eligible:
            idx = learner.dan_eligible[name][0]
            expected[idx] += .1*.3*.2*.2
        torch.testing.assert_close(getattr(model, name), expected)
        assert (getattr(model, name) >= 0).all()
    assert model.edge_weight_i.abs().sum() > 0


@pytest.mark.parametrize('synapse', ['coba', 'cuba'])
@pytest.mark.parametrize('alif,stp', [(False, False), (False, True), (True, False), (True, True)])
def test_nontopographic_modes_keep_all_optional_physical_states(tmp_path, synapse, alif, stp):
    path = tmp_path/'graph.npz'
    np.savez(path, neuron_body_ids=np.arange(4), edge_pre=np.arange(4),
        edge_post=np.roll(np.arange(4), 1), edge_weight=np.array([.1, .1, -.1, -.1], np.float32),
        nt_sign=np.array([1, 1, -1, -1]), superclass_id=np.array([0, 0, 1, 1]),
        superclass_names=np.array(['cb_sensory', 'cb_motor']))
    model = FlyReservoirLM(path, vocab_size=9, d_model=3, injection='sensory',
        read_surface='output', synapse_model=synapse, use_alif=alif, use_stp=stp)
    z = torch.zeros(1, 4)
    u = torch.full_like(z, .4) if stp else model.get_stp_params()[0].detach().expand_as(z).clone()
    state = FlyPhysicalState(z.clone(), tuple(torch.ones_like(z)*.2 for _ in range(4)),
        z.clone(), z.clone(), z.clone(), torch.ones_like(z), u, z.clone())
    learner = FlyBPTTLearner(model, state, settle_ticks=1)
    learner.previous_token = 0
    old_projection = model.input_proj.weight.detach().clone()
    scores, metrics = learner.observe([1, 2, 3, 4])
    assert metrics['writer_grad_norm'] > 0
    assert not torch.equal(old_projection, model.input_proj.weight)
    assert np.isfinite(scores).all()
    assert learner.state.h.shape == learner.state.ge.shape == learner.state.x.shape == z.shape
    assert metrics['writer_gate_grad_norm'] == 0
    if stp:
        assert not torch.equal(learner.state.u, u)


def test_signed_clamp_retains_sign_after_crossing_and_resume(tmp_path):
    model = make_model(tmp_path)
    # A signed layout uses the original sign, including a negative edge at zero.
    model.synapse_model = 'cuba'
    model.register_buffer('edge_weight', torch.tensor([.1, -.1, 0.]))
    learner = FlyBPTTLearner(model, physical(model))
    with torch.no_grad():
        model.edge_weight.copy_(torch.tensor([-.2, .2, -.2]))
    learner.clamp_edges()
    torch.testing.assert_close(model.edge_weight, torch.zeros(3))
    saved = learner.state_dict()
    resumed = FlyBPTTLearner(copy.deepcopy(model), learner.state)
    resumed.load_edge_signs(saved)
    with torch.no_grad():
        resumed.model.edge_weight.copy_(torch.tensor([-.1, .1, -.1]))
    resumed.clamp_edges()
    torch.testing.assert_close(resumed.model.edge_weight, torch.zeros(3))
    with pytest.raises(ValueError, match='coverage'):
        resumed.load_edge_signs({'edge_positive': {}})
    with pytest.raises(ValueError, match='signed continuation'):
        resumed.load_edge_signs({})


def test_resume_inherits_centering_and_dan_modes_only_explicitly_overrides():
    args = argparse.Namespace(read_centering=None, dan_plastic_lr=None)
    resolve_learning_modes(args, {'read_centering': True, 'dan_plastic_lr': .02}, {})
    assert args.read_centering and args.dan_plastic_lr == .02
    args = argparse.Namespace(read_centering=False, dan_plastic_lr=0.)
    resolve_learning_modes(args, {'read_centering': True, 'dan_plastic_lr': .02}, {})
    assert not args.read_centering and args.dan_plastic_lr == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_dan_graph_replays_identical_updates_and_persistent_gate(tmp_path):
    torch.manual_seed(8)
    model = add_dan(make_model(tmp_path).cuda())
    other = copy.deepcopy(model)
    state = physical(model)
    state = FlyPhysicalState(**{k: tuple(t.cuda() for t in v) if k == 'ring' else v.cuda()
                              for k, v in state.state_dict().items()})
    state.ring = tuple(torch.full_like(state.h, .4) for _ in range(4))
    eager = FlyBPTTLearner(model, copy.deepcopy(state), settle_ticks=1)
    captured = FlyBPTTLearner(other, copy.deepcopy(state), settle_ticks=1)
    eager.previous_token = captured.previous_token = 0
    captured.runner = FlyBPTTGraph(captured, 4)
    for labels in ([1, 2, 3, 4], [5, 6, 7, 8]):
        a, _ = eager.observe(labels)
        b, _ = captured.observe(labels)
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(eager.state.dan_gate, captured.state.dan_gate, atol=2e-6, rtol=2e-5)
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
