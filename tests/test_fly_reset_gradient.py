"""Reset derivative and physical-state contracts, not capability studies."""

import copy

import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM, SpikeFn


def test_negative_subthreshold_reset_surrogate_adds_local_gain():
    alpha = .987
    h = torch.tensor(-.3 / alpha, dtype=torch.float64, requires_grad=True)
    v = alpha * h
    s = SpikeFn.apply(v - .1)
    old = v * (1 - s)
    candidate = v * (1 - s.detach())
    old_grad = torch.autograd.grad(old, h, retain_graph=True)[0]
    new_grad = torch.autograd.grad(candidate, h)[0]
    psi = 1 / (1 + (torch.pi * (v.detach() - .1))**2)
    torch.testing.assert_close(old_grad, alpha * (1 - s.detach() - v.detach() * psi))
    assert old_grad.item() > 1
    assert new_grad.item() == pytest.approx(alpha)
    assert torch.equal(old, candidate)


def test_threshold_width_has_unit_peak_positive_tails_and_attached_threshold():
    threshold = torch.tensor([.01, .1, 2.], dtype=torch.float64, requires_grad=True)
    voltage = threshold.detach().clone().requires_grad_()
    spike = SpikeFn.apply(voltage - threshold, threshold.detach())
    dv, dt = torch.autograd.grad(spike.sum(), (voltage, threshold))
    assert torch.equal(dv, torch.ones_like(dv))
    assert torch.equal(dt, -torch.ones_like(dt))
    margin = torch.tensor([-.1, 0., .1, 10.], dtype=torch.float64, requires_grad=True)
    width = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    derivative, width_derivative = torch.autograd.grad(
        SpikeFn.apply(margin, width).sum(), (margin, width), allow_unused=True)
    torch.testing.assert_close(derivative, 1 / (1 + (torch.pi * margin.detach()/.1)**2))
    assert (derivative > 0).all() and derivative[1] == 1
    assert width_derivative is None


def test_threshold_proxy_preserves_all_s14_w32_forward_values(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    torch.manual_seed(29)
    old = make_model(tmp_path, detach_reset=True)
    new = copy.deepcopy(old)
    new.surrogate_mode = 'threshold'
    state = physical(old)
    targets = torch.arange(32).remainder(9)[None]
    results = []
    for model in (old, new):
        learner = FlyBPTTLearner(model, copy.deepcopy(state), settle_ticks=14,
                                use_checkpointing=False)
        with torch.no_grad():
            results.append(learner.forward_window(targets, targets))
    assert torch.equal(results[0][0], results[1][0])
    for key, value in results[0][1].state_dict().items():
        other = getattr(results[1][1], key)
        pairs = zip(value, other) if key == 'ring' else [(value, other)]
        assert all(torch.equal(a, b) for a, b in pairs)


def test_threshold_proxy_retains_silent_first_spike_membrane_and_threshold_credit(tmp_path):
    from test_fly_bptt_learning import make_model, physical
    model = make_model(tmp_path, detach_reset=True, surrogate_mode='threshold').double()
    state = physical(model)
    h = state.h.double().requires_grad_()
    context = model.prepare_coba_tick(h, tuple(v.double() for v in state.ring),
                                     state.ge.double(), state.gi.double(),
                                     state.b.double(), state.x.double(), state.u.double())
    output = model.finish_coba_tick(context, torch.zeros_like(h))
    assert torch.count_nonzero(output[1]) == 0
    dh, dt = torch.autograd.grad(output[2][0].sum(), (h, model.log_threshold))
    assert torch.isfinite(dh).all() and dh.abs().sum() > 0
    assert torch.isfinite(dt).all() and dt.abs().sum() > 0


@pytest.mark.parametrize("synapse_model", ["coba", "cuba"])
def test_reset_option_preserves_forward_and_delayed_spike_credit(tmp_path, synapse_model):
    graph = tmp_path / "graph.npz"
    np.savez(graph, neuron_body_ids=np.arange(6), edge_pre=np.arange(6),
             edge_post=np.roll(np.arange(6), 1),
             edge_weight=np.full(6, .1, dtype=np.float32),
             nt_sign=np.array([1, 1, 1, -1, -1, -1]),
             edge_delay=np.ones(6, dtype=np.int32),
             delay_splits=np.array([0, 6, 6, 6, 6]),
             superclass_id=np.array([0, 0, 0, 1, 1, 1]),
             superclass_names=np.array(['cb_sensory', 'cb_motor']))
    torch.manual_seed(9)
    old = FlyReservoirLM(graph, vocab_size=9, d_model=3,
                         synapse_model=synapse_model, use_alif=True, use_stp=True)
    new = copy.deepcopy(old)
    new.detach_reset = True
    h1 = torch.linspace(-.3, .1, 6)[None].requires_grad_()
    h2 = h1.detach().clone().requires_grad_()
    ring = tuple(torch.full_like(h1, .05) for _ in range(4))
    token = torch.tensor([1])
    outputs = []
    for model, h in ((old, h1), (new, h2)):
        outputs.append(model.step(h, token, spike_ring=ring,
                      i_syn=torch.zeros_like(h), ge=torch.zeros_like(h),
                      gi=torch.zeros_like(h), b=torch.zeros_like(h),
                      x=torch.ones_like(h),
                      u=model.get_stp_params()[0].expand_as(h)))
    def flattened(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from flattened(item)
        else:
            yield value
    for a, b in zip(flattened(outputs[0]), flattened(outputs[1])):
        assert torch.equal(a, b)
    # Both attached emitted spikes and the pending transmitted ring retain
    # dependence on membrane state even when the reset mask is detached.
    for channel in (outputs[1][1], outputs[1][2][0]):
        grad = torch.autograd.grad(channel.sum(), h2, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Adaptation and facilitation likewise keep the current spike path.
    for channel in (outputs[1][-3], outputs[1][-1]):
        grad = torch.autograd.grad(channel.sum(), h2, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
