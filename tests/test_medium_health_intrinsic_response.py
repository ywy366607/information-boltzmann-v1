"""Full-state clock response and observer invariants, without training."""
import copy
from dataclasses import replace
import math
import os

import pytest
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.medium_health import conditional_state_response
from information_boltzmann.runtime.online_credit import credit_tensors, replace_credit
from information_boltzmann.runtime.training import belief_tensors


def setup(device='cpu', intrinsic=True):
    torch.manual_seed(731)
    net = PlasticMediumPorts3D(vocab_size=13, shape=(2, 2, 2), channels=4,
        material_width=2, hidden=4, heads=1, queries=1,
        bath_type='conductance', activity_adaptation=True, short_term_plasticity=True,
        read_mode='temporal', temporal_rates=[1., 3.], temporal_frequencies=[0., 2.],
        intrinsic_time_reference=.043 if intrinsic else None,
        solver_max_step=.011, observer_max_step=.019).to(device)
    if device == 'cpu':
        net.double()
    belief = net.initial_belief()
    with torch.no_grad():
        if intrinsic:
            net.intrinsic_time.head.weight.copy_(net.intrinsic_time.head.weight.new_tensor([[.8, -.3, .2, .1]]))
            net.intrinsic_time.head.bias.fill_(.15)
        field = .3 * torch.randn_like(belief.medium.field)
        flux = tuple(.1 * torch.randn_like(value) for value in belief.medium.flux)
        state = replace(belief.medium, field=field, flux=flux,
                        receptors=.3 + .4 * torch.rand_like(belief.medium.receptors),
                        transmission=.3 + .4 * torch.rand_like(belief.medium.transmission))
        belief = replace(belief, medium=state,
                         temporal=replace(belief.temporal, value=.02 * torch.randn_like(belief.temporal.value)))
    return net, belief, torch.tensor([2], device=device)


def inputs_and_direction(belief):
    values = (*credit_tensors(belief), belief.temporal.value.real, belief.temporal.value.imag)
    generator = torch.Generator(device=values[0].device).manual_seed(0)
    direction = tuple(torch.randn(value.shape, dtype=value.dtype, device=value.device,
                                   generator=generator) for value in values)
    norm = sum(value.square().mean() for value in direction).sqrt()
    return values, tuple(value / norm for value in direction)


def modified_belief(belief, values):
    physical = replace_credit(belief, values[:-2])
    return replace(physical, temporal=replace(belief.temporal,
                   value=torch.complex(values[-2], values[-1])))


def transition(net, belief, observed, *, frozen_duration=None):
    duration = net.event_time(belief, .005) if frozen_duration is None else frozen_duration
    written, _ = net.assimilate(belief, observed, diagnostics=True, training_terms=False)
    outgoing, _ = net.advance(written, duration, diagnostics=True)
    return (*credit_tensors(outgoing), outgoing.temporal.value.real, outgoing.temporal.value.imag)


def test_intrinsic_response_uses_preinput_time_and_complete_clock_jvp():
    net, belief, observed = setup()
    with torch.no_grad():
        duration = net.event_time(belief, .005).detach()
        written, _ = net.assimilate(belief, observed, diagnostics=False, training_terms=False)
        postwrite_duration = net.event_time(written, .005)
    assert abs(float(duration - postwrite_duration)) > 1e-7
    assert abs(float(duration) - .005) > .01
    before_state = [value.clone() for value in belief_tensors(belief)]
    before_parameters = copy.deepcopy(net.state_dict())
    before_rng = torch.get_rng_state().clone()
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_grads = [parameter.grad.clone() for parameter in net.parameters()]
    result = conditional_state_response(net, belief, observed, event_duration=.005, directions=2)
    assert result['duration'] == float(duration)
    assert result['direction_durations'] == [float(duration)] * 2
    assert result['nominal_event_duration'] == .005
    for gain, normalized in zip(result['gains'], result['log_gain_per_physical_time']):
        assert normalized == math.log(gain) / float(duration)
    values, direction = inputs_and_direction(belief)
    epsilon = 1e-6
    with torch.no_grad():
        plus = modified_belief(belief, tuple(x + epsilon * v for x, v in zip(values, direction)))
        minus = modified_belief(belief, tuple(x - epsilon * v for x, v in zip(values, direction)))
        forward, backward = transition(net, plus, observed), transition(net, minus, observed)
        expected = sum(((x - y) / (2 * epsilon)).square().mean()
                       for x, y in zip(forward, backward)).sqrt()
        frozen_plus = transition(net, plus, observed, frozen_duration=duration)
        frozen_minus = transition(net, minus, observed, frozen_duration=duration)
        frozen = sum(((x - y) / (2 * epsilon)).square().mean()
                     for x, y in zip(frozen_plus, frozen_minus)).sqrt()
    assert result['gains'][0] == pytest.approx(float(expected), rel=2e-6, abs=1e-8)
    assert abs(result['gains'][0] - float(frozen)) > 1e-7
    assert torch.equal(torch.get_rng_state(), before_rng)
    for current, saved in zip(belief_tensors(belief), before_state):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
    for name, saved in before_parameters.items():
        torch.testing.assert_close(net.state_dict()[name], saved, rtol=0, atol=0)
    for parameter, saved in zip(net.parameters(), before_grads):
        torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)


def test_fixed_clock_response_keeps_legacy_duration_and_normalization():
    net, belief, observed = setup(intrinsic=False)
    result = conditional_state_response(net, belief, observed, event_duration=.005, directions=1)
    assert result['duration'] == .005
    assert result['direction_durations'] == [.005]
    assert result['log_gain_per_physical_time'][0] == math.log(result['gains'][0]) / .005


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1',
                    reason='Opt-in CUDA allocation; resource calibration owns GPU')
def test_intrinsic_response_cuda_matches_native_forward_ad_with_stp():
    net, belief, observed = setup('cuda')
    net.medium.execution_backend = 'fused'
    before = [value.clone() for value in belief_tensors(belief)]
    # Catch any accidental AOT forward-mode entry, including a warm cache.
    def forbidden(*unused):
        raise AssertionError('STP dual entered the compiled backward path')
    net.medium._compiled_advance = forbidden
    net.medium.short_term_plasticity._compiled_step = forbidden
    result = conditional_state_response(net, belief, observed, event_duration=.005, directions=1)
    net.medium.short_term_plasticity.fuse_execution = False
    reference = conditional_state_response(net, belief, observed, event_duration=.005, directions=1)
    assert result['duration'] == reference['duration']
    assert result['gains'] == reference['gains']
    for current, saved in zip(belief_tensors(belief), before):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
