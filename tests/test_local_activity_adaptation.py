"""Local activity feedback: mathematical/interface tests, no capability claims."""
from dataclasses import replace

import pytest
import torch

from information_boltzmann.core.conductance_response import LocalConductanceResponse
from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticBelief, PlasticMediumPorts3D
from information_boltzmann.core.plastic_feasibility import continuous_rhs
from information_boltzmann.runtime import ContinuousStream


def medium():
    torch.manual_seed(577)
    return PlasticMedium3D((2, 2, 2), channels=3, hidden=8,
                           bath_type='conductance', adaptive_conduction=True,
                           activity_adaptation=True).double()


def ports():
    torch.manual_seed(577)
    return PlasticMediumPorts3D(vocab_size=17, shape=(4, 4, 4), channels=8,
                                hidden=8, heads=2, queries=2,
                                bath_type='conductance', activity_adaptation=True).double()


def test_activity_measures_flow_instead_of_static_storage_and_is_scale_free():
    net = medium()
    empty = net.initial_state()
    field = torch.randn_like(empty.field)
    rule = net.conductance_response
    assert torch.count_nonzero(rule.processing_activity(field * 1e3, empty.flux)) == 0
    flux = tuple(torch.randn_like(x) for x in empty.flux)
    value = rule.processing_activity(field, flux)
    assert (value >= 0).all() and (value <= 1).all()
    torch.testing.assert_close(value, rule.processing_activity(field * 11, tuple(x * 11 for x in flux)))
    assert rule.processing_activity(empty.field, empty.flux).isfinite().all()


def test_inhibitory_history_provides_negative_feedback_and_recovers_at_rest():
    net = medium()
    rule = net.conductance_response
    # Freeze native opening rates to isolate the additional feedback equation.
    with torch.no_grad():
        for parameter in rule.opening.parameters():
            parameter.zero_()
    c = rule.coefficients(net.material_field())
    empty = net.initial_state()
    gates = torch.full_like(empty.receptors, 0.9)
    quiet = rule.opening_rates(empty.field, empty.flux, c, gates)
    moving = rule.opening_rates(empty.field, tuple(torch.ones_like(x) for x in empty.flux), c, gates)
    assert (moving[..., 1, :] > quiet[..., 1, :]).all()
    fresh = rule.opening_rates(empty.field, empty.flux, c, empty.receptors)
    assert (quiet[..., 0, :] < fresh[..., 0, :]).all()
    dt = net._duration(0.2, empty.field)
    recovered = rule.gates_step(gates, quiet, c.closing, dt)
    assert (recovered[..., 1, :] < gates[..., 1, :]).all()
    target_i = quiet[..., 1, :] / (quiet[..., 1, :] + c.closing[None, ..., 1, :])
    expected_i = target_i + (gates[..., 1, :] - target_i) * (
        -dt * (quiet[..., 1, :] + c.closing[None, ..., 1, :])).exp()
    torch.testing.assert_close(recovered[..., 1, :], expected_i, atol=1e-15, rtol=1e-15)


def test_feedback_keeps_receptor_interval_and_electrical_energy_ledger():
    net = medium()
    empty = net.initial_state()
    old = replace(empty, field=torch.randn_like(empty.field),
                  flux=tuple(torch.randn_like(x) for x in empty.flux),
                  receptors=torch.rand_like(empty.receptors))
    rule = net.conductance_response
    field, flux, gates, info = rule(old.field, old.flux, old.receptors,
                                  net._duration(5.0, old.field), rule.coefficients(net.material_field()))
    assert ((gates >= 0) & (gates <= 1)).all()
    new = replace(old, field=field, flux=flux, receptors=gates)
    residual = net.energy(new) - net.energy(old) - info['response_source_work'] + info['response_joule_heat']
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-12, rtol=0)


def test_routing_uses_reciprocal_local_contrast_with_bounded_evidence():
    net = medium()
    empty = net.initial_state()
    rule = net.conduction_plasticity
    gain, _, metric = rule.coefficients(net.material_field())
    field = torch.randn_like(empty.field)
    base = rule.evidence(field, empty.flux, metric)
    uniform = torch.full(field.shape[:-1], 0.8, dtype=field.dtype)
    torch.testing.assert_close(rule.adapted_evidence(field, empty.flux, metric, gain, uniform), base)
    history = torch.zeros_like(uniform)
    history[:, 0] = 1
    evidence = rule.adapted_evidence(field, empty.flux, metric, gain, history)
    assert ((evidence >= -1) & (evidence <= 1)).all()
    assert (evidence[..., 0] >= base[..., 0]).all()
    # Swap endpoints: contrast uses an absolute difference, hence reciprocal.
    forward = (history - torch.roll(history, -1, 1)).abs()
    backward = torch.roll((history - torch.roll(history, 1, 1)).abs(), -1, 1)
    torch.testing.assert_close(forward, backward)


def test_write_competition_moves_probability_without_blanket_shutdown():
    net = ports()
    agent = net.write_agent
    features = torch.zeros(1, 16, dtype=torch.float64)
    zero = torch.zeros(1, 8, dtype=torch.float64)
    gate = agent.chart_policy(features, zero)
    torch.testing.assert_close(gate, agent.chart_policy(features, zero + 0.9))
    fatigue = zero.clone()
    fatigue[:, 0] = 0.9
    changed = agent.chart_policy(features, fatigue)
    assert changed[0, 0] < gate[0, 0]
    assert (changed[0, 1:] > gate[0, 1:]).all()
    torch.testing.assert_close(changed.sum(-1), torch.ones(1, dtype=torch.float64))
    field = net.initial_belief().medium.field
    a, b = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
    chart = lambda feature: agent._packet_chart(net.source, field, features, feature, fatigue)[0]
    torch.testing.assert_close(chart(a + b), chart(a) + chart(b), atol=2e-14, rtol=2e-14)
    changed[0, 0].backward()
    assert agent.log_activity_sensitivity.grad.abs() > 0


def test_feedback_generator_matches_actual_integrator_at_zero():
    net = medium()
    empty = net.initial_state()
    state = replace(empty, field=torch.randn_like(empty.field),
                    flux=tuple(torch.randn_like(x) for x in empty.flux),
                    receptors=torch.rand_like(empty.receptors))
    def integrator(time):
        output, _ = net(state, time)
        return torch.cat((net._pack(output).flatten(), output.receptors.flatten(), output.conduction.flatten()))
    _, derivative = torch.autograd.functional.jvp(integrator, torch.tensor(0.0, dtype=torch.float64),
                                                  torch.tensor(1.0, dtype=torch.float64))
    rhs = continuous_rhs(net, state)
    expected = torch.cat((net._pack(rhs).flatten(), rhs.receptors.flatten(), rhs.conduction.flatten()))
    torch.testing.assert_close(derivative, expected, atol=4e-13, rtol=4e-13)


def test_joint_likelihood_trains_feedback_and_preserves_full_state():
    net = ports()
    belief = net.initial_belief()
    state = replace(belief.medium, field=torch.randn_like(belief.medium.field),
                    flux=tuple(torch.randn_like(x) for x in belief.medium.flux),
                    receptors=torch.rand_like(belief.medium.receptors))
    loss, output, _ = net(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]),
                          PlasticBelief(state, belief.precision), event_duration=0.05)
    loss.backward()
    for parameter in (net.write_agent.log_activity_sensitivity,
                      net.medium.conductance_response.log_adaptation_gain.bias,
                      net.medium.conduction_plasticity.gain.bias):
        assert parameter.grad is not None and parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0
    assert output.medium.receptors.shape == state.receptors.shape
    assert output.medium.elapsed[0] == pytest.approx(0.1)
    snapshot = net.port_snapshot(output)
    assert snapshot['write_port_inhibition'].shape == (1, 8)
    assert snapshot['inhibition_spatial_std'] > 0


def test_write_inhibition_observes_only_its_local_footprint():
    net = ports()
    belief = net.initial_belief()
    receptors = belief.medium.receptors.clone().requires_grad_()
    activity = net.write_port_activity(PlasticBelief(replace(belief.medium, receptors=receptors), belief.precision))
    activity[:, 0].sum().backward()
    outside = (net.write_agent.local_ports.footprint()[0] == 0).reshape(4, 4, 4)
    assert torch.count_nonzero(receptors.grad[:, outside]) == 0


def test_continuation_rejects_silent_feedback_change():
    net = ports()
    stream = ContinuousStream(net, max_step=0.01)
    stream.observe(0, torch.tensor([1]))
    stream.advance_to(0.02)
    saved = stream.state_dict()
    restored = ContinuousStream.from_state_dict(net, saved)
    torch.testing.assert_close(restored.belief.medium.receptors, stream.belief.medium.receptors)
    saved['activity_adaptation'] = False
    with pytest.raises(ValueError, match='activity feedback'):
        ContinuousStream.from_state_dict(net, saved)
