"""Analytic kinetics, electrical work, actual E/I response and stream integration."""
import copy
from dataclasses import replace

import pytest
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.plastic_feasibility import continuous_rhs
from information_boltzmann.core.conductance_response import LocalConductanceResponse
from information_boltzmann.core.conductance_feasibility import conductance_energy_bound, local_ei_certificate
from information_boltzmann.runtime import ContinuousStream


def model():
    torch.manual_seed(487)
    return PlasticMedium3D((2, 2, 2), 3, hidden=8, bath_type='conductance',
                            adaptive_conduction=True).double()


def state(net):
    empty = net.initial_state(2)
    return replace(empty, field=torch.randn_like(empty.field),
                   flux=tuple(torch.randn_like(x) for x in empty.flux),
                   receptors=torch.rand_like(empty.receptors))


def test_receptor_kinetics_exact_semigroup_and_invariant_interval():
    s = torch.rand(2, 2, 2, 2, 2, 3, dtype=torch.float64)
    opening = torch.rand_like(s) + 0.1
    closing = torch.rand(2, 2, 2, 2, 3, dtype=torch.float64) + 0.1
    dt = torch.tensor(0.31, dtype=torch.float64).reshape(1, 1, 1, 1, 1)
    fn = LocalConductanceResponse.gates_step
    full = fn(s, opening, closing, dt)
    half = fn(fn(s, opening, closing, dt / 2), opening, closing, dt / 2)
    torch.testing.assert_close(full, half, atol=2e-15, rtol=2e-15)
    long = fn(s, opening, closing, dt * 1e6)
    assert torch.all((long >= 0) & (long <= 1))
    torch.testing.assert_close(long, opening / (opening + closing[None]))
    torch.testing.assert_close(fn(s, opening, closing, dt * 0), s, atol=0, rtol=0)


def test_balanced_rest_and_learnable_independent_material_times():
    net = model()
    quiet, _ = net.advance(net.initial_state(), 0.2, substeps=4)
    assert quiet.field.abs().max() < 1e-14
    assert quiet.receptors.min() > 0  # Electrical balance permits active receptors at rest.
    before = net.prepare_evolution().response
    with torch.no_grad():
        net.conductance_response.log_parameters.bias[0] += torch.log(torch.tensor(2.0))
    after = net.prepare_evolution().response
    torch.testing.assert_close((after.capacitance/after.leak)[...,0],
                               2*(before.capacitance/before.leak)[...,0])
    torch.testing.assert_close(after.closing, before.closing)


def test_uniform_initial_coefficients_have_an_active_spatial_material_gradient():
    net = model()
    rule = net.conductance_response
    empty = net.initial_state()
    c = rule.coefficients(net.material_field())
    # Removing the double-zero initialization retains the SAME initial law.
    torch.testing.assert_close(c.capacitance, torch.ones_like(c.capacitance), atol=0, rtol=0)
    torch.testing.assert_close(c.leak, torch.ones_like(c.leak), atol=0, rtol=0)
    assert torch.count_nonzero(net.material.coefficients) == 0
    assert rule.log_parameters.weight.abs().sum() > 0
    # Isolate the electrical coefficient path, with no collision, routing,
    # opening actor or training loop able to supply an alternative gradient.
    field = torch.ones_like(empty.field)
    field[:, 0, 0, 0] = 2
    gates = torch.full_like(empty.receptors, 0.25)
    output, _, _ = rule.electrical_step(field, empty.flux, gates,
                                       net._duration(0.1, field), c, diagnostics=False)
    output.square().mean().backward()
    assert torch.isfinite(net.material.coefficients.grad).all()
    assert net.material.coefficients.grad[1:].abs().sum() > 0


def test_persisted_response_modulates_conservative_collision():
    net = model()
    old = state(net)
    other = replace(old, receptors=torch.zeros_like(old.receptors))
    first, second = net.collide(old, 0.1), net.collide(other, 0.1)
    assert (first.field-second.field).abs().max() > 1e-6
    torch.testing.assert_close(net.energy(first), net.energy(old), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(net.energy(second), net.energy(other), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('duration', [0.0, 1e-8, 0.3, 1e3])
def test_frozen_electrical_solution_closes_source_heat_ledger(duration):
    net = model()
    old = state(net)
    rule = net.conductance_response
    c = rule.coefficients(net.material_field())
    dt = net._duration(duration, old.field)
    field, flux, info = rule.electrical_step(old.field, old.flux, old.receptors, dt, c)
    new = replace(old, field=field, flux=flux)
    residual = net.energy(new) - net.energy(old) - info['response_source_work'] + info['response_joule_heat']
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10, rtol=0)
    assert torch.all(info['response_joule_heat'] >= 0)
    total = c.leak[None] + (old.receptors * c.maximum[None]).sum(-2)
    target = (old.receptors * c.maximum[None] * c.reversal[None]).sum(-2) / total
    expected = target + (old.field / c.capacitance.sqrt()[None] - target) * (-dt * total / c.capacitance[None]).exp()
    torch.testing.assert_close(field, expected * c.capacitance.sqrt()[None])
    assert torch.isfinite(field).all() and torch.isfinite(flux[0]).all()


def test_inhibition_is_reversal_dependent_and_changes_effective_time():
    net = model()
    old = state(net)
    c = net.conductance_response.coefficients(net.material_field())
    field = torch.full_like(old.field, 0.5)
    no_gates = torch.zeros_like(old.receptors)
    inhibitory = no_gates.clone()
    inhibitory[..., 1, :] = 1
    off, _, _ = net.conductance_response.rhs(field, old.flux, no_gates, c)
    on, _, _ = net.conductance_response.rhs(field, old.flux, inhibitory, c)
    assert torch.all(on < off)
    below_reversal = torch.full_like(field, -2.0)
    off, _, _ = net.conductance_response.rhs(below_reversal, old.flux, no_gates, c)
    on, _, _ = net.conductance_response.rhs(below_reversal, old.flux, inhibitory, c)
    assert torch.all(on > off)  # A real conductance reverses current below EI.
    tau_off = c.capacitance / c.leak
    tau_on = c.capacitance / (c.leak + c.maximum[..., 1, :])
    assert torch.all(tau_on < tau_off)


def test_time_and_electrical_unit_conversion_preserves_normalized_dynamics():
    net = model()
    old = state(net)
    first, _ = net.respond(old, 0.05)
    other = copy.deepcopy(net)
    rule = other.conductance_response
    rule.time_reference = 7.0
    rule.capacitance_reference = 4.0
    rule.voltage_reference = 0.5  # sqrt(C0)*V0 is the same energy-coordinate unit.
    second, _ = other.respond(old, 7 * 0.05)
    for a, b in zip((first.field, *first.flux, first.receptors),
                    (second.field, *second.flux, second.receptors)):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)


def test_full_split_matches_actual_generator_and_refines_fixed_duration():
    net = model()
    old = state(net)
    derivative = continuous_rhs(net, old)
    tiny, _ = net.advance(old, 1e-7)
    for a, b, d in zip((old.field, *old.flux, old.receptors, old.conduction),
                       (tiny.field, *tiny.flux, tiny.receptors, tiny.conduction),
                       (derivative.field, *derivative.flux, derivative.receptors, derivative.conduction)):
        torch.testing.assert_close((b-a)/1e-7, d, atol=1e-5, rtol=2e-5)
    results = [net.advance(old, 0.02, substeps=k)[0] for k in (4, 8, 64)]
    errors = [(x.field-results[-1].field).norm() for x in results[:2]]
    assert errors[1] < errors[0]
    _, info = net.advance(old, 0.02, substeps=8)
    assert info['response_energy_residual'].abs().max() < 1e-12


def test_energy_bound_and_actual_inhibition_stabilized_oscillatory_witness():
    net = model()
    old = state(net)
    c = net.conductance_response.coefficients(net.material_field())
    bounds = conductance_energy_bound(c)
    df, dj, _ = net.conductance_response.rhs(old.field, old.flux, old.receptors, c)
    power = (old.field*df + sum(x*y for x,y in zip(old.flux,dj))).flatten(1).sum(1) / 8
    m, u2 = bounds['minimum_release_rate'], bounds['source_squared_bound']
    assert torch.all(power <= -m*net.energy(old) + u2/(2*m) + 1e-12)
    dt = 3.0
    advanced, _ = net.advance(old, dt)
    retained = (-m*dt).exp()
    upper = retained*net.energy(old) + (1-retained)*u2/(2*m.square())
    assert torch.all(net.energy(advanced) <= upper + 1e-12)
    proof = local_ei_certificate()
    assert proof['fixed_point_residual'].abs().max() < 1e-14
    torch.testing.assert_close(proof['jacobian'], torch.tensor(
        [[-2.,1.,-1.],[15.,-6.,0.],[20.,0.,-2.]], dtype=torch.float64))
    assert proof['excitation_only_eigenvalues'].real.max() > 0
    assert proof['closed_loop_eigenvalues'].real.max() < 0
    assert proof['closed_loop_eigenvalues'].imag.abs().max() > 0


def test_timestamped_likelihood_trains_response_and_retains_gates_in_continuation(tmp_path):
    torch.manual_seed(488)
    net = PlasticMediumPorts3D(vocab_size=17, shape=(2,2,2), channels=4, hidden=8,
                              bath_type='conductance').double()
    loss, belief, _ = net.forward_timestamped(torch.tensor([[2,3]]), torch.tensor([[3,4]]),
                                             [0,0.02], [0.01,0.03], max_step=0.01)
    loss.backward()
    assert net.medium.conductance_response.opening[-1].weight.grad.abs().sum() > 0
    assert net.medium.conductance_response.log_parameters.bias.grad.abs().sum() > 0
    assert torch.all((belief.medium.receptors >= 0) & (belief.medium.receptors <= 1))
    assert belief.detach().medium.receptors.grad_fn is None
    net.eval()
    with torch.no_grad():
        stream = ContinuousStream(net, max_step=0.01, belief=belief.detach())
        path = tmp_path/'continued.pt'
        torch.save(stream.state_dict(), path)
        resumed = ContinuousStream.from_state_dict(net, torch.load(path, weights_only=True))
        for item in (stream, resumed):
            item.observe(0.03, torch.tensor([5]), training_terms=False)
            item.advance_to(0.04)
        torch.testing.assert_close(stream.belief.medium.receptors, resumed.belief.medium.receptors)
        torch.testing.assert_close(stream.belief.medium.field, resumed.belief.medium.field)
        broken = stream.state_dict()
        broken['receptors'] = None
        with pytest.raises(ValueError, match='receptor'):
            ContinuousStream.from_state_dict(net, broken)


def test_compilation_and_spatial_resolution_load_include_response():
    net = model()
    larger = PlasticMedium3D((4,4,4), 3, hidden=8, bath_type='conductance',
                              adaptive_conduction=True).double()
    larger.load_state_dict(net.state_dict(), strict=True)
    assert larger.initial_state().receptors.shape == (1,4,4,4,2,3)
    old = state(net)
    prepared = net.prepare_evolution()
    def step(state, dt, prepared):
        return net.advance(state, dt, prepared=prepared, diagnostics=False)[0]
    eager = step(old, torch.tensor(0.01), prepared)
    compiled = torch.compile(step, backend='eager', fullgraph=True)(old, torch.tensor(0.01), prepared)
    torch.testing.assert_close(compiled.field, eager.field)
    torch.testing.assert_close(compiled.receptors, eager.receptors)
