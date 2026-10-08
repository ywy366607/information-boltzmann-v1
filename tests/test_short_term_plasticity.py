"""Numerical/continuation verification of pathway STP; no capability study."""
from dataclasses import replace
import os

import pytest
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.plastic_feasibility import continuous_rhs
from information_boltzmann.core.short_term_plasticity import LocalShortTermPlasticity
from information_boltzmann.runtime import ContinuousStream
from information_boltzmann.runtime.training import belief_tensors, clone_belief, quiet_training_chunk
from scripts.ib.train_plastic_conductance import pack_belief, unpack_belief


def medium():
    torch.manual_seed(644)
    return PlasticMedium3D((2, 2, 2), channels=4, hidden=8, bath_type='conductance',
                           adaptive_conduction=True, activity_adaptation=True,
                           short_term_plasticity=True).double()


def test_shared_scalar_energy_preserves_original_expression_and_gradients():
    from scripts.ib.benchmark_stp_execution import original_step
    net = medium()
    rule = net.short_term_plasticity
    state = net.initial_state()
    field = torch.randn_like(state.field).requires_grad_()
    flux = tuple(torch.randn_like(x).requires_grad_() for x in state.flux)
    history = torch.rand_like(state.transmission).requires_grad_()
    rates, baseline = rule.coefficients(net.material_field())
    dt = net._duration(0.013, field).requires_grad_()
    inputs = (history, field, *flux, rates, baseline, dt)
    direction = torch.randn_like(history)
    expected = original_step(history, field, flux, dt, (rates, baseline))
    actual = rule.native_step(history, field, flux, dt, (rates, baseline))
    wanted = torch.autograd.grad(expected, inputs, direction, retain_graph=True)
    got = torch.autograd.grad(actual, inputs, direction)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    for value, reference in zip(got, wanted):
        torch.testing.assert_close(value, reference, atol=2e-14, rtol=2e-14)


def test_frozen_rate_interval_rest_recovery_and_zero_duration_identity():
    net = medium()
    state = net.initial_state()
    rule = net.short_term_plasticity
    coefficients = rule.coefficients(net.material_field())
    damaged = torch.rand_like(state.transmission)
    damaged[..., 0] *= 0.2
    damaged[..., 1] = 0.9
    for duration in (0., 0.1, 10., 1e4):
        output = rule(damaged, state.field, state.flux, net._duration(duration, state.field), coefficients)
        assert ((output >= 0) & (output <= 1)).all()
        if duration == 0:
            torch.testing.assert_close(output, damaged, atol=0, rtol=0)
    rates, baseline = coefficients
    dt = net._duration(0.7, state.field)
    expected_x = 1 + (damaged[..., 0] - 1) * (-dt * rates[None, ..., 0]).exp()
    expected_u = baseline[None] + (damaged[..., 1] - baseline[None]) * (-dt * rates[None, ..., 1]).exp()
    output = rule(damaged, state.field, state.flux, dt, coefficients)
    torch.testing.assert_close(output[..., 0], expected_x, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(output[..., 1], expected_u, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(rule.transmission_gain(state.transmission, coefficients),
                               torch.ones_like(state.conduction), atol=0, rtol=0)


def test_both_facilitating_and_depressing_parameter_regimes_are_reachable():
    rule = LocalShortTermPlasticity(2).double()
    field = torch.zeros(1, 2, 2, 2, 4, dtype=torch.float64)
    flux = tuple(torch.ones_like(field) for _ in range(3))
    shape = (2, 2, 2, 3)
    baseline = torch.full(shape, 0.2, dtype=field.dtype)
    # Analytic steady state: u*=(closing*U+U*r)/(closing+U*r),
    # x*=recovery/(recovery+u*r). Both >1 and <1 gain regimes occur.
    for recovery, expected_direction in ((10., 'facilitate'), (0.01, 'depress')):
        rates = torch.stack((torch.full(shape, recovery, dtype=field.dtype),
                             torch.full(shape, 0.1, dtype=field.dtype),
                             torch.ones(shape, dtype=field.dtype)), -1)
        coefficients = rates, baseline
        state = rule.initial_state(field, coefficients)
        for _ in range(200):
            state = rule(state, field, flux, torch.tensor(0.5), coefficients)
        gain = rule.transmission_gain(state, coefficients)
        assert (gain > 1).all() if expected_direction == 'facilitate' else (gain < 1).all()
        u_star = (0.1 * baseline + baseline) / (0.1 + baseline)
        x_star = recovery / (recovery + u_star)
        torch.testing.assert_close(state[..., 1], u_star[None], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(state[..., 0], x_star[None], atol=1e-10, rtol=1e-10)


def test_stp_is_local_and_leaves_flux_intact_while_reweighting_conservative_transport():
    net = medium()
    empty = net.initial_state()
    state = replace(empty, field=torch.randn_like(empty.field), flux=tuple(torch.randn_like(x) for x in empty.flux))
    c = net.prepare_evolution()
    before = net.energy(state)
    adapted = net.adapt_transmission(state, net._duration(0.4, state.field), c.short_term)
    assert adapted.field is state.field
    assert all(x is y for x, y in zip(adapted.flux, state.flux))
    moved = net.transport(adapted, 0.03, c.material, c.baseline_log_speed, c.short_term)
    original = net.transport(state, 0.03, c.material, c.baseline_log_speed, c.short_term)
    assert (moved.field - original.field).abs().sum() > 0
    torch.testing.assert_close(net.energy(moved), before, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(moved.field.sum((1, 2, 3)), state.field.sum((1, 2, 3)), atol=3e-14, rtol=3e-14)
    activity = net.short_term_plasticity.activity(state.field, state.flux)
    torch.testing.assert_close(activity, net.short_term_plasticity.activity(
        state.field * 3, tuple(x * 3 for x in state.flux)))
    shifted = net.short_term_plasticity.activity(torch.roll(state.field, 1, 1),
                                                tuple(torch.roll(x, 1, 1) for x in state.flux))
    torch.testing.assert_close(shifted, torch.roll(activity, 1, 1))
    # A remote cell cannot influence the positive-x edge at (0,0,0).
    changed = state.field.clone()
    changed[:, :, 1, 1] += 100
    remote = net.short_term_plasticity.activity(changed, state.flux)
    torch.testing.assert_close(remote[:, 0, 0, 0, 0], activity[:, 0, 0, 0, 0], atol=0, rtol=0)


def test_integrator_generator_matches_continuous_equations_including_stp():
    net = medium()
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux),
                    receptors=torch.rand_like(state.receptors), transmission=torch.rand_like(state.transmission))
    pack = lambda s: torch.cat((net._pack(s).flatten(), s.receptors.flatten(),
                                s.conduction.flatten(), s.transmission.flatten()))
    _, derivative = torch.autograd.functional.jvp(lambda t: pack(net.advance(state, t)[0]),
        torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(derivative, pack(continuous_rhs(net, state)), atol=1e-12, rtol=1e-12)


def test_same_physical_duration_converges_with_solver_refinement():
    net = medium()
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux))
    with torch.no_grad():
        result = [net.advance(state, 0.03, substeps=k)[0] for k in (4, 8, 16, 128)]
    def error(value):
        reference = result[-1]
        return ((net._pack(value) - net._pack(reference)).square().sum()
                + (value.transmission - reference.transmission).square().sum()).sqrt()
    assert error(result[2]) < error(result[1]) < error(result[0])
    for value in result:
        assert value.elapsed.item() == pytest.approx(0.03)


def test_spatial_material_can_change_local_rates_and_strict_load_at_finer_grid():
    net = medium()
    with torch.no_grad():
        mode = (net.material.modes.abs() == torch.tensor([1, 0, 0])).all(-1).nonzero()[0, 0]
        net.material.coefficients[1 + mode, 0] = 0.7
    rates, baseline = net.short_term_plasticity.coefficients(net.material_field())
    assert rates.var((0, 1, 2), unbiased=False).sum() > 0
    assert baseline.var((0, 1, 2), unbiased=False).sum() > 0
    fine = PlasticMedium3D((4, 4, 4), channels=4, hidden=8, bath_type='conductance',
                           adaptive_conduction=True, activity_adaptation=True,
                           short_term_plasticity=True).double()
    fine.load_state_dict(net.state_dict(), strict=True)
    fine_rates, fine_baseline = fine.short_term_plasticity.coefficients(fine.material_field())
    torch.testing.assert_close(fine_rates[::2, ::2, ::2], rates)
    torch.testing.assert_close(fine_baseline[::2, ::2, ::2], baseline)


def test_joint_likelihood_and_all_continuation_paths_retain_transmission():
    net = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8, hidden=8,
                              bath_type='conductance', activity_adaptation=True,
                              short_term_plasticity=True).double()
    initial = net.initial_belief()
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    loss, final, _ = quiet_training_chunk(net, ids, targets, initial, event_duration=0.03)
    loss.backward()
    assert net.medium.short_term_plasticity.parameters_map.bias.grad.abs().sum() > 0
    assert net.medium.short_term_plasticity.parameters_map.weight.grad.isfinite().all()
    for copy in (final.detach(), clone_belief(final), unpack_belief(pack_belief(final), 'cpu')):
        for a, b in zip(belief_tensors(final), belief_tensors(copy)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    observed, _ = net.assimilate(final, torch.tensor([4]), diagnostics=False)
    assert observed.medium.transmission is final.medium.transmission
    stream = ContinuousStream(net, max_step=0.01, belief=final.detach())
    saved = stream.state_dict()
    restored = ContinuousStream.from_state_dict(net, saved)
    stream.advance_to(stream.time + 0.02)
    restored.advance_to(restored.time + 0.02)
    for a, b in zip(belief_tensors(stream.belief), belief_tensors(restored.belief)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    changed = dict(saved, short_term_plasticity=False)
    with pytest.raises(ValueError, match='STP law'):
        ContinuousStream.from_state_dict(net, changed)
    damaged = dict(saved, transmission=saved['transmission'].clone())
    damaged['transmission'][..., 0] = -1
    with pytest.raises(RuntimeError, match='Invalid STP'):
        ContinuousStream.from_state_dict(net, damaged)


@pytest.mark.skipif(os.environ.get('IB_ENABLE_RUNTIME_CUDA_TESTS') != '1', reason='Opt-in CUDA allocation')
@pytest.mark.parametrize('duration', [0., 0.005, 0.4])
def test_fused_cuda_step_matches_native_values_and_every_input_gradient(duration):
    torch.manual_seed(827)
    torch.set_num_threads(1)
    rule = LocalShortTermPlasticity(8).cuda()
    field = torch.randn(2, 8, 8, 4, 128, device='cuda', requires_grad=True)
    flux = tuple(torch.randn_like(field).requires_grad_() for _ in range(3))
    state = torch.rand(2, 8, 8, 4, 3, 2, device='cuda', requires_grad=True)
    rates = torch.rand(8, 8, 4, 3, 3, device='cuda').add(0.1).requires_grad_()
    baseline = torch.rand(8, 8, 4, 3, device='cuda').mul(0.8).add(0.1).requires_grad_()
    dt = torch.full((2, 1, 1, 1, 1), duration, device='cuda', requires_grad=True)
    inputs = (state, field, *flux, dt, rates, baseline)
    direction = torch.randn_like(state)
    expected = rule.native_step(state, field, flux, dt, (rates, baseline))
    wanted = torch.autograd.grad(expected, inputs, direction)
    actual = rule(state, field, flux, dt, (rates, baseline))
    got = torch.autograd.grad(actual, inputs, direction)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    for value, reference in zip(got, wanted):
        assert value.isfinite().all()
        torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-5)
