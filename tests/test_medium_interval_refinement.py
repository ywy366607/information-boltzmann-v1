"""Full nonlinear numerical refinement contracts, not capability training.

All persistent mechanisms remain enabled. Refinement changes numerical
resolution at a fixed physical duration; it never creates additional experience.
"""
from dataclasses import replace

import torch

from information_boltzmann.core.intrinsic_time import EvolutionSchedule
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D


def full_medium():
    torch.manual_seed(352)
    net = PlasticMediumPorts3D(
        vocab_size=13, shape=(2, 2, 2), channels=8, material_width=3,
        material_reference_shape=(2, 2, 2), hidden=6, collision_layers=2,
        heads=1, queries=1, port_modes=8, bath_type="conductance",
        adaptive_conduction=True, activity_adaptation=True,
        short_term_plasticity=True, anisotropic_transport=True,
        read_mode="temporal", temporal_rates=[1.2, 3.1],
        temporal_frequencies=[.7, 2.3],
        solver_max_step=.01, observer_max_step=.04,
    ).double()
    with torch.no_grad():
        net.medium.material.coefficients.normal_(std=.08)
        net.medium.transport_shear.bias.copy_(torch.tensor([.2, -.15, .1]))
    belief = net.initial_belief()
    state = belief.medium
    state = replace(
        state, field=.3 * torch.randn_like(state.field),
        flux=tuple(.2 * torch.randn_like(q) for q in state.flux),
        conduction=.1 * torch.randn_like(state.conduction),
        receptors=.15 + .6 * torch.rand_like(state.receptors),
        transmission=.2 + .6 * torch.rand_like(state.transmission),
    )
    history = replace(belief.temporal, value=.03 * torch.randn_like(belief.temporal.value))
    return net, replace(belief, medium=state, temporal=history).detach()


def evaluate(net, initial, observer_count, solver_substeps):
    duration = torch.tensor(.16, dtype=torch.float64, requires_grad=True)
    schedule = EvolutionSchedule(observer_count, solver_substeps, .16)
    out, _, motion = net.advance_interval(initial, duration, schedule=schedule,
                                         diagnostics=False, return_motion=True)
    expression, _ = net.read(out, decode=False, motion=motion)
    state = out.medium
    components = (state.field, *state.flux, state.conduction,
                  state.receptors, state.transmission)
    state_vector = torch.cat([x.flatten() for x in components])
    # A signed downstream read plus storage-sensitive term exercises all rates.
    read_weights = torch.linspace(-.7, 1.1, expression.numel(), dtype=torch.float64)
    loss = (expression.flatten() * read_weights).sum()
    loss = loss + .07 * sum(x.square().mean() for x in components)
    parameters = (
        duration, net.medium.log_speed.weight, net.medium.material.coefficients,
        net.medium.conductance_response.log_parameters.bias,
        net.medium.short_term_plasticity.parameters_map.bias,
    )
    gradients = torch.autograd.grad(loss, parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g.norm() > 1e-10 for g in gradients)
    assert ((state.receptors >= 0) & (state.receptors <= 1)).all()
    assert ((state.transmission >= 0) & (state.transmission <= 1)).all()
    torch.testing.assert_close(state.elapsed, duration.reshape(1), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(out.temporal.elapsed, state.elapsed, atol=1e-14, rtol=1e-14)
    return {
        "state": state_vector.detach(), "read": expression.detach(),
        "history": out.temporal.value.detach(),
        "gradients": tuple(g.detach() for g in gradients),
    }


def relative_error(value, reference):
    return float((value - reference).norm() / reference.norm().clamp_min(1e-12))


def test_full_nonlinear_solver_refines_at_fixed_observer_sampling():
    net, initial = full_medium()
    # Same two observer intervals; total physical steps are 4, 16, 64, 256.
    results = [evaluate(net, initial, 2, substeps) for substeps in (2, 8, 32, 128)]
    reference = results[-1]
    for key in ("state", "read", "history"):
        errors = [relative_error(out[key], reference[key]) for out in results[:-1]]
        assert errors[-1] < .4 * errors[0], (key, errors)
        assert errors[-1] < errors[1], (key, errors)
    names = ("duration", "log_speed", "material", "conductance_rates", "STP")
    for index, name in enumerate(names):
        errors = [relative_error(out["gradients"][index], reference["gradients"][index])
                  for out in results[:-1]]
        # Nonlinear gradient error need not improve at every coarse level.
        assert errors[-1] < .5 * errors[0], (name, errors)


def test_full_nonlinear_observer_refines_with_identical_physical_microsteps():
    net, initial = full_medium()
    # All runs execute the exact same 64 medium microsteps; only temporal
    # measurement quadrature differs. Each observer count divides 64 exactly.
    results = [evaluate(net, initial, samples, 64 // samples)
               for samples in (2, 4, 16, 64)]
    reference = results[-1]
    for out in results[:-1]:
        torch.testing.assert_close(out["state"], reference["state"], atol=3e-13, rtol=3e-13)
    for key in ("read", "history"):
        errors = [relative_error(out[key], reference[key]) for out in results[:-1]]
        assert errors[-1] < .4 * errors[0], (key, errors)
    names = ("duration", "log_speed", "material", "conductance_rates", "STP")
    for index, name in enumerate(names):
        errors = [relative_error(out["gradients"][index], reference["gradients"][index])
                  for out in results[:-1]]
        assert errors[-1] < .5 * errors[0], (name, errors)
