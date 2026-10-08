"""Numerical state closure and causality; no task capability training."""
import copy
from dataclasses import fields

import torch

from test_fly_rtc import individual
from information_boltzmann.core.fly_rtc_motor_flux import MotorFluxForecaster
from information_boltzmann.core.triton_synapse import execute_delayed_synaptic_transmission
from information_boltzmann.core.fly_rtc_learning import (
    copy_physical, initial_student_state, physical_options, quiet_teacher_step,
)


def test_local_split_matches_full_physics_with_actual_boundary(individual):
    model, physical = individual
    student = MotorFluxForecaster(model)
    coefficients = student.coefficients(model)
    local = student.local_state(physical)
    with torch.no_grad():
        for _ in range(14):
            local = student.integrate(local, student.external_arrival(physical), coefficients)
            physical = quiet_teacher_step(model, physical, physical_options(model))
            target = student.local_state(physical)
            for field in fields(local):
                if field.name == 'ring':
                    for predicted, actual in zip(local.ring, target.ring):
                        torch.testing.assert_close(predicted, actual, rtol=1e-5, atol=2e-6)
                else:
                    torch.testing.assert_close(getattr(local, field.name), getattr(target, field.name),
                                               rtol=1e-5, atol=2e-6)


def test_known_queue_contains_only_origin_pulses(individual):
    model, physical = individual
    student = MotorFluxForecaster(model)
    queue = student.known_queue(physical)
    # Shift old pulses with no new emissions. This independently calculates
    # all four known arrival slots using the original transmission function.
    shifted = copy_physical(physical)
    for tick in range(4):
        expected = []
        for sign in ('e', 'i'):
            full = execute_delayed_synaptic_transmission(
                    shifted.ring, getattr(model, 'edge_pre_'+sign),
                    getattr(model, 'edge_post_'+sign), getattr(model, 'edge_weight_'+sign),
                    getattr(model, 'splits_'+sign))[:, student.indices]
            local = student.local_state(shifted)
            internal = execute_delayed_synaptic_transmission(
                    local.ring, getattr(student, 'pre_'+sign),
                    getattr(student, 'post_'+sign), getattr(student, 'weight_'+sign),
                    getattr(student, 'splits_'+sign))
            expected.append(full-internal)
        torch.testing.assert_close(queue[tick], torch.stack(expected, -1))
        shifted.ring = (torch.zeros_like(shifted.h), *shifted.ring[:3])


def test_autonomous_first_tick_exact_and_flux_gradients_finite(individual):
    model, physical = individual
    student = MotorFluxForecaster(model)
    history = initial_student_state(model, physical).history
    local = student.local_state(physical)
    before = copy_physical(physical)
    _, motors, means, fluxes = student(history, local, student.coefficients(model),
                                       student.known_queue(physical))
    with torch.no_grad():
        expected = quiet_teacher_step(model, physical, physical_options(model))
    torch.testing.assert_close(motors[1], expected.h[:, student.indices], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(means[1], expected.h_mean[:, student.indices], rtol=1e-5, atol=2e-6)
    assert torch.equal(physical.h, before.h)
    for actual, saved in zip(physical.ring, before.ring):
        assert torch.equal(actual, saved)
    (motors.square().sum()+fluxes.sum()).backward()
    gradients = [p.grad for p in student.flux_network.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_equal_membrane_observation_does_not_determine_next_membrane(individual):
    model, physical = individual
    student = MotorFluxForecaster(model)
    changed = copy_physical(physical)
    changed.ge[:, student.indices] += .005
    a, b = student.local_state(physical), student.local_state(changed)
    assert torch.equal(a.h, b.h)
    external = student.external_arrival(physical)
    next_a = student.integrate(a, external, student.coefficients(model))
    next_b = student.integrate(b, external, student.coefficients(model))
    assert (next_a.h-next_b.h).abs().max() > 1e-4


def test_closed_full_state_forecast_needs_no_future_boundary(individual):
    model, physical = individual
    complete = copy.deepcopy(model)
    complete.read_indices = torch.arange(model.n_neurons)
    reference = MotorFluxForecaster(complete)
    coefficients = reference.coefficients(complete)
    current = reference.local_state(physical)
    zero = torch.zeros(*physical.h.shape, 2)
    with torch.no_grad():
        for _ in range(14):
            current = reference.integrate(current, zero, coefficients)
            physical = quiet_teacher_step(model, physical, physical_options(model))
            for field in fields(current):
                if field.name == 'ring':
                    for a, b in zip(current.ring, physical.ring):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
                else:
                    target_name = 'h_mean' if field.name == 'mean' else field.name
                    torch.testing.assert_close(getattr(current, field.name), getattr(physical, target_name),
                                               rtol=1e-5, atol=2e-6)
