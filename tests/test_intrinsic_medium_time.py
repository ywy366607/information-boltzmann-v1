"""Deterministic numerical clock/solver contracts; no capability training."""
import math

import pytest
import torch

from information_boltzmann.core.intrinsic_time import (
    EvolutionSchedule, IntrinsicTimePolicy, factor_characteristic_time,
)
from information_boltzmann.core.temporal_probes import CausalProbeFilterBank


def test_policy_reference_gradient_and_explicit_limit():
    policy = IntrinsicTimePolicy(.3, feature_dim=2, max_duration=.6).double()
    features = torch.tensor([[.4, -.2]], dtype=torch.float64)
    duration = policy(features)
    torch.testing.assert_close(duration, duration.new_tensor([.3]))
    duration.sum().backward()
    torch.testing.assert_close(policy.head.bias.grad, duration.new_tensor([.3]))
    torch.testing.assert_close(policy.head.weight.grad, .3 * features)
    with torch.no_grad():
        policy.head.bias.fill_(math.log(3.))
    with pytest.raises(RuntimeError, match="execution budget"):
        policy(features)


def test_schedule_independent_counts_and_duration_gradient():
    duration = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    schedule = EvolutionSchedule.for_duration(duration, solver_max_step=.03,
                                              observer_max_step=.2, max_steps=100)
    assert schedule.observer_count == 4
    assert schedule.solver_substeps == 6
    interval = schedule.interval_duration(duration)
    (interval * schedule.observer_count).backward()
    assert duration.grad == 1
    assert float(interval.detach()) <= .2
    assert float(interval.detach()) / schedule.solver_substeps <= .03
    finer_solver = EvolutionSchedule.for_duration(duration, solver_max_step=.01,
                                                  observer_max_step=.2, max_steps=100)
    assert finer_solver.observer_count == schedule.observer_count
    assert finer_solver.solver_substeps > schedule.solver_substeps
    with pytest.raises(ValueError, match="budget"):
        EvolutionSchedule.for_duration(duration, solver_max_step=.001,
                                       observer_max_step=.2, max_steps=10)
    with pytest.raises(RuntimeError, match="replan"):
        schedule.interval_duration(duration * 2)


def test_zero_interval_preserves_continuous_api_and_its_duration_gradient():
    duration = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    schedule = EvolutionSchedule.for_duration(duration, solver_max_step=.01,
                                              observer_max_step=.1, max_steps=1)
    assert schedule.observer_count == schedule.solver_substeps == 1
    interval = schedule.interval_duration(duration)
    assert interval == 0
    interval.backward()
    assert duration.grad == 1
    with pytest.raises(ValueError, match="nonnegative"):
        EvolutionSchedule.for_duration(-.1, solver_max_step=.01,
                                       observer_max_step=.1, max_steps=1)


def test_remote_skew_wave_and_clock_gradient_match_analytic_solution():
    # Two field sites and their intermediate flux. Remote field responds as
    # (1-cos(sqrt(2)*t))/2; its leading response is second order in time.
    k = torch.tensor([[0., 0., -1.], [0., 0., 1.], [1., -1., 0.]], dtype=torch.float64)
    policy = IntrinsicTimePolicy(.4, feature_dim=1).double()
    duration = policy(torch.zeros(1, 1, dtype=torch.float64)).squeeze(0)
    schedule = EvolutionSchedule.for_duration(duration, solver_max_step=.03,
                                              observer_max_step=.1, max_steps=100)
    dt = schedule.interval_duration(duration) / schedule.solver_substeps
    state = torch.tensor([1., 0., 0.], dtype=torch.float64)
    for _ in range(schedule.observer_count * schedule.solver_substeps):
        state = torch.matrix_exp(k * dt) @ state
    expected = (1 - torch.cos(math.sqrt(2.) * duration)) / 2
    torch.testing.assert_close(state[1], expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(state.square().sum(), state.new_tensor(1.), atol=1e-13, rtol=1e-13)
    gradient = torch.autograd.grad(state[1], policy.head.bias)[0]
    expected_gradient = duration * torch.sin(math.sqrt(2.) * duration) / math.sqrt(2.)
    torch.testing.assert_close(gradient.squeeze(), expected_gradient, atol=1e-12, rtol=1e-12)


def test_observer_sampling_refines_without_changing_exact_medium():
    bank = CausalProbeFilterBank(1, 1, [1.3], [.7]).double()
    duration = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    initial = torch.tensor([1., 0.], dtype=torch.float64)

    def integrate(observer_step):
        schedule = EvolutionSchedule.for_duration(duration, solver_max_step=.02,
                                                  observer_max_step=observer_step, max_steps=1000)
        dt = schedule.interval_duration(duration)
        state, history = initial, bank.initial_state()
        for _ in range(schedule.observer_count):
            state = torch.matrix_exp(k * dt) @ state
            history = bank(state[0].reshape(1, 1, 1), history, dt)
        return state, history

    coarse_state, coarse = integrate(.2)
    fine_state, fine = integrate(.05)
    # Exact augmented ODE: x'=Kx, z'=-(alpha+i*omega)z+alpha*x0.
    generator = torch.zeros(3, 3, dtype=torch.complex128)
    generator[:2, :2] = k
    generator[2, 0] = 1.3
    generator[2, 2] = complex(-1.3, -.7)
    truth = torch.matrix_exp(generator * duration) @ torch.tensor([1., 0., 0.], dtype=torch.complex128)
    torch.testing.assert_close(coarse_state, fine_state, atol=1e-13, rtol=1e-13)
    assert abs(fine.value.squeeze() - truth[2]) < abs(coarse.value.squeeze() - truth[2])
    torch.testing.assert_close(fine.elapsed, duration.reshape(1))
    gradient = torch.autograd.grad(fine.value.real.sum(), duration)[0]
    assert torch.isfinite(gradient) and abs(gradient) > 0


def test_large_angle_lie_splitting_conserves_energy_but_requires_refinement():
    first = torch.tensor([[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    second = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=torch.float64)
    initial = torch.tensor([1., .2, -.1], dtype=torch.float64)
    duration = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    truth = torch.matrix_exp((first + second) * duration) @ initial
    truth_gradient = torch.autograd.grad(truth[1], duration, retain_graph=True)[0]
    errors, gradient_errors = [], []
    for steps in (2, 8, 32):
        dt = duration / steps
        eye = torch.eye(3, dtype=torch.float64)
        # Both generators have unit rotation rate; trigonometric rotations
        # avoid small-matrix matrix_exp approximation error in this invariant.
        left = eye + dt.sin() * first + (1 - dt.cos()) * (first @ first)
        right = eye + dt.sin() * second + (1 - dt.cos()) * (second @ second)
        flow = right @ left
        state = initial
        for _ in range(steps):
            state = flow @ state
        torch.testing.assert_close(state.square().sum(), initial.square().sum(), atol=1e-13, rtol=1e-13)
        errors.append(float((state - truth).norm().detach()))
        gradient = torch.autograd.grad(state[1], duration, retain_graph=True)[0]
        gradient_errors.append(float((gradient - truth_gradient).abs().detach()))
    assert errors[0] > errors[1] > errors[2]
    assert gradient_errors[0] > gradient_errors[1] > gradient_errors[2]


def test_factor_reference_uses_geometry_and_actual_speed():
    factor = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    times = factor_characteristic_time(factor, cell_spacing=(.125, .125, .125))
    torch.testing.assert_close(times.cell_crossing_time, factor.new_tensor(.125 / math.sqrt(6.)))
    torch.testing.assert_close(times.torus_crossing_time, factor.new_tensor(math.sqrt(3.)))
    doubled = factor_characteristic_time(2 * factor, cell_spacing=(.125, .125, .125))
    torch.testing.assert_close(doubled.torus_crossing_time, times.torus_crossing_time / 2)
    zero = factor_characteristic_time(torch.zeros_like(factor), cell_spacing=(.125,) * 3)
    assert torch.isinf(zero.cell_crossing_time) and torch.isinf(zero.torus_crossing_time)
