"""Constructive certificates and exact local Jacobians; no capability training."""
import math

import torch

from information_boltzmann.core.plastic_medium import (
    ContinuousMaterial, MediumState, PlasticMedium3D, complete_periodic_modes,
)
from information_boltzmann.core.plastic_feasibility import (
    bounded_boundary_supply, collision_pair_edges, configure_locking_witness,
    continuous_rhs, discrete_orbit_radius_squared, graph_incidence,
    rotate_back, rotation_generator,
)


def test_default_material_spans_every_reference_location_and_speed_direction():
    model = PlasticMedium3D().double()
    basis = model.material.basis(model.coordinates).reshape(256, -1)
    assert basis.shape == (256, 256)
    assert torch.linalg.matrix_rank(basis) == 256
    assert torch.linalg.matrix_rank(model.log_speed.weight) == 3
    # Exact synthesis of an arbitrary local pattern, without optimization.
    desired = torch.zeros(256, 8, dtype=torch.float64)
    desired[37, 0] = 1
    coefficients = torch.linalg.solve(basis, desired)
    torch.testing.assert_close(basis @ coefficients, desired, atol=2e-14, rtol=2e-14)


def test_rotation_generators_connect_all_free_directions():
    for channels in (3, 64, 128):
        width = 4 * (channels - 1)
        reached = {0}
        edges = collision_pair_edges(width, 2)
        while True:
            expanded = reached | {y for x, y in edges if x in reached} | {x for x, y in edges if y in reached}
            if expanded == reached:
                break
            reached = expanded
        assert len(reached) == width


def test_analytic_generator_matches_actual_integrator_derivative_at_zero():
    torch.manual_seed(409)
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8).double()
    empty = model.initial_state()
    state = MediumState(torch.randn_like(empty.field),
                        tuple(torch.randn_like(x) for x in empty.flux), empty.elapsed)
    def integrator(time):
        output, _ = model(state, time)
        return model._pack(output)
    _, derivative = torch.autograd.functional.jvp(
        integrator, torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(derivative, model._pack(continuous_rhs(model, state)),
                               atol=3e-13, rtol=3e-13)


def test_bounded_power_supply_can_be_realized_by_existing_orthogonal_port():
    from information_boltzmann.core.plastic_medium import boundary_scatter
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8).double()
    plane = configure_locking_witness(model, 1.3)
    empty = model.initial_state()
    field = torch.randn_like(empty.field) * 100
    source = bounded_boundary_supply(field, plane, 1.0, 1.0)
    assert torch.all((source * field).sum(-1) <= 1.0)
    angle = field.new_tensor(0.3)
    desired = field + 0.01 * source
    packet = (desired - angle.cos() * field) / angle.sin()
    output, outgoing = boundary_scatter(empty.with_field(field), packet, angle)
    torch.testing.assert_close(output.field, desired)
    torch.testing.assert_close(output.field.square().sum() + outgoing.square().sum(),
                               field.square().sum() + packet.square().sum())


def test_continuous_phase_locking_orbit_and_transverse_stability():
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8).double()
    omega, power, radius2, beta = 1.3, 1.0, 1.0, 1.0
    plane = configure_locking_witness(model, omega, beta)
    q = 4 * model.channels
    orbit2 = (-radius2 + math.sqrt(radius2**2 + 4*q*power/beta)) / 2
    empty = model.initial_state()
    field = (math.sqrt(orbit2) * plane[:, 0]).expand_as(empty.field)
    state = empty.with_field(field)

    def in_rotating_frame(packed):
        candidate = model._unpack(packed, state)
        rhs = model._pack(continuous_rhs(model, candidate))
        supply = bounded_boundary_supply(candidate.field, plane, power, radius2)
        rhs = rhs + torch.stack((supply, *[torch.zeros_like(supply) for _ in range(3)]), -2)
        return rhs - rotation_generator(packed, plane, omega)

    packed = model._pack(state)
    torch.testing.assert_close(in_rotating_frame(packed), torch.zeros_like(packed), atol=2e-14, rtol=0)
    jacobian = torch.autograd.functional.jacobian(in_rotating_frame, packed).reshape(packed.numel(), -1)
    eigenvalues = torch.linalg.eigvals(jacobian)
    neutral = eigenvalues.abs() < 1e-9
    assert neutral.sum() == 1  # Only global phase is neutral.
    assert eigenvalues.real[~neutral].max() < -0.05

    incidence = graph_incidence(model)
    assert torch.linalg.matrix_rank(incidence) == math.prod(model.shape) - 1
    laplacian_spectrum = torch.linalg.eigvalsh(incidence.T @ incidence)
    eta = beta * orbit2 / q
    # Relative phase eigenvalues solve lambda² + eta*lambda + sigma² = 0.
    phase_roots = (-eta + torch.sqrt((eta**2 - 4*laplacian_spectrum[1:]).to(torch.complex128))) / 2
    assert phase_roots.real.max() < 0


def test_actual_discrete_map_has_a_transversely_stable_phase_locked_orbit():
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8).double()
    omega, duration = 1.3, 0.005
    plane = configure_locking_witness(model, omega)
    orbit2 = discrete_orbit_radius_squared(3, duration, 1.0, 1.0, 1.0)
    state = model.initial_state()
    state = state.with_field((math.sqrt(orbit2) * plane[:, 0]).expand_as(state.field))

    def fixed_frame_map(packed):
        candidate = model._unpack(packed, state)
        source = bounded_boundary_supply(candidate.field, plane, 1.0, 1.0)
        candidate = candidate.with_field(candidate.field + duration * source)
        output, _ = model(candidate, duration)
        return rotate_back(model._pack(output), plane, omega * duration)

    packed = model._pack(state)
    torch.testing.assert_close(fixed_frame_map(packed), packed, atol=2e-14, rtol=2e-14)
    jacobian = torch.autograd.functional.jacobian(fixed_frame_map, packed).reshape(packed.numel(), -1)
    eigenvalues = torch.linalg.eigvals(jacobian)
    neutral = (eigenvalues - 1).abs() < 1e-9
    assert neutral.sum() == 1
    assert eigenvalues.abs()[~neutral].max() < 1 - 1e-4
