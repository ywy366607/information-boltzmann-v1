"""Contracts for velocity-aligned graph finite-volume transport."""
import torch

from scripts.ib_local.geometric_transport import (
    DirectionalCayleyTransport,
    DirectionalFiniteVolumeTransport,
)


def embedded_graph(dtype=torch.float64):
    coordinates = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=dtype)
    adjacency = torch.ones(4, 4, dtype=dtype) - torch.eye(4, dtype=dtype)
    return coordinates, adjacency


def test_directional_transport_preserves_mass_and_positivity():
    coordinates, adjacency = embedded_graph()
    transport = DirectionalFiniteVolumeTransport(
        coordinates, adjacency, stream_fraction=.7).double()
    occupancy = torch.rand(2, 4, 8, 3, dtype=torch.float64)
    output, diagnostics = transport(occupancy)
    torch.testing.assert_close(
        output.sum(1), occupancy.sum(1), atol=2e-12, rtol=2e-12)
    assert output.min() >= 0
    assert diagnostics["transport_mass_residual"] < 2e-12


def test_velocity_selects_anatomical_direction_and_has_opposite_reciprocity():
    coordinates, adjacency = embedded_graph()
    transport = DirectionalFiniteVolumeTransport(
        coordinates, adjacency, stream_fraction=.8).double()
    # q=7 is (+,+,+), q=0 is its (-,-,-) opposite.
    positive = torch.zeros(1, 4, 8, 1, dtype=torch.float64)
    positive[0, 0, 7, 0] = 1
    moved_positive, _ = transport(positive)
    assert moved_positive[0, 1, 7, 0] > 0
    assert moved_positive[0, 2, 7, 0] > 0
    assert moved_positive[0, 3, 7, 0] == 0

    negative = torch.zeros_like(positive)
    negative[0, 0, 0, 0] = 1
    moved_negative, diagnostics = transport(negative)
    assert moved_negative[0, 3, 0, 0] > 0
    assert moved_negative[0, 1, 0, 0] == 0
    assert diagnostics["transport_reciprocity_error"] < 2e-12


def test_physical_axis_scale_changes_geometric_routing():
    coordinates = torch.tensor([
        [0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    adjacency = torch.ones(3, 3) - torch.eye(3)
    isotropic = DirectionalFiniteVolumeTransport(coordinates, adjacency)
    stretched = DirectionalFiniteVolumeTransport(
        coordinates, adjacency, coordinate_scale=torch.tensor([1.0, 20.0, 1.0]))
    assert not torch.allclose(isotropic.transition, stretched.transition)


def test_signed_cayley_transport_preserves_energy_and_reverses_with_velocity():
    coordinates, adjacency = embedded_graph()
    transport = DirectionalCayleyTransport(
        coordinates, adjacency, stream_fraction=.7).double()
    field = torch.randn(2, 4, 8, 3, dtype=torch.float64)
    output, diagnostics = transport(field)
    torch.testing.assert_close(
        output.square().sum((1, 2, 3)), field.square().sum((1, 2, 3)),
        atol=2e-11, rtol=2e-11)
    assert diagnostics["transport_norm_residual"] < 2e-11
    assert diagnostics["transport_reciprocity_error"] < 2e-11
    assert transport.orthogonal_error < 2e-11

    impulse = torch.zeros(1, 4, 8, 1, dtype=torch.float64)
    impulse[0, 0, 7, 0] = 1
    moved, _ = transport(impulse)
    assert moved[0, 1, 7, 0] > 0
    assert moved[0, 2, 7, 0] > 0
    assert moved[0, 3, 7, 0] < 0


def test_signed_cayley_transport_uses_physical_axis_scale():
    coordinates = torch.tensor([
        [0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    adjacency = torch.ones(3, 3) - torch.eye(3)
    isotropic = DirectionalCayleyTransport(coordinates, adjacency)
    stretched = DirectionalCayleyTransport(
        coordinates, adjacency, coordinate_scale=torch.tensor([1., 20., 1.]))
    assert not torch.allclose(isotropic.propagator, stretched.propagator)
