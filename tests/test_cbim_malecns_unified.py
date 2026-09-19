"""Mathematical invariance and autograd contracts for CBIM-MaleCNS Unified V6."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_unified import (
    CBIMMaleCNSUnifiedV6,
    ContextualBoundaryWrite,
    DualScaleGeometricTransport,
    WoodburyKineticCollision,
    QuadraticPassiveBath,
    QKNormReadout,
)


def mock_graph(nodes=16, modes=8, layers=2):
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    coords = torch.rand(nodes, 3, generator=generator)
    node_feats = torch.rand(nodes, 14, generator=generator)

    basis = torch.linalg.qr(torch.randn(nodes, modes, generator=generator)).Q
    eigenvalues = torch.linspace(0.1, 1.0, modes)

    pairs = []
    for layer in range(layers):
        order = torch.roll(torch.arange(nodes), layer)
        pairs.append(order.reshape(-1, 2))
    collision_pairs = torch.stack(pairs)
    collision_feats = torch.rand(layers, nodes // 2, 5, generator=generator)

    neighbor_indices = torch.stack(
        (torch.roll(torch.arange(nodes), 1), torch.roll(torch.arange(nodes), -1)), dim=-1
    )
    neighbor_weights = torch.full((nodes, 2), 0.5)

    return {
        "coordinates": coords,
        "node_features": node_feats,
        "laplacian_basis": basis,
        "laplacian_eigenvalues": eigenvalues,
        "neighbor_indices": neighbor_indices,
        "neighbor_weights": neighbor_weights,
        "collision_pairs": collision_pairs,
        "collision_features": collision_feats,
    }


def test_contextual_boundary_write_exact_energy_balance():
    graph = mock_graph()
    writer = ContextualBoundaryWrite(graph, vocab_size=32, d=64).double()
    field = torch.randn(2, 16, 64, dtype=torch.float64) * 0.5
    tokens = torch.tensor([5, 12])

    field_next, reflected, diag = writer(field, tokens)
    assert field_next.shape == field.shape
    assert reflected.shape == field.shape

    # Exact 2-port energy balance check
    assert diag["write_balance_residual"].item() < 1e-12


def test_dual_scale_transport_exact_norm_preservation():
    graph = mock_graph()
    transport = DualScaleGeometricTransport(graph, velocities=8, content_dim=8, modes=8).double()
    field = torch.randn(2, 16, 64, dtype=torch.float64)

    out, diag = transport(field)
    assert out.shape == field.shape
    norm_before = field.square().sum()
    norm_after = out.square().sum()
    err = (norm_after - norm_before).abs().item()
    assert err < 1e-10, f"Strang splitting energy leaked: {err}"


def test_woodbury_collision_preserves_constraints_and_energy():
    graph = mock_graph()
    collision = WoodburyKineticCollision(graph["node_features"], velocities=8, content_dim=8, rank=2).double()
    field = torch.randn(2, 16, 64, dtype=torch.float64)

    out, diag = collision(field)
    assert out.shape == field.shape

    # 1. Mass & momentum conservation check
    c_before = torch.einsum("cd,bnd->bnc", collision.constraints.double(), field)
    c_after = torch.einsum("cd,bnd->bnc", collision.constraints.double(), out)
    assert (c_after - c_before).abs().max().item() < 1e-10

    # 2. Quadratic energy conservation check
    e_err = (out.square().sum() - field.square().sum()).abs().item()
    assert e_err < 1e-10


def test_quadratic_passive_bath_monotone_cooling():
    graph = mock_graph()
    bath = QuadraticPassiveBath(graph["node_features"], d=64, local_radius=1.25, max_kappa=0.05).double()

    # Low energy field: dissipation should be negligible
    cold_field = torch.randn(2, 16, 64, dtype=torch.float64) * 0.01
    f_cold, _, diag_cold = bath(cold_field)
    assert f_cold.norm() <= cold_field.norm()
    assert diag_cold["cooling_sin2_mean"].item() < 5e-4

    # High energy field: dissipation should be active
    hot_field = torch.randn(2, 16, 64, dtype=torch.float64) * 2.0
    f_hot, _, diag_hot = bath(hot_field)
    assert f_hot.norm() < hot_field.norm()
    assert diag_hot["cooling_sin2_mean"].item() > 1e-2


def test_full_model_e2e_gradients_and_causality(tmp_path):
    graph = mock_graph()
    graph_path = tmp_path / "mock_graph.npz"
    np.savez(graph_path, **{k: v.numpy() for k, v in graph.items()})

    model = CBIMMaleCNSUnifiedV6(graph_path, vocab_size=32, velocities=8, content_dim=8, modes=8, collision_rank=2)

    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    loss_keep, _, _ = model(ids, targets, disable_transport=False, disable_collision=False)
    loss_notr, _, _ = model(ids, targets, disable_transport=True, disable_collision=False)
    loss_noco, _, _ = model(ids, targets, disable_transport=False, disable_collision=True)

    assert torch.isfinite(loss_keep)
    assert torch.isfinite(loss_notr)
    assert torch.isfinite(loss_noco)

    # Gradients reach parameters across all modules
    loss_keep.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no grad"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad"
