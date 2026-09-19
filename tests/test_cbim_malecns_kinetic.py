"""Numerical contracts for the role-separated MaleCNS kinetic CBIM."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_kinetic import (
    CBIMMaleCNSKinetic,
    GraphAdaptiveDissipation,
    GraphBoundarySource,
    GraphVelocityTransport,
    LocalVelocityCollision,
)


def graph_fixture(nodes=8, modes=4):
    generator = torch.Generator().manual_seed(19)
    return {
        "coordinates": torch.rand(nodes, 3, generator=generator),
        "node_features": torch.rand(nodes, 14, generator=generator),
        "laplacian_basis": torch.linalg.qr(
            torch.randn(nodes, modes, generator=generator)).Q,
        "laplacian_eigenvalues": torch.linspace(.1, 1., modes),
    }


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_source_injects_bounded_packet_without_erasing_state():
    graph = graph_fixture()
    source = GraphBoundarySource(graph["coordinates"], vocab_size=32)
    state = torch.randn(2, 8, 64)
    output, diagnostics = source(state, torch.tensor([1, 2]))
    change = (output - state).flatten(1).norm(dim=-1)
    assert torch.all(change <= source.max_injection + 1e-6)
    assert diagnostics["injection_norm"] > 0


def test_transport_preserves_norm_and_velocity_marginal_shape():
    graph = graph_fixture()
    transport = GraphVelocityTransport(
        graph["laplacian_basis"], graph["laplacian_eigenvalues"]).double()
    state = torch.randn(3, 8, 64, dtype=torch.float64)
    output, _ = transport(state)
    torch.testing.assert_close(output.norm(), state.norm(), atol=2e-10, rtol=2e-10)


def test_collision_is_node_local_and_preserves_moments_and_energy():
    collision = LocalVelocityCollision().double()
    state = torch.randn(2, 7, 8, 8, dtype=torch.float64)
    output, _ = collision(state.flatten(2))
    output = output.reshape_as(state)
    before_moments = torch.einsum(
        "cq,bnqa->bnca", collision.constraints, state)
    after_moments = torch.einsum(
        "cq,bnqa->bnca", collision.constraints, output)
    torch.testing.assert_close(after_moments, before_moments,
                               atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(output.square().sum((-1, -2)),
                               state.square().sum((-1, -2)),
                               atol=3e-10, rtol=2e-10)


def test_dissipation_only_contracts_state():
    torch.manual_seed(23)
    model = GraphAdaptiveDissipation(torch.randn(8, 14), d=64).double()
    state = torch.randn(2, 8, 64, dtype=torch.float64) * 100
    output, diagnostics = model(state)
    assert torch.all(output.norm(dim=-1) <= state.norm(dim=-1) + 1e-12)
    energy = .5 * output.square().sum(-1)
    assert energy.max() <= model.target_energy + 2e-10
    assert 0 < diagnostics["alpha_mean"] < 1
    assert diagnostics["gamma_mean"] >= 0


def test_dissipation_rate_parameters_receive_gradient():
    torch.manual_seed(29)
    model = GraphAdaptiveDissipation(torch.randn(8, 14), d=64)
    state = torch.randn(2, 8, 64, requires_grad=True) * .1
    output, _ = model(state)
    output.square().sum().backward()
    for parameter in (model.local_rate, model.node_rate[-1].weight,
                      model.rate[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_dissipation_has_explicit_node_local_rates():
    torch.manual_seed(30)
    features = torch.randn(8, 14)
    model = GraphAdaptiveDissipation(features, d=64)
    state = torch.zeros(2, 8, 64)
    _, diagnostics = model(state)
    assert model.local_rate.shape == (8, 64)
    assert diagnostics["gamma_node_mean"].shape == (8,)
    assert diagnostics["gamma_spatial_std"] > 0
    assert diagnostics["gamma_channel_std"] > 0


def test_complete_operator_energy_ledger_closes(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(31)
    model = CBIMMaleCNSKinetic(path, vocab_size=32).double()
    state = torch.randn(2, 8, 64, dtype=torch.float64) * .1
    output, diagnostics = model.evolve(state, torch.tensor([1, 2]))
    assert diagnostics["energy_balance_residual"].abs() < 2e-10
    local_energy = .5 * output.square().sum(-1)
    assert local_energy.max() <= model.dissipation.target_energy + 2e-10


def test_cross_entropy_reaches_local_collision(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSKinetic(path, vocab_size=32)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, _ = model(ids, targets)
    loss.backward()
    gradient = model.collision.angle[-1].weight.grad
    assert state.shape == (1, 8, 64)
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_training_replay(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer

    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSKinetic(path, vocab_size=32).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, _ = runner.step(
        torch.randint(32, (1, 4), device="cuda"),
        torch.randint(32, (1, 4), device="cuda"),
    )
    assert torch.isfinite(loss)
    assert state.shape == (1, 8, 64)
