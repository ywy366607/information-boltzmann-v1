"""Numerical contracts for the open-boundary MaleCNS kinetic CBIM v3."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_v3 import (
    CBIMMaleCNSV3,
    ContinuousGraphTransport,
    FullRankBoundaryExchange,
    FullStateKineticCollision,
    LocalBoundaryOutflow,
)


def graph_fixture(nodes=8, modes=4, layers=2):
    generator = torch.Generator().manual_seed(41)
    basis = torch.linalg.qr(
        torch.randn(nodes, modes, generator=generator)).Q
    pairs = []
    for layer in range(layers):
        order = torch.roll(torch.arange(nodes), layer)
        pairs.append(order.reshape(-1, 2))
    return {
        "coordinates": torch.rand(nodes, 3, generator=generator),
        "node_features": torch.rand(nodes, 14, generator=generator),
        "laplacian_basis": basis,
        "laplacian_eigenvalues": torch.linspace(.1, 1., modes),
        "neighbor_indices": torch.stack(
            (torch.roll(torch.arange(nodes), 1),
             torch.roll(torch.arange(nodes), -1)), -1),
        "neighbor_weights": torch.full((nodes, 2), .5),
        "collision_pairs": torch.stack(pairs),
        "collision_features": torch.rand(
            layers, nodes // 2, 5, generator=generator),
    }


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_boundary_exchange_is_full_rank_and_resource_bounded():
    torch.manual_seed(43)
    graph = graph_fixture()
    source = FullRankBoundaryExchange(graph, vocab_size=32)
    field = torch.randn(2, 8, 64) * .1
    resource = torch.ones(2, 8)
    fatigue = torch.zeros(2, 8)
    output, resource_next, envelope, diagnostics = source(
        field, resource, fatigue, torch.tensor([1, 2]))
    assert output.shape == field.shape
    assert envelope.shape == resource.shape
    assert torch.all((0 <= resource_next) & (resource_next <= 1))
    assert source.proposal[-1].out_features == 64
    assert diagnostics["write_rate_max"] <= source.max_write_rate


def test_continuous_transport_preserves_energy():
    graph = graph_fixture()
    transport = ContinuousGraphTransport(
        graph["laplacian_basis"], graph["laplacian_eigenvalues"]).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64)
    output, _ = transport(field)
    torch.testing.assert_close(
        output.square().sum(), field.square().sum(), atol=3e-10, rtol=3e-10)


def test_full_state_collision_preserves_only_declared_moments_and_energy():
    torch.manual_seed(53)
    graph = graph_fixture()
    collision = FullStateKineticCollision(graph["node_features"]).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64)
    output, _ = collision(field)
    before = torch.einsum("cd,bnd->bnc", collision.constraints, field)
    after = torch.einsum("cd,bnd->bnc", collision.constraints, output)
    torch.testing.assert_close(after, before, atol=5e-10, rtol=3e-10)
    torch.testing.assert_close(
        output.square().sum(), field.square().sum(), atol=8e-10, rtol=3e-10)
    assert collision.nullity == 60


def test_outflow_is_boundary_local_and_contracting():
    graph = graph_fixture()
    outflow = LocalBoundaryOutflow(graph["node_features"]).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64)
    resource = torch.ones(2, 8, dtype=torch.float64)
    fatigue = torch.zeros(2, 8, dtype=torch.float64)
    envelope = torch.zeros(2, 8, dtype=torch.float64)
    envelope[:, 3] = 1
    output, diagnostics = outflow(field, resource, fatigue, envelope)
    torch.testing.assert_close(output[:, :3], field[:, :3])
    torch.testing.assert_close(output[:, 4:], field[:, 4:])
    assert torch.all(output[:, 3].norm(-1) <= field[:, 3].norm(-1) + 1e-12)
    assert diagnostics["boundary_output_work"] >= 0


def test_complete_step_closes_energy_ledger_and_backpropagates(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(59)
    model = CBIMMaleCNSV3(path, vocab_size=32).double()
    state = model.initial_state(2)
    state[..., :64] = torch.randn(2, 8, 64, dtype=torch.float64) * .1
    next_state, diagnostics = model.evolve(state, torch.tensor([1, 2]))
    assert diagnostics["energy_balance_residual"].abs() < 2e-9
    assert next_state.shape == (2, 8, 66)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, _, _ = model(ids, targets)
    loss.backward()
    for parameter in (
            model.source.proposal[-1].weight,
            model.collision.angle[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v3_cuda_graph_training_step(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer

    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSV3(path, vocab_size=32).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, diagnostics = runner.step(
        torch.randint(32, (1, 4), device="cuda"),
        torch.randint(32, (1, 4), device="cuda"))
    assert torch.isfinite(loss)
    assert state.shape == (1, 8, 66)
    assert torch.isfinite(diagnostics["energy_balance_residual"])
