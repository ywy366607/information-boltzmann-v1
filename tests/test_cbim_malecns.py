"""Numerical invariants for the MaleCNS graph-field CBIM."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns import (
    CBIMMaleCNS,
    GraphConservativeScattering,
    GraphLocalSourceOutflow,
    GraphSpectralTransport,
)


def graph_fixture(nodes=8, modes=4, layers=2):
    generator = torch.Generator().manual_seed(13)
    basis = torch.linalg.qr(torch.randn(nodes, modes, generator=generator)).Q
    pairs = []
    for layer in range(layers):
        order = torch.roll(torch.arange(nodes), layer)
        pairs.append(order.reshape(-1, 2))
    neighbors = torch.stack((torch.roll(torch.arange(nodes), 1),
                             torch.roll(torch.arange(nodes), -1)), 1)
    return {
        "coordinates": torch.rand(nodes, 3, generator=generator),
        "node_features": torch.rand(nodes, 14, generator=generator),
        "adjacency": torch.eye(nodes),
        "neighbor_indices": neighbors,
        "neighbor_weights": torch.full((nodes, 2), .5),
        "laplacian_basis": basis,
        "laplacian_eigenvalues": torch.linspace(.1, 1., modes),
        "collision_pairs": torch.stack(pairs),
        "collision_features": torch.rand(
            layers, nodes // 2, 5, generator=generator),
    }


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_graph_transport_preserves_norm():
    graph = graph_fixture()
    model = GraphSpectralTransport(
        graph["laplacian_basis"], graph["laplacian_eigenvalues"], d=16).double()
    state = torch.randn(3, 8, 16, dtype=torch.float64)
    output, _ = model(state)
    torch.testing.assert_close(output.norm(), state.norm(), atol=2e-10, rtol=2e-10)


def test_graph_scattering_preserves_sum_and_energy():
    graph = graph_fixture()
    model = GraphConservativeScattering(graph, d=16).double()
    state = torch.randn(2, 8, 16, dtype=torch.float64)
    output, _ = model(state)
    torch.testing.assert_close(output.sum(1), state.sum(1), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(output.square().sum((1, 2)),
                               state.square().sum((1, 2)),
                               atol=2e-11, rtol=2e-12)


def test_graph_source_is_bounded():
    graph = graph_fixture()
    model = GraphLocalSourceOutflow(graph, vocab_size=32, d=16)
    state = torch.randn(2, 8, 16) * 100
    output, _ = model(state, torch.tensor([1, 2]))
    assert output.norm(dim=-1).amax() <= 1.250001


def test_end_to_end_gradient_reaches_scattering(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNS(path, vocab_size=32, d=16)
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, _ = model(ids, targets)
    assert state.shape == (1, 8, 16)
    loss.backward()
    gradient = model.scattering.angle[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_cuda_training_replay(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer

    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNS(path, vocab_size=32, d=16).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, _ = runner.step(torch.randint(32, (1, 4), device="cuda"),
                                 torch.randint(32, (1, 4), device="cuda"))
    assert torch.isfinite(loss)
    assert state.shape == (1, 8, 16)
