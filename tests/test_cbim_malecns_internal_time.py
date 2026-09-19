"""Numerical contracts for the explicit internal-time CBIM."""
import numpy as np
import torch

from scripts.ib_local.cbim_malecns_internal_time import (
    CBIMMaleCNSInternalTime,
    LearnedBoundaryWrite,
    LearnedDirectionalEdgeTransport,
)


def graph_fixture(nodes=8):
    generator = torch.Generator().manual_seed(91)
    coordinates = torch.rand(nodes, 3, generator=generator)
    adjacency = torch.ones(nodes, nodes) - torch.eye(nodes)
    pairs = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]],
                          [[1, 2], [3, 4], [5, 6], [7, 0]]])
    return {
        "coordinates": coordinates,
        "coordinate_scale": torch.ones(3),
        "node_features": torch.rand(nodes, 16, generator=generator),
        "adjacency": adjacency,
        "collision_pairs": pairs,
        "collision_features": torch.rand(2, 4, 5, generator=generator),
    }


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_write_and_transport_have_exact_energy_accounts():
    torch.manual_seed(93)
    graph = graph_fixture()
    write = LearnedBoundaryWrite(graph, vocab_size=32).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64) * .1
    written, _, diagnostic = write(field, torch.tensor([1, 2]))
    assert diagnostic["write_balance_residual"] < 2e-10
    transport = LearnedDirectionalEdgeTransport(graph).double()
    moved, diagnostic = transport(written.reshape(2, 8, 8, 8))
    torch.testing.assert_close(moved.square().sum(), written.square().sum(), atol=2e-10, rtol=2e-10)
    assert diagnostic["transport_norm_residual"] < 2e-10
    assert diagnostic["transport_baseline_angle_abs_mean"] > .01
    assert not torch.allclose(moved, written.reshape(2, 8, 8, 8))


def test_full_chain_gives_ce_gradient_to_every_operator(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(97)
    model = CBIMMaleCNSInternalTime(path, vocab_size=32, micro_steps=2,
                                    checkpoint_tokens=2).double()
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, diagnostics = model(ids, targets)
    loss.backward()
    assert state.shape == (1, 8, 64)
    assert diagnostics["write_balance_residual"] < 2e-10
    for parameter in (model.source.address.weight,
                      model.transport.angle[-1].weight,
                      model.collision.angle[-1].weight,
                      model.readout.query):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
