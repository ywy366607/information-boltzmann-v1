"""Contracts for anatomical-port, geometric-transport CBIM."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_port_transport import (
    AnatomicalThreePortBoundary,
    CBIMMaleCNSPortTransport,
)


def graph_fixture(nodes=8):
    generator = torch.Generator().manual_seed(71)
    coordinates = torch.tensor([
        [0., 0., 0.], [.2, .1, .1], [.4, .2, .2], [.6, .3, .3],
        [.8, .4, .4], [1., .5, .5], [.7, .8, .8], [1., 1., 1.]],
        dtype=torch.float64)
    adjacency = (torch.ones(nodes, nodes, dtype=torch.float64)
                 - torch.eye(nodes, dtype=torch.float64))
    input_ports = torch.zeros(4, nodes, dtype=torch.float64)
    input_ports[:, :4] = torch.rand(
        4, 4, generator=generator, dtype=torch.float64) + .1
    output_ports = torch.zeros(4, nodes, dtype=torch.float64)
    output_ports[:, 4:] = torch.rand(
        4, 4, generator=generator, dtype=torch.float64) + .1
    return {
        "coordinates": coordinates,
        "coordinate_scale": torch.tensor([1., 2., 3.], dtype=torch.float64),
        "node_features": torch.rand(
            nodes, 16, generator=generator, dtype=torch.float64),
        "adjacency": adjacency,
        "input_port_weights": input_ports,
        "output_port_weights": output_ports,
    }


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_three_port_boundary_closes_energy_and_has_disjoint_support():
    torch.manual_seed(73)
    boundary = AnatomicalThreePortBoundary(
        graph_fixture(), vocab_size=32, input_topk=3, output_topk=3).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64) * .1
    written, _, write = boundary.write(field, torch.tensor([1, 2]))
    after, response, bath = boundary.read_and_absorb(written)
    assert after.shape == field.shape
    assert response.shape == (2, 4, 64)
    assert write["write_balance_residual"] < 2e-10
    assert bath["bath_balance_residual"] < 2e-10
    input_support = boundary.input_gate_masks.amax(0) > 0
    output_support = boundary.output_gate > 0
    assert not torch.any(input_support & output_support)


def test_output_has_no_global_or_direct_token_shortcut(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(79)
    model = CBIMMaleCNSPortTransport(
        path, vocab_size=32, input_topk=3, output_topk=3).double()
    state = model.initial_state(2)
    _, no_transport_response, _ = model.evolve(
        state, torch.tensor([1, 2]), disable_transport=True)
    _, transported_response, _ = model.evolve(
        state, torch.tensor([1, 2]), disable_transport=False)
    torch.testing.assert_close(
        no_transport_response, torch.zeros_like(no_transport_response))
    assert transported_response.norm() > 0
    connected = list(model.modules())
    assert all(module.__class__.__name__ != "GraphStateReadout"
               for module in connected)


def test_ce_gradient_py_path_uses_collision_transport_and_output_port(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(83)
    model = CBIMMaleCNSPortTransport(
        path, vocab_size=32, input_topk=3, output_topk=3,
        checkpoint_tokens=2).double()
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, diagnostics = model(ids, targets)
    loss.backward()
    assert state.shape == (1, 8, 64)
    assert diagnostics["write_balance_residual"] < 2e-10
    assert diagnostics["bath_balance_residual"] < 2e-10
    assert diagnostics["transport_norm_residual"] < 2e-10
    for parameter in (
            model.boundary.incident[-1].weight,
            model.collision.angle[-1].weight,
            model.port_readout[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_training_step(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSPortTransport(
        path, vocab_size=32, input_topk=3, output_topk=3,
        checkpoint_tokens=2).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, diagnostics = runner.step(
        torch.randint(32, (1, 4), device="cuda"),
        torch.randint(32, (1, 4), device="cuda"))
    assert torch.isfinite(loss)
    assert torch.isfinite(state).all()
    assert torch.isfinite(diagnostics["bath_balance_residual"])
