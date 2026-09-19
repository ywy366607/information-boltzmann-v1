"""Contracts for passive-port CBIM v4; GPU execution waits for v3 training."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_v4 import (
    CBIMMaleCNSV4, PassiveBoundaryScattering)
from tests.test_cbim_malecns_v3 import graph_fixture, save_graph


def test_boundary_scattering_closes_combined_energy():
    torch.manual_seed(71)
    boundary = PassiveBoundaryScattering(
        graph_fixture(), vocab_size=32).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64)
    output, response, diagnostics = boundary(field, torch.tensor([1, 2]))
    assert output.shape == field.shape
    assert response.shape == (2, 64)
    assert diagnostics["boundary_balance_residual"] < 2e-12


def test_scattered_response_has_no_incident_only_shortcut():
    boundary = PassiveBoundaryScattering(graph_fixture(), vocab_size=32)
    zero = torch.zeros(2, 8, 64)
    _, response, _ = boundary(zero, torch.tensor([1, 2]))
    torch.testing.assert_close(response, torch.zeros_like(response))


def test_model_has_only_field_state_and_no_stabilizer(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSV4(path, vocab_size=32)
    assert model.initial_state(2).shape == (2, 8, 64)
    names = set(dict(model.named_modules()))
    assert all("gamma" not in name and "outflow" not in name
               and "fatigue" not in name and "resource" not in name
               for name in names)


def test_end_to_end_gradients_reach_boundary_collision_transport(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(73)
    model = CBIMMaleCNSV4(path, vocab_size=32).double()
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, state, diagnostics = model(ids, targets)
    loss.backward()
    assert state.shape == (1, 8, 64)
    assert diagnostics["boundary_balance_residual"] < 2e-12
    for parameter in (model.boundary.coupling.weight,
                      model.collision.angle[-1].weight,
                      model.transport.symbol[-1].weight,
                      model.readout.output[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v4_cuda_graph_training_step(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSV4(path, vocab_size=32).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, diagnostics = runner.step(
        torch.randint(32, (1, 4), device="cuda"),
        torch.randint(32, (1, 4), device="cuda"))
    assert torch.isfinite(loss)
    assert torch.isfinite(state).all()
    assert torch.isfinite(diagnostics["boundary_balance_residual"])
