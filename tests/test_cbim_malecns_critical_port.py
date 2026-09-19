"""Contracts for the critical Maxwell-boundary MaleCNS model."""
import numpy as np
import pytest
import torch

from scripts.ib_local.cbim_malecns_critical_port import (
    CBIMMaleCNSCriticalPort,
    PassiveCriticalPort,
)
from tests.test_cbim_malecns_v3 import graph_fixture


def save_graph(path, graph):
    np.savez(path, **{key: value.numpy() for key, value in graph.items()})


def test_maxwell_boundary_is_bounded_and_differentiable():
    torch.manual_seed(71)
    graph = graph_fixture()
    boundary = PassiveCriticalPort(graph, vocab_size=32).double()
    field = torch.randn(2, 8, 64, dtype=torch.float64) * .1
    critical_a = torch.zeros(2, 8, dtype=torch.float64)
    output, diagnostics = boundary(field, torch.tensor([1, 2]), critical_a)
    assert output.shape == field.shape
    assert 0 <= diagnostics["accommodation_mean"] <= boundary.max_write
    assert diagnostics["accommodation_max"] <= boundary.max_write
    assert diagnostics["port_balance_residual"].abs() < 1e-12
    output.square().mean().backward()
    assert boundary.write_rate.weight.grad is not None
    assert torch.isfinite(boundary.write_rate.weight.grad).all()
    assert boundary.proposal[-1].weight.grad.norm() > 0


def test_complete_critical_step_has_clean_roles_and_finite_controller(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(73)
    model = CBIMMaleCNSCriticalPort(path, vocab_size=32).double()
    state = model.initial_state(2)
    for token in range(6):
        state, diagnostics = model.evolve(
            state, torch.tensor([token, token + 1]), measure_critical=True)
    field, delta, critical_a, lambda_ema, flux_ema = model.unpack(state)
    assert state.shape == (2, 8, 131)
    assert torch.isfinite(state).all()
    assert torch.all(critical_a >= 0)
    assert torch.all(critical_a <= model.max_critical_a)
    probe_norm = delta.flatten(1).norm(dim=-1)
    torch.testing.assert_close(probe_norm, torch.ones_like(probe_norm))
    assert not hasattr(model, "gamma")
    assert not hasattr(model, "resource")
    assert not hasattr(model, "fatigue")
    assert not hasattr(model, "outflow")
    assert diagnostics["field_energy"] >= 0
    assert field.shape == (2, 8, 64)
    assert lambda_ema.shape == (2, 8)
    assert flux_ema.shape == (2, 8)


def test_cross_entropy_reaches_each_semantic_operator(tmp_path):
    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    torch.manual_seed(79)
    model = CBIMMaleCNSCriticalPort(path, vocab_size=32).double()
    ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss, _, diagnostics = model(ids, targets)
    loss.backward()
    parameters = (
        model.port.proposal[-1].weight,
        model.port.write_rate.weight,
        model.collision.angle[-1].weight,
        model.transport.symbol[-1].weight,
        model.readout.output[-1].weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert torch.isfinite(diagnostics["lambda_mean"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_critical_cuda_graph_training_step(tmp_path):
    from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer

    path = tmp_path / "graph.npz"
    save_graph(path, graph_fixture())
    model = CBIMMaleCNSCriticalPort(path, vocab_size=32).cuda()
    runner = CBIMGraphTrainer(model, tokens=4)
    loss, state, diagnostics = runner.step(
        torch.randint(32, (1, 4), device="cuda"),
        torch.randint(32, (1, 4), device="cuda"))
    assert torch.isfinite(loss)
    assert state.shape == (1, 8, 131)
    assert torch.isfinite(diagnostics["lambda_mean"])

