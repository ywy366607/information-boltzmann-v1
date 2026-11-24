from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.native_supply_regularizer import (
    body_supply, excitation, objective_supply, sigreg_zero, supply, transmission,
)


def test_zero_reference_matches_original_regularizer_and_valid_history():
    g = torch.Generator().manual_seed(9)
    current, history = torch.zeros(7, 5), torch.zeros(6, 5)
    valid = torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.bool)
    directions = torch.randn(5, 16, generator=g)
    torch.testing.assert_close(live_sigreg(current, history, valid, directions),
                               sigreg_zero(10, current))


def test_supply_routes_gradient_to_live_upstream_and_stops_endpoint_and_demand():
    g = torch.Generator().manual_seed(31)
    h = (2+torch.randn(48, 7, generator=g, dtype=torch.double)*.03).requires_grad_()
    z = torch.randn(48, 5, generator=g, dtype=torch.double, requires_grad=True)
    demand = torch.tensor(.05, dtype=torch.double, requires_grad=True)
    loss, info = supply(h, z, demand, torch.tensor(1., dtype=torch.double))
    dh, dz, dd = torch.autograd.grad(loss, (h, z, demand), allow_unused=True)
    assert info['k'] < info['q'] and torch.isfinite(dh).all() and dh.norm() > 0
    assert dz is None and dd is None


def test_scale_derivatives_preserve_relative_ridge_and_excitation_energy():
    g = torch.Generator().manual_seed(17)
    h = torch.randn(25, 6, generator=g, dtype=torch.double).requires_grad_()
    z = torch.randn(25, 4, generator=g, dtype=torch.double)
    for read in (lambda x: excitation(x), lambda x: transmission(x, z)):
        value = read(h)
        grad, = torch.autograd.grad(value, h)
        torch.testing.assert_close(value, read(3.2*h))
        assert abs(float((h.detach()*grad).sum())) < 1e-10


def test_satisfied_supply_has_zero_additional_gradient():
    g = torch.Generator().manual_seed(33)
    h = torch.randn(40, 5, generator=g, dtype=torch.double, requires_grad=True)
    z = h[:, :4].detach()
    loss, info = supply(h, z, torch.tensor(1.), torch.tensor(1.))
    assert info['k'] > info['q'] and loss == 0
    grad, = torch.autograd.grad(loss, h)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_body_port_keeps_current_graph_and_historical_ownership():
    g = torch.Generator().manual_seed(2)
    h = torch.randn(12, 6, generator=g, dtype=torch.double, requires_grad=True)
    history = torch.randn(7, 6, generator=g, dtype=torch.double, requires_grad=True)
    enc = torch.nn.Linear(6, 4, bias=False).double()
    model = SimpleNamespace(encoder=enc, memory=SimpleNamespace(history=history,
        history_valid=torch.tensor([1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)))
    loss, info = body_supply(model, h, torch.tensor(0., dtype=torch.double))
    grad = torch.autograd.grad(loss, (h, history, enc.weight), allow_unused=True)
    assert info['samples'] == 16 and grad[0] is not None
    assert grad[1] is None and grad[2] is None


def test_objective_scope_restores_original_method_on_exception():
    from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody
    original = LiveSigregBody.objective
    with pytest.raises(RuntimeError):
        with objective_supply():
            assert LiveSigregBody.objective is not original
            raise RuntimeError('scope test')
    assert LiveSigregBody.objective is original
