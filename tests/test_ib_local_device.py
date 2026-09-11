import copy

import torch
import numpy as np

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.device_collision import collision_device, prepare_layers
from scripts.ib_local.reference import collision_serial, sample_candidates
from scripts.ib_local.types import FrozenContext
from scripts.ib_local.sampling import sample_window


def test_device_layers_match_serial_values_and_all_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    kernel = CollisionKernel(2, 12).double()
    other = copy.deepcopy(kernel)
    x = (torch.randn(12, 2, dtype=torch.float64) * .3).requires_grad_()
    v = torch.randn_like(x).requires_grad_()
    xx, vv = x.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
    table = sample_candidates(12, 2, 12., 1., 1., dtype=torch.float64)
    actual, lp, accepted = collision_device(x, v, kernel, prepare_layers(table, 12, 'cpu'))
    expected = collision_serial(xx, vv, FrozenContext(xx, vv), table, other)
    torch.testing.assert_close(actual, expected.v, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(lp, expected.log_prob, atol=1e-10, rtol=1e-8)
    assert int(accepted) == expected.stats['accepted']
    weights = torch.randn_like(v)
    ((actual * weights).sum() + lp).backward()
    ((expected.v * weights).sum() + expected.log_prob).backward()
    for a, b in zip((x, v, *kernel.parameters()), (xx, vv, *other.parameters())):
        assert torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)


def test_padding_preserves_candidates_values_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(35)
    kernel = CollisionKernel(4, 8).double()
    other = copy.deepcopy(kernel)
    x = (torch.randn(8, 4, dtype=torch.float64) * .1).requires_grad_()
    v = torch.randn_like(x).requires_grad_()
    xx, vv = x.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
    a, count = sample_window(np.random.default_rng(17), 1, 1, 8, 'cpu', dtype=torch.float64)
    b, count2 = sample_window(np.random.default_rng(17), 1, 1, 8, 'cpu', dtype=torch.float64, padding=True)
    assert count == count2 == sum(int(layer[4].sum()) for layer in b[0])
    va, la, ca = collision_device(x, v, kernel, a[0])
    vb, lb, cb = collision_device(xx, vv, other, b[0])
    torch.testing.assert_close(va, vb, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(la, lb, atol=1e-10, rtol=1e-8)
    assert int(ca) == int(cb)
    weight = torch.randn_like(va)
    ((va * weight).sum() + la).backward()
    ((vb * weight).sum() + lb).backward()
    for a, b in zip((x, v, *kernel.parameters()), (xx, vv, *other.parameters())):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)
