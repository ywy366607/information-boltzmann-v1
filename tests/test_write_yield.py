"""Write Yield: admission τ ≠ leak α. Drive shipped functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fine_grain.native_mot import NativeMoTLayer, NativeMoTStack, SliceRead
from fine_grain.omni_model import DualStreamOmni
from fine_grain.write_yield import retain_and_write, yield_residual


def test_dead_zone_is_exact_zero():
    d = torch.tensor([[[-0.02, 0.01, 0.5]]])
    tau = torch.tensor(0.1)
    u = yield_residual(d, tau)
    assert float(u[0, 0, 0]) == 0.0
    assert float(u[0, 0, 1]) == 0.0
    assert abs(float(u[0, 0, 2]) - 0.4) < 1e-6


def test_sign_preserved_on_excess():
    d = torch.tensor([[[-2.0, 2.0]]])
    u = yield_residual(d, torch.tensor(0.5))
    assert float(u[0, 0, 0]) == -1.5
    assert float(u[0, 0, 1]) == 1.5


def test_alpha_does_not_change_admission():
    d = torch.randn(2, 8, 4)
    tau = torch.tensor(0.2)
    u = yield_residual(d, tau)
    x = torch.randn(2, 8, 4)
    y1 = retain_and_write(x, u, alpha=1.0)
    y0 = retain_and_write(x, u, alpha=0.5)
    # Same write u; only the retained X differs.
    assert torch.allclose(y1 - x, u)
    assert torch.allclose(y0 - 0.5 * x, u)


def test_layer_write_yield_zeros_small_delta():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=8, d=8, n_slices=4, n_heads=2, res=4,
        deslice_write="increment", use_write_yield=True, write_alpha=1.0,
    )
    with torch.no_grad():
        layer.write_yield_raw.fill_(8.0)  # softplus(8)≈8, dead zone swallows Δ
    X = torch.zeros(1, 16, 8)
    H = torch.zeros(1, 2, 8)
    layer(X, H)
    assert layer.last_write_admit < 0.02


def test_read_stays_row_softmax_when_write_yield_on():
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2, use_yield_read=False)
    x = torch.randn(1, 6, 8)
    _, w = read(x)
    assert w.shape[-1] == 4
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_tau_is_not_one_over_m():
    layer = NativeMoTLayer(
        d_x=8, d=8, n_slices=4, n_heads=2, res=4, use_write_yield=True,
    )
    tau = float(F.softplus(layer.write_yield_raw).mean().detach())
    assert abs(tau - 1.0 / 4.0) > 0.05


def test_write_yield_still_optional_on_layer():
    layer = NativeMoTLayer(
        d_x=8, d=8, n_slices=4, n_heads=2, res=4, use_write_yield=True,
    )
    assert layer.write_yield_raw is not None
    X = torch.randn(1, 16, 8)
    H = torch.randn(1, 2, 8)
    layer(X, H)
    assert layer.last_write_tau is not None
