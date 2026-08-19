"""Yield on Bayes surprise: s≤τ ⇒ point field identity. Not L2 recon."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.hard_admit import YieldGate
from fine_grain.native_mot import NativeMoTLayer


def test_yield_below_surprise_threshold_is_zero():
    g = YieldGate(d_x=4)
    with torch.no_grad():
        g.to_tau.bias.fill_(10.0)
    s = torch.tensor([[0.2, 3.0]])
    k = torch.zeros(1, 2, 4)
    scale, tau, excess = g(s, k)
    assert float(tau.detach().min()) > 5.0
    assert float(scale.detach().max()) == 0.0
    assert float(excess.detach().max()) == 0.0


def test_yield_keeps_excess_surprise():
    g = YieldGate(d_x=1)
    with torch.no_grad():
        g.to_tau.weight.zero_()
        g.to_tau.bias.fill_(0.0)  # τ = softplus(0) ≈ 0.693
    s = torch.tensor([[2.0, 0.2]])
    k = torch.zeros(1, 2, 1)
    scale, tau, excess = g(s, k)
    tau0 = float(F.softplus(torch.tensor(0.0)))
    assert float(scale[0, 1].detach()) == 0.0
    assert abs(float(excess[0, 0].detach()) - (2.0 - tau0)) < 1e-5
    assert abs(float(scale[0, 0].detach()) - (2.0 - tau0) / 2.0) < 1e-5


def test_high_tau_is_field_identity():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        hard_admit=True, deslice_write="increment", surprise_mode="v1_bayes",
        gate_on="u",
    )
    with torch.no_grad():
        layer.admit.to_tau.bias.fill_(20.0)
    layer.eval()
    X = torch.randn(2, 16, 16)
    H = torch.randn(2, 4, 16)
    with torch.no_grad():
        X2, H2, _ = layer(X, H, text_mask=torch.ones(2, 4))
    assert float(layer.last_admit_frac) == 0.0
    assert torch.allclose(X2, X, atol=1e-5)
    assert (H2 - H).abs().sum() > 0


def test_baseline_surprise_does_not_fake_l2_refuse():
    """No Bayes U → do not invent ||X-recon||²; leave the write ungated."""
    torch.manual_seed(1)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        hard_admit=True, deslice_write="increment", surprise_mode="baseline",
    )
    X = torch.randn(1, 16, 16)
    H = torch.randn(1, 4, 16)
    X2, _, _ = layer(X, H, text_mask=torch.ones(1, 4))
    assert layer.last_m is None
    assert float((X2 - X).detach().abs().sum()) > 0.0
