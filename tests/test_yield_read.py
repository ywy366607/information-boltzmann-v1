"""Yield on SliceRead: per-head learnable τ, not pinned to 1/M."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import NativeMoTLayer, SliceRead


def test_tau_starts_open_not_one_over_m():
    read = SliceRead(d_x=16, d=16, n_slices=4, n_heads=2, use_yield_read=True)
    tau = F.softplus(read.yield_raw)
    assert float(tau.max()) < 0.05
    assert float(tau.max()) < 0.25  # 1/M for M=4; must not sit there
    with torch.no_grad():
        read.to_logits.weight.zero_()
        read.to_logits.bias.zero_()
    x = torch.randn(2, 8, 16)
    _, w = read(x)
    # Small τ: uniform 1/4 still enters so θ_h can learn.
    assert float(w.detach().abs().max()) > 0.1


def test_high_tau_kills_uniform():
    torch.manual_seed(0)
    read = SliceRead(d_x=16, d=16, n_slices=4, n_heads=2, use_yield_read=True)
    with torch.no_grad():
        read.to_logits.weight.zero_()
        read.to_logits.bias.zero_()
        read.yield_raw.fill_(4.0)  # softplus(4) ≈ 4 > 1
    x = torch.randn(2, 8, 16)
    _, w = read(x)
    assert float(w.detach().abs().max()) < 1e-5
    assert float(read.last_null.min()) > 0.99


def test_tau_gets_gradient_when_mass_exceeds_it():
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2, use_yield_read=True)
    x = torch.randn(1, 5, 8, requires_grad=False)
    _, w = read(x)
    w.sum().backward()
    assert read.yield_raw.grad is not None
    assert float(read.yield_raw.grad.abs().sum()) > 0.0


def test_peaked_assignment_survives_dead_zone():
    torch.manual_seed(1)
    M = 4
    read = SliceRead(d_x=16, d=16, n_slices=M, n_heads=2, use_yield_read=True)
    with torch.no_grad():
        read.to_logits.weight.zero_()
        read.to_logits.bias.zero_()
        read.to_logits.bias[0] = 8.0
    x = torch.randn(1, 6, 16)
    S, w = read(x)
    assert float(w[..., 0].detach().mean()) > 0.5
    assert float(w[..., 1:].abs().max()) < 1e-4
    assert S.shape == (1, M, 16)


def test_default_read_still_full_softmax():
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2)
    assert read.use_yield_read is False
    x = torch.randn(1, 5, 8)
    _, w = read(x)
    assert torch.allclose(w.sum(-1), torch.ones(1, 5), atol=1e-5)


def test_layer_preserve_mass_with_yield_read():
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4, use_yield_read=True,
    )
    assert layer.read.use_yield_read is True
    assert layer.deslice.preserve_mass is True
