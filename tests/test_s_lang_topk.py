"""Slice-selective language Δ and increment write identity."""
from __future__ import annotations

import torch

from fine_grain.native_mot import NativeMoTLayer


def test_s_lang_topk_zero_keeps_all_delta():
    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=8, n_heads=4, res=8,
        surprise_mode="baseline", s_lang_topk=0,
    )
    S = torch.randn(2, 8, 32)
    delta = torch.randn(2, 8, 32)
    out = layer._mask_lang_delta(delta, 8)
    assert torch.allclose(out, delta)
    assert layer.last_lang_frac == 1.0


def test_s_lang_topk_keeps_exactly_k_slices():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=8, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="rms_dir",
        s_lang_topk=2, deslice_write="increment",
        gate_h_local=False,
    )
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 6, 32)
    layer(X, H, text_mask=torch.ones(2, 6))
    keep = layer.last_lang_keep
    assert keep is not None
    assert keep.shape == (2, 8, 1)
    assert torch.allclose(keep.sum(dim=1), torch.full((2, 1), 2.0))
    assert abs(layer.last_lang_frac - 2 / 8) < 1e-5


def test_increment_zero_delta_writes_zero():
    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=8, n_heads=4, res=8,
        surprise_mode="baseline", deslice_write="increment",
    )
    w = torch.softmax(torch.randn(2, 64, 8), dim=-1)
    S = torch.randn(2, 8, 32)
    dx = layer.deslice.write_delta(S - S, w)
    assert float(dx.abs().max()) < 1e-6
