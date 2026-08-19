"""Null slice ∅: a point may not enter any content slice."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.models import sparse_deslice_weights
from fine_grain.native_mot import DesliceWrite, NativeMoTLayer, SliceRead, slice_mass_loss_weights


def test_default_read_still_partitions_unity():
    torch.manual_seed(0)
    read = SliceRead(d_x=16, d=16, n_slices=4, n_heads=2)
    assert read.use_null_slice is False
    assert read.to_null is None
    x = torch.randn(2, 12, 16)
    S, w = read(x)
    assert S.shape == (2, 4, 16)
    assert w.shape == (2, 12, 4)
    assert torch.allclose(w.sum(-1), torch.ones(2, 12), atol=1e-5)
    assert read.last_null is None


def test_null_softmax_leaves_mass_on_sink():
    torch.manual_seed(1)
    read = SliceRead(d_x=16, d=16, n_slices=4, n_heads=2, use_null_slice=True)
    assert read.to_null is not None
    x = torch.randn(2, 12, 16)
    S, w = read(x)
    leftover = 1.0 - w.sum(dim=-1)
    assert S.shape == (2, 4, 16)
    assert w.shape == (2, 12, 4)
    assert float(leftover.detach().min()) >= -1e-6
    assert float(leftover.detach().mean()) > 0.0
    assert torch.allclose(leftover, read.last_null, atol=1e-5)
    # Content masses are a leftover of a partition of M+1, not renormalized.
    assert float(w.detach().sum(-1).mean()) < 0.999


def test_override_does_not_renorm_when_null_on():
    read = SliceRead(d_x=8, d=8, n_slices=3, n_heads=2, use_null_slice=True)
    x = torch.randn(1, 5, 8)
    w = torch.zeros(1, 5, 3)
    w[0, 0, 0] = 0.4
    S, w_out = read(x, w_override=w)
    assert torch.allclose(w_out[0, 0], w[0, 0])
    assert float(w_out[0, 1].sum()) == 0.0
    assert abs(float(read.last_null[0, 0]) - 0.6) < 1e-6
    assert abs(float(read.last_null[0, 1]) - 1.0) < 1e-6


def test_one_point_can_monopolize_a_slice():
    """1px exclusive on slice 0; background on ∅. S[0] is that point, not the mean."""
    torch.manual_seed(2)
    d_x, d, N, M = 8, 8, 16, 4
    read = SliceRead(d_x=d_x, d=d, n_slices=M, n_heads=2, use_null_slice=True)
    x = torch.randn(1, N, d_x)
    w = torch.zeros(1, N, M)
    w[0, 0, 0] = 1.0
    S, w_out = read(x, w_override=w)
    xp = read.proj_in(x)
    got = S[:, 0]
    exclusive = read.out(xp[:, 0])
    meanish = read.out(xp.mean(dim=1))
    empty = read.out(torch.zeros_like(xp[:, 0]))
    assert torch.allclose(got, exclusive, atol=1e-5)
    assert float((got - exclusive).abs().mean()) < float((got - meanish).abs().mean())
    assert torch.allclose(S[:, 1], empty, atol=1e-4)
    pi = slice_mass_loss_weights(w_out)
    assert float(pi[0, 0]) > float(pi[0, 1:].max()) * 5


def test_deslice_does_not_inflate_sink_points():
    torch.manual_seed(3)
    B, N, M, d, d_x = 1, 8, 4, 8, 8
    inc = torch.ones(B, M, d)
    w = torch.full((B, N, M), 0.02 / M)
    w[0, 0, 0] = 0.8
    keep = DesliceWrite(d, d_x, deslice_topk=2, preserve_mass=True)
    blow = DesliceWrite(d, d_x, deslice_topk=2, preserve_mass=False)
    with torch.no_grad():
        keep.proj.weight.copy_(blow.proj.weight)
        keep.proj.bias.copy_(blow.proj.bias)
    d_keep = keep.write_delta(inc, w)
    d_blow = blow.write_delta(inc, w)
    # Sink-ish points (tiny content mass) must not be renormalized up to 1.
    assert float(d_keep[0, 1].abs().mean()) < 0.05
    assert float(d_blow[0, 1].abs().mean()) > float(d_keep[0, 1].abs().mean()) * 5
    assert float(d_keep[0, 0].abs().mean()) > float(d_keep[0, 1].abs().mean())


def test_sparse_deslice_renorm_false_keeps_leftover():
    torch.manual_seed(4)
    w = torch.softmax(torch.randn(2, 1, 6, 5), dim=-1) * 0.2
    w_sp = sparse_deslice_weights(w, topk=2, renorm=False)
    assert w_sp.shape == w.shape
    assert float(w_sp.sum(-1).mean()) < 0.25
    assert float((w_sp > 1e-8).float().sum(-1).mean()) == 2.0
    w_on = sparse_deslice_weights(w, topk=2, renorm=True)
    assert torch.allclose(w_on.sum(-1), torch.ones(2, 1, 6), atol=1e-5)


def test_layer_with_null_zero_increment_writes_nothing():
    torch.manual_seed(5)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_null_slice=True, deslice_write="increment", deslice_topk=2,
    )
    X = torch.randn(1, 16, 16)
    H = torch.randn(1, 4, 16)
    S, w = layer.read(X)
    delta = layer.deslice.write_delta(torch.zeros_like(S), w)
    assert float(delta.abs().max()) < 1e-6
    assert layer.deslice.preserve_mass is True
    leftover = 1.0 - w.sum(-1)
    assert float(leftover.mean()) > 0.0
