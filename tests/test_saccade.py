"""Residual pixel-mass saccade: no IG, no grid, Q unchanged."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import NativeMoTStack, SliceRead
from fine_grain.saccade import residual_pixel_mass


def test_residual_mass_highlights_errors():
    X = torch.zeros(1, 4, 3)
    recon = torch.zeros(1, 4, 3)
    X[0, 0] = 2.0
    pm = residual_pixel_mass(X, recon, gain=1.0)
    assert pm.shape == (1, 4)
    assert float(pm[0, 0]) > float(pm[0, 1])


def test_pixel_mass_changes_pool_not_assignment():
    rd = SliceRead(d_x=8, d=16, n_slices=4, n_heads=2)
    x = torch.randn(2, 16, 8)
    S0, w0 = rd(x)
    pm = torch.ones(2, 16)
    pm[:, :4] = 3.0
    S1, w1 = rd(x, pixel_mass=pm)
    assert torch.allclose(w0, w1)
    assert (S0 - S1).abs().sum() > 0


def test_saccade_off_matches_default():
    kw = dict(d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4)
    torch.manual_seed(0)
    a = NativeMoTStack(**kw)
    torch.manual_seed(0)
    b = NativeMoTStack(**kw, saccade=False)
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    a.eval(); b.eval()
    with torch.no_grad():
        Xa, Ha, _, _ = a.forward_native(img, emb, torch.ones(1, 3))
        Xb, Hb, _, _ = b.forward_native(img, emb, torch.ones(1, 3))
    assert torch.allclose(Xa, Xb)
    assert torch.allclose(Ha, Hb)


def test_saccade_sets_pixel_mass():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4,
        saccade=True, saccade_gain=1.0,
    )
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 4, 32)
    stack.eval()
    with torch.no_grad():
        stack.forward_native(img, emb, torch.ones(2, 4))
    pm = stack.layers[0].last_pixel_mass
    assert pm is not None and pm.shape[-1] == 64
    assert float(pm.mean()) > 0


def test_halt_skips_second_look_when_residual_tiny():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=2, n_heads=4,
        saccade=True, saccade_inner=2, saccade_halt=True, saccade_halt_eps=1e9,
    )
    stack.eval()
    img = torch.rand(3, 3, 8, 8)
    emb = torch.randn(3, 3, 32)
    with torch.no_grad():
        _, _, _, tr = stack.forward_native(img, emb, torch.ones(3, 3))
    # eps huge → after look 1, never take look 2. 2 layers × 1 = 2 traces.
    assert len(tr) == 2
    assert float(stack._last_looks.mean()) == 2.0


def test_four_layer_inner_time():
    """Same param count as F2: 4 unique layers, 2 looks each."""
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=4, n_heads=4,
        share_layers=False, saccade=True, saccade_inner=2,
    )
    assert len(stack.layers) == 4
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    _, _, _, tr = stack.forward_native(img, emb, torch.ones(1, 3))
    assert len(tr) == 8


def test_shared_saccade_no_lti():
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=8, n_layers=4, n_heads=4,
        share_layers=True, n_loops=3, lti_inject=False, saccade=True,
    )
    img = torch.rand(1, 3, 8, 8)
    emb = torch.randn(1, 3, 32)
    _, _, _, tr = stack.forward_native(img, emb, torch.ones(1, 3))
    assert len(tr) == 3
    assert stack.layers[0].last_pixel_mass is not None
