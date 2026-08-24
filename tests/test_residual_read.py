"""Read the residual r=X−S0. ℓ_∅=τ−γ e so explained points are not read."""
from __future__ import annotations

import torch

from fine_grain.native_mot import NativeMoTLayer, SliceRead
from fine_grain.omni_model import DualStreamOmni


def test_zero_residual_goes_to_null():
    torch.manual_seed(0)
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2, use_null_slice=True)
    r = torch.zeros(2, 10, 8)
    e = r.pow(2).mean(-1)
    _, w = read(r, point_pe=e)
    leftover = 1.0 - w.sum(-1)
    assert float(leftover.mean().detach()) > 0.5
    assert float(w.abs().sum().detach()) < float(leftover.sum().detach())


def test_high_pe_avoids_null_relative_to_zero():
    torch.manual_seed(1)
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2, use_null_slice=True)
    r0 = torch.zeros(1, 8, 8)
    r1 = torch.zeros(1, 8, 8)
    r1[0, 0] = 4.0
    _, w0 = read(r0, point_pe=r0.pow(2).mean(-1))
    _, w1 = read(r1, point_pe=r1.pow(2).mean(-1))
    null0 = 1.0 - w0.sum(-1)
    null1 = 1.0 - w1.sum(-1)
    assert float(null1[0, 0].detach()) < float(null0[0, 0].detach())
    assert float(w1[0, 0].sum().detach()) > float(w1[0, 1].sum().detach())


def test_layer_matched_prior_not_read():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_residual_read=True, deslice_write="increment",
    )
    assert layer.lang_s0 is not None
    assert layer.read.use_null_slice is True
    X = torch.zeros(1, 16, 16)
    H = torch.zeros(1, 3, 16)
    s0 = layer.language_prior(H, torch.ones(1, 3), 16, X.device, X.dtype)
    with torch.no_grad():
        # Force S0 ≈ X so residual is ~0 (explained).
        pass
    X = s0.detach()
    X2, _, _ = layer(X, H, torch.ones(1, 3))
    null = layer.last_null
    assert null is not None
    assert float(null.mean()) > 0.3


def test_band_pane_punishes_filling_neighbors():
    """Local flood around structure is expensive. No stroke mask."""
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True,
    )
    tgt = torch.full((1, 3, 8, 8), -1.0)
    tgt[:, 1, 3, 4] = 1.0
    fig, band, far = m._structure_panes(tgt)
    err_trace = torch.zeros(1, 64, 1)
    err_trace[0, 3 * 8 + 4] = 1.0
    err_fill = err_trace.clone()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = 3 + dy, 4 + dx
            if 0 <= y < 8 and 0 <= x < 8:
                err_fill[0, y * 8 + x] = 1.0
    lt = float(m._three_pane(err_trace, fig, band, far).detach())
    lf = float(m._three_pane(err_fill, fig, band, far).detach())
    assert lf > lt * 1.5


def test_s0_loss_is_not_area_mean():
    """A single unusual pixel in o gets a real vote. No stroke mask."""
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True,
    )
    tgt = torch.full((1, 3, 8, 8), -1.0)
    tgt[:, 1, 3, 4] = 1.0
    pi = m._observation_pi(tgt)
    err = torch.zeros(1, 64, 1)
    err[0, 3 * 8 + 4] = 1.0
    pane = float(m._two_pane(err, err, pi).detach())
    area = float(err.mean().detach())
    assert pane > 10.0 * area


def test_s0_accuracy_still_reaches_prior():
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True,
    )
    assert m.mot_stack.use_residual_read is True
    z = torch.randn(1, 3, 8, 8)
    tgt = torch.randn(1, 3, 8, 8)
    out = m(z, ["Draw digit 2 with a thin red stroke"], need_pix=[True], t=torch.ones(1))
    loss, meta = m.omni_loss(out, {
        "need_text": [False], "need_pix": [True],
        "answer": ["2"], "target_rgb": tgt,
        "stroke": torch.zeros(1, 8, 8), "t": None,
    }, z.device)
    assert "s0_acc" in meta
    m.zero_grad(set_to_none=True)
    loss.backward()
    w = m.mot_stack.layers[0].lang_s0[0].weight
    assert w.grad is not None
    assert float(w.grad.abs().sum()) > 0.0


def test_content_softmax_without_null_ignores_uniform_pe_boost():
    """Same log e on every content slice cancels — why we need ∅."""
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 4)
    e = torch.rand(1, 5, 1)
    w0 = torch.softmax(logits, dim=-1)
    w1 = torch.softmax(logits + e.log(), dim=-1)
    assert torch.allclose(w0, w1, atol=1e-5)
