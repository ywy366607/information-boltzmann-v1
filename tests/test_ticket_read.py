"""Read tickets = optical flow: Δ = X − pred, s ≤ τ → invisible."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import (
    NativeMoTLayer,
    NativeMoTStack,
    SliceRead,
    admit_read_weights,
    flow_residual,
    xy_features,
)


def test_zero_residual_is_invisible():
    w = torch.full((1, 6, 4), 0.25)
    s = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0, 0.0]])
    tau = torch.tensor(0.1)
    w2, gate = admit_read_weights(w, s, tau)
    assert float(gate[0, 3]) > 0.0
    assert float(gate[0, :3].max()) == 0.0
    assert float(w2[0, :3].abs().sum()) == 0.0
    assert float(w2[0, 3].sum()) > 0.0


def test_matched_prediction_zeroes_read():
    x = torch.randn(1, 8, 4)
    s = flow_residual(x, x)
    assert float(s.max()) < 1e-6
    w = torch.softmax(torch.randn(1, 8, 4), dim=-1)
    w2, gate = admit_read_weights(w, s, tau=torch.tensor(1e-4))
    assert float(w2.abs().sum()) == 0.0
    assert float(gate.max()) == 0.0


def test_one_pixel_vs_zero_pred_monopolizes():
    torch.manual_seed(0)
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2)
    x = torch.zeros(1, 8, 8)
    x[0, 0] = 4.0
    s = flow_residual(x, torch.zeros_like(x))
    tau = torch.tensor(0.1)
    _, w = read(x, point_admit=s, admit_tau=tau)
    assert float(w[0, 1:].detach().abs().sum()) == 0.0
    assert float(w[0, 0].detach().sum()) > 0.0


def test_layer_computes_flow_tickets():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_ticket_read=True, surprise_mode="baseline",
    )
    assert layer.lang_s0 is not None
    assert layer.lang_proto is not None
    X = torch.zeros(1, 16, 16)
    X[0, 0] = 3.0
    H = torch.zeros(1, 4, 16)
    s = layer.flow_tickets(X, H, torch.ones(1, 4), X_prior=torch.zeros_like(X))
    assert float(s[0, 0]) > float(s[0, 1:].max())


def test_xy_features_have_fourier():
    xy = torch.zeros(1, 4, 2)
    xy[0, 1] = torch.tensor([1.0, -1.0])
    feat = xy_features(xy, n_freq=4)
    assert feat.shape == (1, 4, 2 + 16)
    assert float((feat[0, 0] - feat[0, 1]).pow(2).sum()) > 0.0


def test_same_h_different_xy_different_pred():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_ticket_read=True, surprise_mode="baseline",
    )
    H = torch.randn(1, 4, 16)
    pred = layer.language_prior(H, torch.ones(1, 4), 16, H.device, H.dtype)
    assert pred.shape == (1, 16, 16)
    # broadcast proto would be identical at every n
    assert float((pred[0, 0] - pred[0, 5]).detach().pow(2).sum()) > 1e-8
    assert float(pred.detach().var(dim=1).mean()) > 1e-8


def test_same_xy_different_h_different_pred():
    torch.manual_seed(1)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_ticket_read=True, surprise_mode="baseline",
    )
    Ha = torch.randn(1, 4, 16)
    Hb = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4)
    pa = layer.language_prior(Ha, mask, 16, Ha.device, Ha.dtype)
    pb = layer.language_prior(Hb, mask, 16, Hb.device, Hb.dtype)
    assert float((pa - pb).detach().pow(2).mean()) > 1e-8
    # Token-wise: swapping one token must move S0, not just a pooled vector.
    Hc = Ha.clone()
    Hc[0, 0] = Hb[0, 0]
    pc = layer.language_prior(Hc, mask, 16, Hc.device, Hc.dtype)
    assert float((pa - pc).detach().pow(2).mean()) > 1e-8


def test_flow_tickets_spatial_on_uniform_x():
    """Black/uniform X still gets spatially varying tickets from f(H,xy)."""
    torch.manual_seed(2)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_ticket_read=True, surprise_mode="baseline",
    )
    X = torch.zeros(1, 16, 16)
    H = torch.randn(1, 4, 16)
    s = layer.flow_tickets(X, H, torch.ones(1, 4), X_prior=None)
    assert s.shape == (1, 16)
    assert float(s.std()) > 1e-6


def test_stack_ticket_read_runs():
    torch.manual_seed(0)
    stack = NativeMoTStack(
        d_llm=16, res=4, d_x=16, d=16, n_slices=4, n_layers=2, n_heads=2,
        use_ticket_read=True, surprise_mode="v1_bayes",
    )
    img = torch.zeros(1, 3, 4, 4)
    img[0, :, 0, 0] = 1.0
    emb = torch.randn(1, 3, 16)
    X, _, tok, _ = stack.forward_native(img, emb, torch.ones(1, 3))
    assert X.shape[1] == 16
    assert tok.shape[1] == 4
    assert stack.layers[0].last_admit_alpha is not None
    pred = stack.layers[0].last_lang_pred
    assert pred is not None
    assert pred.shape[1] == 16
    assert float(pred.var(dim=1).mean()) > 1e-8
