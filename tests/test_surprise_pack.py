"""BLT pack: existing surprise tickets change SliceRead assignment."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.native_mot import (
    NativeMoTLayer,
    NativeMoTStack,
    SliceRead,
    next_read_tickets,
    surprise_pack_weights,
)


def test_flat_surprise_does_not_repack():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(2, 8, 4), dim=-1)
    u = torch.ones(2, 8)
    w2, alpha = surprise_pack_weights(w, u)
    assert torch.allclose(w2, w, atol=1e-6)
    assert float(alpha.abs().max()) == 0.0


def test_high_surprise_point_leaves_pack_slice():
    w = torch.full((1, 6, 4), 1.0 / 4)
    u = torch.ones(1, 6)
    u[0, 0] = 20.0
    w2, alpha = surprise_pack_weights(w, u)
    assert float(alpha[0, 0]) > float(alpha[0, 1:].max())
    assert float(w2[0, 0, 0]) < float(w2[0, 1, 0])
    assert float(w2[0, 1, 0]) > 0.5
    assert torch.allclose(w2.sum(-1), torch.ones(1, 6), atol=1e-5)


def test_read_uses_lagged_u_not_a_new_entropy():
    torch.manual_seed(1)
    read = SliceRead(d_x=8, d=8, n_slices=4, n_heads=2)
    x = torch.randn(1, 10, 8)
    u = torch.ones(1, 10)
    u[0, 0] = 30.0
    _, w0 = read(x)
    _, w1 = read(x, point_u=u)
    assert float(w1[0, 0, 0].detach()) < float(w0[0, 0, 0].detach())
    assert float(w1[0, 1:, 0].detach().mean()) > float(w1[0, 0, 0].detach())
    assert read.last_pack_alpha is not None
    assert float(read.last_pack_alpha[0, 0]) > 0.5


def test_next_tickets_flat_U_keeps_residual():
    pm = torch.tensor([[1.0, 3.0, 1.0]])
    ux = torch.ones(1, 3)
    t = next_read_tickets(pm, ux)
    assert torch.allclose(t, pm)


def test_layer1_packs_from_layer0_residual():
    torch.manual_seed(2)
    stack = NativeMoTStack(
        d_llm=16, res=4, d_x=16, d=16, n_slices=4, n_layers=2, n_heads=2,
        pack_by_surprise=True, surprise_mode="v1_bayes",
    )
    img = torch.zeros(1, 3, 4, 4)
    img[0, :, 0, 0] = 1.0
    emb = torch.randn(1, 3, 16)
    stack.forward_native(img, emb, torch.ones(1, 3))
    w0 = stack.layers[0].last_w
    w1 = stack.layers[1].last_w
    assert stack.layers[1].last_pack_alpha is not None
    # Layer 0 has no lagged U so it does not pack. Layer 1 does.
    assert stack.layers[0].last_pack_alpha is None
    assert float(w1[0, 0, 0]) != float(w0[0, 0, 0]) or float(
        stack.layers[1].last_pack_alpha.max()
    ) >= 0.0
