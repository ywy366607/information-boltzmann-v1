"""P0 residual cell: identity fixed point, increment deslice, π_X freeze.

Drives the shipped NativeMoTLayer / NativeMoTStack / DualStreamOmni forward —
no reimplementation of the update.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.bayesian_surprise import global_gate_from_surprise
from fine_grain.native_mot import DesliceWrite, NativeMoTLayer, NativeMoTStack


def _layer(**kw) -> NativeMoTLayer:
    cfg = dict(d_x=32, d=64, n_slices=8, n_heads=4, res=8)
    cfg.update(kw)
    return NativeMoTLayer(**cfg)


def test_forced_zero_gate_three_state_identity():
    torch.manual_seed(0)
    layer = _layer(
        surprise_mode="v0_jepa", s_update="rms", local_kind="dw3",
        deslice_write="increment", gate_h_local=True,
    )
    layer.eval()
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 10, 64)
    z = torch.zeros(2, 8, 1)
    with torch.no_grad():
        X2, H2, tr = layer(X, H, text_mask=torch.ones(2, 10), force_gate=z)
    assert torch.allclose(X2, X, atol=1e-5, rtol=1e-5), float((X2 - X).abs().max())
    assert torch.allclose(H2, H, atol=1e-5, rtol=1e-5), float((H2 - H).abs().max())
    assert torch.allclose(layer.last_S_write, layer.last_S, atol=1e-5, rtol=1e-5)
    assert tr.x_delta < 1e-5 and tr.h_delta < 1e-5
    assert float(layer.last_g_G.abs().max()) < 1e-6


def test_constant_zero_gate_is_identity():
    torch.manual_seed(1)
    layer = _layer(surprise_mode="constant", deslice_write="increment", gate_h_local=True)
    layer.surprise_gate.constant_val = 0.0
    layer.eval()
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 6, 64)
    with torch.no_grad():
        X2, H2, _ = layer(X, H, text_mask=torch.ones(2, 6))
    assert torch.allclose(X2, X, atol=1e-5)
    assert torch.allclose(H2, H, atol=1e-5)
    assert torch.allclose(layer.last_S_write, layer.last_S, atol=1e-5)


def test_increment_write_zero_when_s_unchanged():
    torch.manual_seed(2)
    dw = DesliceWrite(d=16, d_x=8, deslice_topk=0)
    with torch.no_grad():
        dw.proj.bias.fill_(7.5)
    zeros = torch.zeros(2, 4, 16)
    w = torch.softmax(torch.randn(2, 16, 4), dim=-1)
    delta = dw.write_delta(zeros, w)
    assert delta.abs().max().item() < 1e-6, float(delta.abs().max())

    layer = _layer(surprise_mode="baseline", deslice_write="increment", gate_h_local=True)
    layer.eval()
    X = torch.randn(1, 64, 32)
    H = torch.randn(1, 5, 64)
    with torch.no_grad():
        layer(X, H, text_mask=torch.ones(1, 5), force_gate=torch.zeros(1, 8, 1))
        dS = layer.last_S_write - layer.last_S
        scatter = layer.deslice.write_delta(dS, layer.last_w)
    assert dS.abs().max().item() < 1e-6
    assert scatter.abs().max().item() < 1e-6


def test_recognition_default_is_write_A_ungated():
    layer = _layer()
    assert layer.deslice_write == "absolute"
    assert layer.gate_h_local is False
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=1, n_heads=4,
    )
    assert stack.deslice_write == "absolute"
    assert stack.gate_h_local is False
    assert stack.layers[0].deslice_write == "absolute"


def test_baseline_still_moves_field():
    """Ungated residual cell must keep the existing x_delta > 0 contract."""
    torch.manual_seed(3)
    layer = _layer()
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 10, 64)
    X2, H2, tr = layer(X, H, text_mask=torch.ones(2, 10))
    assert X2.shape == X.shape and H2.shape == H.shape
    assert tr.x_delta > 0 and tr.h_delta > 0


def test_pi_x_zero_freezes_stack_to_stem():
    torch.manual_seed(4)
    stack = NativeMoTStack(
        d_llm=32, res=8, d_x=32, d=32, n_slices=4, n_layers=2, n_heads=4,
    )
    stack.eval()
    img = torch.rand(2, 3, 8, 8)
    emb = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6)
    with torch.no_grad():
        stem = stack.encode_X(img)
        X0, H0, _, _ = stack.forward_native(img, emb, mask, pi_x=0.0)
        X1, H1, _, _ = stack.forward_native(img, emb, mask, pi_x=1.0)
    assert torch.allclose(X0, stem, atol=1e-5), float((X0 - stem).abs().max())
    assert torch.allclose(X0, stack._last_X_stem, atol=1e-5)
    assert (X1 - stem).abs().mean().item() > 1e-4
    # language residual is still live on a read-only port
    h_passthru = stack.text_out(stack.text_in(emb)) + emb
    assert (H0 - h_passthru).abs().mean().item() > 1e-5


def test_omni_need_pix_only_selects_readout():
    from fine_grain.omni_model import DualStreamOmni

    torch.manual_seed(5)
    model = DualStreamOmni(
        d_model=32, n_slices=4, n_layers=2, res=8, n_heads=4,
        surprise_mode="baseline", s_update="raw", use_stiefel=False, deslice_topk=0,
    )
    model.eval()
    img = torch.rand(2, 3, 8, 8)
    prompts = ["What is this", "What is this"]
    with torch.no_grad():
        stem = model.mot_stack.encode_X(img)
        out_text_only = model(img, prompts, need_pix=[False, False])
        out_pix = model(img, prompts, need_pix=[True, True])
        out_clamped = model(img, prompts, need_pix=[False, False], pi_x=0.0)
    assert (out_text_only["X"] - stem).abs().mean().item() > 1e-4
    assert torch.allclose(out_text_only["X"], out_pix["X"], atol=1e-5)
    assert torch.allclose(out_clamped["X"], stem, atol=1e-5)


def test_global_gate_lse_not_mean_on_1px():
    U = torch.zeros(2, 32, 1)
    U[:, 0] = 4.0
    g = global_gate_from_surprise(U, beta=1.0, kind="lse")
    g_mean = 1.0 - torch.exp(-U.mean())
    assert g.shape == (2, 1, 1)
    assert float(g.min()) > float(g_mean) + 0.3
    g_top = global_gate_from_surprise(U, beta=1.0, kind="topk", topk=1)
    assert float(g_top.min()) > 0.95


def test_workspace_write_is_not_time_increment():
    """C uses Read(W), not S_read=Read(E+W). On empty W they differ when E ≠ 0."""
    torch.manual_seed(8)
    layer = _layer(surprise_mode="baseline", local_kind="none")
    layer.deslice_write = "workspace"
    layer.gate_h_local = False
    layer.eval()
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 6, 64)
    W = torch.zeros_like(X)
    with torch.no_grad():
        S_e, _ = layer.read(X)
        S_w, _ = layer.read(W)
        X_c, _, _ = layer(X, H, text_mask=torch.ones(2, 6), W=W)
        layer.deslice_write = "increment"
        X_b, _, _ = layer(X, H, text_mask=torch.ones(2, 6))
    # empty workspace read is not the evidence read
    assert (S_e - S_w).abs().mean().item() > 1e-3
    # so C and P0 (time increment) must move X differently
    assert (X_c - X_b).abs().mean().item() > 1e-4


def test_absolute_write_moves_x_when_s_frozen():
    """Absolute Deslice(S) is a broadcast: g=0 still changes X, Read stays near S."""
    torch.manual_seed(6)
    layer = _layer(surprise_mode="constant", local_kind="none")
    layer.surprise_gate.constant_val = 0.0
    layer.deslice_write = "absolute"
    layer.gate_h_local = False
    layer.eval()
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 6, 64)
    with torch.no_grad():
        S0, _ = layer.read(X)
        X2, H2, _ = layer(X, H, text_mask=torch.ones(2, 6))
        S1, _ = layer.read(X2)
    # field is allowed to move (working-memory refresh)
    assert (X2 - X).abs().mean().item() > 1e-4
    # increment path still identity
    layer.deslice_write = "increment"
    layer.gate_h_local = True
    with torch.no_grad():
        X3, H3, _ = layer(X, H, text_mask=torch.ones(2, 6))
    assert torch.allclose(X3, X, atol=1e-5)
    assert torch.allclose(H3, H, atol=1e-5)
    _ = (S0, S1, H2)  # keep reads for the first branch


def test_global_gate_baseline_open_zero_closed():
    z = torch.zeros(3, 16, 1)
    ones = torch.ones(3, 16, 1)
    g_open = global_gate_from_surprise(z, slice_gate=ones, kind="lse")
    g_shut = global_gate_from_surprise(z, slice_gate=z, kind="lse")
    assert torch.allclose(g_open, torch.ones_like(g_open), atol=1e-5)
    assert torch.allclose(g_shut, torch.zeros_like(g_shut), atol=1e-5)


if __name__ == "__main__":
    test_forced_zero_gate_three_state_identity()
    test_constant_zero_gate_is_identity()
    test_increment_write_zero_when_s_unchanged()
    test_baseline_still_moves_field()
    test_pi_x_zero_freezes_stack_to_stem()
    test_omni_need_pix_only_selects_readout()
    test_global_gate_lse_not_mean_on_1px()
    test_global_gate_baseline_open_zero_closed()
    print("ALL P0 IDENTITY TESTS PASSED")
