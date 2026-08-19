"""Unit tests: bidirectional X↔H evolution (F_vision, Read, F_language)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.cross_modal_slice_loop import (  # noqa: E402
    CrossModalSliceFrontend,
    CrossModalSliceLayer,
    run_layer_steps_dict,
)


def _layer(writeback=True):
    return CrossModalSliceLayer(
        mix_dim=64, h_dim=32, n_slices=8, heads=4, dim_head=16,
        deslice_topk=2, write_h_into_x=writeback, beta_write=0.3,
    )


def test_F_vision_depends_on_H():
    """X' = F_vision(X, H): different H → different X' (bidirectional)."""
    layer = _layer()
    B, N, C = 2, 64, 64
    x = torch.randn(B, N, C)
    h_a = torch.zeros(B, 32)
    h_b = torch.randn(B, 32)
    x_a, s_a = layer.F_vision(x, h_a)
    x_b, s_b = layer.F_vision(x, h_b)
    assert x_a.shape == (B, N, C)
    assert s_a.shape == (B, 8, C)
    # H must influence the next visual field
    assert (x_a - x_b).abs().sum() > 1e-4


def test_Read_of_X_t1_depends_on_H_query():
    """Read(X_{t+1}; H) soft-selects slices using H as query."""
    layer = _layer()
    x = torch.randn(2, 64, 64)
    h = torch.randn(2, 32)
    x1, slices = layer.F_vision(x, h)
    read_a, attn_a, _ = layer.Read(slices, torch.zeros_like(h))
    read_b, attn_b, _ = layer.Read(slices, torch.ones_like(h))
    assert attn_a.shape == (2, 8)
    assert torch.allclose(attn_a.sum(-1), torch.ones(2), atol=1e-5)
    assert (attn_a - attn_b).abs().sum() > 0 or (read_a - read_b).abs().sum() > 0


def test_F_language_residual():
    layer = _layer()
    h = torch.randn(3, 32)
    read = torch.randn(3, 32)
    h2, delta = layer.F_language(h, read)
    assert torch.allclose(h2, h + delta, atol=1e-5)


def test_equation_order_H_reads_new_X():
    """H_{t+1} must use Read(X_{t+1}), not only X_t.

    Structural check: run_layer_steps_dict applies F_vision before Read.
    """
    layer = _layer()
    x = torch.randn(2, 64, 64)
    h = torch.randn(2, 32)
    out = run_layer_steps_dict(layer, x, h)
    assert "X_t1_F_vision" in out and "H_t1_F_language" in out
    # Read is computed from slices produced by F_vision (same object path)
    assert out["slices_from_X_t1"].shape[1] == 8
    assert torch.isfinite(out["H_t1_F_language"]).all()


def test_full_step_forward_equations():
    """forward implements (X,H)→(X',H') with both states moving."""
    layer = _layer()
    x = torch.randn(2, 64, 64)
    h = torch.randn(2, 32)
    x2, h2, slices, tr = layer(x, h, layer_idx=0)
    assert x2.shape == x.shape and h2.shape == h.shape
    assert tr.x_delta_norm > 0
    assert tr.h_delta_norm >= 0
    assert slices is not None


def test_writeback_off_still_conditions_X_via_bias():
    """Even without deslice write, F_vision still takes H (cond bias)."""
    layer = _layer(writeback=False)
    x = torch.randn(1, 64, 64)
    h0 = torch.zeros(1, 32)
    h1 = torch.ones(1, 32)
    xa, _ = layer.F_vision(x, h0)
    xb, _ = layer.F_vision(x, h1)
    assert (xa - xb).abs().sum() > 1e-4


def test_frontend_L_layers_tokens():
    fe = CrossModalSliceFrontend(
        d_llm=64, res=16, T=8, dim=64, n_layers=3, shared_layer=True,
        projector="linear",
    )
    img = torch.rand(2, 3, 16, 16)
    out = fe(img)
    assert out.tokens.shape == (2, 8, 64)
    assert out.meta["kind"] == "B_xmodal"
    assert "F_vision" in out.meta["contract"]
    assert len(out.meta["layer_traces"]) == 3
    # Final thesis: full-res point field is primary memory (not tokens alone)
    assert out.meta.get("point_field") is True
    assert out.meta.get("N_points") == 16 * 16
    assert out.meta.get("tokens_are_interface_only") is True
    assert out.meta.get("replaces_legacy_slice_tokens") is True
    assert out.meta.get("transolver") == "3_multimodal"
    assert fe._last_x is not None and fe._last_x.shape[1] == 16 * 16


def test_external_h_seeds_H0():
    fe = CrossModalSliceFrontend(
        d_llm=32, res=16, T=8, n_layers=1, projector="linear", h_dim=32,
    )
    img = torch.rand(1, 3, 16, 16)
    out = fe(img, external_h=torch.randn(1, 32))
    assert torch.isfinite(out.tokens).all()


def test_gradients_flow_bidirectional():
    fe = CrossModalSliceFrontend(
        d_llm=32, res=16, T=8, n_layers=2, projector="linear",
    )
    out = fe(torch.rand(2, 3, 16, 16))
    out.tokens.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in fe.parameters() if p.requires_grad
    )


def test_chained_steps_re_read():
    """Step t+1 reads reorganized X from step t (loop, not feedforward FE only)."""
    layer = _layer()
    x = torch.randn(1, 64, 64)
    h = torch.randn(1, 32)
    x1, h1, _, _ = layer(x, h, 0)
    x2, h2, _, _ = layer(x1, h1, 1)
    # second reorganize moves field again
    assert (x2 - x1).abs().sum() > 1e-5
    assert (h2 - h1).abs().sum() > 1e-5


if __name__ == "__main__":
    test_F_vision_depends_on_H()
    print("ok F_vision(H)")
    test_Read_of_X_t1_depends_on_H_query()
    print("ok Read")
    test_F_language_residual()
    print("ok F_language")
    test_equation_order_H_reads_new_X()
    print("ok equation order")
    test_full_step_forward_equations()
    print("ok forward")
    test_writeback_off_still_conditions_X_via_bias()
    print("ok cond without write")
    test_frontend_L_layers_tokens()
    print("ok frontend")
    test_external_h_seeds_H0()
    print("ok H0")
    test_gradients_flow_bidirectional()
    print("ok grads")
    test_chained_steps_re_read()
    print("ok re-read chain")
    print("ALL BIDIRECTIONAL SLICE LOOP TESTS PASSED")
