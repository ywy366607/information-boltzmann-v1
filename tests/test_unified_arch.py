"""Unified graph: residual write, no t2i-only forks."""
from __future__ import annotations

import torch

from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_KNOBS, unified_kwargs


def test_unified_kwargs_are_residual_not_replace():
    kw = unified_kwargs()
    assert kw["deslice_write"] == "increment"
    assert kw["vfe_coef"] == 0.1
    assert kw["s_lang_topk"] == 0
    assert kw["s_kalman_update"] is False
    assert kw["gate_on"] == "u"
    assert kw["prior_write"] == 0.0
    assert kw["deslice_topk"] == 2
    assert kw["use_null_slice"] is True
    assert kw["pack_by_surprise"] is False
    assert kw["hard_admit"] is False
    assert kw["use_yield_read"] is False
    assert kw["use_ticket_read"] is False
    assert kw["use_write_yield"] is False
    assert kw["write_alpha"] == 1.0
    assert kw["use_residual_read"] is True


def test_omni_unified_factory_wires_stack():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    assert m.mot_stack.deslice_write == "increment"
    assert m.vfe_coef == 0.1
    assert m.mot_stack.s_lang_topk == 0
    assert m.mot_stack.use_null_slice is True
    assert m.mot_stack.hard_admit is False
    assert m.mot_stack.use_yield_read is False
    assert m.mot_stack.use_ticket_read is False
    assert m.mot_stack.use_write_yield is False
    assert m.mot_stack.use_residual_read is True
    for layer in m.mot_stack.layers:
        assert layer.deslice_write == "increment"
        assert layer.s_lang_topk == 0
        assert layer.read.use_null_slice is True
        assert layer.pack_by_surprise is False
        assert layer.hard_admit is False
        assert layer.read.use_yield_read is False
        assert layer.use_ticket_read is False
        assert layer.use_write_yield is False
        assert layer.use_residual_read is True
        assert layer.lang_s0 is not None
        assert layer.deslice.preserve_mass is True


def test_toy_language_default_has_no_lm():
    m = DualStreamOmni(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    assert m.lm is None
    h, mask = m.encode_text(["Draw digit 3 with a thin red stroke"], torch.device("cpu"))
    assert h.shape[-1] == 32
    assert mask.any()


def test_old_omni_default_not_silently_unified():
    # Constructing DualStreamOmni() without unified() must not force increment.
    assert UNIFIED_KNOBS["deslice_write"] == "increment"
    m = DualStreamOmni(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    # DualStreamVQA default is still absolute unless caller passes increment.
    assert m.mot_stack.deslice_write == "absolute"
    assert m.mot_stack.use_null_slice is False
    assert m.mot_stack.pack_by_surprise is False
    assert m.mot_stack.hard_admit is False
    assert m.mot_stack.use_yield_read is False
    assert m.mot_stack.use_ticket_read is False
    assert m.mot_stack.use_write_yield is False
    assert m.mot_stack.use_residual_read is False


def test_text_readout_port_still_evolves_latent_visual_field():
    """need_pix selects output loss; it must not turn joint evolution off."""
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
    ).eval()
    img = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        out = m(img, ["What is this"], need_pix=[False])
    stem = m.mot_stack._last_X_stem
    assert stem is not None
    assert (out["X"] - stem).abs().mean() > 1e-6


def test_explicit_zero_pi_x_remains_a_causal_clamp():
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
    ).eval()
    img = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        out = m(img, ["What is this"], pi_x=0.0, need_pix=[False])
    assert torch.allclose(out["X"], m.mot_stack._last_X_stem)
