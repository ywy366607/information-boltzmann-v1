"""Unified graph: residual write, no t2i-only forks."""
from __future__ import annotations

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
    assert kw["use_null_slice"] is False
    assert kw["pack_by_surprise"] is False
    assert kw["hard_admit"] is False
    assert kw["use_yield_read"] is True


def test_omni_unified_factory_wires_stack():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    assert m.mot_stack.deslice_write == "increment"
    assert m.vfe_coef == 0.1
    assert m.mot_stack.s_lang_topk == 0
    assert m.mot_stack.use_null_slice is False
    assert m.mot_stack.hard_admit is False
    assert m.mot_stack.use_yield_read is True
    for layer in m.mot_stack.layers:
        assert layer.deslice_write == "increment"
        assert layer.s_lang_topk == 0
        assert layer.read.use_null_slice is False
        assert layer.pack_by_surprise is False
        assert layer.hard_admit is False
        assert layer.read.use_yield_read is True
        assert layer.deslice.preserve_mass is True


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
