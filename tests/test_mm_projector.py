"""Tests for LLaVA-1.5 style MM projector on real frontends."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.frontends import build_frontend  # noqa: E402
from fine_grain.mm_projector import MMProjector  # noqa: E402


def test_mlp_projector_shape_and_gelu():
    p = MMProjector(32, 64, kind="mlp")
    assert isinstance(p.net, nn.Sequential)
    assert len(p.net) == 3
    assert isinstance(p.net[1], nn.GELU)
    x = torch.randn(2, 8, 32)
    y = p(x)
    assert y.shape == (2, 8, 64)
    y.sum().backward()  # real grad path


def test_linear_projector_ablation():
    p = MMProjector(16, 16, kind="linear")
    assert isinstance(p.net, nn.Linear)
    y = p(torch.randn(1, 4, 16))
    assert y.shape == (1, 4, 16)


def test_frontend_default_is_mlp_not_single_linear():
    """A/B/C last map is MMProjector(mlp), not bare nn.Linear."""
    for kind in ("A", "B", "C"):
        fe = build_frontend(kind, d_llm=48, res=32, T=16, patch=4, dim=32, depth=1)
        # A/B: fe.proj; C: nested
        if kind == "C":
            assert isinstance(fe.patch_fe.proj, MMProjector)
            assert fe.patch_fe.proj.kind == "mlp"
            assert isinstance(fe.slice_fe.proj, MMProjector)
        else:
            assert isinstance(fe.proj, MMProjector), kind
            assert fe.proj.kind == "mlp"
        x = torch.rand(1, 3, 32, 32)
        out = fe(x)
        assert out.tokens.shape[-1] == 48
        assert out.meta.get("projector") == "mlp"


def test_frontend_linear_flag():
    fe = build_frontend("A", d_llm=32, res=32, T=16, patch=4, dim=32, depth=1, projector="linear")
    assert fe.proj.kind == "linear"
    assert isinstance(fe.proj.net, nn.Linear)
    y = fe(torch.rand(2, 3, 32, 32))
    assert y.tokens.shape == (2, 16, 32)
    assert y.meta["projector"] == "linear"


if __name__ == "__main__":
    test_mlp_projector_shape_and_gelu()
    print("ok mlp")
    test_linear_projector_ablation()
    print("ok linear")
    test_frontend_default_is_mlp_not_single_linear()
    print("ok frontend mlp")
    test_frontend_linear_flag()
    print("ok frontend linear")
    print("ALL MM PROJECTOR TESTS PASSED")
