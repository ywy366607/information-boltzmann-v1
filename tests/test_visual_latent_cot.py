"""Tests for Visual Latent CoT slice frontend."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.visual_latent_cot import VisualLatentCoTSlice  # noqa: E402


def test_forward_shape_and_contract():
    fe = VisualLatentCoTSlice(
        d_llm=64, res=32, T=16, dim=64, depth=2,
        latent_steps=2, beta=0.5, gamma=0.9, projector="mlp",
    )
    x = torch.rand(2, 3, 32, 32)
    out = fe(x)
    assert out.tokens.shape == (2, 16, 64)
    assert out.meta["kind"] == "B_vlcot"
    assert out.meta["latent_steps"] == 2
    assert len(fe._last_z_traj) == 2
    # thought norms shouldn't explode
    for z in fe._last_z_traj:
        assert torch.isfinite(z).all()
        assert z.norm(dim=-1).max() < 50


def test_nextlat_aux_and_external_h():
    fe = VisualLatentCoTSlice(d_llm=32, res=32, T=8, depth=1, latent_steps=3, projector="linear")
    x = torch.rand(1, 3, 32, 32)
    out = fe(x)
    loss = fe.nextlat_loss()
    assert torch.isfinite(loss)
    loss.backward()
    # external hidden path
    fe.zero_grad(set_to_none=True)
    h = torch.randn(1, 32)
    out2 = fe(x, external_h=h, steps=2)
    assert out2.tokens.shape[-1] == 32


def test_gamma_damps():
    fe = VisualLatentCoTSlice(d_llm=32, res=32, T=8, depth=1, latent_steps=4, gamma=0.5, projector="linear")
    out = fe(torch.rand(1, 3, 32, 32))
    norms = [z.norm().item() for z in fe._last_z_traj]
    # not a strict proof, but should stay bounded
    assert max(norms) < 20


def test_encode_frontend_dispatch():
    from fine_grain.frontends import PatchFrontend
    from fine_grain.visual_latent_cot import encode_frontend

    a = PatchFrontend(32, res=32, patch=4, dim=32, depth=1, T=16, projector="linear")
    b = VisualLatentCoTSlice(32, res=32, T=8, depth=1, latent_steps=2, projector="linear")
    img = torch.rand(1, 3, 32, 32)
    ya = encode_frontend(a, img)
    yb = encode_frontend(b, img, external_h=torch.randn(1, 32))
    assert ya.tokens.shape[-1] == 32 and yb.tokens.shape[-1] == 32


if __name__ == "__main__":
    test_forward_shape_and_contract()
    print("ok shape")
    test_nextlat_aux_and_external_h()
    print("ok nextlat")
    test_gamma_damps()
    print("ok gamma")
    test_encode_frontend_dispatch()
    print("ok dispatch")
    print("ALL VISUAL LATENT COT TESTS PASSED")

