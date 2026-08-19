"""Tests: MoT joint self-attn on slices ‖ text; deslice to points."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.mot_coevolve import MoTCoEvolveFrontend, MoTJointLayer  # noqa: E402
from fine_grain.transolver3_vlm import MoTOCRBridge, build_ocr_bridge  # noqa: E402


def test_joint_attn_updates_both_x_and_text():
    layer = MoTJointLayer(dim=64, n_slices=8, n_heads=4, deslice_topk=2)
    x = torch.randn(2, 64, 64)
    text = torch.randn(2, 12, 64)
    mask = torch.ones(2, 12)
    x2, t2, slices, tr = layer(x, text, text_mask=mask)
    assert x2.shape == x.shape
    assert t2.shape == text.shape
    assert slices.shape == (2, 8, 64)
    assert tr.x_delta > 0
    assert tr.text_delta > 0
    # different text → different X' (joint)
    x3, _, _, _ = layer(x, torch.randn_like(text), text_mask=mask)
    assert (x2 - x3).abs().sum() > 1e-4


def test_sequence_is_slices_then_text():
    """Structural: G+L length sequence (verified via L in trace)."""
    layer = MoTJointLayer(dim=64, n_slices=8, n_heads=4)
    x = torch.randn(1, 36, 64)
    text = torch.randn(1, 10, 64)
    _, _, _, tr = layer(x, text)
    assert tr.G == 8 and tr.L == 10


def test_frontend_mot_shapes():
    fe = MoTCoEvolveFrontend(d_llm=32, res=16, T=8, n_layers=2, projector="linear")
    img = torch.rand(2, 3, 16, 16)
    emb = torch.randn(2, 14, 32)
    mask = torch.ones(2, 14)
    x, text_out, slices, traces = fe.forward_mot(img, emb, mask)
    assert x.shape[1] == 16 * 16
    assert text_out.shape == emb.shape
    assert slices.shape[1] == 8
    assert len(traces) == 2
    out = fe(img, text_emb=emb, text_mask=mask)
    assert out.meta["mot"] is True
    assert out.meta["joint_self_attn"] is True
    assert out.tokens.shape == (2, 8, 32)


def test_mot_bridge_ce_grads():
    class TinyLM(nn.Module):
        def __init__(self, d=32, V=64):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.block = nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=64, batch_first=True,
            )
            self.lm_head = nn.Linear(d, V)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kw):
            h = self.block(inputs_embeds)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            class O:
                pass

            o = O()
            o.loss = loss if loss is not None else torch.zeros(())
            o.logits = logits
            return o

    d = 32
    fe = MoTCoEvolveFrontend(d, res=16, T=8, n_layers=2, projector="linear")
    llm = TinyLM(d=d)
    bridge = MoTOCRBridge(fe, llm)
    img = torch.rand(2, 3, 16, 16)
    ids = torch.randint(1, 60, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    lab = ids.clone()
    lab[:, :4] = -100
    loss, meta = bridge(img, ids, mask, text_labels=lab)
    assert meta["mot"] is True
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bridge.frontend.parameters() if p.requires_grad
    )


def test_build_mot_kind():
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 32)
            self.f = nn.Linear(32, 32)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kw):
            class O:
                pass
            o = O()
            o.loss = torch.tensor(0.0)
            o.logits = kw["inputs_embeds"]
            return o

    args = SimpleNamespace(
        res=16, patch=4, T=8, dim=32, depth=2, topk=2, projector="linear",
        n_layers=2, coevolve_rounds=1, pass1_w=0.0, use_h_token=True,
    )
    b = build_ocr_bridge("B_mot", 32, TinyLM(), args)
    assert isinstance(b, MoTOCRBridge)


if __name__ == "__main__":
    test_joint_attn_updates_both_x_and_text()
    print("ok joint both")
    test_sequence_is_slices_then_text()
    print("ok seq")
    test_frontend_mot_shapes()
    print("ok frontend")
    test_mot_bridge_ce_grads()
    print("ok bridge")
    test_build_mot_kind()
    print("ok build")
    print("ALL MOT TESTS PASSED")
