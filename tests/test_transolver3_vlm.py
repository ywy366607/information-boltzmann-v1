"""Tests for Transolver3 native multimodal OCR bridge (co-evolve path)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.cross_modal_slice_loop import CrossModalSliceFrontend  # noqa: E402
from fine_grain.transolver3_vlm import (  # noqa: E402
    Transolver3OCRBridge,
    build_ocr_bridge,
)


class _TinyLM(nn.Module):
    """Minimal causal LM stub with embeddings + linear head."""

    def __init__(self, d=32, V=64):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.block = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=64, batch_first=True,
        )
        self.lm_head = nn.Linear(d, V)
        self.config = SimpleNamespace(hidden_size=d, vocab_size=V)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self, inputs_embeds=None, attention_mask=None, labels=None,
        output_hidden_states=False, use_cache=False, input_ids=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        h = self.block(inputs_embeds)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # shift CE
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        hs = (h,) if output_hidden_states else None

        class O:
            pass

        o = O()
        o.loss = loss if loss is not None else torch.tensor(0.0)
        o.logits = logits
        o.hidden_states = (None, h) if output_hidden_states else None
        return o


def test_evolve_keeps_point_field_and_moves_H():
    d = 32
    fe = CrossModalSliceFrontend(
        d, res=16, T=8, n_layers=2, projector="linear", h_dim=d,
    )
    llm = _TinyLM(d=d)
    bridge = Transolver3OCRBridge(fe, llm, coevolve_rounds=1, use_h_token=True)
    img = torch.rand(2, 3, 16, 16)
    h0 = torch.zeros(2, d)
    tok, h, out = bridge.evolve_field(img, h_seed=h0)
    assert tok.shape == (2, 8, d)
    assert h.shape == (2, d)
    assert out.meta["point_field"] is True
    assert out.meta["N_points"] == 16 * 16
    # different seed → different H (almost surely)
    tok_b, h_b, _ = bridge.evolve_field(img, h_seed=torch.ones(2, d))
    assert (h - h_b).abs().sum() > 0 or (tok - tok_b).abs().sum() > 0


def test_forward_ce_and_grads():
    d = 32
    fe = CrossModalSliceFrontend(
        d, res=16, T=8, n_layers=2, projector="linear", h_dim=d,
    )
    llm = _TinyLM(d=d)
    bridge = Transolver3OCRBridge(fe, llm, coevolve_rounds=1, use_h_token=True)
    img = torch.rand(2, 3, 16, 16)
    ids = torch.randint(1, 60, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    lab = ids.clone()
    lab[:, :4] = -100  # prompt
    loss, meta = bridge(img, ids, mask, text_labels=lab)
    assert torch.isfinite(loss)
    assert meta.get("transolver") == "3_multimodal" or meta.get("kind") == "B_xmodal"
    loss.backward()
    grads = [
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bridge.frontend.parameters() if p.requires_grad
    ]
    assert any(grads)
    # LLM frozen
    for p in bridge.llm.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0 or not p.requires_grad


def test_build_rejects_legacy_B_tokens():
    args = SimpleNamespace(
        res=16, patch=4, T=8, dim=32, depth=2, topk=2, projector="linear",
        n_layers=2, coevolve_rounds=1, pass1_w=0.0, use_h_token=True,
    )
    llm = _TinyLM(d=32)
    b = build_ocr_bridge("B_xmodal", 32, llm, args)
    assert isinstance(b, Transolver3OCRBridge)
    try:
        build_ocr_bridge("B", 32, llm, args)
        raise AssertionError("legacy B should raise")
    except ValueError as e:
        assert "slice-token" in str(e).lower() or "unknown" in str(e).lower() or "B" in str(e)


def test_coevolve_two_rounds_runs():
    d = 32
    fe = CrossModalSliceFrontend(
        d, res=16, T=8, n_layers=1, projector="linear", h_dim=d,
    )
    llm = _TinyLM(d=d)
    bridge = Transolver3OCRBridge(fe, llm, coevolve_rounds=2, pass1_ce_weight=0.1)
    img = torch.rand(1, 3, 16, 16)
    ids = torch.randint(1, 60, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.long)
    lab = ids.clone()
    lab[:, :3] = -100
    loss, meta = bridge(img, ids, mask, text_labels=lab)
    assert meta.get("two_look") is True
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_evolve_keeps_point_field_and_moves_H()
    print("ok evolve")
    test_forward_ce_and_grads()
    print("ok forward grads")
    test_build_rejects_legacy_B_tokens()
    print("ok reject legacy B")
    test_coevolve_two_rounds_runs()
    print("ok coevolve2")
    print("ALL TRANSOLVER3 VLM TESTS PASSED")
