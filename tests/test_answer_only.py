"""Tests for answer-only labels (no LLM required)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.vlm_data import (  # noqa: E402
    answer_candidates,
    answer_only_labels,
    format_prompt,
    format_vqa,
    make_vqa_batch,
)


class _FakeTok:
    """Minimal whitespace tokenizer for unit tests."""

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=48,
                 padding=False, return_tensors=None, **kw):
        if isinstance(text, list):
            batch_ids = []
            for t in text:
                ids = self._enc(t, max_length)
                batch_ids.append(ids)
            L = max(len(x) for x in batch_ids)
            import torch
            ids = torch.zeros(len(batch_ids), L, dtype=torch.long)
            mask = torch.zeros(len(batch_ids), L, dtype=torch.long)
            for i, row in enumerate(batch_ids):
                ids[i, : len(row)] = torch.tensor(row)
                mask[i, : len(row)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        ids = self._enc(text, max_length)
        if return_tensors == "pt":
            import torch
            t = torch.tensor([ids])
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}
        return {"input_ids": ids}

    def _enc(self, text, max_length):
        # map tokens to stable ints
        toks = text.strip().split()
        ids = [abs(hash(t)) % 10000 + 1 for t in toks][:max_length]
        return ids or [1]


def test_answer_only_masks_prompt_keeps_answer():
    tok = _FakeTok()
    q = "What color is the small square?"
    ans = "red"
    prompt = format_prompt(q)
    text = format_vqa(q, ans)
    ids, mask, lab = answer_only_labels(tok, [prompt], [text], max_length=48)
    assert ids.shape == lab.shape
    # some supervised tokens
    assert (lab[0] != -100).any()
    # some masked tokens (prompt)
    assert (lab[0] == -100).any()
    # supervised positions equal original ids
    sup = lab[0] != -100
    assert (lab[0, sup] == ids[0, sup]).all()


def test_answer_candidates_closed():
    assert set(answer_candidates("color")) == {"red", "green", "blue", "yellow"}
    assert set(answer_candidates("kinks")) == {"5", "6", "7", "8"}


def test_batch_answer_only_frac():
    tok = _FakeTok()
    data = make_vqa_batch(np.random.default_rng(0), 8, res=32)
    ids, mask, lab = answer_only_labels(tok, data["prompt"], data["text"], 48)
    frac = float((lab != -100).sum() / mask.sum())
    assert 0.0 < frac < 0.6  # answers short vs full Q+A


if __name__ == "__main__":
    test_answer_only_masks_prompt_keeps_answer()
    print("ok labels")
    test_answer_candidates_closed()
    print("ok cands")
    test_batch_answer_only_frac()
    print("ok frac")
    print("ALL ANSWER-ONLY TESTS PASSED")
