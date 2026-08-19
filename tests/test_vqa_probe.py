"""Tests for synthetic VQA data + exact-match probe helpers."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.vlm_data import (  # noqa: E402
    exact_match,
    format_vqa,
    make_vqa_batch,
    normalize_answer,
)


def test_exact_match_basic():
    assert exact_match("red", "red")
    assert exact_match("Red.", "red")
    assert exact_match("  BLUE\n", "blue")
    assert exact_match("7 corners", "7")
    assert not exact_match("green", "red")
    assert not exact_match("", "red")


def test_make_vqa_batch_no_angles_has_answers():
    rng = np.random.default_rng(0)
    b = make_vqa_batch(rng, 8, res=32, mix=("color", "kinks"))
    assert b["image"].shape == (8, 3, 32, 32)
    assert len(b["answer"]) == 8
    assert all(a for a in b["answer"])
    assert all(p in ("color", "kinks") for p in b["probe"])
    assert "angles" not in b["probe"]
    for text, ans in zip(b["text"], b["answer"]):
        assert text.endswith(f"Answer: {ans}")
        assert text.startswith("Question:")


def test_train_val_seeds_differ():
    """Disjoint seeds should not produce identical answer sequences."""
    a = make_vqa_batch(np.random.default_rng(0), 32, res=32)["answer"]
    b = make_vqa_batch(np.random.default_rng(90_001), 32, res=32)["answer"]
    # extremely unlikely to be fully identical
    assert a != b


if __name__ == "__main__":
    test_exact_match_basic()
    print("ok exact")
    test_make_vqa_batch_no_angles_has_answers()
    print("ok batch")
    test_train_val_seeds_differ()
    print("ok seeds")
    print("ALL VQA PROBE TESTS PASSED")
