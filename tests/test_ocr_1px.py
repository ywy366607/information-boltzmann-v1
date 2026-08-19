"""Tests for 1px OCR renderer + VQA batch."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.ocr_1px import OCR_DIGITS, make_ocr_1px, render_digit_mask  # noqa: E402
from fine_grain.vlm_data import answer_candidates, make_ocr_vqa_batch  # noqa: E402


def test_render_all_digits_have_ink():
    for d in OCR_DIGITS:
        m = render_digit_mask(d, R=32, box=16, y0=8, x0=8)
        assert m.sum() >= 5, (d, int(m.sum()))


def test_make_ocr_batch_shape_and_labels():
    rng = np.random.default_rng(0)
    img, lab, msk = make_ocr_1px(rng, n=16, res=32)
    assert img.shape == (16, 3, 32, 32)
    assert lab.shape == (16,)
    assert lab.min() >= 0 and lab.max() <= 9
    # ink present
    assert msk.reshape(16, -1).sum(dim=1).min() >= 3


def test_ocr_vqa_batch():
    b = make_ocr_vqa_batch(np.random.default_rng(1), 8, res=32)
    assert b["image"].shape[0] == 8
    assert all(p == "ocr" for p in b["probe"])
    assert all(a in OCR_DIGITS for a in b["answer"])
    assert all(t.endswith(f"Answer: {a}") for t, a in zip(b["text"], b["answer"]))
    assert answer_candidates("ocr") == OCR_DIGITS


def test_ocr_string_and_cer():
    from fine_grain.ocr_1px import make_ocr_string_1px
    from fine_grain.vlm_data import char_error_rate, exact_string_match, make_ocr_string_vqa_batch

    img, texts, msk = make_ocr_string_1px(np.random.default_rng(0), batch=4, res=32, min_len=2, max_len=3)
    assert img.shape[0] == 4
    assert all(2 <= len(t) <= 3 and t.isdigit() for t in texts)
    assert msk.reshape(4, -1).sum(1).min() >= 6
    b = make_ocr_string_vqa_batch(np.random.default_rng(1), 3, res=32, min_len=2, max_len=4)
    assert all(p == "ocr_str" for p in b["probe"])
    assert exact_string_match(" 12\n", "12")
    assert char_error_rate("13", "12") == 0.5
    assert char_error_rate("12", "12") == 0.0


if __name__ == "__main__":
    test_render_all_digits_have_ink()
    print("ok render")
    test_make_ocr_batch_shape_and_labels()
    print("ok make")
    test_ocr_vqa_batch()
    print("ok vqa")
    test_ocr_string_and_cer()
    print("ok string")
    print("ALL OCR 1PX TESTS PASSED")

