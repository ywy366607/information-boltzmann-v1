"""Position-invariant t2i scores and black-canvas samples."""
from __future__ import annotations

import numpy as np
import torch

from fine_grain.gen_metrics import (
    digit_shift_scores,
    free_color_acc,
    gen_free_scores,
    parse_draw_prompt,
)
from fine_grain.omni_tasks import one_sample
from fine_grain.ocr_1px import render_digit_mask
from fine_grain.tasks import SIGNAL


def test_parse_draw_prompt():
    d, c = parse_draw_prompt("Draw digit 6 with a thin yellow stroke")
    assert d == "6" and c == "yellow"
    d, c = parse_draw_prompt("Draw digit 3 with a thin red stroke blank image")
    assert d == "3" and c == "red"


def test_free_metrics_on_perfect_digit():
    res = 32
    m = render_digit_mask("5", res, 16, 8, 8)
    img = torch.zeros(1, 3, res, res)
    ink = torch.tensor(SIGNAL[0], dtype=torch.float32).view(1, 3, 1, 1)  # red
    img = img + ink * torch.from_numpy(m.astype("float32")).view(1, 1, res, res)
    rec = gen_free_scores(img, "5", "red")
    assert rec["color_acc"] > 0.9
    assert rec["digit_top1"] == 1.0
    assert rec["digit_iou"] > 0.5
    wrong = digit_shift_scores(img, "1", "red")
    assert wrong["digit_iou"] < rec["digit_iou"]


def test_color_acc_zero_on_black():
    img = torch.zeros(1, 3, 16, 16)
    assert free_color_acc(img, "red") == 0.0


def test_t2i_black_canvas_is_black_off_stroke():
    rng = np.random.default_rng(0)
    s = one_sample(rng, 32, "t2i", t2i_canvas="black")
    paper, tgt, st = s["image"], s["target_rgb"], s["stroke"]
    bg = (1.0 - st).unsqueeze(1)
    assert float((paper * bg).abs().sum()) == 0.0
    assert float((tgt * bg).abs().sum()) == 0.0
    assert "blank image" in s["prompt"]
    assert s["digit"] in "0123456789"
    assert s["color"] in ("red", "green", "blue", "yellow")


def test_t2i_paper_prompt_unchanged():
    rng = np.random.default_rng(1)
    s = one_sample(rng, 32, "t2i", t2i_canvas="paper")
    assert s["prompt"].startswith("Draw digit")
    assert "blank" not in s["prompt"]
    bg = (1.0 - s["stroke"]).unsqueeze(1)
    assert torch.allclose(s["image"] * bg, s["target_rgb"] * bg, atol=1e-6)
