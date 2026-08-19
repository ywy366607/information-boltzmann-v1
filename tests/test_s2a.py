"""S2a: answer-IG teacher + gaze head; eval can skip 2nd look."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.s2a import AnswerGazeHead, kl_cat
from scripts.run_v0_surprise_eval import DualStreamVQAModel


def test_kl_cat_zero_on_self():
    p = torch.softmax(torch.randn(4, 6), dim=-1)
    assert float(kl_cat(p, p).max()) < 1e-5


def test_s2a_train_loss_hits_gaze_head():
    m = DualStreamVQAModel(
        d_model=32, n_slices=8, n_layers=2, res=8, n_heads=4,
        surprise_mode="v1_bayes", saccade=True, saccade_inner=2, s2a=True,
    )
    imgs = torch.rand(3, 3, 8, 8)
    prompts = ["What color is the small square ?"] * 3
    targets = torch.zeros(3, dtype=torch.long)
    out = m(imgs, prompts)
    assert len(out["logits_k"]) == 4
    assert len(out["s2a_g_logits"]) == 2
    assert len(out["s2a_ig"]) == 2
    loss = m.task_loss(out, targets)
    loss.backward()
    assert m.gaze_head.net[-1].weight.grad is not None
    assert float(m.gaze_head.net[-1].weight.grad.abs().sum()) > 0


def test_s2a_eval_closed_gate_one_look_per_layer():
    m = DualStreamVQAModel(
        d_model=32, n_slices=8, n_layers=2, res=8, n_heads=4,
        surprise_mode="v1_bayes", saccade=True, saccade_inner=2, s2a=True,
    )
    nn.init.constant_(m.gaze_head.net[-1].bias, -20.0)
    m.eval()
    imgs = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        out = m(imgs, ["What color is the small square ?"] * 2)
    assert len(out["traces"]) == 2
    assert float(out["looks"].mean()) == 2.0


def test_s2a_eval_open_gate_two_looks():
    m = DualStreamVQAModel(
        d_model=32, n_slices=8, n_layers=2, res=8, n_heads=4,
        surprise_mode="v1_bayes", saccade=True, saccade_inner=2, s2a=True,
    )
    nn.init.constant_(m.gaze_head.net[-1].bias, 20.0)
    m.eval()
    imgs = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        out = m(imgs, ["What color is the small square ?"] * 2)
    assert len(out["traces"]) == 4
    assert float(out["looks"].mean()) == 4.0
