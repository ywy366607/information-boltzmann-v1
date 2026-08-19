"""Unit tests for non-generative linear probe on real synthetic batches."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fine_grain.frontends import build_frontend  # noqa: E402
from fine_grain.linear_probe import (  # noqa: E402
    N_COLOR,
    N_KINKS,
    VisionLinearProbe,
    eval_linear_probe,
    labels_from_batch,
    probe_loss,
    train_linear_probe,
)
from fine_grain.vlm_data import make_vqa_batch  # noqa: E402


def test_labels_from_batch_color_kinks_no_angles():
    rng = np.random.default_rng(1)
    data = make_vqa_batch(rng, 16, res=32, mix=("color", "kinks"))
    lab = labels_from_batch(data)
    assert lab["color_y"].shape[0] == 16
    assert lab["kinks_y"].shape[0] == 16
    assert lab["is_color"].any() or lab["is_kinks"].any()
    assert "angles" not in data["probe"]
    # labels in range
    for i, kind in enumerate(data["probe"]):
        if kind == "color":
            assert 0 <= int(lab["color_y"][i]) < N_COLOR
            assert lab["is_color"][i]
            assert not lab["is_kinks"][i]
        elif kind == "kinks":
            assert 0 <= int(lab["kinks_y"][i]) < N_KINKS
            assert lab["is_kinks"][i]
            assert not lab["is_color"][i]
        else:
            raise AssertionError(f"unexpected probe kind {kind}")


def test_vision_linear_probe_forward_and_loss():
    torch.manual_seed(0)
    fe = build_frontend("A", d_llm=32, res=32, T=16, patch=4, dim=32, depth=1)
    model = VisionLinearProbe(fe, d_feat=32)
    rng = np.random.default_rng(2)
    data = make_vqa_batch(rng, 4, res=32)
    out = model(data["image"])
    assert out["color_logits"].shape == (4, N_COLOR)
    assert out["kinks_logits"].shape == (4, N_KINKS)
    assert out["pooled"].shape == (4, 32)
    lab = labels_from_batch(data)
    loss = probe_loss(out, lab)
    assert torch.isfinite(loss)
    loss.backward()  # real path through frontend + heads


def test_eval_keys_and_short_train():
    """Short real train on A; metrics keys present; no angles in mix."""
    row = train_linear_probe(
        "A", T=16, d_feat=32, res=32, patch=4, dim=32, depth=1,
        steps=5, batch=4, probe_n=16, seed=0, val_seed=90_001,
        device=torch.device("cpu"), log_every=5, collect_failures=2,
    )
    assert row["status"] == "ok"
    assert row["protocol"] == "linear_probe"
    for k in ("acc_color", "acc_kinks", "acc_overall", "n_color", "n_kinks"):
        assert k in row["probe"], k
    assert 0.0 <= row["acc_color"] <= 1.0
    assert 0.0 <= row["acc_kinks"] <= 1.0
    # eval path only uses color/kinks
    assert row["probe"]["n_color"] + row["probe"]["n_kinks"] == row["probe"]["n_overall"]


def test_slice_frontend_probe_path():
    """B frontend also trains without LLM."""
    row = train_linear_probe(
        "B", T=8, d_feat=32, res=32, patch=4, dim=32, depth=1,
        steps=3, batch=2, probe_n=8, seed=1,
        device=torch.device("cpu"), log_every=3, collect_failures=0,
    )
    assert row["status"] == "ok"
    assert row["kind"] == "B"
    assert "acc_kinks" in row["probe"]


if __name__ == "__main__":
    test_labels_from_batch_color_kinks_no_angles()
    print("ok labels")
    test_vision_linear_probe_forward_and_loss()
    print("ok forward")
    test_eval_keys_and_short_train()
    print("ok train")
    test_slice_frontend_probe_path()
    print("ok B")
    print("ALL LINEAR PROBE TESTS PASSED")
