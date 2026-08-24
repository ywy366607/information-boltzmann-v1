"""t2i must paint the digit onto the *same* input paper."""
from __future__ import annotations

import numpy as np
import torch

from fine_grain.omni_tasks import GRID_PLACES, make_omni_batch, one_sample


def test_t2i_target_shares_input_background():
    rng = np.random.default_rng(0)
    s = one_sample(rng, 32, "t2i")
    paper, tgt, st = s["image"], s["target_rgb"], s["stroke"]
    assert paper.shape == tgt.shape == (1, 3, 32, 32)
    bg = (1.0 - st).unsqueeze(1)
    # Off-stroke pixels are the input paper, not a second random canvas.
    assert torch.allclose(paper * bg, tgt * bg, atol=1e-6)
    # Stroke is actually painted.
    assert float((st > 0.5).sum()) >= 3
    ink_err = ((paper - tgt).abs() * st.unsqueeze(1)).sum()
    assert float(ink_err) > 0.0
    assert s["need_pix"] is True and s["need_text"] is False
    assert "Draw digit" in s["prompt"]


def test_t2i_gray_hint_only_paints_stroke():
    from fine_grain.omni_tasks import apply_t2i_gray_hint, make_omni_batch

    rng = np.random.default_rng(1)
    b = make_omni_batch(rng, 8, 32, mix=["t2i"])
    hinted = apply_t2i_gray_hint(b["image"], b["stroke"], b["kind"], 1.0, rng)
    st = b["stroke"].unsqueeze(1)
    bg = 1.0 - st
    assert torch.allclose(hinted * bg, b["image"] * bg, atol=1e-5)
    assert float(((hinted - b["image"]).abs() * st).sum()) > 0.0


def test_t2i_can_lock_digit_and_color():
    rng = np.random.default_rng(0)
    s = one_sample(
        rng, 32, "t2i", t2i_canvas="black", t2i_place="center",
        t2i_digit=7, t2i_color="green",
    )
    assert s["digit"] == "7"
    assert s["color"] == "green"
    assert "digit 7" in s["prompt"] and "green" in s["prompt"]


def test_t2i_center_is_fixed_mid_box():
    rng = np.random.default_rng(0)
    a = one_sample(rng, 32, "t2i", t2i_canvas="black", t2i_place="center")
    b = one_sample(rng, 32, "t2i", t2i_canvas="black", t2i_place="center")
    # Same digit may differ, but ink lives in the central 16x16.
    for s in (a, b):
        ys, xs = torch.where(s["stroke"][0] > 0.5)
        assert int(ys.min()) >= 7 and int(ys.max()) <= 24
        assert int(xs.min()) >= 7 and int(xs.max()) <= 24


def test_t2i_grid_names_the_full_resolution_address():
    rng = np.random.default_rng(3)
    corners = {
        "top_left": (0, 0),
        "top_right": (0, 2),
        "bottom_left": (2, 0),
        "bottom_right": (2, 2),
    }
    for place, (row, col) in corners.items():
        s = one_sample(
            rng, 32, "t2i", t2i_canvas="black", t2i_place=place,
            t2i_digit=7, t2i_color="green",
        )
        ys, xs = torch.where(s["stroke"][0] > 0.5)
        assert int(float(ys.float().mean()) // (32 / 3)) == row
        assert int(float(xs.float().mean()) // (32 / 3)) == col
        assert f"at {place.replace('_', ' ')}" in s["prompt"]
        assert s["placement"] == place

    batch = make_omni_batch(
        rng, len(GRID_PLACES), 16, mix=["t2i"],
        t2i_canvas="black", t2i_place="grid",
    )
    assert len(batch["placement"]) == len(GRID_PLACES)
    assert set(batch["placement"]) <= set(GRID_PLACES)


def test_spatial_prompt_vocabulary_is_explicit_and_opt_in():
    from fine_grain.omni_model import DualStreamOmni

    old = DualStreamOmni(d_model=16, n_slices=4, n_layers=1, res=8, n_heads=4)
    grid = DualStreamOmni(
        d_model=16, n_slices=4, n_layers=1, res=8, n_heads=4,
        spatial_prompt_vocab=True,
    )
    assert "top" not in old.vocab
    assert all(word in grid.vocab for word in grid.EXTRA_SPATIAL_WORDS)


def test_equal_energy_ink_unit_l2():
    from fine_grain.omni_tasks import equal_energy_ink
    from fine_grain.vlm_data import COLORS

    norms = [float(np.linalg.norm(equal_energy_ink(c))) for c in COLORS]
    assert all(abs(n - 1.0) < 1e-5 for n in norms)
    y = equal_energy_ink("yellow")
    g = equal_energy_ink("green")
    assert abs(float((y ** 2).sum()) - float((g ** 2).sum())) < 1e-6
    # Saturated SIGNAL yellow was 2×; unit yellow is not [1,1,0].
    assert float(y.max()) < 0.9


def test_rgb_energy_weights_equalize_yellow_vs_red():
    from fine_grain.omni_model import rgb_energy_weights

    bg = torch.full((1, 3, 4, 4), -1.0)
    red = bg.clone()
    red[:, 0] = 1.0
    yel = bg.clone()
    yel[:, 0] = 1.0
    yel[:, 1] = 1.0
    miss_r = ((bg - red).pow(2) * rgb_energy_weights(red, True)).mean()
    miss_y = ((bg - yel).pow(2) * rgb_energy_weights(yel, True)).mean()
    assert torch.allclose(miss_r, miss_y, rtol=1e-5)
    assert float((bg - yel).pow(2).mean()) > 1.5 * float((bg - red).pow(2).mean())


def test_balanced_observation_bce_prefers_exact_thin_figure():
    from fine_grain.omni_model import balanced_observation_bce

    target = torch.zeros(1, 3, 8, 8)
    target[:, 1, 2:6, 4] = 1.0
    exact = target.clamp(1e-4, 1.0 - 1e-4)
    flood = exact.clone()
    flood[:, 1] = 1.0 - 1e-4
    blank = torch.full_like(target, 1e-4)
    le = balanced_observation_bce(exact, target)
    assert le < balanced_observation_bce(flood, target)
    assert le < balanced_observation_bce(blank, target)


def test_fgen_deslice_write_topk_is_2():
    from fine_grain.omni_model import DualStreamOmni

    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
    )
    assert m.mot_stack.deslice_topk == 2
    assert m.mot_stack.layers[0].deslice.deslice_topk == 2


def test_thicken_stroke_grows_mask():
    from fine_grain.omni_tasks import thicken_stroke

    m = torch.zeros(1, 32, 32)
    m[0, 16, 16] = 1
    t5 = thicken_stroke(m, 5)
    assert float(t5.sum()) > float(m.sum())
    assert int(t5[0, 16, 16]) == 1
    assert int(t5[0, 16, 18]) == 1
