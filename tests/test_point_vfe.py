"""Terminal free energy lives on points. Slices are workspace only."""
from __future__ import annotations

import torch

from fine_grain.bayesian_surprise import compute_point_vfe, compute_slice_vfe
from fine_grain.omni_model import DualStreamOmni


def test_point_vfe_same_algebra_as_slices():
    torch.manual_seed(0)
    mu_p = torch.randn(2, 16, 3)
    lv_p = torch.zeros(2, 16, 3)
    mu_q = mu_p + 0.2 * torch.randn(2, 16, 3)
    lv_q = torch.full((2, 16, 3), -0.4)
    y = mu_q + 0.1 * torch.randn(2, 16, 3)
    a = compute_point_vfe(mu_p, lv_p, mu_q, lv_q, y)
    b = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, y)
    assert torch.allclose(a["F"], b["F"])
    assert a["F"].shape == (2, 16, 1)


def test_decode_gauss_sigma_one_at_init():
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
    )
    X = torch.randn(2, 64, 32)
    mu, lv = m.decode_gauss(X)
    assert mu.shape == lv.shape == (2, 64, 3)
    assert float(lv.detach().abs().max()) < 1e-6


def test_language_prior_field_is_on_points():
    torch.manual_seed(0)
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=2, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
    )
    z = torch.randn(2, 3, 8, 8)
    t = torch.ones(2)
    out = m(z, ["Draw digit 3 with a thin yellow stroke"] * 2, need_pix=[True, True], t=t)
    Xp = m.mot_stack._last_X_prior
    assert Xp is not None
    assert Xp.shape == m.mot_stack._last_X.shape
    assert out["rgb_lv"] is not None
    assert out["rgb_mu_p"] is not None
    assert out["x_pred"].shape == z.shape


def test_two_pane_punishes_black_and_flood():
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
    )
    tgt = torch.full((1, 3, 8, 8), -1.0)
    tgt[:, 1, 2:5, 3:6] = 1.0
    pi = m._observation_pi(tgt)
    ones = torch.ones(1, 64, 1)
    zeros = torch.zeros(1, 64, 1)
    # Pred matches target figure, misses bg vs vice versa — both panes matter.
    black = ones.clone()
    flood = zeros.clone()
    # acc: high = bad. black canvas: acc_y high on figure, acc_bg low
    acc_y_black = ones
    acc_bg_black = zeros
    acc_y_flood = zeros
    acc_bg_flood = ones
    lb = m._two_pane(acc_y_black, acc_bg_black, pi)
    lf = m._two_pane(acc_y_flood, acc_bg_flood, pi)
    lgood = m._two_pane(zeros, zeros, pi)
    assert float(lb) > float(lgood)
    assert float(lf) > float(lgood)


def test_omni_loss_includes_sy_sigreg():
    torch.manual_seed(0)
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
        sigreg_coef=0.1,
    )
    z = torch.randn(2, 3, 8, 8)
    tgt = torch.randn(2, 3, 8, 8)
    out = m(z, ["Draw digit 1 with a thin red stroke"] * 2, need_pix=[True, True], t=torch.ones(2))
    _, meta = m.omni_loss(out, {
        "need_text": [False, False], "need_pix": [True, True],
        "answer": ["1", "1"], "target_rgb": tgt,
        "stroke": torch.zeros(2, 8, 8), "t": None,
    }, z.device)
    assert "sigreg_sy" in meta
    assert "see_acc" in meta or "point_F" in meta


def test_omni_loss_lands_f_on_points():
    torch.manual_seed(0)
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2, vfe_coef=0.1,
    )
    z = torch.randn(2, 3, 8, 8)
    tgt = torch.randn(2, 3, 8, 8)
    t = torch.ones(2)
    out = m(z, ["Draw digit 1 with a thin red stroke"] * 2, need_pix=[True, True], t=t)
    batch = {
        "need_text": [False, False],
        "need_pix": [True, True],
        "answer": ["1", "1"],
        "target_rgb": tgt,
        "stroke": torch.zeros(2, 8, 8),
        "t": None,
    }
    loss, meta = m.omni_loss(out, batch, z.device)
    assert loss.requires_grad
    assert "point_F" in meta
    assert out.get("point_vfe") is not None
    loss.backward()


def test_slice_mass_loss_weights_inverse_size():
    """One exclusive point gets as much vote as three shared points combined."""
    from fine_grain.native_mot import slice_mass_loss_weights

    w = torch.zeros(1, 4, 2)
    w[0, 0, 0] = 1.0
    w[0, 1:, 1] = 1.0 / 3.0
    pi = slice_mass_loss_weights(w).reshape(-1)
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(pi[0], torch.tensor(0.5), atol=1e-5)
    assert torch.allclose(pi[1:], torch.full((3,), 1.0 / 6.0), atol=1e-5)


def test_obs_rarity_boosts_outlier_pixel_not_stroke_mask():
    """One unusual pixel in the observation — any kind — gets the vote."""
    m = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, res=8, n_heads=4,
        surprise_mode="v1_bayes", deslice_write="increment",
        fm_pred="x", fm_signed=True, prior_write=1.0, deslice_topk=2,
    )
    tgt = torch.full((1, 3, 8, 8), -1.0)
    tgt[:, 1, 3, 4] = 1.0
    r = m._obs_rarity(tgt).reshape(8, 8)
    assert float(r[3, 4]) > float(r.mean()) * 10.0
    w = m._obs_assign(tgt)
    from fine_grain.native_mot import slice_mass_loss_weights
    pi = slice_mass_loss_weights(w)
    extra = r.reshape(1, -1, 1)
    wn = pi * extra
    wn = wn / wn.sum(dim=1, keepdim=True)
    peak = float(wn.reshape(8, 8)[3, 4])
    assert peak > 0.2


def test_uniform_assignment_is_area_mean():
    from fine_grain.native_mot import slice_mass_loss_weights

    B, N, M = 2, 16, 4
    w = torch.full((B, N, M), 1.0 / M)
    pi = slice_mass_loss_weights(w)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / N), atol=1e-5)


def test_recognition_still_uses_slice_vfe_when_no_pix():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    assert m.vfe_coef == 0.1
    img = torch.rand(1, 3, 8, 8)
    out = m(img, ["What digit is drawn with the thin stroke?"], need_pix=[False])
    assert out.get("rgb_lv") is not None
    # text-only path does not set x_pred; slice vfe_loss still present
    assert out.get("x_pred") is None
    assert out.get("vfe_loss") is not None
