"""F0/F1: shipped slice VFE and free-energy gap (not a reimplementation)."""
from __future__ import annotations

import math

import torch

from fine_grain.bayesian_surprise import BayesianSurpriseGate, compute_slice_vfe


def _kalman_q(mu_p, lv_p, S, sigma_r=1.0):
    var_p = lv_p.exp()
    sr2 = sigma_r ** 2
    var_star = 1.0 / (1.0 / var_p + 1.0 / sr2)
    mu_star = var_star * (S / sr2 + mu_p / var_p)
    return mu_star, var_star.log()


def test_compute_slice_vfe_shapes_and_split():
    torch.manual_seed(0)
    B, M, d = 3, 5, 8
    mu_p = torch.randn(B, M, d)
    lv_p = torch.zeros(B, M, d)
    mu_q = mu_p + 0.3 * torch.randn(B, M, d)
    lv_q = torch.full((B, M, d), -0.5)
    S = mu_q + 0.2 * torch.randn(B, M, d)
    out = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=1.0)
    for k in ("U", "acc_mean", "acc_tr", "F", "gap", "F_min", "s_err"):
        assert out[k].shape == (B, M, 1), k
    # F is complexity + accuracy (mean + trace + const)
    recon = out["U"] + out["acc_mean"] + out["acc_tr"] + 0.5 * math.log(1.0)
    assert torch.allclose(out["F"], recon, atol=1e-5)


def test_elephant_high_F_low_gap():
    """Surprising S, q = exact Kalman posterior: explained but expensive."""
    B, M, d = 2, 4, 16
    mu_p = torch.zeros(B, M, d)
    lv_p = torch.zeros(B, M, d)  # σp=1
    S = torch.full((B, M, d), 8.0)  # purple elephant
    mu_q, lv_q = _kalman_q(mu_p, lv_p, S, sigma_r=1.0)
    out = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=1.0)
    assert float(out["gap"].max()) < 1e-4, float(out["gap"].max())
    assert float(out["F"].mean()) > 5.0
    assert float(out["F_min"].mean()) > 5.0
    # gap << F  — raw F is not a halt zero
    assert float(out["F"].mean()) > 10.0 * float(out["gap"].mean() + 1e-8)


def test_lazy_q_high_gap_low_U():
    """q = p, μq far from S: no complexity, inference not done."""
    B, M, d = 2, 4, 16
    mu_p = torch.zeros(B, M, d)
    lv_p = torch.zeros(B, M, d)
    mu_q = mu_p.clone()
    lv_q = lv_p.clone()
    S = torch.full((B, M, d), 6.0)
    out = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=1.0)
    assert float(out["U"].max()) < 1e-5
    assert float(out["gap"].mean()) > 1.0
    assert float(out["s_err"].mean()) > 10.0
    assert float(out["gap"].mean()) > float(out["U"].mean()) + 1.0


def test_f0_v1_gate_still_tracks_U_not_gap():
    """Default gate_on='u': g = 1-exp(-βU), unchanged by F/gap."""
    torch.manual_seed(1)
    g = BayesianSurpriseGate(d_model=32, n_slices=8, mode="v1_bayes", beta=1.0, gate_on="u")
    S = torch.randn(2, 8, 32)
    H = torch.randn(2, 6, 32)
    gate, meta = g(S, H)
    assert "F" in meta and "gap" in meta and "acc_tr" in meta
    expect = 1.0 - torch.exp(-1.0 * meta["surprise"])
    assert torch.allclose(gate, expect, atol=1e-5)
    # logging present
    assert meta["F"].shape == (2, 8, 1)
    assert meta["pred_loss"].ndim == 0


def test_f1_gap_gate_zero_when_inference_done():
    """gate_on='gap': perfect q* ⇒ g≈0 even though F is large (elephant)."""
    gate_mod = BayesianSurpriseGate(
        d_model=16, n_slices=4, mode="v1_bayes", beta=1.0, gate_on="gap", n_heads=4,
    )
    B, M, d = 2, 4, 16
    S = torch.full((B, M, d), 7.0)
    H = torch.zeros(B, 3, d)
    # Force prior/post to Kalman elephant: p = N(0,1), q = q*
    mu_p = torch.zeros(B, M, d)
    lv_p = torch.zeros(B, M, d)
    mu_q, lv_q = _kalman_q(mu_p, lv_p, S, sigma_r=1.0)
    vfe = compute_slice_vfe(mu_p, lv_p, mu_q, lv_q, S, sigma_r=1.0)
    g = 1.0 - torch.exp(-1.0 * vfe["gap"])
    assert float(vfe["F"].mean()) > 4.0
    assert float(g.max()) < 1e-4

    # Lazy: q=p, S far → gap gate open
    vfe_lazy = compute_slice_vfe(mu_p, lv_p, mu_p, lv_p, S, sigma_r=1.0)
    g_lazy = 1.0 - torch.exp(-1.0 * vfe_lazy["gap"])
    assert float(vfe_lazy["U"].max()) < 1e-5
    assert float(g_lazy.mean()) > 0.5

    # Live module still forwards and exposes gap
    gate, meta = gate_mod(S, H)
    assert gate.shape == (B, M, 1)
    assert "gap" in meta


def test_q_infer_star_zeros_gap_at_eval():
    torch.manual_seed(4)
    g = BayesianSurpriseGate(d_model=32, n_slices=4, mode="v1_bayes", n_heads=4)
    S = torch.randn(2, 4, 32) * 3
    H = torch.randn(2, 5, 32)
    g.eval()
    g.q_infer = "amortized"
    _, m0 = g(S, H)
    g.q_infer = "star"
    _, m1 = g(S, H)
    assert float(m1["gap"].max()) < 1e-4
    # snapping q* changes U (now KL(q*||p)) vs amortized
    assert m0["gap"].shape == m1["gap"].shape


def test_v1_meta_exposes_live_mu_star():
    torch.manual_seed(5)
    g = BayesianSurpriseGate(d_model=32, n_slices=4, mode="v1_bayes", n_heads=4)
    S = torch.randn(2, 4, 32, requires_grad=True)
    H = torch.randn(2, 5, 32)
    _, meta = g(S, H)
    assert meta["mu_star"].shape == S.shape
    assert meta["mu_star"].requires_grad
    meta["mu_star"].sum().backward()
    assert S.grad is not None and float(S.grad.abs().sum()) > 0


def test_f2_vfe_train_loss_hits_post_not_prior():
    """F2: gap(q‖q*) with frozen p,S trains post_head only."""
    torch.manual_seed(3)
    g = BayesianSurpriseGate(d_model=32, n_slices=4, mode="v1_bayes", n_heads=4)
    S = torch.randn(2, 4, 32)
    H = torch.randn(2, 5, 32)
    _, meta = g(S, H)
    loss = meta["vfe_train_loss"]
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    post_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in g.post_head.parameters()
    )
    prior_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in g.prior_head.parameters()
    )
    assert post_grad, "amortized q must receive VFE grad"
    assert not prior_grad, "p and S are detached; prior_head must not get VFE grad"


def test_kalman_gain_and_mu_star_identity():
    """S' = μp + K(S−μp) equals the precision-weighted μ*."""
    B, M, d = 2, 4, 8
    mu_p = torch.randn(B, M, d)
    lv_p = torch.zeros(B, M, d)
    S = torch.randn(B, M, d)
    dummy_q = mu_p.clone()
    out = compute_slice_vfe(mu_p, lv_p, dummy_q, lv_p, S, sigma_r=1.0)
    var_p = lv_p.exp()
    K = var_p / (var_p + 1.0)
    recon = mu_p + K * (S - mu_p)
    assert torch.allclose(out["mu_star"], recon, atol=1e-5)
    assert torch.allclose(out["K"], K, atol=1e-5)
    # Clamp floor lv=-5 ⇒ σp²≈0.0067, K small, μ* near μp
    tiny = compute_slice_vfe(mu_p, torch.full_like(lv_p, -5.0), dummy_q, lv_p, S, sigma_r=1.0)
    assert float(tiny["K"].mean()) < 0.01
    assert torch.allclose(tiny["mu_star"], mu_p, atol=0.05)
    # Clamp cap lv=2 ⇒ σp²≈7.39, K large, μ* nearer S than μp
    wide = compute_slice_vfe(mu_p, torch.full_like(lv_p, 2.0), dummy_q, lv_p, S, sigma_r=1.0)
    assert float(wide["K"].mean()) > 0.85
    err_s = (wide["mu_star"] - S).pow(2).mean()
    err_p = (wide["mu_star"] - mu_p).pow(2).mean()
    assert float(err_s) < float(err_p)


def test_s_kalman_update_writes_mu_star_not_mot_delta():
    """s_kalman_update: S_write is live μ*, independent of MoT Δ."""
    from fine_grain.native_mot import NativeMoTLayer

    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=4, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="rms_dir",
        s_kalman_update=True, deslice_write="absolute",
        gate_h_local=False,
    )
    X = torch.randn(2, 64, 32)
    H = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6)
    X2, H2, tr = layer(X, H, text_mask=mask)
    assert X2.shape == X.shape and H2.shape == H.shape
    assert layer.last_S_write is not None
    _, meta = layer.surprise_gate(layer.last_S, H, text_mask=mask)
    assert torch.allclose(layer.last_S_write, meta["mu_star"].detach(), atol=1e-5)
    # H still moved (MoT ran); S write is not the MoT residual step.
    assert (H2 - H).abs().sum() > 0
    assert tr.kalman_k >= 0.0


def test_s_kalman_default_off_keeps_rms_dir():
    from fine_grain.native_mot import NativeMoTLayer

    layer = NativeMoTLayer(
        d_x=32, d=32, n_slices=4, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="rms_dir",
    )
    assert layer.s_kalman_update is False
    S = torch.randn(2, 4, 32)
    delta = torch.randn(2, 4, 32)
    gate = torch.ones(2, 4, 1)
    S2 = layer._apply_s_update(S, delta, gate)
    assert not torch.allclose(S2, S)


def test_pred_loss_still_prior_not_vfe_accuracy():
    """pred_loss remains ||S-μp||², not ||S-μq||²."""
    torch.manual_seed(2)
    g = BayesianSurpriseGate(d_model=32, n_slices=4, mode="v1_bayes", n_heads=4)
    S = torch.randn(2, 4, 32)
    H = torch.randn(2, 5, 32)
    _, meta = g(S, H)
    # pred_loss is a scalar; just check it is finite and does not require F
    assert meta["pred_loss"].ndim == 0
    assert torch.isfinite(meta["pred_loss"])
    assert "acc_mean" in meta
