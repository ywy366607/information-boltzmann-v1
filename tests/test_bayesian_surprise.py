"""Unit tests for BayesianSurpriseGate and spatial_surprise_map."""
import pytest
import torch
import torch.nn as nn
from fine_grain.bayesian_surprise import (
    BayesianSurpriseGate,
    global_gate_from_surprise,
    spatial_surprise_map,
)


def test_baseline_mode():
    B, M, d, T = 2, 8, 32, 10
    gate_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="baseline")
    S = torch.randn(B, M, d)
    H = torch.randn(B, T, d)

    g, meta = gate_mod(S, H)
    assert g.shape == (B, M, 1)
    assert torch.allclose(g, torch.ones_like(g))
    assert meta["mean_gate"] == 1.0


def test_constant_and_random_mode():
    B, M, d, T = 2, 8, 32, 10
    c_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="constant", constant_val=0.42)
    S = torch.randn(B, M, d)
    H = torch.randn(B, T, d)

    g_c, meta_c = c_mod(S, H)
    assert torch.allclose(g_c, torch.full_like(g_c, 0.42))

    r_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="random")
    g_r, meta_r = r_mod(S, H)
    assert g_r.shape == (B, M, 1)
    assert (g_r >= 0.0).all() and (g_r <= 1.0).all()


def test_v0_jepa_mode():
    B, M, d, T = 2, 8, 32, 10
    gate_mod = BayesianSurpriseGate(d_model=d, n_slices=M, beta=2.0, mode="v0_jepa", detach_gate=True)
    S = torch.randn(B, M, d, requires_grad=True)
    H = torch.randn(B, T, d, requires_grad=True)

    g, meta = gate_mod(S, H)
    assert g.shape == (B, M, 1)
    assert (g >= 0.0).all() and (g <= 1.0).all()
    assert "surprise" in meta
    assert meta["surprise"].shape == (B, M, 1)
    assert (meta["surprise"] >= 0.0).all()

    # Verify detach_gate ensures no gradient leaks through gate into S
    loss = (g * S).sum()
    loss.backward()
    assert S.grad is not None
    # S.grad should equal g
    assert torch.allclose(S.grad, g.expand_as(S))


def test_v0_shuffled_and_reverse():
    B, M, d, T = 2, 8, 32, 10
    shuf_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v0_shuffled")
    rev_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v0_reverse")
    S = torch.randn(B, M, d)
    H = torch.randn(B, T, d)

    g_s, meta_s = shuf_mod(S, H)
    assert g_s.shape == (B, M, 1)

    g_r, meta_r = rev_mod(S, H)
    assert g_r.shape == (B, M, 1)
    assert (g_r >= 0.0).all() and (g_r <= 1.0).all()


def test_v0_global_and_spatial_only():
    B, M, d, T = 2, 8, 32, 10
    g_glob = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v0_global_only")
    g_spat = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v0_spatial_only")
    S = torch.randn(B, M, d)
    H = torch.randn(B, T, d)

    gate_g, meta_g = g_glob(S, H)
    assert gate_g.shape == (B, M, 1)
    # in global_only, all slices in the same batch item have the exact same gate value
    assert torch.allclose(gate_g[:, 0:1, :], gate_g[:, 1:2, :])

    gate_s, meta_s = g_spat(S, H)
    assert gate_s.shape == (B, M, 1)
    assert (gate_s >= 0.0).all() and (gate_s <= 1.0).all()


def test_v1_bayes_kl_correctness():
    B, M, d, T = 2, 8, 32, 10
    v1_mod = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v1_bayes")
    S = torch.randn(B, M, d)
    H = torch.randn(B, T, d)

    g, meta = v1_mod(S, H)
    assert g.shape == (B, M, 1)
    assert (g >= 0.0).all() and (g <= 1.0).all()
    assert meta["surprise"].shape == (B, M, 1)
    assert (meta["surprise"] >= 0.0).all()
    assert "u_mu" in meta and "u_sigma" in meta
    assert meta["u_mu"].shape == (B, M, 1)
    assert meta["u_sigma"].shape == (B, M, 1)
    assert torch.allclose(meta["surprise"], meta["u_mu"] + meta["u_sigma"], atol=1e-5)


def _erank(tokens: torch.Tensor) -> float:
    """tokens: [M, d] → Roy–Vetterli effective rank."""
    s = torch.linalg.svdvals(tokens.detach().float())
    p = (s * s).clamp_min(1e-12)
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def test_prior_readout_not_uniform_at_init():
    """RMSNorm + orthogonal Q must not collapse H_ctx to a single mean-pool."""
    torch.manual_seed(0)
    gate_mod = BayesianSurpriseGate(d_model=64, n_slices=16, mode="v0_jepa")
    H = torch.randn(2, 10, 64) * 6.0  # match the real ||H|| ≫ ||Q_old|| regime
    H_ctx = gate_mod._predict_prior_from_h(H)
    u = torch.nn.functional.normalize(H_ctx, dim=-1)
    sim = u @ u.transpose(-1, -2)
    eye = torch.eye(16, dtype=torch.bool)
    cos = float(sim[0].masked_select(~eye).abs().mean().detach())
    assert cos < 0.6, f"H_ctx pairwise |cos|={cos:.3f} still collapsed"
    assert _erank(H_ctx[0]) > 4.0, f"H_ctx erank={_erank(H_ctx[0]):.2f}"


def test_pred_loss_trains_prior_not_S():
    """Aux prior loss reaches Q / lang_to_prior; stopgrad(S) blocks S."""
    torch.manual_seed(1)
    gate_mod = BayesianSurpriseGate(d_model=32, n_slices=8, mode="v0_jepa", detach_gate=True)
    S = torch.randn(2, 8, 32, requires_grad=True)
    H = torch.randn(2, 10, 32, requires_grad=True)
    _, meta = gate_mod(S, H)
    assert meta["pred_loss"].ndim == 0
    meta["pred_loss"].backward()
    assert gate_mod.slice_queries.grad is not None
    assert float(gate_mod.slice_queries.grad.norm()) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate_mod.lang_to_prior.parameters())
    assert S.grad is None


def test_pred_loss_trains_bayes_prior():
    torch.manual_seed(2)
    gate_mod = BayesianSurpriseGate(d_model=32, n_slices=8, mode="v1_bayes", detach_gate=True)
    S = torch.randn(2, 8, 32, requires_grad=True)
    H = torch.randn(2, 10, 32, requires_grad=True)
    _, meta = gate_mod(S, H)
    meta["pred_loss"].backward()
    assert float(gate_mod.slice_queries.grad.norm()) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate_mod.prior_head.parameters())
    assert S.grad is None


def test_pred_loss_live_S_reaches_encoder():
    """Without detach_pred_target, pred_loss sends grad into S."""
    torch.manual_seed(3)
    gate_mod = BayesianSurpriseGate(
        d_model=32, n_slices=8, mode="v0_jepa",
        detach_gate=True, detach_pred_target=False,
    )
    S = torch.randn(2, 8, 32, requires_grad=True)
    H = torch.randn(2, 10, 32, requires_grad=True)
    _, meta = gate_mod(S, H)
    meta["pred_loss"].backward()
    assert S.grad is not None and float(S.grad.norm()) > 0
    assert float(gate_mod.slice_queries.grad.norm()) > 0


def test_sigreg_on_live_S():
    """SIGReg(S) must not be stop-grad: encoder S receives gradient."""
    torch.manual_seed(4)
    gate_mod = BayesianSurpriseGate(d_model=32, n_slices=8, mode="v0_jepa")
    S = torch.randn(2, 8, 32, requires_grad=True)
    H = torch.randn(2, 10, 32, requires_grad=True)
    _, meta = gate_mod(S, H)
    assert "sigreg_loss" in meta
    meta["sigreg_loss"].backward()
    assert S.grad is not None and float(S.grad.norm()) > 0


def test_spatial_surprise_map():
    B, M, N, res = 2, 8, 64, 8
    U = torch.rand(B, M, 1)
    # router assignment A: [B, N, M] summing to 1 over M
    A = torch.softmax(torch.randn(B, N, M), dim=-1)

    u_pt = spatial_surprise_map(U, A)
    assert u_pt.shape == (B, N)

    u_img = spatial_surprise_map(U, A, res=res)
    assert u_img.shape == (B, 1, res, res)


def test_stiefel_queries_orthogonality():
    """Verify Stiefel manifold polar decomposition forces Q @ Q.T = I_M."""
    torch.manual_seed(42)
    B, M, d, T = 2, 16, 64, 10
    gate_stiefel = BayesianSurpriseGate(d_model=d, n_slices=M, mode="v1_bayes", stiefel_queries=True)
    Q = gate_stiefel._get_queries().squeeze(0)  # [M, d]

    # Check orthogonality: Q @ Q.T should equal identity matrix
    gram = Q @ Q.T
    eye = torch.eye(M, device=Q.device)
    assert torch.allclose(gram, eye, atol=1e-3), f"Stiefel Gram max error: {torch.max(torch.abs(gram - eye)):.5f}"

    # Check effective rank
    svals = torch.linalg.svdvals(Q)
    stable_rank = float((svals.pow(2).sum()) / (svals.max().pow(2)))
    assert abs(stable_rank - M) < 0.1, f"Stable rank {stable_rank} should equal M={M}"


def test_official_epps_pulley_sigreg():
    """Official SIGReg: isotropic Gaussian cloud scores lower than a collapsed spike."""
    from fine_grain.sigreg import compute_sigreg_loss

    torch.manual_seed(42)
    N, d = 256, 16
    Z_gauss = torch.randn(N, d)
    Z_spike = torch.ones(N, d)
    sig_g = compute_sigreg_loss(Z_gauss, num_slices=64)
    sig_s = compute_sigreg_loss(Z_spike, num_slices=64)
    assert sig_g["sigreg_total"].ndim == 0
    assert sig_s["sigreg_total"] > sig_g["sigreg_total"]


def test_prior_is_multihead():
    g = BayesianSurpriseGate(d_model=32, n_slices=8, mode="v0_jepa", n_heads=4)
    assert g.n_heads == 4 and g.dh == 8
    H = torch.randn(2, 10, 32)
    ctx = g._predict_prior_from_h(H)
    assert ctx.shape == (2, 8, 32)
    assert g.last_attn.shape == (2, 4, 8, 10)


def test_global_gate_from_surprise_not_mean():
    U = torch.zeros(1, 32, 1)
    U[0, 3] = 5.0
    g = global_gate_from_surprise(U, beta=1.0, kind="lse")
    g_mean = 1.0 - torch.exp(-U.mean())
    assert g.shape == (1, 1, 1)
    assert float(g) > float(g_mean) + 0.3

