"""Active-inference VFE: one observation, action in-graph, S0 accuracy."""
from __future__ import annotations

import torch

from fine_grain.bayesian_surprise import compute_point_vfe, reduce_observation_f
from fine_grain.native_mot import NativeMoTLayer
from fine_grain.omni_model import DualStreamOmni


def _pix_model(**kw):
    defaults = dict(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True, use_ticket_read=True,
        deslice_write="increment", deslice_topk=2,
        surprise_mode="v1_bayes",
    )
    defaults.update(kw)
    return DualStreamOmni.unified(**defaults)


def test_fixed_sigma_accuracy_is_prediction_error():
    torch.manual_seed(0)
    mu_p = torch.randn(2, 16, 3)
    lv_p = torch.zeros(2, 16, 3)
    mu_q = mu_p + 0.3 * torch.randn(2, 16, 3)
    lv_q = torch.zeros(2, 16, 3)
    y = mu_q + 0.2 * torch.randn(2, 16, 3)
    vfe = compute_point_vfe(mu_p, lv_p, mu_q, lv_q, y, sigma_r=1.0)
    expected = 0.5 * (y - mu_q).pow(2).mean(dim=-1, keepdim=True)
    assert torch.allclose(vfe["acc_mean"], expected, atol=1e-5)


def test_omni_loss_single_observation_not_black():
    torch.manual_seed(0)
    m = _pix_model(prior_write=1.0)
    z = torch.randn(2, 3, 8, 8)
    tgt = torch.randn(2, 3, 8, 8)
    prompts = ["Draw digit 1 with a thin red stroke"] * 2
    out = m(z, prompts, need_pix=[True, True], t=torch.ones(2))
    loss, meta = m.omni_loss(out, {
        "need_text": [False, False], "need_pix": [True, True],
        "answer": ["1", "1"], "target_rgb": tgt,
        "stroke": torch.zeros(2, 8, 8), "t": None,
    }, z.device)
    assert meta.get("n_obs") == 1
    assert "point_F" in meta
    vfe = out["point_vfe_terms"]
    mu_q = m._img_to_pts(out["x_pred"])
    y = m._img_to_pts(tgt)
    bg = torch.full_like(y, -1.0)
    mu_pr = m._img_to_pts(out["rgb_mu_p"])
    lv_pr = m._img_to_pts(out["rgb_lv_p"])
    lv_q = m._img_to_pts(out["rgb_lv"])
    fy = compute_point_vfe(mu_pr, lv_pr, mu_q, lv_q, y, sigma_r=1.0)
    fb = compute_point_vfe(mu_pr, lv_pr, mu_q, lv_q, bg, sigma_r=1.0)
    pi = m._observation_pi(tgt)
    one = float(m._two_pane(fy["F"], fy["F"], pi).detach())
    mixed = one + float(reduce_observation_f(fb).detach())
    got = float(out["point_vfe"].detach())
    assert abs(got - one) < 1e-4
    assert abs(got - mixed) > 1e-4
    assert torch.allclose(vfe["F"], fy["F"], atol=1e-5)
    assert loss.requires_grad


def test_action_writes_bptt_reaches_write_params():
    torch.manual_seed(0)
    m = _pix_model(prior_write=1.0)
    # Zero-init pix_head (JiT) blocks ∂rgb/∂X; a live head is the trained state.
    torch.nn.init.normal_(m.pix_head[-1].weight, std=0.05)
    torch.nn.init.zeros_(m.pix_head[-1].bias)
    x = torch.randn(1, 3, 8, 8)
    prompts = ["Draw digit 3 with a thin yellow stroke"]
    out = m.action_writes(x, prompts, need_pix=[True], n_steps=2, t=torch.ones(1))
    assert int(out["n_action"]) == 2
    tgt = torch.randn(1, 3, 8, 8)
    loss = (out["x_pred"] - tgt).pow(2).mean()
    m.zero_grad(set_to_none=True)
    loss.backward()
    g = m.pix_head[-1].weight.grad
    assert g is not None
    assert float(g.abs().sum()) > 0.0
    dg = m.mot_stack.layers[0].deslice.proj.weight.grad
    assert dg is not None
    assert float(dg.abs().sum()) > 0.0


def test_s0_prior_receives_accuracy_grad():
    torch.manual_seed(0)
    m = _pix_model()
    z = torch.randn(1, 3, 8, 8)
    tgt = torch.randn(1, 3, 8, 8)
    prompts = ["Draw digit 7 with a thin green stroke"]
    out = m(z, prompts, need_pix=[True], t=torch.ones(1))
    loss, meta = m.omni_loss(out, {
        "need_text": [False], "need_pix": [True],
        "answer": ["7"], "target_rgb": tgt,
        "stroke": torch.zeros(1, 8, 8), "t": None,
    }, z.device)
    assert "s0_acc" in meta
    m.zero_grad(set_to_none=True)
    loss.backward()
    w = m.mot_stack.layers[0].lang_s0[0].weight
    assert w.grad is not None
    assert float(w.grad.abs().sum()) > 0.0


def test_tickets_detach_s0_from_gate():
    torch.manual_seed(0)
    layer = NativeMoTLayer(
        d_x=16, d=16, n_slices=4, n_heads=2, res=4,
        use_ticket_read=True, surprise_mode="baseline",
    )
    X = torch.randn(1, 16, 16)
    H = torch.randn(1, 4, 16)
    s = layer.flow_tickets(X, H, torch.ones(1, 4), X_prior=None)
    assert layer.last_s0 is not None
    assert layer.last_s0.requires_grad
    assert not s.requires_grad
    layer.zero_grad(set_to_none=True)
    layer.last_s0.sum().backward()
    g2 = layer.lang_s0[0].weight.grad
    assert g2 is not None
    assert float(g2.abs().sum()) > 0.0
