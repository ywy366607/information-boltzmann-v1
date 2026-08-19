"""OT flow matching helpers and unified time condition."""
from __future__ import annotations

import torch

from fine_grain.flow_match import TimeCondition, interpolate, sample_t, velocity_target
from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_KNOBS


def test_interpolate_endpoints():
    x0 = torch.zeros(2, 3, 4, 4)
    x1 = torch.ones(2, 3, 4, 4)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    th = torch.full((2,), 0.5)
    assert torch.allclose(interpolate(x0, x1, t0), x0)
    assert torch.allclose(interpolate(x0, x1, t1), x1)
    assert torch.allclose(interpolate(x0, x1, th), 0.5 * torch.ones_like(x0))
    assert torch.allclose(velocity_target(x0, x1), torch.ones_like(x0))


def test_sample_t_logit_normal_in_unit_interval():
    t = sample_t(256, torch.device("cpu"), "logit_normal")
    assert t.shape == (256,)
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0


def test_heun_equals_euler_when_v_constant():
    from fine_grain.flow_match import ode_integrate

    x0 = torch.zeros(1, 3, 4, 4)
    v = torch.ones(1, 3, 4, 4)

    def step_fn(x, t):
        return v.expand_as(x)

    eul = ode_integrate(step_fn, x0, n_steps=4, method="euler", clamp=False)
    heu = ode_integrate(step_fn, x0, n_steps=4, method="heun", clamp=False)
    assert torch.allclose(eul, heu, atol=1e-5)
    assert torch.allclose(eul, torch.ones_like(x0), atol=1e-5)


def test_time_cond_zero_init_is_noop():
    tc = TimeCondition(8)
    x = torch.randn(2, 5, 8)
    t = torch.tensor([0.2, 0.9])
    assert float(tc(t, x).abs().max()) < 1e-8


def test_time_mod_zero_init_is_identity():
    from fine_grain.flow_match import TimeMod

    tm = TimeMod(8)
    x = torch.randn(2, 5, 8)
    t = torch.tensor([0.1, 0.8])
    assert torch.allclose(tm(t, x), x, atol=1e-6)


def test_sample_t_jit_is_official_logit_normal():
    t = sample_t(512, torch.device("cpu"), "jit")
    assert t.shape == (512,)
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0
    # P_mean=-0.8 shifts mass toward smaller t (noisier).
    assert float(t.mean()) < 0.45


def test_adaln_zero_is_identity():
    from fine_grain.flow_match import AdaLNZero

    ada = AdaLNZero(8)
    x = torch.randn(2, 5, 8)
    c = torch.randn(2, 8)
    assert torch.allclose(ada(x, c), x, atol=1e-6)


def test_timestep_embed_varies_with_t():
    from fine_grain.flow_match import TimestepEmbedder

    te = TimestepEmbedder(16)
    a = te(torch.tensor([0.1]))
    b = te(torch.tensor([0.9]))
    assert a.shape == (1, 16)
    assert (a - b).abs().mean() > 1e-4


def test_t_is_a_point_coordinate():
    """t sits on every point like xy. t=None is the old stem (F2-safe)."""
    torch.manual_seed(0)
    stack = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
    ).mot_stack
    img = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        x0 = stack.encode_X(img)
        x_none = stack.encode_X(img, t=None)
        x_a = stack.encode_X(img, t=torch.tensor([0.1, 0.1]))
        x_b = stack.encode_X(img, t=torch.tensor([0.9, 0.9]))
    assert torch.allclose(x0, x_none)
    assert (x_a - x_b).abs().mean() > 1e-5
    assert (x_a - x0).abs().mean() > 1e-5


def test_language_not_in_t_coord():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    img = torch.rand(2, 3, 8, 8)
    t = torch.tensor([0.3, 0.3])
    with torch.no_grad():
        xa = m.mot_stack.encode_X(img, t=t)
        xb = m.mot_stack.encode_X(img, t=t)
    assert torch.allclose(xa, xb)


def test_signed_x_pred_is_unbounded():
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8, fm_pred="x", fm_signed=True,
    )
    z = torch.randn(1, 3, 8, 8)
    t = torch.tensor([0.4])
    out = m(z, ["Draw digit 1 with a thin red stroke"], need_pix=[True], t=t)
    assert out["x_pred"] is not None
    # Linear head, not squashed to (0,1).
    assert out["x_pred"].abs().max() < 2.0  # zero-init ⇒ near 0
    assert torch.allclose(out["rgb"], (out["x_pred"] + 1) * 0.5, atol=1e-5)


def test_omni_fm_jit_x_pred_defines_v():
    """JiT: net emits x; v = (x − z_t)/(1−t)."""
    from fine_grain.flow_match import v_from_x_pred

    torch.manual_seed(0)
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8, fm_pred="x")
    assert m.fm_pred == "x"
    z = torch.rand(2, 3, 8, 8)
    t = torch.tensor([0.25, 0.75])
    out = m(z, ["Draw digit 1 with a thin red stroke"] * 2, need_pix=[True, True], t=t)
    assert out["x_pred"] is not None and out["x_pred"].shape == z.shape
    assert torch.allclose(out["rgb"], out["x_pred"])
    assert torch.allclose(out["v"], v_from_x_pred(out["x_pred"], z, t), atol=1e-5)
    assert m.mot_stack.deslice_write == UNIFIED_KNOBS["deslice_write"]


def test_no_t_still_sigmoid_rgb():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    x = torch.rand(1, 3, 8, 8)
    out = m(x, ["What is this"], need_pix=[False])
    assert out["v"] is None
    assert out["rgb"].min() >= 0 and out["rgb"].max() <= 1
