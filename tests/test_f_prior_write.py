"""Language prior write is the F-action: Deslice(μp − S)."""
from __future__ import annotations

import torch

from fine_grain.omni_model import DualStreamOmni


def test_recognition_default_does_not_prior_write():
    m = DualStreamOmni.unified(d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8)
    assert m.mot_stack.prior_write == 0.0
    assert m.mot_stack.layers[0].prior_write == 0.0


def test_prior_write_uses_language_mu_p():
    torch.manual_seed(0)
    a = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=2, res=8,
        fm_pred="x", fm_signed=True, prior_write=0.0,
    )
    b = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=2, res=8,
        fm_pred="x", fm_signed=True, prior_write=1.0,
    )
    b.load_state_dict(a.state_dict())
    z = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.3, 0.7])
    p = ["Draw digit 3 with a thin yellow stroke"] * 2
    with torch.no_grad():
        a(z, p, need_pix=[True, True], t=t)
        xa = a.mot_stack._last_X
        b(z, p, need_pix=[True, True], t=t)
        xb = b.mot_stack._last_X
    assert b.mot_stack.layers[0].last_mu_p is not None
    # Same weights, extra Deslice(μp−S) must move the field.
    assert (xa - xb).abs().mean() > 1e-6


def test_active_generation_prior_write_is_not_zeroed_at_t0():
    torch.manual_seed(7)
    base = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=2, res=8,
        fm_pred="x", fm_signed=False, prior_write=0.0,
    )
    scaled = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=2, res=8,
        fm_pred="x", fm_signed=False, prior_write=1.0, prior_write_by_t=True,
    )
    active = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=2, res=8,
        fm_pred="x", fm_signed=False, prior_write=1.0, prior_write_by_t=False,
    )
    scaled.load_state_dict(base.state_dict())
    active.load_state_dict(base.state_dict())
    z = torch.zeros(1, 3, 8, 8)
    t0 = torch.zeros(1)
    prompt = ["Draw digit 3 with a thin yellow stroke"]
    with torch.no_grad():
        x_base = base(z, prompt, need_pix=[True], t=t0)["X"]
        x_scaled = scaled(z, prompt, need_pix=[True], t=t0)["X"]
        x_active = active(z, prompt, need_pix=[True], t=t0)["X"]
    assert torch.allclose(x_base, x_scaled, atol=1e-6)
    assert not torch.allclose(x_base, x_active, atol=1e-6)


def test_noisy_pred_loss_skipped_when_prior_write():
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True, prior_write=1.0, prior_loss_coef=0.1,
    )
    z = torch.randn(2, 3, 8, 8)
    tgt = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.8])
    out = m(z, ["Draw digit 1 with a thin red stroke"] * 2, need_pix=[True, True], t=t)
    batch = {
        "need_text": [False, False],
        "need_pix": [True, True],
        "answer": ["1", "1"],
        "target_rgb": tgt,
        "stroke": torch.zeros(2, 8, 8),
        "t": t,
    }
    _, meta = m.omni_loss(out, batch, z.device)
    assert "prior_clean" in meta
    assert meta["prior_clean_coef"] == 0.1


def test_clean_prior_target_uses_the_generation_chart_time():
    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=False, prior_write=1.0, prior_loss_coef=0.1,
    )
    z = torch.zeros(2, 3, 8, 8)
    t0 = torch.zeros(2)
    prompts = ["Draw digit 1 with a thin red stroke"] * 2
    m(z, prompts, need_pix=[True, True], t=t0)
    seen = []
    original = m.mot_stack.encode_X

    def record_t(img, t=None):
        seen.append(None if t is None else t.detach().clone())
        return original(img, t=t)

    m.mot_stack.encode_X = record_t
    idx = torch.arange(2)
    clean = torch.rand_like(z)
    loss = m._prior_clean_loss(clean, idx, z.device, t=t0)
    assert loss is not None
    assert len(seen) == 1 and torch.equal(seen[0], t0)


def test_f_generate_uses_one_native_field_pass_at_source_time():
    from scripts.train_omni_probe import f_generate

    class ProbeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, x, prompts, pi_x=None, need_pix=None, t=None):
            self.calls.append({"x": x.clone(), "pi_x": pi_x, "t": t.clone()})
            return {"x_pred": x + 0.25}

    m = ProbeModel()
    z = torch.randn(1, 3, 8, 8)
    y, n = f_generate(
        m, z, ["Draw digit 2 with a thin red stroke"], [True],
        n_steps=8, halt_eps=0.03, return_steps=True,
    )
    assert len(m.calls) == 1
    assert torch.equal(m.calls[0]["x"], z)
    assert torch.equal(m.calls[0]["t"], torch.zeros(1))
    assert m.calls[0]["pi_x"] == 1.0
    assert torch.allclose(y, z + 0.25)
    assert torch.equal(n, torch.ones(1))


def test_f_iterate_halts_when_field_stops():
    from scripts.train_omni_probe import f_iterate

    x = torch.ones(2, 3, 4, 4)

    def contract(z):
        return z * 0.4

    y, n = f_iterate(contract, x, n_steps=20, halt_eps=0.02)
    assert y.shape == x.shape
    assert float(n.max()) < 20
    assert float(n.min()) >= 1
    y2, n2 = f_iterate(contract, x, n_steps=3, halt_eps=0.0)
    assert (n2 == 3).all()


def test_f_iterate_per_sample_halt():
    from scripts.train_omni_probe import f_iterate

    x = torch.stack([torch.ones(3, 4, 4), torch.ones(3, 4, 4) * 4.0], 0)

    def shrink(z):
        return z * 0.5

    _, n = f_iterate(shrink, x, n_steps=10, halt_eps=0.3)
    # smaller start hits the floor first
    assert float(n[0]) < float(n[1])


def test_eval_score_prefers_identity():
    from scripts.train_omni_probe import eval_score

    low = {"t2i": {"digit_top1": 0.3, "psnr": 30.0}}
    high = {"t2i": {"digit_top1": 0.5, "psnr": 18.0}}
    assert eval_score(high, ["t2i"]) > eval_score(low, ["t2i"])
