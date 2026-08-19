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


def test_f_generate_iterates():
    from scripts.train_omni_probe import f_generate

    torch.manual_seed(0)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True, prior_write=1.0, prior_write_by_t=False,
    )
    z = torch.randn(1, 3, 8, 8)
    y = f_generate(m, z, ["Draw digit 2 with a thin red stroke"], [True], n_steps=2)
    assert y.shape == z.shape
    # pred_loss exists on the forward but must not be the noisy-S term in the loss.


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
