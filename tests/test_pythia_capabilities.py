"""B2 bridge: Pythia binds to the proven reconstruction/segmentation chart."""
from __future__ import annotations

import numpy as np
import torch

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    CAPABILITY_CHAMPION_PATH,
    capability_champion_kwargs,
)
from scripts.train_pythia_capabilities import (
    current_gate,
    edit_gate,
    make_static_bank,
    paired_digit_likelihood,
    sample_digit_group,
    static_one_step,
)


def _model() -> DualStreamOmni:
    return DualStreamOmni(
        d_model=32,
        n_slices=8,
        n_layers=1,
        n_heads=4,
        res=8,
        surprise_mode="v1_bayes",
        s_update="raw",
        prior_loss_coef=0.0,
        vfe_coef=0.0,
        use_stiefel=False,
        deslice_topk=0,
        prior_write=1.0,
        prior_write_by_t=False,
        pixel_loss_mode="balanced_bce",
        spatial_prompt_vocab=True,
        capability_vocab=True,
        use_modal_precision=True,
        use_target_time=True,
        seg_classes=2,
        s0_acc_coef=0.0,
    )


def test_capability_bridge_uses_proven_static_graph():
    kw = capability_champion_kwargs()
    assert CAPABILITY_CHAMPION_PATH.name == "northstar_slice_capability_best.pt"
    assert kw["use_modal_precision"] is True
    assert kw["use_target_time"] is True
    assert kw["seg_classes"] == 2
    assert kw["prior_write"] == 1.0
    assert "use_active_gdn2" not in kw or kw["use_active_gdn2"] is False


def test_current_gate_requires_reconstruction_and_source_dependence():
    good = {
        "digit_top1": 0.90,
        "color_acc": 0.95,
        "paired_iou": 0.90,
        "seg_iou": 0.95,
        "flood": 0.02,
        "source_iou_gap": 0.40,
    }
    assert current_gate(good)
    bad = dict(good, source_iou_gap=0.0)
    assert not current_gate(bad)
    bad = dict(good, paired_iou=0.20)
    assert not current_gate(bad)


def test_edit_gate_requires_geometry_color_and_source_dependence():
    good = {
        "digit_top1": 0.80,
        "color_acc": 0.92,
        "paired_iou": 0.70,
        "seg_iou": 0.97,
        "flood": 0.02,
        "source_digit_gap": 0.60,
    }
    assert edit_gate(good)
    assert not edit_gate(dict(good, source_digit_gap=0.0))
    assert not edit_gate(dict(good, color_acc=0.50))
    assert not edit_gate(dict(good, paired_iou=0.20))


def test_current_step_trains_language_readers_not_visual_core():
    torch.manual_seed(0)
    model = _model()
    names = set(model.set_optimization_phase("language"))
    bank = make_static_bank(8, "image_to_current")
    samples = bank[:16]
    loss, meta = static_one_step(
        model, samples, bank, np.random.default_rng(0),
        current_shuffle_coef=0.1,
    )
    assert torch.isfinite(loss)
    assert meta["case"] == "image_to_current"
    assert "source_shuffle" in meta
    assert meta["image_precision"] == 1.0
    assert meta["text_precision"] == 0.0
    loss.backward()
    assert any("surprise_gate.prior_head" in name for name in names)
    assert model.mot_stack.stem.weight.grad is None
    assert model.pix_head[-1].weight.grad is None
    assert model.seg_head[-1].weight.grad is None
    assert not model.mot_stack.stem.weight.requires_grad


def test_digit_group_is_identified_by_existing_likelihoods():
    bank = make_static_bank(8, "text_to_both")
    samples = sample_digit_group(bank, np.random.default_rng(2))
    assert {sample["digit"] for sample in samples} == {str(i) for i in range(10)}
    assert len({sample["target_color"] for sample in samples}) == 1
    assert len({sample["source_place"] for sample in samples}) == 1
    model = _model().eval()
    from fine_grain.capability_tasks import collate_capability

    batch = collate_capability(samples)
    with torch.no_grad():
        out = model(
            batch["image"], batch["prompt"], t=torch.zeros(10),
            image_precision=batch["image_precision"],
            text_precision=batch["text_precision"],
            target_time=batch["target_time"],
        )
        loss, meta = paired_digit_likelihood(out, batch)
    assert torch.isfinite(loss)
    assert meta["digit_group_active"] is True
    assert "digit_energy_gap" in meta
