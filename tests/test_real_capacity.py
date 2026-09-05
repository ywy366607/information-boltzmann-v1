import json

import numpy as np
from PIL import Image
import torch

from fine_grain.real_capacity import add_zero_horizon_controls, build_real_bank, letterbox_foreground
from fine_grain.omni_model import DualStreamOmni
from test_pythia_omni import _tiny_causal_lm


def test_annotation_letterbox_preserves_discrete_foreground(tmp_path):
    pixels = np.zeros((8, 16), dtype=np.uint8)
    pixels[2:6, 5:10] = 3
    path = tmp_path / "mask.png"
    Image.fromarray(pixels).save(path)
    mask = letterbox_foreground(path, 32)
    assert mask.shape == (32, 32)
    assert set(mask.unique().tolist()) == {0, 1}
    assert mask[:8].sum() == 0
    assert mask[12:20, 10:20].sum() == 80


def test_token_future_target_cannot_leak_into_predictive_pass():
    torch.manual_seed(7)
    model = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=2, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="raw", use_stiefel=False,
        prior_write=1.0, seg_classes=2, use_modal_precision=True,
        use_target_time=True, use_task_tokens=True, control_prefix_attention=True,
        gaussian_head_layout="per_head", history_size=2,
        use_active_gdn2=True, active_gdn2_initial_trust=0.1,
        lm=_tiny_causal_lm(),
    ).eval()
    image = torch.rand(1, 3, 8, 8)
    history = torch.rand(1, 2, 3, 8, 8)
    ids = torch.zeros(1, 1, dtype=torch.long)
    kwargs = dict(
        score_tokens=False, image_precision=torch.ones(1),
        text_precision=torch.zeros(1, 1), target_time=torch.ones(1),
        history_images=history, history_precision=torch.ones(1, 2),
        task_id=torch.tensor([3]), visual_prompt_mask=torch.zeros_like(ids, dtype=torch.bool),
    )
    target = torch.rand_like(image)
    first = model.forward_tokens_with_future_posterior(image, ids, torch.ones_like(ids), target, **kwargs)
    second = model.forward_tokens_with_future_posterior(image, ids, torch.ones_like(ids), 1 - target, **kwargs)
    torch.testing.assert_close(first["rgb"], second["rgb"], rtol=0, atol=0)
    assert not torch.allclose(first["transition_q_mu"], second["transition_q_mu"])
    (first["belief_mu"].square().mean() + first["transition_q_rgb"].square().mean()).backward()
    assert all(p.grad is None for p in model.lm.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if not n.startswith("lm."))


def test_real_bank_boundaries_identity_edits_and_temporal_order(tmp_path):
    images = []
    for i in range(4):
        path = tmp_path / f"frame{i}.png"
        Image.fromarray(np.full((12, 16, 3), 30 + i * 50, dtype=np.uint8)).save(path)
        images.append(str(path))
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:8, 4:11] = 1
    Image.fromarray(mask).save(mask_path)
    rows = []
    for task, ids in {
        "t2i": ("freedom-t2i-17808", "freedom-t2i-32449"),
        "it2i": ("freedom-it2i-34244", "freedom-it2i-24875"),
        "i2t": ("opengv-1", "opengv-4"),
    }.items():
        for key in ids:
            rows.append({"id": key, "task": task, "prompt": "Edit the picture.",
                         "answer": "A real caption sentence. Another sentence.",
                         "source_files": [] if task == "t2i" else [images[0]],
                         "target_file": images[1]})
    share = tmp_path / "share.json"
    share.write_text(json.dumps({"records": rows}))
    davis = tmp_path / "davis.json"
    davis.write_text(json.dumps({"records": [
        {"sequence": sequence, "frame": i, "image": images[i], "mask": str(mask_path)}
        for sequence in ("blackswan", "camel") for i in range(3)
    ]}))
    bank = build_real_bank(share, davis, 32)
    assert len(bank) == 14
    for row in bank:
        if row["task"] == "future":
            assert row["history_frames"] == [0, 1] and row["target_frame"] == 2
            torch.testing.assert_close(row["image"], row["history_images"][-1])
            assert not torch.equal(row["target_rgb"], row["image"])
            assert row["text_missing"] and row["target_time"] == 1
        else:
            assert row["history_precision"].sum() == 0
        if row["id"].endswith("-identity"):
            assert row["task_id"] == 2
            torch.testing.assert_close(row["image"], row["target_rgb"])
        if row["task"] == "t2i":
            assert row["image_precision"] == 0 and row["image"].sum() == 0
        if row["task"] == "i2t":
            assert row["answer"] == "A real caption sentence."
    mixed_bank = add_zero_horizon_controls(bank)
    assert len(bank) == 14 and len(mixed_bank) == 16
    for control in mixed_bank[14:]:
        original = next(s for s in bank if s["id"] == control["id"].removesuffix("-tau0"))
        observed = next(s for s in bank if s["task"] == "segmentation"
                        and s["sequence"] == control["sequence"])
        assert control["task_id"] == original["task_id"] == 3
        assert control["target_time"] == 0 and original["target_time"] == 1
        assert control["target_frame"] == 1 and original["target_frame"] == 2
        for key in ("image", "history_images", "history_precision"):
            torch.testing.assert_close(control[key], original[key], rtol=0, atol=0)
        torch.testing.assert_close(control["target_rgb"], control["image"], rtol=0, atol=0)
        torch.testing.assert_close(control["target_seg"], observed["target_seg"], rtol=0, atol=0)
        assert control["target_file"] == observed["target_file"]
        assert control["mask_file"] == observed["mask_file"]
