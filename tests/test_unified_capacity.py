import pytest
import torch

from fine_grain.unified_capacity import extend_unified_bank


@pytest.mark.parametrize("res", [64, 256])
def test_full_bank_additions_preserve_modal_boundaries_and_native_ink(res):
    originals = [dict(id=key, target_rgb=torch.full((3, res, res), float(i)),
                      target_file=f"source-{i}.png")
                 for i, key in enumerate(("freedom-t2i-17808", "freedom-t2i-32449"))]
    bank = extend_unified_bank(originals, res)
    assert len(bank) == 28 and len(originals) == 2
    by_id = {s["id"]: s for s in bank}
    for s in bank[2:]:
        assert s["history_precision"].sum() == 0 and s["target_time"] == 0
        assert s["image"].shape == (3, res, res)
        if s["task"] == "it2t":
            image_control = by_id[s["qa_image_control_id"]]
            text_control = by_id[s["qa_text_control_id"]]
            assert image_control["prompt"] == s["prompt"] and image_control["answer"] != s["answer"]
            assert not torch.equal(image_control["image"], s["image"])
            assert torch.equal(text_control["image"], s["image"])
            assert text_control["prompt"] != s["prompt"] and text_control["answer"] != s["answer"]
            assert s["image_precision"] == 1 and not s["text_missing"]
        if s["task"] == "t2t":
            assert not s["need_pix"] and s["image_precision"] == 0 and s["image"].sum() == 0
        if s.get("family") == "ocr1px":
            assert s["text_missing"] and s["image_precision"] == 1
            assert 3 <= s["target_seg"].sum() < 100
        if s.get("family") == "generation1px":
            assert s["image_precision"] == 0 and s["image"].sum() == 0
            assert s["target_rgb"][1:].sum() == 0
            assert torch.equal(s["target_rgb"][0], s["target_seg"].float())
            assert 3 <= s["target_seg"].sum() < 100
    with pytest.raises(ValueError, match="unique"):
        extend_unified_bank(bank, res)
