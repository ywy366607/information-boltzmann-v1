import pytest

from scripts.train_mixed_native import capability_failures


def text_row():
    return dict(id="qa", task="it2t", decoded=dict(exact=True, eos=True),
                need_pix=False, image_shuffle_nll_gap=1.0, text_shuffle_nll_gap=1.0)


def test_language_gate_requires_both_modalities_despite_perfect_answer():
    good = text_row()
    assert capability_failures({"samples": [good]}) == []
    bad = {**good, "text_shuffle_nll_gap": 0.0}
    assert capability_failures({"samples": [bad]}) == ["qa:question_used"]
    bad = {**good, "image_shuffle_nll_gap": 0.0}
    assert capability_failures({"samples": [bad]}) == ["qa:image_used"]


def test_sparse_generation_gate_does_not_accept_blank_high_psnr():
    row = dict(id="ink", task="t2i", family="generation1px", psnr=33,
               rgb_stroke_iou=0.0, seg_iou=0.0, prompt_stroke_iou_gap=0.0)
    assert capability_failures({"samples": [row]}) == ["ink:rgb_stroke", "ink:seg_stroke", "ink:prompt_used"]


def test_admission_cannot_silently_omit_required_cases():
    assert capability_failures({"samples": [text_row()]}, ["qa", "ocr"]) == ["ocr:missing_result"]
    with pytest.raises(ValueError, match="nonempty"):
        capability_failures({"samples": []})
    with pytest.raises(ValueError, match="unique"):
        capability_failures({"samples": [text_row(), text_row()]})
    with pytest.raises(ValueError, match="registered"):
        capability_failures({"samples": [{"id": "unknown", "task": "other"}]})
