from scripts.train_sharegpt4o_i2t_curriculum import (
    compact_answer,
    make_fixed_ocr_samples,
)


def test_compact_answer_keeps_one_bounded_sentence():
    text = "The image shows a red bird on a branch. More details follow."
    assert compact_answer(text, max_words=6) == "The image shows a red bird."


def test_compact_answer_removes_markdown_and_adds_period():
    assert compact_answer("**Golden logo** on white", 3) == "Golden logo on."


def test_compact_answer_stops_after_quoted_sentence():
    text = 'An ad called "Whitening Strips." These strips are quick.'
    assert compact_answer(text, 12) == 'An ad called "Whitening Strips."'


def test_fixed_ocr_bank_has_one_hard_1px_sample_per_digit():
    samples = make_fixed_ocr_samples(32, 7)
    assert [sample["answer"] for sample in samples] == list("0123456789")
    assert all(sample["image"].shape == (3, 32, 32) for sample in samples)
    assert all(sample["source_group"] == "ocr_1px" for sample in samples)
