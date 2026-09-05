import numpy as np
import pytest

from fine_grain.capability_tasks import capability_sample
from fine_grain.omni_tasks import GRID_PLACES
from fine_grain.gen_metrics import paired_digit_scores


@pytest.mark.parametrize("res", [16, 32, 64])
def test_perfect_registered_targets_pass_all_digits_and_addresses(res):
    rng = np.random.default_rng(12)
    for digit in map(str, range(10)):
        for place in GRID_PLACES:
            sample = capability_sample(rng, res, "text_to_both", digit=digit, color="red", place=place)
            score = paired_digit_scores(sample["target_rgb"], digit, "red", place)
            assert score["paired_digit_top1"] == 1, (res, digit, place, score)
            assert score["paired_digit_iou"] == 1


def test_blank_prediction_is_not_a_digit():
    rng = np.random.default_rng(1)
    s = capability_sample(rng, 32, "text_to_both", digit="0", color="red", place="middle_center")
    scores = paired_digit_scores(s["target_rgb"] * 0, "0", "red", "middle_center")
    assert scores["paired_digit_top1"] == 0


def test_cached_registered_templates_do_not_change_metric_results():
    from fine_grain.gen_metrics import _registered_templates

    _registered_templates.cache_clear()
    s = capability_sample(np.random.default_rng(1), 32, "text_to_both",
                          digit="7", color="yellow", place="top_right")
    first = paired_digit_scores(s["target_rgb"], "7", "yellow", "top_right")
    second = paired_digit_scores(s["target_rgb"], "7", "yellow", "top_right")
    assert first == second
    assert _registered_templates.cache_info().hits == 1
    assert first["paired_digit_top1"] == 1
