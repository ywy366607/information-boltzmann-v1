import pytest
import torch

from scripts.ib_local.credit_scale_diagnostics import (
    equal_layer_norms, magnitude, match_layer_norms, parameter_effect,
)


def test_wrong_direction_has_same_norm_per_layer_not_only_globally():
    ref = [torch.tensor([3., 4.]), torch.tensor([2.])]
    wrong = [torch.tensor([-1., 0.]), torch.tensor([10.])]
    result, _ = match_layer_norms(ref, wrong, {2:[0], 13:[1]})
    assert magnitude([result[0]]) == 5
    assert magnitude([result[1]]) == 2
    assert result[0][0] < 0


def test_equal_norm_amplification_preserves_direction_and_empty_credit():
    current = [torch.tensor([3., 4.]), torch.tensor([9.])]
    old = [torch.tensor([.03, .04]), torch.tensor([0.])]
    result, scales = equal_layer_norms(current, old, {2:[0], 13:[1]})
    torch.testing.assert_close(result[0], current[0])
    assert result[1].item() == 0 and scales['13'] == 0


def test_fp16_counts_are_actual_stored_parameter_differences():
    source = [torch.tensor([1., 1.], dtype=torch.float16)]
    held = [source[0].clone()]
    full = [torch.tensor([1.0001, 1.002], dtype=torch.float16)]
    score = parameter_effect(held, full, source, 100)
    assert score['changed_coordinates'] == 1
    assert score['fraction_trainable_coordinates'] == .5
    assert score['fraction_whole_model'] == .01
    assert score['difference_to_reference_step_norm'] is None


def test_no_nonzero_importance_created_from_zero_direction():
    with pytest.raises(ValueError):
        match_layer_norms([torch.ones(2)], [torch.zeros(2)], {2:[0]})
