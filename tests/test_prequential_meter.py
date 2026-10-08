"""Numerical accounting tests; these make no model capability claims."""
import json

import pytest

from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter


REFERENCE = {'train': 'train.npy', 'counts_sha256': 'fixed-training-only-counts'}


def test_full_totals_weight_targets_and_tail_is_bounded():
    meter = FirstPassPredictionMeter(100, 200, REFERENCE, recent_windows=2)
    meter.record([1., 3.], [4., 4.])
    meter.record([10.], [5.])
    meter.record([2., 4., 6.], [6., 6., 6.])
    row = meter.summary()
    assert row['first_pass_measured_train_targets'] == 6
    assert row['first_pass_train_nll'] == pytest.approx(26/6)
    assert row['first_pass_fixed_unigram_nll'] == pytest.approx(31/6)
    assert row['first_pass_gain_over_fixed_unigram'] == pytest.approx(5/6)
    assert row['recent_first_pass_measured_targets'] == 4
    assert row['recent_first_pass_train_nll'] == pytest.approx(22/4)
    assert len(meter.recent) == 2
    meter.verify_cursor(106, 206)


def test_resume_preserves_all_measured_history_without_inventing_earlier_scores():
    meter = FirstPassPredictionMeter(819136, 919136, REFERENCE)
    assert meter.summary()['first_pass_train_nll'] is None
    meter.record([3., 5.], [6., 6.])
    restored = FirstPassPredictionMeter.from_state_dict(
        json.loads(json.dumps(meter.state_dict())), REFERENCE)
    assert restored.summary() == meter.summary()
    for instance in (meter, restored):
        instance.record([2.], [4.])
        instance.verify_cursor(819139, 919139)
    assert restored.summary() == meter.summary()
    assert restored.targets == 3
    assert restored.summary()['first_pass_train_nll'] == pytest.approx(10/3)


@pytest.mark.parametrize('scores,reference', [([], []), ([1.], []),
    ([float('nan')], [2.]), ([1.], [float('inf')]), ([-1.], [2.])])
def test_rejected_scores_leave_meter_unchanged(scores, reference):
    meter = FirstPassPredictionMeter(0, 0, REFERENCE)
    before = meter.state_dict()
    with pytest.raises(ValueError):
        meter.record(scores, reference)
    assert meter.state_dict() == before


def test_cursor_check_rejects_missing_or_replay_accounting():
    meter = FirstPassPredictionMeter(100, 200, REFERENCE)
    meter.record([1., 2.], [3., 4.])
    with pytest.raises(ValueError):
        meter.verify_cursor(104, 202)
    with pytest.raises(ValueError):
        meter.verify_cursor(102, 204)


def test_resume_rejects_different_reference_or_corrupt_tail():
    meter = FirstPassPredictionMeter(0, 0, REFERENCE, recent_windows=1)
    meter.record([2.], [3.])
    with pytest.raises(ValueError):
        FirstPassPredictionMeter.from_state_dict(meter.state_dict(), {'train': 'validation.npy'})
    state = meter.state_dict()
    state['recent'] = [[2, 3., 4.]]
    with pytest.raises(ValueError):
        FirstPassPredictionMeter.from_state_dict(state, REFERENCE)
