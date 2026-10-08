"""Numerical bookkeeping checks for recorded, pre-update lifelong evaluation."""
import numpy as np
import pytest

from scripts.ib.summarize_fly_live_timing import summarize_encounter, build_summary


def recorded_row(ticks=2):
    return {'bptt_train_tokens': 32, 'B_curve': [2.]*8,
            'A1_curve': [100., 5., 5., 5., 5., 5., 5., 5.],
            'A2_replay_curve': [0., 4., 4., 4., 3., 3., 3., 3.],
            'fresh_validation_cursor': 8, 'physical_ticks_per_input': ticks,
            'event_interval': [100, 116], 'update_interval': [25, 29],
            'physical_tick_interval': [200, 200+16*ticks]}


def test_summary_excludes_bridge_and_separates_return_update():
    row = summarize_encounter(recorded_row(), np.zeros(8, dtype=int), np.array([3.]), window=4)
    revisit = row['revisit']
    assert revisit['matched_nll_gain'] == pytest.approx(11/7)
    assert revisit['first_return_window_preupdate_nll_gain'] == 1.
    assert revisit['first_return_window_matched_pairs'] == 3
    assert revisit['intervening_events_before_first_matched_pair'] == 9
    assert revisit['intervening_physical_ticks_before_first_matched_pair'] == 18
    assert row['first_pass']['gain_over_fixed_unigram'] == 1.


def test_summary_rejects_fabricated_exposure_or_updates():
    for field, value in [('event_interval', [100, 117]),
                         ('physical_tick_interval', [200, 231]),
                         ('update_interval', [25, 28])]:
        source = recorded_row()
        source[field] = value
        with pytest.raises(ValueError):
            summarize_encounter(source, np.zeros(8, dtype=int), np.array([3.]), window=4)


def test_historical_timing_is_not_counted_as_repaired_evidence():
    old, new = recorded_row(1), recorded_row(2)
    old['bptt_train_tokens'], new['bptt_train_tokens'] = 16, 32
    config = {'window': 4, 'physical_ticks_per_input': 2,
              'origin': {'timing_migration': {'bptt_train_tokens': 16}}}
    result = build_summary([old, new], config, {'bptt_train_tokens': 32},
                           np.zeros(8, dtype=int), np.array([3.]))
    assert result['new_timing_validation']['encounters'] == 1
    assert result['new_timing_validation']['fresh_targets'] == 8
    assert result['new_timing_fresh_training_updates'] == 4
    assert result['capability_assessment'] == 'insufficient_joint_updates'


def test_clock_migration_does_not_inherit_old_phase_validation():
    old, new = recorded_row(), recorded_row()
    old['bptt_train_tokens'], new['bptt_train_tokens'] = 100, 132
    new['writer_baseline_clock'] = 'input'
    config = {'window': 4, 'physical_ticks_per_input': 2, 'writer_baseline_clock': 'input',
              'origin': {'writer_clock_migration': {'bptt_train_tokens': 100}}}
    result = build_summary([old, new], config, {'bptt_train_tokens': 132},
                          np.zeros(8, dtype=int), np.array([3.]))
    phase = result['current_implementation_phase']
    assert result['new_timing_validation']['encounters'] == 2
    assert phase['encounters'] == 1
    assert phase['fresh_validation_targets'] == 8
    assert phase['fresh_training_updates'] == 8
    assert result['encounters'][0]['writer_baseline_clock'] == 'physical'
