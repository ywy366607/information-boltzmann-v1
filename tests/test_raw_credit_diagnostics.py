"""Direction diagnostics must not silently reinstate exact-gradient gates."""
import importlib.util
from pathlib import Path

import numpy as np


path = Path(__file__).resolve().parents[1] / 'scratch/bounded_causal_tags_20261108/analyze_raw_credit.py'
spec = importlib.util.spec_from_file_location('raw_credit_diagnostic', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(raw, reference, event=0):
    return dict(event=event, raw=np.asarray(raw), reference=np.asarray(reference),
                margin=-np.ones(6), error_bound=np.ones(6))


def test_large_magnitude_error_does_not_discard_direction_information():
    reference = np.array([1., 2., -3., 4., -5., 6.])
    result = audit.summarize([row(reference * .01, reference)])
    assert result['conservative_margin_pass_fraction'] == 0
    assert result['positive_pairing_fraction'] == 1
    assert result['raw_cosine']['mean'] == 1
    assert result['beats_all_three_wrong_fraction'] == 1


def test_mismatch_preserves_each_family_norm():
    raw = np.array([1., 2., 3., 4., 5., 6.])
    for wrong in audit.mismatched(raw):
        np.testing.assert_array_equal(np.sort(wrong[:4]), raw[:4])
        np.testing.assert_array_equal(np.sort(wrong[4:]), raw[4:])
        assert np.linalg.norm(wrong) == np.linalg.norm(raw)
        assert not np.array_equal(wrong, raw)


def test_zero_events_are_counted_and_not_turned_into_cosine_failures():
    result = audit.summarize([row(np.zeros(6), np.zeros(6)), row(np.zeros(6), np.ones(6), 1)])
    assert result['inactive_events'] == 2 and result['active_events'] == 0
    assert result['raw_cosine'] is None and result['positive_pairing_fraction'] is None
    assert result['coordinate_sign_agreement'] == 0
    assert result['nonzero_reference_coordinates'] == 6


def test_positive_pairing_alone_does_not_imply_receiver_selectivity():
    result = audit.summarize([row(np.arange(1., 7.), np.ones(6))])
    assert result['positive_pairing_fraction'] == 1
    assert result['beats_wrong_mean_fraction'] == 0
    negative = audit.summarize([row(-np.arange(1., 7.), np.arange(1., 7.))])
    assert negative['negative_pairing_events'] == 1
    assert negative['raw_cosine']['mean'] == -1


def test_optional_trace_observes_all_events_without_changing_fixture():
    fixture = audit.load_fixture()
    original = fixture.integrated_case()
    trace = []
    observed = fixture.integrated_case(credit_sink=trace.append)
    for key in original:
        if key != 'elapsed_seconds':
            assert original[key] == observed[key]
    assert len(trace) == fixture.STEPS
    assert all(len(r['raw']) == fixture.OWNERS for r in trace)
    assert all((r['margin'] <= 0).all() for r in trace)
