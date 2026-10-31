"""Numerical contracts for interpreting acute policy contrasts, not capability tests."""
import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / 'scripts/ib/fly_interface/native_causal_audit.py'
_spec = importlib.util.spec_from_file_location('_native_decision_audit', _path)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def test_mother_tick_change_cannot_reverse_the_measured_child_decision_effect():
    a = [0.8, 0.8, 0.1, 0.1]
    b = [0.0, 0.0, 0.2, 0.2]
    result = _audit.decision_hazard_stats(a, b, [(1, 2, 2, 0)])
    assert sum(b) < sum(a)  # The full-clock contrast points the opposite way.
    assert result['ticks'] == 2
    assert result['delta_mean'] == pytest.approx(0.1)
    assert result['intact_mean'] == pytest.approx(0.1)


def test_missing_decisions_are_unmeasured_and_clock_mismatch_fails():
    assert _audit.decision_hazard_stats([.1], [.2], [])['delta_mean'] is None
    with pytest.raises(ValueError, match='outside the replay clock'):
        _audit.decision_hazard_stats([.1], [.2], [(0, 0, 2, 0)])
