"""Exact numerical filter checks, not capability experiments."""
import pytest

from scripts.ib.audit_fly_event_clock import writer_clock_coefficients


@pytest.mark.parametrize('quiet_ticks', [0, 1, 3])
@pytest.mark.parametrize('clock', ['physical', 'input'])
def test_baseline_fixed_point_matches_its_exact_event_recurrence(quiet_ticks, clock):
    decay = .9
    result = writer_clock_coefficients(decay, quiet_ticks, clock)
    baseline = 0.0
    for _ in range(1000):
        baseline = decay * baseline + (1 - decay)
        if clock == 'physical':
            baseline *= decay ** quiet_ticks
    assert baseline == pytest.approx(result['stationary_baseline_fraction_of_constant_packet'])
    if clock == 'input' or quiet_ticks == 0:
        assert baseline == pytest.approx(1.0)
    else:
        assert result['stationary_innovation_fraction_of_constant_packet'] > 0


def test_one_quiet_tick_leaves_half_of_constant_packet_at_input_pulse():
    result = writer_clock_coefficients(.9512, 1, 'physical')
    assert result['stationary_innovation_fraction_of_constant_packet'] == pytest.approx(1 / 1.9512)
    assert writer_clock_coefficients(.9512, 1, 'input')[
        'stationary_innovation_fraction_of_constant_packet'] == pytest.approx(0.0)
