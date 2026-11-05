"""Report arithmetic checks, without training or environment simulation."""
import pytest

from scripts.ib_local.compare_navigation_settlement import room_opportunities, signal_checks


def test_opportunity_probability_counts_candidate_mass_not_number_of_visits():
    rows = [dict(candidate_rooms=[0, 2, 0], choice_probability=[.2, .5, .3],
                 world={'anchor': 2}),
            dict(candidate_rooms=[0, 0], choice_probability=[.4, .6],
                 world={'anchor': 0})]
    result = room_opportunities(rows, 0)
    assert result['decisions'] == 1
    assert result['policy_probability'] == .5
    assert result['uniform_candidate_probability'] == 2 / 3
    assert result['actual_fraction'] == 0


def test_no_available_opportunity_is_missing_not_zero_probability():
    result = room_opportunities([], 1)
    assert result['decisions'] == 0 and result['policy_probability'] is None


def test_delayed_signal_check_rejects_old_reward_hidden_as_continuation():
    row = dict(progress=.02, settlement_kind='continuation', settlement=.02,
               metal_g=.02, usable_signal=.02, witness_g=-.01,
               witness_pre=.5, witness_post=.51, wood_target=[0, 0, 0, .2],
               interoception=[0, 2., 0], td_previous=dict(g=.1, y=.1, objective_boundary=True))
    assert signal_checks([row], 'continuation')['checked_rounds'] == 1
    row['metal_g'] = -.01
    with pytest.raises(AssertionError):
        signal_checks([row], 'continuation')


def test_joint_delivery_uses_joint_not_cached_progress():
    row = dict(progress=-.001, joint_progress=.36, settlement_kind='continuation_joint',
               settlement=.36, metal_g=.36, usable_signal=.36, witness_g=-.01,
               witness_pre=.5, witness_post=.51, wood_target=[0, 0, 0, 3.6],
               interoception=[0, 36., 0], td_previous=dict(g=.1, y=.1, objective_boundary=True))
    assert signal_checks([row], 'continuation_joint')['checked_rounds'] == 1
    row['joint_progress'] = row['progress']
    with pytest.raises(AssertionError):
        signal_checks([row], 'continuation_joint')
