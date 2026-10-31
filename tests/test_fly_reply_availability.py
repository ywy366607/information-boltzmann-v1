"""Reply-opportunity mathematics and durable event attribution contracts."""
import copy
import json
import math

import pytest

from information_boltzmann.core.fly_reply_availability import (
    ReplyAvailability, beta_occurrence_information,
)


def test_uniform_beta_expected_information_matches_the_closed_form():
    p, information = beta_occurrence_information(1, 1)
    assert p == .5
    assert information == pytest.approx(math.log(2) - .5, abs=2e-15)
    p2, i2 = beta_occurrence_information(2, 1)
    entropy = -p2 * math.log(p2) - (1 - p2) * math.log1p(-p2)
    assert i2 == pytest.approx(entropy - .5, abs=2e-15)


def test_repeat_no_replies_reduce_both_probability_and_opportunity_value():
    model = ReplyAvailability({})
    values, probabilities = [], []
    for _ in range(80):
        p, information = model.expected(False)
        probabilities.append(p)
        values.append(information + p * 2.)
        trial = model.begin(False)
        assert model.settle(False, trial) == information
    assert all(b < a for a, b in zip(probabilities, probabilities[1:]))
    assert all(b < a for a, b in zip(values, values[1:]))
    assert model.expected(False)[0] == pytest.approx(1 / 82)
    assert model.expected(True)[0] == .5
    assert model.state['settled_trials'] == 80


def test_each_yes_is_one_trial_not_each_word_and_duplicate_settlement_is_zero():
    model = ReplyAvailability({})
    trial = model.begin(True)
    before = model.expected(True)[1]
    assert model.settle(True, trial) == before
    saved = copy.deepcopy(model.state)
    assert model.settle(True, trial) == 0.
    assert model.settle(False, trial) == 0.
    assert model.settle(None, trial) == 0.
    assert model.state == saved
    assert model.expected(True)[0] == pytest.approx(2 / 3)
    assert model.state['current']['replied'] is True


def test_cumulative_ids_reject_a_stale_reply_for_the_newer_pending_turn():
    model = ReplyAvailability({})
    first = model.begin(False)
    model.settle(False, first)
    second = model.begin(True)
    assert second == first + 1
    before = copy.deepcopy(model.state)
    assert model.settle(True, first) == 0.
    assert model.settle(True, second + 1) == 0.
    assert model.state == before
    assert model.pending['id'] == second
    assert model.settle(True, second) > 0.


def test_pending_opportunity_survives_json_save_resume_without_restarting_the_trial():
    original = ReplyAvailability({})
    original.settle(False)  # No trial is fabricated by an outcome.
    trial = original.begin(True)
    saved = json.loads(json.dumps(original.state))
    restored = ReplyAvailability(saved)
    assert restored.pending['id'] == trial
    assert restored.state['trial_counter'] == trial
    assert restored.settle(True, trial) == original.settle(True, trial)
    assert restored.state == original.state
    assert restored.begin(False) == trial + 1


def test_infrastructure_failure_is_censored_without_reward_or_posterior_update():
    model = ReplyAvailability({})
    initial = copy.deepcopy(model.state['buckets'])
    trial = model.begin(True)
    assert model.settle(None, trial) == 0.
    assert model.state['buckets'] == initial
    assert model.state['censored_trials'] == 1
    assert model.state['settled_trials'] == 0
    assert model.pending is None
    assert model.state['current']['status'] == 'censored'
    assert model.state['current']['information_awarded'] == 0.
    assert model.begin(True) == trial + 1


def test_unknown_or_unbegun_reply_cannot_create_an_observation():
    model = ReplyAvailability({})
    saved = copy.deepcopy(model.state)
    assert model.settle(True) == 0.
    assert model.settle(False, 30) == 0.
    assert model.state == saved


def test_pending_async_request_cannot_be_overwritten_or_counted_as_absent():
    model = ReplyAvailability({})
    trial = model.begin(False)
    saved = copy.deepcopy(model.state)
    with pytest.raises(ValueError, match='unresolved'):
        model.begin(True)
    assert model.state == saved
    assert model.pending['id'] == trial
    assert model.expected(False) == beta_occurrence_information(1, 1)


def test_has_spoken_is_the_callers_semantic_flag_not_a_nonzero_token_count():
    model = ReplyAvailability({})
    quiet = model.begin(False)  # Can be many raw boundary/whitespace tokens.
    assert model.pending['bucket'] == 'quiet'
    model.settle(False, quiet)
    spoken = model.begin(True)
    assert model.pending['bucket'] == 'spoken'
    model.settle(True, spoken)
    with pytest.raises(ValueError, match='boolean'):
        model.begin(1)
    with pytest.raises(ValueError):
        model.settle(0)


@pytest.mark.parametrize('alpha,beta', [(1, 100), (65, 97), (1000, 2000)])
def test_harmonic_information_matches_direct_integer_formula(alpha, beta):
    p, information = beta_occurrence_information(alpha, beta)
    harmonic = lambda n: math.fsum(1 / k for k in range(1, n + 1))
    entropy = -p * math.log(p) - (1 - p) * math.log1p(-p)
    direct = entropy - harmonic(alpha + beta) + p * harmonic(alpha) + (1 - p) * harmonic(beta)
    assert information == pytest.approx(direct, rel=0., abs=2e-15)


def test_large_counts_keep_small_positive_information_without_log_cancellation():
    p, information = beta_occurrence_information(10 ** 12, 10 ** 12)
    assert p == .5
    assert math.isfinite(information)
    assert information == pytest.approx(1 / (4 * 10 ** 12), rel=1e-11)
    _, skewed = beta_occurrence_information(1, 10 ** 12)
    assert 0. < skewed < 1e-12


def test_diagnostics_are_inspection_copies_and_keep_the_last_settled_result():
    model = ReplyAvailability({'unrelated': 'preserved'})
    trial = model.begin(True)
    model.settle(True, trial)
    report = model.diagnostics()
    report['current']['status'] = 'pending'
    report['buckets']['spoken']['alpha'] = 1000
    assert model.state['current']['status'] == 'settled'
    assert model.state['buckets']['spoken']['alpha'] == 2
    assert model.state['unrelated'] == 'preserved'


def test_incompatible_schema_is_rejected_without_resetting_the_individual():
    state = {'format': 'other-version', 'keep': 'original'}
    with pytest.raises(ValueError, match='migration'):
        ReplyAvailability(state)
    assert state == {'format': 'other-version', 'keep': 'original'}
