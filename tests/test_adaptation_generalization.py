"""Measurement identities and censoring, not trained capability studies."""
import math
import pytest
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary


def summarize(values, **kwargs):
    return recovery_summary(values, block_tokens=1, hold_blocks=2, **kwargs)['generalization']


def test_exact_appl_gain_and_all_tokens_included():
    losses=[9.,8.,7.,6.,6.,6.,6.,6.,6.5]
    result=summarize(losses,reference_nll=[8.]*len(losses))
    assert result['appl_h']==pytest.approx(math.exp(sum(losses)/len(losses)))
    assert result['gain_ratio_over_reference']==pytest.approx(math.exp(8.-sum(losses)/len(losses)))
    assert result['scored_tokens']==len(losses)


def test_faster_recovery_and_lower_plateau_both_improve_appl():
    slow=summarize([10.]*4+[6.]*6)
    fast=summarize([10.]*2+[6.]*8)
    better=summarize([10.]*2+[5.]*8)
    assert better['appl_h']<fast['appl_h']<slow['appl_h']
    assert fast['plateau_nll']==6.
    assert fast['recovery_tokens']<slow['recovery_tokens']


def test_constant_good_stream_has_zero_recovery_but_retains_quality():
    good,bad=summarize([4.]*10),summarize([8.]*10)
    assert good['recovery_tokens']==bad['recovery_tokens']==0
    assert good['appl_h']<bad['appl_h']


def test_changing_tail_is_candidate_not_confirmed_platform():
    result=summarize([10.,9.,8.,7.,6.,5.,4.,3.])
    assert result['plateau_nll'] is None
    assert result['plateau_estimate_nll'] is not None
    assert result['recovery_tokens'] is None
    assert result['plateau_status']=='still_changing'
    assert result['appl_h']>0


def test_short_observation_still_has_primary_score():
    result=summarize([8.,7.])
    assert result['appl_h']==pytest.approx(math.exp(7.5))
    assert result['plateau_nll'] is None


def test_relapse_is_not_early_recovery():
    result=summarize([10.,6.,6.,10.,6.,6.,6.,6.])
    assert result['recovery_tokens']==6


def test_reference_is_matched_and_nonfinite_is_rejected():
    with pytest.raises(ValueError,match='exact same'):
        summarize([8.,7.],reference_nll=[8.])
    with pytest.raises(ValueError,match='Finite'):
        summarize([float('nan')])


def test_legacy_recovery_fields_stay_compatible():
    result=recovery_summary([10.,2.,9.,8.,3.,3.],block_tokens=1,hold_blocks=2)
    assert result['half_recovery_tokens']==6
    assert result['legacy_half_recovery']
    assert 'appl_h' in result['generalization']


def test_ag_requires_confirmed_recovery_and_uses_same_tail_reference():
    censored = summarize([10., 10., 10., 9., 6., 5., 5., 6.], reference_nll=[8.] * 8)
    assert censored['plateau_nll'] == 5.5
    assert censored['recovery_tokens'] is None
    assert 'ag' not in censored
    result = summarize([10., 9., 6., 6., 6., 6., 6., 6.],
                       reference_nll=[20., 20., 8., 8., 8., 8., 8., 8.])
    assert result['plateau_reference_nll'] == 8.
    assert result['q_quality'] == pytest.approx(1 / (1 + math.exp(-2)))
