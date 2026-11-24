"""Finite randomization identities for the real 4-by-3 selectors; no training."""
import math

from scripts.audit_logged_choice_effect import propensity,effects


def row(g, *, choice='generate', phase=None):
    return dict(metal_g=g,choice=choice,choice_probability=[.2,.5,.3],phase=phase,
        habit_logits=[0.,0.,0.],
        joint_language_search=dict(selected=1,distribution=[.1,.2,.3,.4],candidates=['a','b','c','d']))


def test_factorial_expectation_and_predictable_baseline_cancel_exactly():
    pj = [.1,.2,.3,.4]
    total,actual = 0.,0.
    for j in range(4):
        pa = [[.2,.5,.3],[.5,.3,.2],[.3,.2,.5],[.4,.4,.2]][j]
        for a in range(3):
            reward = .1*j*a-.03*j+.02*a
            wj,_,_ = propensity(pj,j); wa,_,_ = propensity(pa,a)
            total += pj[j]*pa[a]*(reward-3.7)*(1-wa)*(1-wj)
            actual += (pj[j]-.25)*(pa[a]-1/3)*reward
    assert math.isclose(total,actual,abs_tol=1e-14)


def test_reader_uses_only_earlier_gains_for_its_baseline_and_real_horizon():
    rows = [row(.1),row(.2),row(.3)]
    result = effects(rows,horizon=2,discount=.8,block=2)
    wa = (1/3)/.5
    # First return .1+.8*.2; second .2+.8*.3, centered on past .1 only.
    expected = ((.1+.8*.2)+(.2+.8*.3-1.8*.1))*(1-wa)/2
    assert abs(result['fire']['mean_lift']-expected)<1e-12
    assert result['events']==2 and result['skipped']['future_not_observed']==1
    assert result['fire']['full_blocks']==1


def test_uniform_choices_have_zero_preference_effect_without_invented_success():
    rows = [row(.1),row(-.2),row(.4)]
    for r in rows:
        r['choice_probability']=[1/3]*3
        r['joint_language_search']['distribution']=[.25]*4
    result = effects(rows,block=2)
    for key in ('fire','draft','both','interaction'):
        assert result[key]['mean_lift']==0
    assert result['draft']['expected_maximally_coupled_choice_changes']==0


def test_missing_support_is_not_assigned_a_fake_propensity():
    r = row(.1)
    r['choice_probability']=[1.,0.,0.]
    result = effects([r])
    assert result['events']==0
    assert result['skipped']['missing_or_invalid_randomization']==1


def test_imagination_has_no_effect_when_actual_policy_is_the_habit():
    rows=[row(.1),row(.2),row(-.1)]
    for r in rows:
        r['habit_logits']=[math.log(p) for p in r['choice_probability']]
        r['imagination_offset']=[0.,0.,0.]
    result=effects(rows,block=2)
    assert abs(result['imagination']['mean_lift'])<1e-15
    assert abs(result['draft_imagination_interaction']['mean_lift'])<1e-15
