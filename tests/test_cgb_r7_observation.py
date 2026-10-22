"""Observation/likelihood invariants, not a synthetic capability study."""
import numpy as np
from scipy.optimize import check_grad

from scripts.ib.generative.r7_lifetime_constraints import mixture_objective, observed_intervals, score
from scripts.ib.generative.r7_observation_audit import trajectory_summary
from scripts.ib.generative.r7_generator_bridge import interval_mass,retention_sector
from information_boltzmann.cgb.mobile_exchange import MobileExchangeParams


def test_interval_event_and_right_censor_probabilities_partition_one():
    rates=np.array([.013,.72]);weights=np.array([.31,.69])
    event=np.sum(weights[:,None]*np.exp(-rates[:,None]*np.arange(60))
                 *(-np.expm1(-rates[:,None])))
    right=np.sum(weights*np.exp(-rates*60))
    assert abs(event+right-1.)<1e-14
    model=dict(rates_per_minute=rates.tolist(),weights_at_first_observation=weights.tolist())
    result=score(model,np.array([0.,59.]),np.array([1.,0.]))
    assert np.all(np.isfinite(result['per_track_nll']))


def test_mixture_likelihood_derivative_matches_interval_probability():
    d=np.array([0.,2.,10.,59.]);ended=np.array([1.,1.,1.,0.])
    theta=np.array([np.log(.02),np.log(.7),-.3])
    err=check_grad(lambda x:mixture_objective(x,d,ended)[0],
                   lambda x:mixture_objective(x,d,ended)[1],theta)
    assert err<1e-7


def test_initially_existing_tracks_retained_in_counts_not_incident_likelihood():
    records=[dict(first_frame=0,last_frame=59),dict(first_frame=2,last_frame=2),
             dict(first_frame=4,last_frame=59)]
    d,ended=observed_intervals([dict(records=records)])
    np.testing.assert_array_equal(d,[0.,55.])
    np.testing.assert_array_equal(ended,[1.,0.])
    counts=trajectory_summary(records)
    assert counts['counts'][0]==1 and counts['counts'][2]==2 and counts['counts'][-1]==2
    assert sum(counts['counts'])==sum(r['last_frame']-r['first_frame']+1 for r in records)


def test_actual_support_generator_preserves_killed_probability_accounting():
    transient,rates=retention_sector(MobileExchangeParams(),3)
    np.testing.assert_allclose(-transient.sum(axis=1),
        [rates['disappearance_unbound'],rates['disappearance_bound']],atol=1e-17)
    survival,event=interval_mass(transient,17.,.37)
    assert np.all(np.diff(survival)<=0) and np.all(event>=0)
    assert abs(event[:59].sum()+survival[59]-1)<1e-14
