"""Numerical explanatory counterexamples, not synthetic capability claims."""
import torch

from scripts.ib_local.audit_relation_cause import minimal_models, forecast_audit
from scripts.ib_local.causal_relation_program import posterior


def test_same_history_can_hide_opposite_causal_models_and_crossing_resolves_it():
    model=minimal_models()['nonidentification']
    assert model['observation_max_difference']==0
    assert model['mechanism_1_action_slope']!=model['mechanism_2_action_slope']
    assert model['crossed_observation_rank']==2
    assert model['crossed_action_predictions'][1]-model['crossed_action_predictions'][0]>1.9


def test_factual_candidate_margin_does_not_identify_action_contrast_erasure():
    model=minimal_models()['factual_margin_counterexample']
    assert model['margin_before']>0>model['margin_after']
    assert model['mean_mse_after']<model['mean_mse_before']
    assert model['action_difference_before']==model['action_difference_after']


def test_perfect_residuals_retain_fixed_prior_noise_and_radial_action_artifact():
    model=minimal_models()
    assert model['residual_prior_floor']['residual_mse']==0
    assert model['residual_prior_floor']['noise']==[.2]
    assert abs(model['radial_leak']['invented_action_difference'])>.28
    a,b=model['radial_leak']['condition_norm_fixed_prediction']
    assert a==b


def test_actual_nullspace_witness_preserves_fit_but_reverses_query_contrast():
    x=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]],dtype=torch.float64)
    y=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
    q=torch.tensor([[.5,.5,.5,0.],[.5,.5,-.5,0.]],dtype=torch.float64)
    d=posterior(x,y,q,.1)
    d['base']=torch.zeros(2,2,dtype=torch.float64)
    d['query']=q
    values=forecast_audit(d,torch.zeros(2),0)['same_history_opposite_action_operators']
    for v in values:
        assert abs(v['achieved_contrast']-v['desired_contrast'])<1e-12
        assert v['historical_prediction_change']<1e-12
