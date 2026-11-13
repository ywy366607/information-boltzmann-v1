"""Numerical contracts for occurrence-calibrated historical read increments."""
from types import SimpleNamespace

import numpy as np
import torch

from scripts.ib_local.navigation_direct_read import (
    DirectReadGainEngine, causal_prefix, gain_forecast, gain_state, observe_gain)
from scripts.ib_local.language_navigation_kanli_v4i import FilmEngine


def test_first_observation_has_bias_corrected_error_scale():
    state=gain_state(.95)
    f=gain_forecast(state,np.zeros(3),np.ones(3),np.ones(3,dtype=bool))
    observe_gain(state,1,2.,f)
    nxt=gain_forecast(state,np.zeros(3),np.ones(3),np.ones(3,dtype=bool))
    assert state['count'].tolist()==[0,1,0]
    np.testing.assert_allclose(nxt['rms'],[0.,2.,0.])
    assert not nxt['alpha'].any()


def test_unrelated_observations_do_not_age_participating_evidence():
    state=gain_state(.95)
    f=gain_forecast(state,np.zeros(3),np.ones(3),np.array([False,True,False]))
    before={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    for _ in range(100):observe_gain(state,0,100.,f)
    for k in state:np.testing.assert_array_equal(state[k],before[k])


def test_shrinkage_can_help_even_when_unshrunk_read_is_worse():
    # E[(target-base)*delta]=1, E[delta^2]=10. Raw loss rises by 8,
    # whereas alpha=.1 lowers loss by .1. R-squared clipped at zero misses it.
    state=gain_state(.95)
    state['cross'][:]=1.;state['second'][:]=10.
    state['count'][:]=20;state['error'][:]=1.
    f=gain_forecast(state,np.zeros(3),np.ones(3),np.ones(3,dtype=bool))
    np.testing.assert_allclose(f['alpha'],.1)
    assert 10-2>0 and 10*.1**2-2*.1<0


def test_normalized_choice_increment_is_invariant_to_positive_target_units():
    states=[gain_state(.95),gain_state(.95)]
    for i in range(30):
        base=np.array([.3,.5,.7])+i*.01
        delta=np.array([.2,-.1,.4]);target=.8+i*.005
        fs=[]
        for scale,state in zip((1.,7.),states):
            f=gain_forecast(state,scale*base,scale*delta,np.ones(3,dtype=bool))
            fs.append(f);observe_gain(state,i%3,scale*target,f)
        np.testing.assert_allclose(fs[0]['alpha'],fs[1]['alpha'],atol=1e-14)
        np.testing.assert_allclose(fs[0]['offset'],fs[1]['offset'],atol=1e-14)
        np.testing.assert_allclose(fs[0]['mean']*7,fs[1]['mean'],atol=1e-14)


def test_unavailable_channels_zero_variance_and_negative_alignment_are_neutral():
    state=gain_state(.95);state['count'][:]=5
    state['cross'][:]=[1.,-1.,1.];state['second'][:]=[0.,1.,1.]
    f=gain_forecast(state,np.zeros(3),np.ones(3),np.array([True,True,False]))
    assert not f['alpha'].any() and not f['offset'].any()


def test_default_choice_hook_exactly_preserves_legacy_rule():
    engine=SimpleNamespace(precision=lambda:.37)
    values=torch.tensor([.8,-.1,4.]);available=torch.tensor([True,True,False])
    old=(.37*(values-values[available].mean())).masked_fill(~available,0.)
    torch.testing.assert_close(FilmEngine._choice_offset(engine,values,available),old,rtol=0,atol=0)


def test_new_increment_reaches_choice_independently_of_old_critic_precision():
    engine=object.__new__(DirectReadGainEngine)
    engine.precision=lambda:0.
    engine._direct_read_offset=torch.tensor([.2,-.2,0.])
    values=torch.tensor([1.,2.,3.]);available=torch.tensor([True,True,False])
    engine.episodic_wood_mode='real'
    actual=engine._choice_offset(values,available)
    torch.testing.assert_close(actual,engine._direct_read_offset,rtol=0,atol=0)
    engine.episodic_wood_mode='blocked'
    assert not engine._choice_offset(values,available).any()


def test_reading_or_replaying_forecasts_does_not_count_as_a_new_outcome():
    state=gain_state(.95)
    for _ in range(100):gain_forecast(state,np.zeros(3),np.ones(3),np.ones(3,dtype=bool))
    assert not state['count'].any() and not state['cross'].any()


def test_prefix_credit_uses_only_fixed_boundary_and_actual_prior_bytes():
    weight=torch.tensor(.2,requires_grad=True);seen=[]
    class Model:
        def predict_observed(self,ids,state,logp):
            seen.append((ids.item(),logp.shape))
            state=state+weight*ids.item()
            return state.expand(1,1,4),state,{},state.reshape(1,1),None
    e=SimpleNamespace(model=Model(),recurrent=torch.tensor(1.,requires_grad=True),pending=2,device='cpu')
    record=dict(probs=[np.ones(256,dtype=np.float32)/256]*3)
    features,logits=causal_prefix(e,torch.tensor([[3,4,99,88]]),3,record)
    assert [x[0] for x in seen]==[2,3,4]
    assert features.shape==(3,1) and logits.shape==(1,3,4)
    logits.sum().backward()
    assert weight.grad!=0 and e.recurrent.item()==1. and e.recurrent.grad is None


def test_silent_prefix_uses_its_observed_boundary_probability():
    class Model:
        def predict_observed(self,ids,state,logp):
            assert ids.item()==7
            torch.testing.assert_close(logp.exp(),torch.full((1,1,256),1/256))
            return torch.zeros(1,1,256),state,{},torch.zeros(1,2),None
    e=SimpleNamespace(model=Model(),recurrent=torch.tensor(1.),pending=7,device='cpu')
    causal_prefix(e,torch.tensor([[99]]),0,dict(probs=[],boundary_probability=np.ones(256,dtype=np.float32)/256))


def test_matching_batch_policy_keeps_exact_existing_gradient_object(monkeypatch):
    from scripts.ib_local.navigation_causal_wood import CausalWoodEngine
    policy=torch.tensor(1.,requires_grad=True)
    result=(policy,dict(choice=[.1,.2,.3],stops=[]))
    monkeypatch.setattr(CausalWoodEngine,'_policy_logp',lambda *a:result)
    e=object.__new__(DirectReadGainEngine)
    record=dict(choice_logits=[.1,.2,.3],habit_logits=[.1,.2,.3],stop_logits=[])
    rng=torch.get_rng_state().clone()
    assert e._policy_logp(None,None,None,0,record) is result
    assert not e._policy_replay_fallback['used']
    assert torch.equal(torch.get_rng_state(),rng)
