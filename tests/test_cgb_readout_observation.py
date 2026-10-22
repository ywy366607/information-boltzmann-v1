"""Likelihood/calibration numerics and one-way observation, not capability."""
import copy

import numpy as np
import pytest

from information_boltzmann.cgb.readout_observation import before_arrival_features, bridge_pool_port_current, objective, fit_readout, FrozenBitReadout
from scripts.ib.generative.compare_support_transport import same
from scripts.ib.generative.native_learning_stream import new_session
from information_boltzmann.cgb.mobile_exchange import MobileExchange


@pytest.mark.parametrize('diagonal',[False,True])
def test_objective_derivative_and_strictly_regularized_fit(diagonal):
    # Deterministic numerical vectors test the objective, not memory/data skill.
    x=np.arange(256,dtype=float).reshape(32,8)/100
    y=np.arange(32,dtype=np.int64)
    bits=((y[:,None]>>np.arange(8))&1).astype(float)
    theta=np.linspace(-.2,.2,16 if diagonal else 72)
    prior=np.linspace(-1,1,8)
    value,grad=objective(theta,x,bits,prior,.01,diagonal)
    for index in (0,len(theta)//2,len(theta)-1):
        a,b=theta.copy(),theta.copy();a[index]+=1e-5;b[index]-=1e-5
        difference=(objective(a,x,bits,prior,.01,diagonal)[0]-objective(b,x,bits,prior,.01,diagonal)[0])/2e-5
        assert difference==pytest.approx(grad[index],abs=1e-9)
    readout=fit_readout(x,y,diagonal=diagonal)
    assert readout.fit_info['converged'] and readout.fit_info['gradient_inf']<1e-6
    restored=FrozenBitReadout.from_state_dict(readout.state_dict())
    np.testing.assert_array_equal(readout.logits(x[0]),restored.logits(x[0]))


def test_feature_measurement_and_fitting_cannot_update_physics_or_see_next_target():
    m=MobileExchange([[1.,1.,1.],[1.8,1.,1.]],[3,3],[3,3],[4,4],[4,4],pre=[0],post=[1],port_nodes=[0,1]*4)
    session=new_session(m,b'actual prior');state=m.state_dict();old_session=copy.deepcopy(session)
    rows=[before_arrival_features(m,session,65) for _ in range(16)]
    fit=fit_readout([x['brain'] for x in rows],np.arange(16,dtype=np.int64))
    assert fit.fit_info['constant_features']==list(range(8))
    prediction=fit.logits(rows[0]['brain'])
    np.testing.assert_array_equal(prediction,fit.logits(before_arrival_features(m,session,65)['brain']))
    same(m.state_dict(),state);assert session==old_session


def test_invalid_or_corrupt_instruments_are_not_accepted():
    with pytest.raises(ValueError):fit_readout(np.zeros((2,8)),np.array([0,256]))
    fit=fit_readout(np.zeros((16,8)),np.arange(16,dtype=np.int64))
    state=fit.state_dict();state['scale'][0]=0.
    with pytest.raises(ValueError):FrozenBitReadout.from_state_dict(state)


def test_conditional_current_matches_actual_ou_limit_and_native_jump_generator():
    m=MobileExchange([[1.,1.,1.],[1.8,1.,1.]],[3,3],[3,3],[4,4],[4,4],pre=[0],post=[1],
        port_nodes=[0,1]*4,charge=[.4,-.2],world_charge=np.linspace(-.5,.5,8))
    state=m.state_dict()
    jump=np.zeros(m.n)
    for event in m.events():
        other=MobileExchange.from_state_dict(state)
        other.apply_event(event)
        jump+=m.event_rate(event)*(other.charge-m.charge)
    expected_field=bridge_pool_port_current(m)-jump[m.port_nodes]
    errors=[]
    class ZeroNoise:
        def normal(self,size=None):return 0. if size is None else np.zeros(size)
    for dt in (1e-3,1e-4,1e-5,1e-6):
        other=MobileExchange.from_state_dict(state);other.rng_field=ZeroNoise()
        other._field(dt,None)
        actual=(other.charge-m.charge)[m.port_nodes]/dt
        errors.append(np.max(np.abs(actual-expected_field)))
    assert errors[0]>errors[1]>errors[2]
    assert errors[-1]<1e-7
    same(m.state_dict(),state)
