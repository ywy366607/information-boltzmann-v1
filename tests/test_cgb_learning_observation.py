"""Read-only native-learning/evidence interfaces, not capability studies."""
import numpy as np
import pytest

from information_boltzmann.cgb.mobile_exchange import MobileExchange, MobileExchangeParams
from information_boltzmann.cgb.learning_observation import (
    port_bit_logits, byte_nll, native_learning_drift, parameter_quench)
from information_boltzmann.cgb.exchange_checkpoint import save_exchange,load_exchange


def panel():
    xyz=np.stack(np.meshgrid(*([np.array([1.,1.8])]*3),indexing='ij'),axis=-1).reshape(-1,3)
    return MobileExchange(xyz,np.full(8,3),np.full(8,2),np.full(8,4),np.full(8,2),
        pre=[0,1,2],post=[1,3,6],bound=[0,1,0],trace=[1,0,1],carried=1,
        charge=np.linspace(-.3,.3,8),port_nodes=np.arange(8),seed=94)


def same_physical(a,b):
    for name in a._arrays+('pre','post','capacity'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_energy_difference_gives_normalized_target_free_byte_likelihood():
    m=panel()
    before=m.state_dict()
    logits=port_bit_logits(m)
    v=m.charge[m.port_nodes]/m.capacitance[m.port_nodes]
    c=1/(1/m.capacitance[m.port_nodes]+1/m.world_capacitance)
    plus=.5*c*(3-v)**2
    minus=.5*c*(-3-v)**2
    np.testing.assert_allclose(logits,minus-plus,atol=1e-15)
    probabilities=np.exp([-byte_nll(logits,b) for b in range(256)])
    assert abs(probabilities.sum()-1.)<1e-14
    assert np.isfinite(byte_nll(np.array([-1000.,1000.]*4),0))
    after=MobileExchange.from_state_dict(before)
    same_physical(m,after)
    for rng in ('rng_reaction','rng_motion','rng_field'):
        assert getattr(m,rng).bit_generator.state==getattr(after,rng).bit_generator.state


def test_learning_generator_matches_actual_allowed_event_observable_jumps():
    m=panel();before=m.state_dict()
    observed=native_learning_drift(m)
    expected={k:0. for k in ('support','trace','bridges','conductance','charged')}
    def values(obj):
        return dict(support=float(obj.bound.sum()),trace=float(obj.trace.sum()),
            bridges=float(len(obj.pre)),conductance=float(obj.bridge_conductance().sum()),
            charged=float(obj.charged.sum()))
    old=values(m)
    for event in m.events():
        clone=MobileExchange.from_state_dict(before)
        rate=m.event_rate(event)
        clone.apply_event(event)
        new=values(clone)
        for name in expected:expected[name]+=rate*(new[name]-old[name])
    for name,value in expected.items():
        assert abs(observed['totals'][name]['net']-value)<1e-14
        row=observed['totals'][name]
        assert row['positive']>=0 and row['negative']<=0
        assert abs(row['net']-row['positive']-row['negative'])<1e-14
    same_physical(m,MobileExchange.from_state_dict(before))
    assert observed['by_reaction']['binding']['support_drift']!=0.
    assert observed['by_reaction']['transport']['rate']>0.


def test_parameter_quench_pays_work_and_retains_all_history():
    m=panel();before=MobileExchange.from_state_dict(m.state_dict())
    work0=m.work_in
    expected=m.params.capture_coupling*float(np.dot(m.trace,m.bound))
    # Include an actually occupied mutual configuration for a nonzero quench.
    if expected==0:
        m.trace[1]=1
        m.initial_energy,m.initial_free_energy=m.energy(),m.free_energy()
        before=MobileExchange.from_state_dict(m.state_dict())
        expected=m.params.capture_coupling*float(np.dot(m.trace,m.bound))
    record=parameter_quench(m,capture_coupling=0.)
    same_physical(m,before)
    assert record['external_work']==expected
    assert m.work_in-work0==expected
    for value in m.balances().values():assert abs(float(value))<1e-12
    parameter_quench(m,capture_coupling=before.params.capture_coupling)
    assert m.work_in==before.work_in
    assert m.energy()==before.energy()


def test_frozen_storage_control_keeps_fast_physical_exchange_running():
    m=panel();before=MobileExchange.from_state_dict(m.state_dict())
    parameter_quench(m,mobility=0.,assembly_rate=0.,binding_rate=0.,transport_rate=0.,
        release_rate=0.,write_rate=0.,recovery_rate=0.)
    for _ in range(4):m.step(1.,np.array([3.,-3.]*4))
    for name in ('coordinates','pre','post','bound','trace'):
        np.testing.assert_array_equal(getattr(m,name),getattr(before,name))
    # Reservoir exchange remains a physical process; freezing plasticity
    # does not imply freezing every molecular coordinate of the material.
    assert m.material.sum()==before.material.sum()
    assert np.linalg.norm(m.charge-before.charge)>0.
    for value in m.balances().values():assert abs(float(value))<1e-11


def test_bad_observations_and_unaccounted_interventions_rejected():
    m=panel();before=m.state_dict()
    with pytest.raises(ValueError):byte_nll(np.zeros(8),256)
    with pytest.raises(ValueError):port_bit_logits(m,amplitude=-1.)
    with pytest.raises(ValueError):parameter_quench(m,end_capacity=9)
    with pytest.raises(ValueError):parameter_quench(m,capture_coupling=-1.)
    assert m.params==MobileExchangeParams(**before['params'])


def test_stream_cursor_and_fixed_statistics_restore_with_physical_future(tmp_path):
    m=panel()
    meta=dict(next_byte=12,phase='validation',fixed_counts=[1]*256,source_only_voltage=[.1]*8,
              scored=12,nll_sum=64.2,byte_ms=4.,dt_ms=2.)
    save_exchange(tmp_path/'stream.npz',m,{'session':meta})
    restored,p=load_exchange(tmp_path/'stream.npz')
    assert p['session']==meta
    for byte in (65,66,67):
        assert byte_nll(port_bit_logits(m),byte)==byte_nll(port_bit_logits(restored),byte)
        target=np.array([3. if (byte>>bit)&1 else -3. for bit in range(8)])
        for _ in range(2):
            m.step(2.,target);restored.step(2.,target)
    same_physical(m,restored)
