"""Joint polarized configuration/charge thermodynamics and complete state."""
from dataclasses import replace
import math

import numpy as np
import pytest

from information_boltzmann.cgb.charge_storage import select_storage_constitution
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange,MaterialEvent
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import same
from test_cgb_mobile_exchange import material,opposite
from test_cgb_growing_material import growing
from test_cgb_native_port import prepare_charge


@pytest.mark.parametrize('c,r',[(0,0),(0,1),(1,0),(1,1)])
def test_all_polarized_events_use_actual_energy_and_reverse_state(c,r):
    m=material(c,r,polarized_storage=True)
    for event in m.events():
        n=MobileExchange.from_state_dict(m.state_dict())
        u,g=n.energy(),n.free_energy()
        du,dg=n.event_delta(event);forward=n.event_rate(event)
        before=n.storage_charge_partition()
        n.apply_event(event)
        assert n.energy()-u==pytest.approx(du,abs=4e-14)
        assert n.free_energy()-g==pytest.approx(dg,abs=1e-13)
        reverse=opposite(event,len(m.pre))
        assert reverse in n.events()
        assert math.log(forward/n.event_rate(reverse))==pytest.approx(-dg,abs=3e-14)
        after=n.storage_charge_partition()
        if event.kind in ('write','thermal'):
            np.testing.assert_allclose(after['mobile_brain'],before['mobile_brain'],atol=4e-16,rtol=0)
            assert np.any(after['bound_brain']!=before['bound_brain'])
        assert max(abs(float(v)) for v in n.balances().values())<1e-10


@pytest.mark.parametrize('r',[0,1])
def test_boundary_bound_dipole_and_world_field_share_one_transition(r):
    m=growing();select_storage_constitution(m,True)
    if r==0:
        m.apply_event(next(e for e in m.events() if e.interface==0 and e.kind=='thermal' and e.sign==-1))
    event=next(e for e in m.events() if e.interface==0 and e.kind=='thermal')
    before=m.storage_charge_partition();u,g=m.energy(),m.free_energy()
    du,dg=m.event_delta(event);rate=m.event_rate(event)
    m.apply_event(event)
    after=m.storage_charge_partition()
    for key in ('mobile_brain','mobile_world'):
        np.testing.assert_allclose(before[key],after[key],atol=2e-16,rtol=0)
    assert after['bound_world'][0]-before['bound_world'][0]==-event.sign*m.params.charge_per_event
    assert m.energy()-u==pytest.approx(du,abs=2e-14)
    assert m.free_energy()-g==pytest.approx(dg,abs=2e-14)
    assert math.log(rate/m.event_rate(replace(event,sign=-event.sign)))==pytest.approx(-dg,abs=2e-14)


def test_thermal_field_sensitivity_is_reciprocal_not_a_new_timing_rule():
    m=material(r=0,polarized_storage=True)
    event=next(e for e in m.events() if e.kind=='thermal' and e.edge==0)
    du,dg=m.event_delta(event)
    n=fork_material(m)
    n.charge[event.i]+=.25*n.capacitance[event.i]
    assert n.event_delta(event)[1]-dg==pytest.approx(-.25*m.params.charge_per_event)
    old=material(r=0)
    before=old.event_delta(event)[1]
    old.charge[event.i]+=.25*old.capacitance[event.i]
    assert old.event_delta(event)[1]==before


def test_partition_reindexes_with_paid_domains_and_restores_actual_future(tmp_path):
    m=growing();select_storage_constitution(m,True)
    initial=m.storage_charge_partition()
    receipt=m.birth_trial(0,[1.,1.8,1.],.3,[0,1,1,1,2],0.,insertion_index=0)
    assert receipt['accepted']
    partition=m.storage_charge_partition()
    np.testing.assert_array_equal(partition['bound_brain'][1:],initial['bound_brain'])
    assert partition['bound_brain'][0]==0.
    path=tmp_path/'polarized.npz';save_exchange(path,m,{'constitution':'polarized'})
    n,_=load_exchange(path)
    assert n.params.polarized_storage and n.state_dict()['params']['polarized_storage']
    for _ in range(8):
        m.step(2.,np.array([3.]));n.step(2.,np.array([3.]))
    same(m.state_dict(),n.state_dict())
    assert max(abs(float(v)) for v in m.balances().values())<1e-10


def test_new_constitution_is_explicit_and_legacy_payload_is_unchanged():
    m=material()
    legacy=m.state_dict()
    assert 'polarized_storage' not in legacy['params']
    before=m.energy(),m.free_energy()
    receipt=select_storage_constitution(m,True)
    assert receipt['net_state_energy_change']==0
    assert (m.energy(),m.free_energy())==before
    select_storage_constitution(m,False)
    same(m.state_dict(),legacy)
    with pytest.raises(ValueError):
        replace(m.params,polarized_storage=1)


def test_native_port_includes_polarized_thermal_charge_feedback():
    m=growing();select_storage_constitution(m,True)
    native=m.port_constitution()
    assert any(e.kind=='thermal' for e in native.sensor_jumps)
    neutral=fork_material(m)
    prepare_charge(neutral,np.zeros(len(m.port_nodes)),np.zeros(len(m.port_nodes)))
    total=np.zeros(m.n);sense=np.zeros(len(m.port_nodes))
    for event in neutral.events():
        rate=neutral.event_rate(event)
        for node,dq in neutral._charge_changes(event):
            if node>=neutral.n:continue
            if event.interface>=0:sense[event.interface]+=rate*dq
            else:total[node]+=rate*dq
    v=neutral.charge/neutral.capacitance
    for edge,(i,j) in enumerate(zip(neutral.pre,neutral.post)):
        current=neutral.bridge_conductance()[edge]*(v[i]-v[j])
        total[i]-=current;total[j]+=current
    np.testing.assert_allclose(native.body_current,total[m.port_nodes],atol=2e-15,rtol=1e-13)
    np.testing.assert_allclose(native.sensory_current(np.zeros(len(m.port_nodes))),sense,atol=2e-15,rtol=1e-13)
