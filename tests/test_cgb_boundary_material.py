"""The same finite reaction graph at real electrical boundary terminals."""
from dataclasses import replace

import numpy as np
import pytest

from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.identity_material import IdentityMaterialProgram,IdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from information_boltzmann.cgb.exchange_checkpoint import save_exchange,load_exchange
from information_boltzmann.cgb.sensory_channel import frozen_field_channel
from information_boltzmann.cgb.learning_observation import native_learning_drift
from scripts.ib.generative.compare_support_transport import same


def material(n=1,c=0,r=0):
    program=IdentityMaterialProgram(dict(code=np.ones((1,1)),occurrence_affinity=np.zeros((1,1)),
        occurrence_out=np.zeros(1),occurrence_in=np.zeros(1),strength_affinity=np.zeros((1,1)),
        strength_out=np.zeros(1),strength_in=np.zeros(1)),['numerical'],{})
    return BoundaryIdentityMobileExchange([[1.,1.,1.],[1.8,1.,1.]], [3,3],[3,3],[4,4],[4,4],
        carried=2,end_capacity=4,pre=[0],post=[1],bound=0,trace=1,charge=[.4,-.2],
        port_nodes=[0],world_charge=[-.3],program=program,cell_type=[0,0],recognition_energy_fraction=.5,
        port_installed=n,port_bound=c,port_trace=r,seed=46)


@pytest.mark.parametrize('n,c,r',[(0,0,0),(1,0,0),(1,1,0),(1,0,1),(1,1,1)])
def test_every_boundary_event_uses_joint_energy_and_reverse_rate(n,c,r):
    m=material(n,c,r)
    events=[e for e in m.events() if e.interface>=0]
    assert events
    for event in events:
        current=BoundaryIdentityMobileExchange.from_state_dict(m.state_dict())
        u,g=current.energy(),current.free_energy()
        du,dg=current.event_delta(event)
        rate=current.event_rate(event)
        current.apply_event(event)
        assert abs(current.energy()-u-du)<2e-13
        assert abs(current.free_energy()-g-dg)<2e-13
        reverse=replace(event,sign=-event.sign)
        assert reverse in current.events()
        assert abs(np.log(rate/current.event_rate(reverse))+dg)<2e-13
        for error in current.balances().values():
            assert abs(float(error))<2e-13


def test_growth_and_capture_draw_from_existing_local_compositions():
    m=material(0,0,0)
    old_b=m.material.copy();old_p=m.charged.copy();old_d=m.depleted.copy()
    m.apply_event(MaterialEvent('assembly',0,0,0,interface=0))
    assert m.material[0]==old_b[0]-1 and m.charged[0]==old_p[0]-1 and m.depleted[0]==old_d[0]+1
    assert m.port_installed[0]==1
    before=m.free[0]
    m.apply_event(MaterialEvent('binding',0,0,0,interface=0))
    assert m.port_bound[0]==1 and m.free[0]==before-1
    q=m.charge.copy();w=m.world_charge.copy()
    m.apply_event(MaterialEvent('write',0,0,0,interface=0))
    assert m.port_trace[0]==1
    assert m.charge[0]==q[0]+m.params.charge_per_event
    assert m.world_charge[0]==w[0]-m.params.charge_per_event
    assert MaterialEvent('assembly',0,0,0,-1,interface=0) not in m.events()


def test_empty_boundary_adds_no_conductance_and_migration_preserves_old_state():
    m=material(0,0,0)
    np.testing.assert_array_equal(m.effective_port_conductance(),m.port_conductance)
    state=m.state_dict()
    for key in ('port_installed','port_bound','port_trace'):
        state['arrays'].pop(key)
    for key in ('boundary_reactions_enabled','boundary_conduction_enabled','field_scheme'):
        state.pop(key)
    state.update(kind='identity_mobile_exchange',schema=1)
    old=IdentityMobileExchange.from_state_dict(state)
    migrated=BoundaryIdentityMobileExchange.from_identity(old)
    for name in old._arrays:
        np.testing.assert_array_equal(getattr(old,name),getattr(migrated,name))
    for name in old._ledgers:
        assert getattr(old,name)==getattr(migrated,name)
    assert old.energy()==migrated.energy() and old.free_energy()==migrated.free_energy()
    migrated.boundary_reactions_enabled=False
    for _ in range(16):
        old.step(2.,np.array([2.]))
        migrated.step(2.,np.array([2.]))
    for name in old._arrays:
        np.testing.assert_array_equal(getattr(old,name),getattr(migrated,name))
    for name in old._ledgers:
        assert getattr(old,name)==getattr(migrated,name)
    assert old.rng_reaction.bit_generator.state==migrated.rng_reaction.bit_generator.state


def test_trace_and_material_change_actual_passive_body_channel():
    m=material(1,0,0)
    baseline=m.effective_port_conductance().copy()
    channel=frozen_field_channel(m,[2.])
    m.apply_event(MaterialEvent('write',0,0,0,interface=0))
    assert m.effective_port_conductance()[0]>baseline[0]
    changed=frozen_field_channel(m,[2.])
    assert np.linalg.norm(changed['gain']-channel['gain'])>0.
    before=m.state_dict()
    sham=BoundaryIdentityMobileExchange.from_state_dict(before)
    blocked=BoundaryIdentityMobileExchange.from_state_dict(before)
    blocked.boundary_conduction_enabled=False
    assert blocked.energy()==m.energy() and blocked.free_energy()==m.free_energy()
    m._field(2.,np.array([3.]));sham._field(2.,np.array([3.]));blocked._field(2.,np.array([3.]))
    same(m.state_dict(),sham.state_dict())
    assert np.linalg.norm(m.charge-blocked.charge)>0.
    for obj in (m,blocked):
        assert max(abs(float(v)) for v in obj.balances().values())<1e-12


def test_native_generator_counts_installed_boundary_contacts_and_gates():
    m=material(1,0,0)
    report=native_learning_drift(m)
    expected={key:0. for key in report['totals']}

    def observables(obj):
        return dict(support=float(obj.bound.sum()+obj.port_bound.sum()),
            trace=float(obj.trace.sum()+obj.port_trace.sum()),bridges=float(len(obj.pre)+obj.port_installed.sum()),
            conductance=float(obj.bridge_conductance().sum()+obj.effective_port_conductance().sum()),
            charged=float(obj.charged.sum()))
    before=observables(m)
    for event in m.events():
        other=BoundaryIdentityMobileExchange.from_state_dict(m.state_dict())
        other.apply_event(event)
        after=observables(other)
        for key in expected:expected[key]+=m.event_rate(event)*(after[key]-before[key])
    for key in expected:
        assert abs(expected[key]-report['totals'][key]['net'])<1e-13


def test_boundary_archive_restores_storage_barriers_and_complete_future(tmp_path):
    m=material(1,1,1);m.field_scheme='symmetric_ou'
    for _ in range(4):m.step(2.,np.array([2.]))
    save_exchange(tmp_path/'boundary.npz',m,{'cursor':4})
    other,meta=load_exchange(tmp_path/'boundary.npz')
    assert meta['cursor']==4
    assert other.field_scheme=='symmetric_ou'
    for _ in range(16):
        m.step(2.,np.array([-2.]));other.step(2.,np.array([-2.]))
    same(m.state_dict(),other.state_dict())
    state=m.state_dict();state['arrays']['port_installed'][:]=0;state['arrays']['port_trace'][:]=1
    with pytest.raises(ValueError):BoundaryIdentityMobileExchange.from_state_dict(state)


def test_native_matched_jumps_isolate_memory_without_an_unpaid_reset():
    source=material(1,1,1)
    write=BoundaryIdentityMobileExchange.from_state_dict(source.state_dict())
    active=BoundaryIdentityMobileExchange.from_state_dict(source.state_dict())
    write_event=MaterialEvent('write',0,0,0,-1,interface=0)
    active_event=replace(write_event,kind='active')
    dw=write.event_delta(write_event)[0];da=active.event_delta(active_event)[0]
    write.apply_event(write_event);active.apply_event(active_event)
    for name in source._arrays:
        if name!='port_trace':np.testing.assert_array_equal(getattr(write,name),getattr(active,name))
    assert write.port_trace[0]==0 and active.port_trace[0]==1
    expected=-(source.params.trace_energy-source.params.capture_coupling)
    assert abs(dw-da-expected)<1e-13
    unbind=MaterialEvent('binding',0,0,0,-1,interface=0)
    assert write.event_rate(unbind)>active.event_rate(unbind)
    assert write.effective_port_conductance()[0]<active.effective_port_conductance()[0]
    for model in (write,active):
        assert max(abs(float(v)) for v in model.balances().values())<1e-12
