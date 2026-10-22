"""Native constitutive identities; no synthetic predictive-capability task."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.native_port import native_port_constitution
from scripts.ib.generative.compare_support_transport import same
from test_cgb_growing_material import growing


def prepare_charge(m,brain=None,world=None):
    u,g=m.energy(),m.free_energy()
    if brain is not None:
        delta=np.asarray(brain)-m.charge[m.port_nodes]
        m.charge[m.port_nodes]=brain
        m.electrode_charge-=float(delta.sum())
    if world is not None:
        delta=np.asarray(world)-m.world_charge
        m.world_charge[:]=world
        m.source_charge+=float(delta.sum())
    du,dg=m.energy()-u,m.free_energy()-g
    m.work_in+=du;m.entropy_balance+=du-dg
    assert max(abs(float(v)) for v in m.balances().values())<1e-10


def test_interface_uses_hidden_native_state_not_current_sensory_charge():
    m=growing();state=m.state_dict();p=m.port_constitution()
    other=fork_material(m)
    prepare_charge(other,[1.7],[-1.4])
    q=native_port_constitution(other)
    np.testing.assert_allclose(p.body_current,q.body_current,atol=2e-15)
    np.testing.assert_allclose(p.match_sensation()['matching_world_voltage'],q.match_sensation()['matching_world_voltage'],atol=1e-13)
    result=p.match_sensation()
    assert result['sensor_status']==['unique_finite']
    np.testing.assert_allclose(p.sensory_current(result['matching_world_voltage'])+p.body_current,0.,atol=2e-14)
    assert np.linalg.eigvalsh(p.fixed_carrier_admittance).min()>=-1e-15
    assert np.all(p.phase_admittance_diagonal_bounds>=0)
    same(m.state_dict(),state)


def test_virtual_neutral_rates_match_direct_native_preparation_and_feedback():
    m=growing();p=m.port_constitution()
    neutral=fork_material(m)
    prepare_charge(neutral,np.zeros(len(m.port_nodes)),np.zeros(len(m.port_nodes)))
    body=np.zeros(neutral.n);sensor=np.zeros(len(neutral.port_nodes))
    for event in neutral.events():
        rate=neutral.event_rate(event)
        candidate=fork_material(neutral);candidate.apply_event(event)
        change=candidate.charge-neutral.charge
        if event.interface>=0:sensor+=rate*change[neutral.port_nodes]
        else:body+=rate*change
    v=neutral.charge/neutral.capacitance
    for e,(i,j) in enumerate(zip(neutral.pre,neutral.post)):
        flow=neutral.bridge_conductance()[e]*(v[i]-v[j]);body[i]-=flow;body[j]+=flow
    np.testing.assert_allclose(p.body_passive_current+p.body_reaction_current,body[neutral.port_nodes],rtol=1e-13,atol=2e-16)
    np.testing.assert_allclose(p.sensory_current(np.zeros(1)),sensor,rtol=1e-13,atol=2e-16)
    # Fixed-carrier tangent is measured from the actual reaction law, not
    # the interface's own virtual evaluator.
    def actual_internal(voltage):
        candidate=fork_material(neutral)
        prepare_charge(candidate,voltage*candidate.capacitance[candidate.port_nodes])
        total=np.zeros(candidate.n)
        vv=candidate.charge/candidate.capacitance
        for e,(i,j) in enumerate(zip(candidate.pre,candidate.post)):
            flow=candidate.bridge_conductance()[e]*(vv[i]-vv[j]);total[i]-=flow;total[j]+=flow
        for event in candidate.events():
            if event.interface>=0:continue
            for node,delta in candidate._charge_changes(event):
                if node<candidate.n:total[node]+=candidate.event_rate(event)*delta
        return total[candidate.port_nodes]
    eps=1e-5
    derivative=(actual_internal(np.array([eps]))-actual_internal(np.array([-eps])))/(2*eps)
    np.testing.assert_allclose(derivative,-p.fixed_carrier_admittance[:,0],rtol=1e-8,atol=1e-11)


def test_neutral_birth_mean_is_zero_but_noise_and_recovery_are_retained():
    m=growing();prepare_charge(m,[0.])
    nodes,weights=hermgauss(7);weights=weights/np.sqrt(np.pi)
    samples=[];ratios=[]
    for charge in nodes*np.sqrt(m.capacitance[0]):
        candidate=fork_material(m)
        receipt=candidate.birth_trial(0,[1.,1.8,1.],float(charge),[0,1,1,1,2],0.,insertion_index=2)
        assert receipt['accepted']
        samples.append(candidate.charge[candidate.port_nodes[0]])
        ratios.append(receipt['log_birth_acceptance_ratio'])
    assert abs(np.dot(weights,samples))<1e-15
    assert np.dot(weights,np.square(samples))==pytest.approx(m.capacitance[0]/2,abs=1e-15)
    np.testing.assert_allclose(ratios,ratios[0],atol=2e-14,rtol=0)
    m.birth_trial(0,[1.,1.8,1.],.3,[0,1,1,1,2],0.,insertion_index=2)
    p=m.port_constitution()
    neutral=fork_material(m);prepare_charge(neutral,[0.])
    expected=0.
    for row in neutral.domain_generator_rates()['inverse_channels']:
        if row['parent']==neutral.port_nodes[0]:
            expected+=row['accepted_rate']*neutral.charge[row['child']]
    assert expected!=0.
    assert p.body_recovery_current[0]==pytest.approx(expected,rel=1e-12,abs=1e-30)
    assert p.phase_receipts and np.all(p.phase_admittance_diagonal_bounds>=0)


def test_active_sensing_can_remain_identifiable_without_passive_leak():
    template=growing()
    m=BoundaryIdentityMobileExchange([[1.,1.,1.]],[2],[2],[4],[4],
        port_nodes=[0],port_installed=1,port_trace=1,port_bound=1,port_conductance=0.,
        boundary_conduction_enabled=False,program=template.program,cell_type=[0],recognition_energy_fraction=.5,
        params=replace(template.params,conversion_rate=0.))
    result=m.port_constitution().match_sensation()
    assert result['sensor_status']==['unique_finite']
    assert abs(result['current_residual'][0])<1e-13
    m.boundary_reactions_enabled=False
    assert m.port_constitution().match_sensation()['sensor_status']==['non_unique']
    m.params=replace(m.params,conversion_rate=.1)
    assert m.port_constitution().match_sensation()['sensor_status']==['unreachable']


def test_material_reindexing_preserves_constitution_and_duplicate_roles_fail():
    m=growing();m.birth_trial(0,[1.,1.8,1.],.3,[0,1,1,1,2],0.,insertion_index=0)
    before=m.port_constitution()
    permutation=np.array([2,0,1]);inverse=np.argsort(permutation)
    for name in m._node_arrays:
        value=getattr(m,name)[permutation].copy()
        if not getattr(m,name).flags.writeable:value.setflags(write=False)
        setattr(m,name,value)
    m.pre=inverse[m.pre];m.post=inverse[m.post];m.port_nodes=inverse[m.port_nodes]
    after=m.port_constitution()
    np.testing.assert_allclose(before.body_current,after.body_current,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(before.match_sensation()['matching_world_voltage'],after.match_sensation()['matching_world_voltage'],rtol=1e-13,atol=1e-13)
    m.port_nodes=np.r_[m.port_nodes,m.port_nodes]
    with pytest.raises(ValueError,match='Distinct'):native_port_constitution(m)
