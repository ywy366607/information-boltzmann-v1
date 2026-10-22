"""Matched physical intervention interfaces, not a memory-capacity study."""
import numpy as np
import pytest

from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.memory_intervention import matched_trace_condition, assert_matched_trace_states
from scripts.ib.generative.compare_support_transport import same
from test_cgb_growing_material import growing
from scripts.ib.generative.analyze_trace_fate import shared_fate


def test_condition_uses_allowed_reactions_with_same_charge_resources_and_rngs():
    source=growing()
    keep,erase=fork_material(source),fork_material(source)
    state=source.state_dict()
    keys={'i:0:1':0,'p:0':1}
    a=matched_trace_condition(keep,[0,1],keys,erase=False)
    b=matched_trace_condition(erase,[0,1],keys,erase=True)
    assert a['applied']==b['applied']==2
    assert_matched_trace_states(keep,erase)
    assert a['delta_u']+a['reaction_heat_out']==pytest.approx(0.,abs=1e-14)
    assert b['delta_u']+b['reaction_heat_out']==pytest.approx(0.,abs=1e-14)
    assert a['delta_u']-b['delta_u']==pytest.approx(sum(x['trace_energy_if_kept'] for x in a['receipts']))
    assert keep.trace.sum()+keep.port_trace.sum()==2
    assert erase.trace.sum()+erase.port_trace.sum()==0
    for name in ('charge','charged','depleted','bound','port_bound'):
        np.testing.assert_array_equal(getattr(keep,name),getattr(erase,name))
    same(source.state_dict(),state)


def test_finite_local_composition_limits_both_conditions_without_clipping():
    source=growing()
    # Move original local D to P, preserving its conserved inventory and
    # explicitly account the corresponding energy as diagnostic work.
    u,g=source.energy(),source.free_energy()
    source.charged+=source.depleted
    source.depleted[:]=0
    du,dg=source.energy()-u,source.free_energy()-g
    source.work_in+=du
    source.entropy_balance+=du-dg
    keep,erase=fork_material(source),fork_material(source)
    a=matched_trace_condition(keep,[0,1],{'i:0:1':0,'p:0':1},erase=False)
    b=matched_trace_condition(erase,[0,1],{'i:0:1':0,'p:0':1},erase=True)
    assert a['applied']==b['applied']==0
    same(keep.state_dict(),source.state_dict())
    same(erase.state_dict(),source.state_dict())


def test_mismatched_states_are_not_called_memory_specific():
    m=growing();other=fork_material(m)
    other.charge[0]+=.1
    with pytest.raises(AssertionError):
        assert_matched_trace_states(m,other)
    with pytest.raises(ValueError):
        matched_trace_condition(m,[0,0],{},erase=True)


def test_parent_history_is_not_redefined_by_erasing_or_rebuilding_a_contact():
    parent={'a':dict(carrier_id=0,trace=1,bound=1),'b':dict(carrier_id=1,trace=1,bound=0)}
    live={'a':dict(carrier_id=0,trace=0,bound=1),'b':dict(carrier_id=2,trace=1,bound=0)}
    result=shared_fate(parent,live)
    assert result['groups']['traced']==dict(initial=2,retained=1,current_trace=0,current_bound=1)
    assert result['retained']==result['rebuilt_old_positions']==1
