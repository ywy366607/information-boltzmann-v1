"""Spatial source biochemistry/resource and continuation contracts only."""
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from test_fly_kc_dopamine import setup as receiver_setup
from information_boltzmann.core.fly_dan_release import (
    attach_dan_release,dan_state,advance_dan_release,reciprocal_arrival,dan_arrival,
    DAN_RELEASE_STATE_FIELDS,DAN_RELEASE_PARAMETER_NAMES,
)
from information_boltzmann.core.fly_kc_terminals import terminal_state
from information_boltzmann.core.fly_kc_dopamine import spatial_arrival
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights


def setup(tmp_path):
    m,s,_=receiver_setup(tmp_path)
    m.kcd_post=torch.tensor([0,2]);path=tmp_path/'dan_source.npz'
    data=dict(dan_index=np.array([0,2]),node_ids=np.array([[0,10],[0,11],[2,10]]),
        owner=np.array([0,0,1]),volume=np.ones(3),u=np.array([0]),v=np.array([1]),g=np.ones(1),
        site_ids=np.array([[0,0,0,0],[0,1,0,0],[2,0,0,0]]),site_node=np.array([0,1,2]),site_owner=np.array([0,0,1]),
        in_edge=np.array([0,1]),in_pre=np.array([3,4]),in_post=np.array([0,2]),
        in_delay=np.array([1,2]),in_site=np.array([0,2]),in_node=np.array([0,2]),in_fraction=np.ones(2),
        out_edge=np.array([0,0,1]),out_pre=np.array([0,0,2]),out_post=np.array([3,3,4]),
        out_delay=np.array([1,1,2]),out_site=np.array([0,1,2]),out_fraction=np.array([.5,.5,1.]),
        outkc_edge=np.array([0,0,1]),outkc_node=np.array([0,1,2]),outkc_delay=np.array([1,1,2]),
        outkc_site=np.array([0,1,2]),outkc_fraction=np.array([.5,.5,1.]))
    np.savez_compressed(path,**data)
    path.with_suffix('.json').write_text(json.dumps({'kc_geometry_sha256':bytes(m.kc_terminal_digest.tolist()).hex()}),encoding='utf-8')
    attach_dan_release(m,path)
    return m,s,path


def test_reciprocal_contact_location_delay_and_no_cross_cell_transport(tmp_path):
    m,s,_=setup(tmp_path);v=terminal_state(m,s)
    v[3]=torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    torch.testing.assert_close(reciprocal_arrival(m,v),torch.tensor([[2.,0.,0.]]))
    fields=advance_dan_release(m,s,s.ring,v)
    assert fields[0][0,0]>0 and fields[0][0,1]==0
    assert fields[1][0,0]>fields[1][0,1]>0 and fields[1][0,2]==0
    assert fields[3][0,0]>fields[3][0,1]>0 and fields[3][0,2]==0
    m.dan_local_feedback_enabled=False
    blocked=advance_dan_release(m,s,s.ring,v)
    assert torch.count_nonzero(blocked[1])==torch.count_nonzero(blocked[3])==0


def test_uniform_source_preserves_old_graph_totals_and_local_output_reaches_KC(tmp_path):
    m,s,_=setup(tmp_path)
    values=dan_state(m,s);values[3]=torch.ones(4,3)
    expected=torch.zeros_like(s.h);expected[0,3]=.2;expected[0,4]=.3
    torch.testing.assert_close(dan_arrival(m,values),expected)
    torch.testing.assert_close(dan_arrival(m,values,'kc'),torch.tensor([[.1,.1,.3]]))
    routed=spatial_arrival(m,s.ring,dan_values=values)
    torch.testing.assert_close(routed,torch.tensor([[.2,.2,.3]]))
    values[3]=torch.tensor([[2.,0.,1.]]*4)
    routed=spatial_arrival(m,s.ring,dan_values=values)
    assert routed[0,0]>0 and routed[0,1]==0


def test_shared_resource_mass_bounds_and_first_AP_reference(tmp_path):
    m,s,_=setup(tmp_path);v=terminal_state(m,s)
    v[3]=torch.zeros_like(v[3]);ring=[torch.ones_like(s.h)]*4
    first=advance_dan_release(m,s,ring,v)
    torch.testing.assert_close(first[3][0],torch.ones(3))
    current=replace(s,**dict(zip(DAN_RELEASE_STATE_FIELDS,first)))
    second=advance_dan_release(m,current,ring,v)
    assert torch.all(second[3][0]<first[3][0])
    rate=torch.exp(m.dan_local_log_release_rate)[m.danl_site_owner].unsqueeze(0)
    lost=first[3][0:1]*(-torch.expm1(-rate))
    rho=torch.sigmoid(m.dan_local_logit_resource_recovery)[m.danl_site_owner].unsqueeze(0)
    torch.testing.assert_close(first[2],1-lost*rho)
    assert torch.all((second[2]>=0)&(second[2]<=1))
    after=advance_dan_release(m,replace(s,**dict(zip(DAN_RELEASE_STATE_FIELDS,second))),s.ring,v)
    assert torch.all(after[2]>second[2])


def test_material_gradient_identity_and_complete_state_archive(tmp_path):
    m,s,path=setup(tmp_path);v=terminal_state(m,s);v[3]=torch.tensor([[1.,0.,0.]]*4)
    fields=advance_dan_release(m,s,s.ring,v)
    gradients=torch.autograd.grad(fields[3][0,0]+2*fields[3][0,1],
        (m.dan_local_log_transport,m.dan_local_log_affinity))
    assert all(g[0]!=0 and g[1]==0 for g in gradients)
    current=replace(s,**dict(zip(DAN_RELEASE_STATE_FIELDS,fields)))
    for name in DAN_RELEASE_STATE_FIELDS:torch.testing.assert_close(getattr(current.detached(),name),getattr(current,name))
    with pytest.raises(ValueError,match='Incomplete'):
        dan_state(m,replace(s,dan_local_calcium=fields[1]))
    with pytest.raises(ValueError,match='identity'):
        attach_dan_release(copy.deepcopy(m),path,saved={'dan_release_digest':torch.zeros(32,dtype=torch.uint8)})
    assert 'dan_release_digest' in checkpoint_model_weights(m)


def test_one_token_whole_tick_recompute_optimizer_and_source_reference(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner,FlyPhysicalState
    from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
    m,s,_=setup(tmp_path);s=replace(s,h=s.h.index_fill(1,m.kct_index,3.))
    a=FlyBPTTLearner(m,copy.deepcopy(s),settle_ticks=3,use_checkpointing=False)
    b=FlyBPTTLearner(copy.deepcopy(m),copy.deepcopy(s),settle_ticks=3,use_checkpointing=True)
    assert set(DAN_RELEASE_PARAMETER_NAMES)<=set(a.adam_names)
    assert set(DAN_RELEASE_PARAMETER_NAMES)<=set(native_parameter_names(a.model))
    a.previous_token=b.previous_token=0
    x,_=a.observe([1]);y,_=b.observe([1]);torch.testing.assert_close(torch.tensor(x),torch.tensor(y))
    assert a.physical_ticks==b.physical_ticks and a.physical_ticks>1
    for name in DAN_RELEASE_STATE_FIELDS:torch.testing.assert_close(getattr(a.state,name),getattr(b.state,name))
    saved=a.state_dict();c=FlyBPTTLearner(copy.deepcopy(a.model),FlyPhysicalState(**saved['physical']),settle_ticks=3)
    c.optimizer.load_state_dict(saved['optimizer'])
    for name in DAN_RELEASE_STATE_FIELDS:torch.testing.assert_close(getattr(c.state,name),getattr(a.state,name),rtol=0,atol=0)


def test_disabled_source_exactly_restores_existing_reference_without_double_count(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event
    m,s,_=setup(tmp_path);s=replace(s,h=s.h.index_fill(1,m.kct_index,3.))
    reference=copy.deepcopy(m);reference.dan_local_enabled=False
    m.dan_local_source_enabled=False
    with torch.no_grad():
        a=advance_fly_input_event(m,copy.deepcopy(s),torch.tensor([1]),settle_ticks=3)
        b=advance_fly_input_event(reference,copy.deepcopy(s),torch.tensor([1]),settle_ticks=3)
    for name in ('h','dopamine','kc_contact_memory','kc_dopamine_signal'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
