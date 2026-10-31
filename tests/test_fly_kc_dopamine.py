"""Spatial teacher routing, continuity and derivatives; no capability claims."""
from dataclasses import replace
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from test_fly_kc_contact_memory import local_setup
from information_boltzmann.core.fly_kc_dopamine import (
    attach_kc_dopamine, dopamine_state, spatial_arrival, advance_kc_dopamine,
    contact_teacher, KC_DOPAMINE_STATE_FIELDS,
)
from information_boltzmann.core.fly_carried_credit import open_window, apply_carried_credit
from information_boltzmann.core.fly_kc_contact_memory import advance_contact_learning
from information_boltzmann.core.fly_kc_terminals import terminal_state
from information_boltzmann.core.fly_bptt_learning import advance_compartment_learning
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights


def setup(tmp_path):
    model,state=local_setup(tmp_path)
    model.dan_edge_pre=torch.tensor([0,2])
    model.dan_edge_post=torch.tensor([3,4])
    model.dan_edge_weight=torch.tensor([.2,.3])
    model.dan_delay_splits=(0,1,2,2,2)
    path=tmp_path/'dopamine.npz'
    np.savez_compressed(path,edge=np.array([0,1]),pre=np.array([0,2]),post=np.array([3,4]),
        node=np.array([0,2]),delay=np.array([1,2]),fraction=np.array([1.,1.]),
        kc_index=np.array([3,4]),node_ids=np.array([[3,1],[3,2],[4,1]]))
    path.with_suffix('.json').write_text(json.dumps({'kc_geometry_sha256':bytes(model.kc_terminal_digest.tolist()).hex()}),encoding='utf-8')
    attach_kc_dopamine(model,path)
    state=replace(state,dopamine=torch.zeros_like(state.dopamine))
    return model,state,path


def test_actual_contacts_delay_conservation_and_no_cell_crossing(tmp_path):
    m,s,_=setup(tmp_path)
    packets=[torch.zeros_like(s.h) for _ in range(4)]
    packets[0][0,0]=1.
    arrival=spatial_arrival(m,packets)
    torch.testing.assert_close(arrival,torch.tensor([[.4,0.,0.]]))
    assert float(arrival[0,:2].mean())==pytest.approx(.2)
    before,credit=dopamine_state(m,s)
    out=advance_kc_dopamine(m,replace(s,kc_dopamine_signal=before,kc_dopamine_credit=credit),packets)
    assert out[0][0,0]>out[0][0,1]>0 and out[0][0,2]==0
    packets[0].zero_();packets[1][0,2]=1.
    torch.testing.assert_close(spatial_arrival(m,packets),torch.tensor([[0.,0.,.3]]))
    m.kc_dopamine_delivery_enabled=False
    assert torch.count_nonzero(spatial_arrival(m,packets))==0


def test_history_prolongation_and_fail_closed_identity(tmp_path):
    m,s,path=setup(tmp_path)
    s=replace(s,dopamine=torch.arange(12,dtype=torch.float32).reshape(2,6),
              credit_dopamine=torch.arange(18,dtype=torch.float32).reshape(3,6))
    values=dopamine_state(m,s)
    torch.testing.assert_close(values[0],s.dopamine[:,[3,3,4]])
    torch.testing.assert_close(values[1],s.credit_dopamine[:,[3,3,4]])
    current=replace(s,**dict(zip(KC_DOPAMINE_STATE_FIELDS,values)))
    for n in KC_DOPAMINE_STATE_FIELDS:
        torch.testing.assert_close(getattr(current.detached(),n),getattr(current,n))
    with pytest.raises(ValueError,match='Incomplete'):
        dopamine_state(m,replace(s,kc_dopamine_credit=values[1]))
    with pytest.raises(ValueError,match='identity'):
        attach_kc_dopamine(m,path,saved={'kc_dopamine_digest':torch.zeros(32,dtype=torch.uint8)})
    assert 'kc_dopamine_digest' in checkpoint_model_weights(m)


def test_local_teacher_changes_phi_not_MBON_reference(tmp_path):
    m,s,_=setup(tmp_path)
    local=torch.tensor([[2.,0.,0.],[.5,0.,0.]])
    teacher=contact_teacher(m,(local,torch.zeros(3,3)))
    assert teacher[0].shape==(2,3)
    values=terminal_state(m,s)
    values[3]=torch.tensor([[.7,.7,.7]]*4)
    def update():
        return advance_contact_learning(m,s,s.h,torch.zeros_like(s.h),s.ring,
            s.dopamine,s.credit_dopamine,s.dpm_trace,values,advance_compartment_learning,
            spatial_dopamine_values=(local,torch.zeros(3,3)))
    learned=update()
    assert learned[0][0,0]<0 and learned[0][0,1]==0 and learned[0][0,2]==0
    m.kc_dopamine_teacher_enabled=False
    assert torch.count_nonzero(update()[0])==0
    torch.testing.assert_close(s.comp_memory,torch.zeros_like(s.comp_memory))


def test_conditional_clock_credit_and_transport_gradient(tmp_path):
    m,s,_=setup(tmp_path)
    packets=[torch.zeros_like(s.h) for _ in range(4)]
    packets[0][0,0]=1.
    def tick(state):
        values=advance_kc_dopamine(m,state,packets)
        return replace(state,**dict(zip(KC_DOPAMINE_STATE_FIELDS,values)))
    full=tick(tick(s))
    loss=full.kc_dopamine_signal[0,0]+3*full.kc_dopamine_signal[1,0]
    params=(m.dopamine_logit_fast,m.dopamine_logit_slow,m.kc_dopamine_log_transport)
    exact=torch.autograd.grad(loss,params,retain_graph=True)
    torch.testing.assert_close(exact[0],(full.kc_dopamine_credit[0,0]+3*full.kc_dopamine_credit[1,0]),rtol=2e-5,atol=1e-8)
    torch.testing.assert_close(exact[1],3*full.kc_dopamine_credit[2,0],rtol=2e-5,atol=1e-8)
    assert exact[2][0]!=0 and exact[2][1]==0
    start=tick(s).detached()
    opened,leaves,history=open_window(start,('dopamine',))
    end=tick(opened)
    (end.kc_dopamine_signal[0,0]+3*end.kc_dopamine_signal[1,0]).backward()
    # Shared two clock parameters include exact conditional historical paths;
    # the new transport material has in-window derivatives, full history later.
    m.carried_credit=('dopamine',)
    apply_carried_credit(m,leaves,history)
    torch.testing.assert_close(m.dopamine_logit_fast.grad,exact[0],rtol=2e-5,atol=1e-8)
    torch.testing.assert_close(m.dopamine_logit_slow.grad,exact[1],rtol=2e-5,atol=1e-8)


def test_one_token_actual_tick_recompute_optimizer_and_restore(tmp_path):
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner
    from information_boltzmann.core.fly_native_policy_credit import native_parameter_names
    m,s,_=setup(tmp_path)
    m.dan_delay_splits=(0,1,2,2,2)
    a=FlyBPTTLearner(m,copy.deepcopy(s),use_checkpointing=False,settle_ticks=3)
    b=FlyBPTTLearner(copy.deepcopy(m),copy.deepcopy(s),use_checkpointing=True,settle_ticks=3)
    assert 'kc_dopamine_log_transport' in a.adam_names
    assert 'kc_dopamine_log_transport' in native_parameter_names(a.model)
    a.previous_token=b.previous_token=0
    a.observe([1]);b.observe([1])
    assert a.physical_ticks==b.physical_ticks and a.physical_ticks>1
    for name in KC_DOPAMINE_STATE_FIELDS:
        torch.testing.assert_close(getattr(a.state,name),getattr(b.state,name))
    assert a.state.kc_dopamine_signal.shape==(2,3)
    saved=a.state_dict()
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
    c=FlyBPTTLearner(copy.deepcopy(a.model),FlyPhysicalState(**saved['physical']),settle_ticks=3)
    c.optimizer.load_state_dict(saved['optimizer'])
    for name in KC_DOPAMINE_STATE_FIELDS:
        torch.testing.assert_close(getattr(c.state,name),getattr(a.state,name),rtol=0,atol=0)
