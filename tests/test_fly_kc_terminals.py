"""Numerical/local routing contracts, not language capability experiments."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn

from test_fly_apl_cable import setup as apl_setup
from information_boltzmann.core.fly_bptt_learning import (
    attach_dopamine_modulation,attach_compartment_memory,FlyBPTTLearner,FlyPhysicalState,
    step_fly_physical_tick,
)
from information_boltzmann.core.fly_mb_terminal import step_mb_terminal
from information_boltzmann.core.fly_kc_terminals import (
    attach_kc_terminals,terminal_state,advance_kc_terminals,kc_arrival,KC_TERMINAL_STATE_FIELDS,
    CellwiseGeometry,
)
from information_boltzmann.core.fly_apl_cable import CableGeometry,diffuse
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from information_boltzmann.core.fly_native_policy_credit import native_parameter_names


@pytest.mark.parametrize('corruption', [0., float('nan')])
def test_owned_workspace_still_rejects_an_invalid_tree_solution(corruption):
    from types import SimpleNamespace
    geometry = CellwiseGeometry(np.array([2., 3.]), np.array([0]), np.array([1]),
                                np.array([.7]), np.array([0, 0]))
    matrix, _, _ = geometry.factor(torch.zeros(1))
    bad_factor = SimpleNamespace(solve=lambda rhs: np.full_like(rhs, corruption))
    with pytest.raises(FloatingPointError):
        geometry.solve_owned_rhs(matrix, bad_factor, np.array([[.4], [2.1]]))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_sparse_delay_accumulation_preserves_dense_values_and_vjp(device):
    from types import SimpleNamespace
    from information_boltzmann.core.fly_kc_terminals import _add_receptor_arrival, _sum
    model = SimpleNamespace()
    for key, value in [('order', [2, 5, 0, 1, 4, 3]), ('count', [2, 1, 2, 1]),
                       ('target', [0, 5, 7, 11])]:
        setattr(model, 'route_' + key, torch.tensor(value, device=device))
    model.route_gaba = torch.tensor([.2, .7, -.3, .5, .1, .6], device=device)
    model.route_serotonin = model.route_gaba * .4
    release = torch.tensor([.4, .2, .8, -.5, .3, .7], device=device, requires_grad=True)
    actual, amines = release.new_zeros(15), release.new_zeros(10)
    expected, expected_amines = release.new_zeros(15), release.new_zeros(15)
    for amount in (release, release * 1.3):
        _add_receptor_arrival(model, 'route', amount, actual, amines)
        expected = expected + _sum(model, 'route', amount * model.route_gaba, 15)
        expected_amines = expected_amines + _sum(model, 'route', amount * model.route_serotonin, 15)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(amines, expected_amines[:10], rtol=0, atol=0)
    ga = torch.autograd.grad(actual.square().sum() + amines.square().sum(), release, retain_graph=True)[0]
    gb = torch.autograd.grad(expected.square().sum() + expected_amines[:10].square().sum(), release)[0]
    torch.testing.assert_close(ga, gb)


@pytest.mark.parametrize('voltage_grad,material_grad', [(True, True), (True, False), (False, True)])
def test_streamed_rhs_matches_dense_solve_and_selective_gradients(voltage_grad, material_grad):
    args = (np.array([1., 2., 3., 4.]), np.array([0, 1]), np.array([1, 2]),
            np.array([.7, 1.2]), np.array([0, 0, 0, 1]))
    geometry = CellwiseGeometry(*args)
    assert all(getattr(geometry, name).dtype == np.int32
               for name in ('first', 'second', 'owner', 'parent', 'order'))
    original_check = geometry.solve_owned_rhs
    widths = []
    def check_one_rhs(matrix, factor, rhs):
        widths.append(rhs.shape[1])
        return original_check(matrix, factor, rhs)
    geometry.solve_owned_rhs = check_one_rhs
    x = torch.tensor([[.4, -.2, .7, .3], [.1, .8, -.4, 1.], [2., .1, .2, -.3]],
                     dtype=torch.double, requires_grad=voltage_grad)
    rate = torch.tensor([.2, -.1], dtype=torch.double, requires_grad=material_grad)
    actual = diffuse(x, rate, geometry)
    incidence = torch.tensor([[1., -1., 0., 0.], [0., 1., -1., 0.]], dtype=torch.double)
    area = torch.tensor(args[0])
    matrix = torch.diag(area) + incidence.T @ torch.diag(torch.tensor(args[3]) * rate[0].exp()) @ incidence
    expected = torch.linalg.solve(matrix, (area * x).T).T
    # the cell-wise forests compute in float32 (CellwiseGeometry.DTYPE): float32-level agreement
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    variables = [v for v in (x, rate) if v.requires_grad]
    weights = torch.arange(1, 13, dtype=torch.double).reshape(3, 4)
    a = torch.autograd.grad((actual.square() * weights).sum(), variables)
    b = torch.autograd.grad((expected.square() * weights).sum(), variables)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    assert widths == [1] * 6
    with torch.no_grad():
        torch.testing.assert_close(diffuse(x, rate, geometry), expected)


def setup(tmp_path):
    model,state,_=apl_setup(tmp_path)
    state=replace(state,apl_packets=torch.zeros_like(state.apl_packets))
    model.has_dopamine=True
    for name,value in [('dan_edge_pre',[0]),('dan_edge_post',[5])]:
        model.register_buffer(name,torch.tensor(value),persistent=False)
    model.dan_delay_splits=(0,1,1,1,1)
    model.register_buffer('dan_edge_weight',torch.tensor([.1]),persistent=False)
    model.dan_scale=1.
    attach_dopamine_modulation(model)
    attach_compartment_memory(model,tmp_path/'graph.npz',tmp_path/'ann.feather')
    model.comp_unit_traces=True
    model.register_buffer('kcd_pre',model.comp_pre.clone(),persistent=False)
    model.register_buffer('kcd_post',model.comp_post.clone(),persistent=False)
    model.register_buffer('kcd_weight',torch.ones(2),persistent=False)
    model.log_kc_release=nn.Parameter(torch.zeros(model.n_superclasses))
    _,g,c,r=step_mb_terminal(model,state)
    state=replace(state,mb_terminal_gaba=g,mb_dpm_concentration=c,mb_dpm_receptor=r,
        comp_trace=torch.zeros(1,2),comp_memory=torch.zeros(1,2))
    data=dict(site_ids=np.array([[3,10,0,0],[3,20,0,0],[4,10,0,0]]),
        kc_index=np.array([3,4]),source_index=np.array([0,2,1]),
        owner=np.array([0,0,1]),site_owner=np.array([0,0,1]),site_node=np.array([0,1,2]),
        volume=np.array([1.,1.,1.]),u=np.array([0]),v=np.array([1]),g=np.array([1.]),
        comp_pre=np.array([3,4]),comp_post=np.array([5,5]),
        fast_edge=np.array([0,0,1]),fast_pre=np.array([3,3,4]),fast_post=np.array([5,5,5]),
        fast_site=np.array([0,1,2]),fast_delay=np.array([1,1,1]),fast_fraction=np.array([.5,.5,1.]),
        apl_edge=np.array([],dtype=np.int64),apl_delay=np.array([],dtype=np.int64),
        apl_site=np.array([],dtype=np.int64),apl_node=np.array([],dtype=np.int64),apl_fraction=np.array([]),
        receptor_source=np.array([0]),receptor_kc=np.array([0]),receptor_delay=np.array([1]),
        receptor_node=np.array([0]),receptor_pre_node=np.array([-1]),receptor_fraction=np.array([1.]))
    path=tmp_path/'kc.npz';np.savez(path,**data)
    path.with_suffix('.json').write_text(json.dumps({'sites':3,'nodes':3}))
    attach_kc_terminals(model,path)
    return model,state.detached(),path


@pytest.mark.parametrize('nested', [False, True])
def test_spatial_checkpoint_preserves_multiple_vjps_and_native_state(tmp_path, nested):
    from torch.utils.checkpoint import checkpoint
    from information_boltzmann.core.fly_bptt_learning import checkpoint_native_spatial
    model, state, _ = setup(tmp_path)
    state = replace(state, ring=(torch.tensor([[1., 0., 0., 0., 0., 0.]]), *state.ring[1:]))
    raw = torch.tensor([[0., 0., 0., 1., 1., 0.]])
    original = {name: value.clone() for name, value in state.state_dict().items()
                if torch.is_tensor(value)}
    model._native_spatial_checkpointing = False
    plain = checkpoint_native_spatial(advance_kc_terminals, model, state, raw)
    model._native_spatial_checkpointing = True
    def run(packet):
        return checkpoint_native_spatial(advance_kc_terminals, model, state, packet)
    checked = checkpoint(run, raw, use_reentrant=False) if nested else run(raw)
    for a, b in zip(plain, checked):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index in (3, 0):
        a = torch.autograd.grad(plain[index].square().sum(), model.kc_terminal_log_diffusion,
                                retain_graph=True)[0]
        b = torch.autograd.grad(checked[index].square().sum(), model.kc_terminal_log_diffusion,
                                retain_graph=True)[0]
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for name, value in original.items():
        torch.testing.assert_close(getattr(state, name), value, rtol=0, atol=0)


def test_same_kc_sites_can_release_differently_and_use_real_delay(tmp_path):
    model,state,_=setup(tmp_path)
    state=replace(state,ring=(torch.tensor([[1.,0.,0.,0.,0.,0.]]),*state.ring[1:]))
    model.kc_terminal_transport_enabled=False
    raw=torch.tensor([[0.,0.,0.,1.,1.,0.]])
    values=advance_kc_terminals(model,state,raw)
    assert 0<values[3][0,0]<values[3][0,1]==values[3][0,2]==1
    # One source acts locally; the other tree has exactly no new signal.
    assert values[0][0,0]>0 and torch.count_nonzero(values[0])==1
    torch.testing.assert_close(kc_arrival(model,state,'comp',new_values=values,instant=True),
        torch.tensor([[(values[3][0,0]+1)/2,1.]]))
    # Historical point packets cannot be a second physical path.
    changed=replace(state,ring=tuple(p.index_fill(1,model.kct_index,900.) for p in state.ring),
        **dict(zip(KC_TERMINAL_STATE_FIELDS,values)))
    torch.testing.assert_close(kc_arrival(model,changed,'comp'),kc_arrival(model,state,'comp',new_values=values))
    model.kc_terminal_gating_enabled=False
    bypass=advance_kc_terminals(model,state,raw)
    torch.testing.assert_close(bypass[3][0],torch.ones(3))


def test_native_transport_couples_connected_locations_but_not_other_cells(tmp_path):
    model,state,_=setup(tmp_path)
    state=replace(state,ring=(torch.tensor([[1.,0.,0.,0.,0.,0.]]),*state.ring[1:]))
    raw=torch.tensor([[0.,0.,0.,1.,1.,0.]])
    values=advance_kc_terminals(model,state,raw)
    assert values[0][0,0]>values[0][0,1]>0 and values[0][:,2].sum()==0
    loss=values[3][0,0]+2*values[3][0,1]
    grad=torch.autograd.grad(loss,model.kc_terminal_log_diffusion)[0]
    assert grad[:,0].abs().sum()>0 and torch.count_nonzero(grad[:,1])==0


def test_local_state_detach_checkpoint_and_identity_fail_closed(tmp_path):
    model,state,path=setup(tmp_path)
    values=terminal_state(model,state)
    local=replace(state,**dict(zip(KC_TERMINAL_STATE_FIELDS,values)))
    detached=local.detached()
    for key in KC_TERMINAL_STATE_FIELDS:torch.testing.assert_close(getattr(detached,key),getattr(local,key))
    assert 'kc_terminal_log_diffusion' in native_parameter_names(model)
    saved=checkpoint_model_weights(model)
    saved['kc_terminal_digest']=saved['kc_terminal_digest']^1
    with pytest.raises(ValueError,match='identity mismatch'):attach_kc_terminals(copy.deepcopy(model),path,saved=saved)
    with pytest.raises(ValueError,match='incompatible shape'):
        terminal_state(model,replace(state,kc_terminal_packets=torch.zeros(4,2)))


def test_first_activation_initializes_only_absent_history(tmp_path):
    model,state,_=setup(tmp_path)
    newborn=replace(state,mb_terminal_gaba=torch.empty(0),mb_dpm_concentration=torch.empty(0),
        mb_dpm_receptor=torch.empty(0),comp_trace=torch.empty(0))
    values=terminal_state(model,newborn)
    assert all(v.numel()>0 and torch.count_nonzero(v)==0 for v in values[:6])
    assert all(v.numel()==0 for v in values[6:])
    # Nonempty malformed history is rejected, never discarded as a newborn.
    with pytest.raises((RuntimeError,ValueError,IndexError)):
        terminal_state(model,replace(state,mb_terminal_gaba=torch.ones(3,1)))


def test_cell_blocks_match_whole_forest_values_and_adjoint():
    args=([1.,2.,1.,3.],[0,2],[1,3],[.5,2.],[0,0,1,1])
    x=torch.tensor([[.7,.2,.9,.1]],dtype=torch.double,requires_grad=True)
    rate=torch.tensor([-.4,.2],dtype=torch.double,requires_grad=True)
    a=diffuse(x,rate,CellwiseGeometry(*args));b=diffuse(x,rate,CableGeometry(*args))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)   # float32 forest vs float64 sparse LU
    ga=torch.autograd.grad(a.square().sum(),(x,rate));gb=torch.autograd.grad(b.square().sum(),(x,rate))
    for left,right in zip(ga,gb):torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize("site_stp", [False, True])
def test_one_window_bptt_and_recomputation_preserve_all_site_fields(tmp_path, site_stp):
    torch.manual_seed(7)
    model,state,_=setup(tmp_path)
    state=replace(state,h=state.h.index_fill(1,model.kct_index,3.))
    if site_stp:
        from information_boltzmann.core.fly_kc_site_stp import attach_kc_site_stp
        attach_kc_site_stp(model)
    a=FlyBPTTLearner(model,copy.deepcopy(state),use_checkpointing=False,settle_ticks=3)
    b=FlyBPTTLearner(copy.deepcopy(model),copy.deepcopy(state),use_checkpointing=True,settle_ticks=3)
    a.previous_token=b.previous_token=0
    p,_=a.observe([1]);q,_=b.observe([1])
    torch.testing.assert_close(torch.tensor(p),torch.tensor(q))
    for key in KC_TERMINAL_STATE_FIELDS:
        torch.testing.assert_close(getattr(a.state,key),getattr(b.state,key))
    assert torch.isfinite(a.model.kc_terminal_log_diffusion).all()
    assert a.state.kc_terminal_eligibility.shape==a.state.comp_memory.shape


def test_local_release_teaches_phi_without_optimizer_and_carried_trace_matches_vjp(tmp_path):
    model,state,_=setup(tmp_path)
    model.kc_terminal_transport_enabled=False
    model.carried_credit=('comp','dopamine')
    state=replace(state,h=state.h.index_fill(1,model.kct_index,3.),
        ring=(torch.tensor([[1.,0.,0.,0.,0.,0.]]),*state.ring[1:]),
        comp_trace=torch.tensor([[.4,.7]]),dopamine=torch.tensor([[2.]*6,[.5]*6]))
    out=step_fly_physical_tick(model,state,None,torch.zeros_like(state.h),state.baseline,{})
    assert out.kc_terminal_eligibility[0,0]<out.kc_terminal_eligibility[0,1]
    assert 0>out.comp_memory[0,0]>out.comp_memory[0,1]
    exact=torch.autograd.grad(out.comp_memory.sum(),model.comp_logit_trace)[0]
    assert exact.abs()>0
    torch.testing.assert_close(exact,out.credit_comp[2].sum(),rtol=2e-5,atol=1e-8)
