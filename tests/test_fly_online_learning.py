"""Numerical derivatives and complete continuing-state tests (no capability claims)."""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from information_boltzmann.core.fly_online_learning import rmsnorm_vjp, advance_local_jacobian, FlyOnlineLearner
from information_boltzmann.core.fly_reservoir import SpikeFn, FlyReservoirLM, BiologicalTopographicWriter
from information_boltzmann.core.eprop_credit_assignment import SensoryWriterEligibilityState, DopamineReceptorState


def test_rmsnorm_matches_actual_forward_epsilon():
    torch.manual_seed(4)
    for eps in (None,1e-6):
        norm=nn.RMSNorm(7,eps=eps).double()
        z=(torch.randn(2,7,dtype=torch.float64)*1e-4).requires_grad_()
        error=torch.randn_like(z)
        expected=torch.autograd.grad((norm(z)*error).sum(),(z,norm.weight))
        actual=rmsnorm_vjp(z.detach(),error,norm)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


def test_writer_old_baseline_reset_alif_matches_autograd():
    torch.manual_seed(8)
    d=3; n=3
    weight=torch.randn(n,d,dtype=torch.float64,requires_grad=True)
    state=SensoryWriterEligibilityState.init_zero(d,'cpu',1,1,1)
    state.c_gate=state.c_gate.double(); state.e_proj=state.e_proj.double(); state.e_adaptation=state.e_adaptation.double()
    h=torch.zeros(n,dtype=torch.float64); b=torch.zeros_like(h); baseline=torch.zeros_like(h)
    alpha=torch.full_like(h,.83); integ=torch.full_like(h,.2)
    beta=torch.full_like(h,.3); rho=torch.full_like(h,.91)
    for step in range(5):
        x=torch.randn(d,dtype=torch.float64); gates=torch.rand(n,dtype=torch.float64)*2
        proposal=gates*(weight*x).sum(-1)
        drive=proposal-baseline
        baseline=.7*baseline+.3*proposal
        v=alpha*h+integ*drive
        theta=.1+beta*b
        spikes=SpikeFn.apply(v-theta)
        h=v*(1-spikes); b=rho*b+(1-rho)*spikes
        psi=1/(1+(torch.pi*(v-theta)).square())
        with torch.no_grad():
            actual=state.update_input_eligibility(x[None],gates[None],alpha,integ,.7,
                v_pre=v,spikes=spikes,psi=psi,beta_adaptation=beta,rho_adaptation=rho)
        expected=torch.autograd.grad(h.sum(),weight,retain_graph=True)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def test_dopamine_exact_relaxation_and_delay():
    state=DopamineReceptorState.init_zero(3,'cpu',k_on=100,k_off=.05)
    ring=(torch.tensor([[1.,0,0]]),torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,3))
    state.update_from_dan_activity(torch.zeros(1,3),torch.zeros(1,3),
        dan_edge_pre=torch.tensor([0]),dan_edge_post=torch.tensor([2]),dan_edge_weight=torch.tensor([5.]),
        delayed_pulses=ring,delay_splits=(0,0,1,1,1))
    assert state.q[2]==0 # edge at tier 2, pulse currently at tier 1
    ring=(ring[1],ring[0],ring[2],ring[3])
    state.update_from_dan_activity(torch.zeros(1,3),torch.zeros(1,3),
        dan_edge_pre=torch.tensor([0]),dan_edge_post=torch.tensor([2]),dan_edge_weight=torch.tensor([5.]),
        delayed_pulses=ring,delay_splits=(0,0,1,1,1))
    torch.testing.assert_close(state.q[2],torch.tensor(500/500.05))
    assert torch.all((state.q>=0)&(state.q<=1))


def make_model(tmp_path):
    graph=tmp_path/'local_coba.npz'
    np.savez(graph,neuron_body_ids=np.arange(6),edge_pre=np.arange(6),edge_post=np.roll(np.arange(6),1),
        edge_weight=np.full(6,.01,dtype=np.float32),nt_sign=np.array([1,1,1,-1,-1,-1]),
        edge_delay=np.ones(6,dtype=np.int32),superclass_id=np.array([0,0,0,1,1,1]),superclass_names=np.array(['sensory','output']))
    parts=tmp_path/'parts.npz'
    np.savez(parts,visual_idx=np.array([0]),chemo_idx=np.array([1]),mechano_idx=np.array([2]))
    model=FlyReservoirLM(graph,vocab_size=9,d_model=3,synapse_model='coba',use_alif=True,use_stp=True)
    writer=BiologicalTopographicWriter(3,parts)
    model.read_mask[:3]=0
    model.topographic_writer=writer; model.injection_mode='topographic'; model.input_proj=None
    return model


def test_complete_resume_with_pending_updates(tmp_path):
    torch.manual_seed(7)
    a=make_model(tmp_path)
    learner=FlyOnlineLearner(a,lr=1e-4,grad_accum_tokens=4,synapse_update_interval=2)
    for i in range(3): learner.step(torch.tensor([i]),torch.tensor([i+1]))
    snapshot=copy.deepcopy(learner.state_dict())
    parameters=copy.deepcopy(a.state_dict())
    edges=[a.edge_weight_e.clone(),a.edge_weight_i.clone()]
    b=make_model(tmp_path); b.load_state_dict(parameters)
    b.edge_weight_e.copy_(edges[0]); b.edge_weight_i.copy_(edges[1])
    resumed=FlyOnlineLearner(b,lr=1e-4,grad_accum_tokens=4,synapse_update_interval=2)
    resumed.load_state_dict(snapshot)
    for i in range(3,6):
        first=learner.step(torch.tensor([i]),torch.tensor([i+1]))
        second=resumed.step(torch.tensor([i]),torch.tensor([i+1]))
        assert first==second
    assert learner.events==resumed.events==6
    assert learner.pending==resumed.pending==2
    for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(a.edge_weight_e,b.edge_weight_e,rtol=0,atol=0)
    for key in learner.syn: torch.testing.assert_close(learner.syn[key],resumed.syn[key],rtol=0,atol=0)


def test_fused_credit_matches_pytorch_cuda():
    import pytest
    from information_boltzmann.core import triton_online_credit as fused
    if not torch.cuda.is_available() or fused.triton is None: pytest.skip('CUDA Triton required')
    torch.manual_seed(33)
    dev='cuda'; n=9; d=5
    modality=torch.tensor([0,0,0,1,1,1,2,2,2],device=dev)
    x=torch.randn(d,device=dev); packet=torch.randn(n,device=dev)
    gates=torch.softmax(torch.randn(3,device=dev),0)*3
    jac=torch.diag(gates)-gates[:,None]*(gates/3)[None,:]
    a=torch.rand(n,device=dev); integ=torch.rand(n,device=dev)
    v=torch.randn(n,device=dev); spikes=(v>0).float(); psi=torch.rand(n,device=dev)
    beta=torch.rand(n,device=dev); rho=torch.rand(n,device=dev)
    h=torch.randn(n,3*(d+1),device=dev); b=torch.randn_like(h); c=torch.randn_like(h)
    expected_h=h.clone(); expected_b=b.clone(); expected_c=c.clone()
    raw=(packet[:,None]*jac[modality])[:,:,None]*torch.cat((x,torch.ones(1,device=dev)))[None,None,:]
    raw=raw.flatten(1)
    direct=integ[:,None]*(raw-expected_c)
    expected_c.mul_(.8).add_(raw,alpha=.2)
    advance_local_jacobian(expected_h,expected_b,direct,0,0,a,v,spikes,psi,beta,rho)
    fused.update_gate_traces(h,b,c,packet,jac,modality,x,a,integ,v,spikes,psi,beta,rho,.8)
    for first,second in zip((h,b,c),(expected_h,expected_b,expected_c)):
        torch.testing.assert_close(first,second,rtol=1e-5,atol=2e-6)
    state=SensoryWriterEligibilityState.init_zero(d,dev,3,3,3)
    state.e_proj.normal_();state.e_adaptation.normal_();state.c_gate.normal_()
    expected=copy.deepcopy(state)
    expected.update_input_eligibility(x[None],gates[None],a,integ,.8,v_pre=v,spikes=spikes,psi=psi,
                                      beta_adaptation=beta,rho_adaptation=rho)
    fused.update_projection_traces(state,gates,modality,x,a,integ,v,spikes,psi,beta,rho,.8)
    for name in ('e_proj','e_adaptation','c_gate'):
        torch.testing.assert_close(getattr(state,name),getattr(expected,name),rtol=1e-5,atol=2e-6)
    weight=torch.randn(n,d,device=dev); factor=torch.randn(n,device=dev)
    expected=weight*(1-.001*.0001)-.001*factor[:,None]*state.e_proj
    fused.update_projection_weights(weight,state.e_proj,factor,.001,.0001)
    torch.testing.assert_close(weight,expected,rtol=1e-6,atol=1e-6)
    from information_boltzmann.core.eprop_credit_assignment import EPropCreditAssignment
    edges=19; neurons=10
    pre=torch.randint(neurons,(edges,),device=dev,dtype=torch.int32)
    post=torch.randint(neurons,(edges,),device=dev,dtype=torch.int32)
    L=torch.randn(1,neurons,device=dev); phi=torch.randn_like(L); q=torch.rand(neurons,device=dev)
    z=torch.randn(1,4,neurons,device=dev); splits=(0,3,8,10,19)
    weight=torch.rand(edges,device=dev); pending=torch.zeros_like(weight)
    reference_weight=weight.clone(); reference_pending=pending.clone()
    for apply in (False,True):
        delta=EPropCreditAssignment.compute_synaptic_updates_sparse(L*q[None],phi,z,pre,post,splits,scale=.001)
        reference_pending.add_(delta)
        if apply:
            reference_weight.add_(reference_pending).clamp_(min=0,max=5); reference_pending.zero_()
        fused.update_edge_traces(weight,pending,pre,post,L,phi,q,z,splits,.001,apply)
        torch.testing.assert_close(weight,reference_weight,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(pending,reference_pending,rtol=1e-6,atol=1e-6)


def test_physical_local_sensitivities_match_autograd(tmp_path):
    torch.manual_seed(61)
    model=make_model(tmp_path)
    learner=FlyOnlineLearner(model,train_synapses=False,train_sensory=False)
    for name in learner.physical_names: getattr(model,name).requires_grad_(True)
    h=torch.zeros_like(learner.h); b=torch.zeros_like(h)
    ge=torch.zeros_like(h); gi=torch.zeros_like(h)
    x=torch.ones_like(h); u=model.get_stp_params()[0].detach()
    # Arriving recurrent currents are conditioned on, as documented by this learner.
    ring=tuple(torch.rand_like(h)*.1 for _ in range(4))
    for event in range(4):
        h0,b0,ge0,gi0=h,b,ge,gi
        result,bio=model.step(h,torch.tensor([event]),spike_ring=ring,ge=ge,gi=gi,b=b,x=x,u=u,
                              return_biophysics=True)
        h,spikes,_,ge,gi,b,x,u=result
        L=torch.arange(1,7,dtype=torch.float32)[None,:]/6
        psi=1/(1+(torch.pi*(bio['v_pre']-bio['eff_threshold'])).square())
        with torch.no_grad():
            learner.syn=dict(ge=ge,gi=gi,b=b,x=x,u=u)
            learner.refresh()
            for grad in learner.grads.values(): grad.zero_()
            learner._physical_gradients(L,bio,h0,b0,ge0,gi0,spikes,psi)
        expected=torch.autograd.grad((L*h).sum(),[getattr(model,name) for name in learner.physical_names],retain_graph=True,allow_unused=True)
        for name,reference in zip(learner.physical_names,expected):
            if reference is None: reference=torch.zeros_like(getattr(model,name))
            torch.testing.assert_close(learner.grads[name],reference,rtol=3e-4,atol=2e-5,msg=name)


def test_fused_physical_credit_matches_pytorch_cuda(tmp_path):
    import pytest
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from information_boltzmann.core import triton_online_credit as fused
    if fused.triton is None: pytest.skip('Triton required')
    torch.manual_seed(40)
    a=FlyOnlineLearner(make_model(tmp_path),train_synapses=False,train_sensory=False)
    b=FlyOnlineLearner(copy.deepcopy(a.model).cuda(),train_synapses=False,train_sensory=False)
    for step in range(3):
        hprev=torch.randn(1,6)*.2; bprev=torch.rand(1,6)*.1
        geprev=torch.rand(1,6)*.1; giprev=torch.rand(1,6)*.1
        gen=torch.rand(1,6)*.1; gin=torch.rand(1,6)*.1
        spikes=(torch.rand(1,6)>.8).float(); psi=torch.rand(1,6); L=torch.randn(1,6)
        a.syn['ge']=gen; a.syn['gi']=gin
        b.syn['ge']=gen.cuda(); b.syn['gi']=gin.cuda()
        G=torch.rand(1,6)*.5+.05
        alpha=torch.exp(-G)
        bio=dict(v_pre=torch.randn(1,6)*.2,alpha_eff=alpha,beta_int=(1-alpha)/G,g_total=G)
        with torch.no_grad():
            a._physical_gradients(L,bio,hprev,bprev,geprev,giprev,spikes,psi)
            b._physical_gradients(L.cuda(),{k:v.cuda() for k,v in bio.items()},hprev.cuda(),bprev.cuda(),
                                  geprev.cuda(),giprev.cuda(),spikes.cuda(),psi.cuda())
        for attr in ('phys_h','phys_b','ge_derivative','gi_derivative'):
            torch.testing.assert_close(getattr(a,attr),getattr(b,attr).cpu(),rtol=1e-4,atol=2e-6)
        for name in a.physical_names:
            torch.testing.assert_close(a.grads[name],b.grads[name].cpu(),rtol=1e-4,atol=2e-6)
