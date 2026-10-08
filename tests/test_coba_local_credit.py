"""Numerical verification of local eligibility and representable plasticity."""
import copy
import pytest
import torch
from information_boltzmann.core.coba_local_credit import (
    LocalCobaEdgeCredit, compensated_add_, voltage_conductance_derivative,
    update_local_edges_pytorch,
)
from information_boltzmann.core.fly_reservoir import SpikeFn
from information_boltzmann.core import triton_online_credit as fused


def test_sub_ulp_updates_accumulate_and_bounds_discard_outward_residual():
    weight = torch.ones(4)
    residual = torch.zeros_like(weight)
    update = torch.full_like(weight, 2**-27)
    for _ in range(32): compensated_add_(weight, residual, update)
    torch.testing.assert_close(weight.double()+residual.double(), torch.full((4,),1+32*2**-27,dtype=torch.float64),rtol=0,atol=1e-14)
    assert torch.all(weight > 1)
    weight.zero_(); residual.zero_()
    compensated_add_(weight,residual,-update,(0,5))
    assert not weight.any() and not residual.any()


@pytest.mark.parametrize('reversal',[1.0,-.2])
def test_heterogeneous_destination_credit_matches_multistep_autograd(reversal):
    torch.manual_seed(23)
    n, e = 5, 8
    pre=torch.tensor([0,1,2,3,4,2,0,1]); post=torch.tensor([4,2,3,1,0,4,4,2])
    weight=(torch.rand(e,dtype=torch.float64)*.3).requires_grad_()
    state=LocalCobaEdgeCredit.zeros_like(weight)
    pending=torch.zeros_like(weight)
    h=torch.zeros(n,dtype=torch.float64); b=h.clone(); g=h.clone()
    leak=torch.linspace(.6,.94,n,dtype=torch.float64)
    rho=torch.linspace(.8,.99,n,dtype=torch.float64)
    beta=torch.linspace(.03,.2,n,dtype=torch.float64)
    gain=1.3; splits=(0,2,4,6,8)
    tiers=torch.arange(e)//2
    for event in range(7):
        pulses=tuple(torch.rand(n,dtype=torch.float64) for _ in range(4))
        arriving=torch.stack(pulses)[tiers,pre]
        old_h=h
        g=leak*g+(1-leak)*torch.zeros_like(g).index_add(0,post,weight*arriving)
        G=.04+gain*g
        alpha=torch.exp(-G); integ=(1-alpha)/G
        drive=torch.randn(n,dtype=torch.float64)*.2
        v=alpha*h+integ*(gain*g*reversal+drive)
        theta=.1+beta*b
        spikes=SpikeFn.apply(v-theta)
        h=v*(1-spikes); b=rho*b+(1-rho)*spikes
        bio={'v_pre':v,'alpha_eff':alpha,'beta_int':integ,'g_total':G}
        psi=1/(1+(torch.pi*(v-theta)).square())
        direct=gain*(voltage_conductance_derivative(bio,old_h)+integ*reversal)
        update_local_edges_pytorch(state,weight,pending,pre,post,splits,pulses,
            torch.zeros(n,dtype=torch.float64),bio,direct,leak,spikes,psi,beta,rho,0,False)
        expected=torch.autograd.grad(h.sum(),weight,retain_graph=True)[0]
        torch.testing.assert_close(state.voltage,expected,rtol=2e-11,atol=2e-11)
        torch.testing.assert_close(state.adaptation,torch.autograd.grad(b.sum(),weight,retain_graph=True)[0],rtol=2e-11,atol=2e-11)


@pytest.mark.parametrize('mask_dormant',[False,True])
def test_cuda_local_credit_and_compensation_match_reference(mask_dormant):
    if not torch.cuda.is_available() or fused.triton is None: pytest.skip('CUDA Triton required')
    torch.manual_seed(44)
    n,e=17,43; dev='cuda'
    pre=torch.randint(n,(e,),device=dev,dtype=torch.int32)
    post=torch.randint(n,(e,),device=dev,dtype=torch.int32)
    weight=torch.rand(e,device=dev); ref_weight=weight.clone()
    state=LocalCobaEdgeCredit.zeros_like(weight); ref=copy.deepcopy(state)
    pending=torch.zeros_like(weight); ref_pending=pending.clone(); splits=(0,8,21,35,e)
    for event in range(5):
        pulses=tuple(torch.rand(1,n,device=dev) for _ in range(4))
        if event == 0:
            pulses=tuple(torch.zeros_like(pulse) for pulse in pulses)
            pending[:3]=1e-6; ref_pending[:3]=1e-6
            state.rounding_residual[3:6]=1e-8; ref.rounding_residual[3:6]=1e-8
        signal=torch.randn(1,n,device=dev)*.1
        G=torch.rand(1,n,device=dev)*.3+.05; a=torch.exp(-G)
        bio={'alpha_eff':a,'v_pre':torch.randn(1,n,device=dev)*.1,'beta_int':(1-a)/G,'g_total':G}
        direct=torch.randn(1,n,device=dev)*.05
        decay=torch.rand(1,n,device=dev)*.3+.6
        spikes=(torch.rand(1,n,device=dev)>.7).float()
        psi=torch.rand(1,n,device=dev); beta=torch.rand(1,n,device=dev)*.2;rho=torch.rand(1,n,device=dev)*.1+.85
        apply=event%2==1 or event==0
        update_local_edges_pytorch(ref,ref_weight,ref_pending,pre,post,splits,pulses,signal,bio,direct,decay,spikes,psi,beta,rho,.001,apply)
        fused.update_local_coba_edges(state,weight,pending,pre,post,torch.stack(pulses),signal,bio,direct,decay,
                                     spikes,psi,beta,rho,splits,.001,apply,mask_dormant=mask_dormant)
        for name in ('conductance','voltage','adaptation','rounding_residual'):
            torch.testing.assert_close(getattr(state,name),getattr(ref,name),rtol=1e-5,atol=2e-7)
        torch.testing.assert_close(weight,ref_weight,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(pending,ref_pending,rtol=1e-5,atol=1e-8)
    w=torch.ones(3,5,device=dev); r=torch.zeros_like(w)
    h=torch.full_like(w,2**-27); factor=torch.ones(3,device=dev)
    for _ in range(32): fused.update_projection_weights(w,h,factor,1.,0.,r)
    torch.testing.assert_close(w.double()+r.double(),torch.full_like(w,1-32*2**-27).double(),rtol=0,atol=1e-14)
