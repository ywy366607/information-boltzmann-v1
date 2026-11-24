"""Numerical contracts for the actual-body measurement, not capability tasks."""
import numpy as np
import torch

from scripts.audit_higher_order_dynamics import gradient, pairs, signals, omega, optimizer_regulation
from scripts.ib_local.adaptive_hamiltonian import potential
from information_boltzmann.budgeted_adamw import BudgetedAdamW


def saved_case():
    rng=np.random.default_rng(23)
    return dict(theta=rng.normal(0,.3,(2,7)),context=rng.normal(0,.1,(2,1,3,1,1)),
        frequency_scale=np.ones((2,1,3,1,1)),impedance=np.full((2,1,3,1,1),.8)),rng


def test_pair_force_includes_content_weight_gradient_of_production_potential():
    saved,rng=saved_case()
    s=rng.normal(0,.3,(2,3,2,2))
    x=torch.tensor(s[:,None],dtype=torch.float64,requires_grad=True)
    theta=torch.tensor(saved['theta'],dtype=torch.float64)
    value=potential(x,theta,3).sum()
    g,=torch.autograd.grad(value,x)
    np.testing.assert_allclose(gradient(s,saved['theta']),g[:,0].numpy(),atol=1e-14,rtol=1e-12)
    np.testing.assert_allclose(pairs(s,saved['theta'])[2].sum(),value.detach().numpy(),atol=1e-14)


def test_actual_pair_energy_flux_satisfies_node_energy_continuity_and_hodge_kernel():
    saved,rng=saved_case()
    s,p=rng.normal(0,.3,(2,3,2,2)),rng.normal(0,.3,(2,3,2,2))
    energy,j,harm,b=signals(s,p,saved)
    w=omega(saved,s)[...,None,None];z=saved['impedance'].reshape(w.shape)
    ds=w/z*p;dp=-w*z*s-gradient(s,saved['theta'])
    eps=1e-6
    derivative=(signals(s+eps*ds,p+eps*dp,saved)[0]-signals(s-eps*ds,p-eps*dp,saved)[0])/(2*eps)
    np.testing.assert_allclose(derivative,j@b.T,atol=1e-10,rtol=1e-6)
    np.testing.assert_allclose(harm@b.T,0,atol=1e-14)


def test_saved_gradient_budget_proposal_matches_real_pytorch_optimizer():
    saved,rng=saved_case()
    shape=saved['theta'].shape
    g=rng.normal(0,.2,shape)
    m=rng.normal(0,.05,shape);v=rng.uniform(.05,.1,shape)
    saved.update(real_gradient=g,optimizer_exp_avg=m,optimizer_exp_avg_sq=v,
        optimizer_budget_energy=np.array(.05),optimizer_budget_prior_rms=np.array(.02),
        optimizer_budget_gate=np.array(.6),optimizer_step=np.array(10.),
        optimizer_scalars={'budget_count':4},optimizer_group=dict(
            betas=(.9,.999),lr=3e-4,eps=1e-8,weight_decay=.01))
    result=optimizer_regulation(saved)

    def actual(gradient):
        p=torch.nn.Parameter(torch.tensor(saved['theta'],dtype=torch.float64))
        opt=BudgetedAdamW([p],**saved['optimizer_group'],cooling=True,fused=False)
        state=opt.state[p]
        for name in ('exp_avg','exp_avg_sq','budget_energy','budget_prior_rms','budget_gate','step'):
            state[name]=torch.tensor(saved['optimizer_'+name],dtype=torch.float64)
        state['budget_phase']=torch.tensor(0.,dtype=torch.float64)
        state['budget_count']=4
        p.grad=torch.tensor(gradient,dtype=torch.float64)
        opt.step()
        return p.detach().numpy()-saved['theta']

    target=result['target_coordinate'];a,b=result['perturbed_coordinates']
    d1=np.zeros_like(g);d2=np.zeros_like(g)
    d1.flat[a]=.05*g.flat[a];d2.flat[b]=.05*g.flat[b]
    baseline=actual(g)
    mixed=actual(g+d1+d2)-actual(g+d1)-actual(g+d2)+baseline
    np.testing.assert_allclose(result['target_step'],baseline.flat[target],atol=1e-14,rtol=1e-10)
    np.testing.assert_allclose(result['mixed_target_step'],mixed.flat[target],atol=1e-14,rtol=1e-6)
