"""Coherent successor law and the production complete-loop interfaces."""
import torch

from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local.causal_relation_program import posterior,log_score,Port,mechanism,compose
from scripts.ib_local.joint_successor import coordinate,reduce_cells,standardized
from scripts.ib_local.successor_distribution import training_distribution
from test_relation_retention_use import configured


def test_joint_score_scalar_limit_and_frozen_draw_rng():
    location=torch.tensor([[.1],[.8]],dtype=torch.float64)
    scale=torch.tensor([[.3],[1.2]],dtype=torch.float64)
    d=dict(mean=location,scale2=scale,df=9.)
    target=torch.tensor([[.4],[-.3]],dtype=torch.float64)
    torch.testing.assert_close(log_score(dict(d,scale_matrix=scale[:,:,None]),target),log_score(d,target))
    rng=torch.get_rng_state().clone()
    draw=standardized(9.,13)
    assert bool(torch.isfinite(draw).all()) and torch.equal(rng,torch.get_rng_state())
    assert torch.equal(draw,standardized(9.,13))


def test_coherent_matrix_keeps_marginals_and_joint_score_rotates():
    rng=torch.Generator().manual_seed(11)
    x=torch.randn(7,4,generator=rng,dtype=torch.float64)
    y=torch.randn(7,12,generator=rng,dtype=torch.float64)
    q=torch.randn(3,4,generator=rng,dtype=torch.float64)
    a=posterior(x,y,q,.1,noise_prior=torch.ones(12,dtype=x.dtype))
    b=posterior(x,y*.7,q,.1,noise_prior=torch.ones(12,dtype=x.dtype))
    a['mean'].requires_grad_();b['mean'].requires_grad_()
    weights=torch.tensor([.4,.6],dtype=x.dtype)
    d=coordinate((a,b),weights)
    torch.testing.assert_close(d['scale_matrix'].diagonal(dim1=-2,dim2=-1),d['scale2'])
    assert torch.linalg.eigvalsh(d['scale_matrix']).min()>0
    target=torch.randn(3,12,generator=rng,dtype=x.dtype)
    rotation=torch.linalg.qr(torch.randn(12,12,generator=rng,dtype=x.dtype)).Q
    rotated=dict(d,mean=d['mean']@rotation,scale_matrix=rotation.T@d['scale_matrix']@rotation)
    torch.testing.assert_close(log_score(d,target),log_score(rotated,target@rotation))
    (-log_score(d,target).mean()).backward()
    assert a['mean'].grad.norm()>0 and b['mean'].grad.norm()>0


def test_cell_quadrature_preserves_mass_and_native_representatives():
    points=torch.arange(30,dtype=torch.float32).reshape(10,3)
    cells=torch.tensor([-1,-1,-1,-1,7,7,7,8,8,9])
    reduced,weights,trace=reduce_cells(points,points,cells,4)
    assert weights.sum()==1 and len(reduced)==4
    values={-1:2.,7:4.,8:1.,9:-2.}
    predicted=sum(float(w)*values[t['representative_cell']] for w,t in zip(weights,trace))
    assert abs(predicted-sum(values[int(i)] for i in cells)/10)<1e-6
    for p,t in zip(reduced,trace):
        index=(points==p).all(-1).nonzero()[0,0]
        assert int(cells[index])==t['representative_cell']
    fewer,w,t=reduce_cells(points,points,cells,2)
    assert len(fewer)==2 and w.sum()==1 and t[-1]['stratum']=='remaining_mass'


def test_real_native_joint_cycle_uses_same_law_for_late_learning(tmp_path):
    e,a=configured(tmp_path)
    try:
        e.gain_units_schema=1;ev.configure(e,schema=5,word_execution='native_anchor')
        first=e.step(torch.randn(8,4))
        assert e.successor_distribution_last['joint']['samples']==4096
        assert e.successor_distribution_last['branches']<=e.joint_proposal_budget
        second=e.step(torch.randn(8,4))
        learned=second['causal_program']['successor_learning']
        assert learned['target_excluded'] and learned['event'] in learned['evidence']
        state=e.internal_pending['causal_program']['state']
        source=e._internal_source[None];action=e._internal_action[1:2]
        mother=e.model.encoder(e._mother_boundary_target[None]).detach()
        p=e._momentum_decision_read[None]
        mediated=mechanism(e.model,state,source,action,kind=Port.DECISION,mother=mother,ridge=e.relation_ridge)
        direct=mechanism(e.model,state,source,action,kind=Port.DECISION,momentum=p,ridge=e.relation_ridge)
        trained=training_distribution(e.model,state,mediated,direct)
        path=compose(e.model,state,source,action,mother=mother,momentum=p,ridge=e.relation_ridge,
            reward_valid=e.reward_label_valid(wood=True))
        for key in ('mean','scale2','scale_matrix','df'):
            torch.testing.assert_close(trained[key],path['decision'][key])
    finally:a.close()
