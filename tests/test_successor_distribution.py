"""Production successor integration and delayed-learning tensor contracts."""
from dataclasses import replace

import pytest
import torch

from scripts.ib_local import successor_distribution as sd
from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local.causal_relation_program import Port, compose, log_score, mechanism
from scripts.ib_local.relation_evidence import mechanism_evidence
from test_relation_retention_use import configured


def test_cubature_keeps_marginals_center_identity_and_rng():
    mean=torch.arange(12,dtype=torch.float64).reshape(3,4)
    distribution=dict(mean=mean,scale2=torch.ones_like(mean),df=5.)
    rng=torch.get_rng_state().clone()
    points,weights=sd.quadrature(distribution,4)
    assert points.shape==(3,4,4) and weights.sum()==1
    torch.testing.assert_close((points*weights[None,:,None]).sum(1),mean)
    torch.testing.assert_close(points[:,:2]-mean[:,None],-(points[:,2:]-mean[:,None]))
    perm=torch.tensor([2,0,1])
    reordered,_=sd.quadrature(dict(distribution,mean=mean[perm]),4)
    torch.testing.assert_close(reordered,points[perm],rtol=0,atol=0)
    thicker,_=sd.quadrature(dict(distribution,df=3.),4)
    assert bool(((thicker-mean[:,None]).abs()>=(points-mean[:,None]).abs()-1e-12).all())
    assert torch.equal(rng,torch.get_rng_state())
    collapsed,w=sd.quadrature(dict(distribution,scale2=torch.zeros_like(mean)),4)
    assert torch.equal(collapsed[:,0],mean) and w.tolist()==[1.]


def test_same_program_consensus_and_gradient(tmp_path):
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        st=e.internal_pending['causal_program']['state']
        s=e._internal_source.reshape(1,-1);action=e._internal_action[1:2]
        p=e._momentum_decision_read[None]
        mother=e.model.encoder(e._mother_boundary_target[None]).detach()
        mediated=mechanism(e.model,st,s,action,kind=Port.DECISION,mother=mother,ridge=e.relation_ridge)
        direct=mechanism(e.model,st,s,action,kind=Port.DECISION,momentum=p,ridge=e.relation_ridge)
        coordinated=sd.training_distribution(e.model,st,mediated,direct)
        path=compose(e.model,st,s,action,mother=mother,momentum=p,ridge=e.relation_ridge,
                     reward_valid=e.reward_label_valid(wood=True))
        for key in ('mean','scale2','df'):
            torch.testing.assert_close(coordinated[key],path['decision'][key])
        mediated['mean'].retain_grad();direct['mean'].retain_grad()
        (-log_score(coordinated,torch.ones_like(coordinated['mean'])).mean()).backward()
        assert mediated['mean'].grad.norm()>0 and direct['mean'].grad.norm()>0
    finally:a.close()


def test_distribution_native_loop_and_finite_post_U_learning(tmp_path):
    e,a=configured(tmp_path)
    try:
        e.gain_units_schema=1
        ev.configure(e,schema=4,word_execution='native_anchor')
        first=e.step(torch.randn(8,4))
        pending=e.internal_pending['causal_program']
        frozen=pending['decision_learning_state']
        assert frozen.organization is None and frozen.evidence_bound
        actual=pending['episode_identity']
        assert actual in [f.fact_id for f in mechanism_evidence(e.model,frozen,Port.MOTHER)]
        frozen_fact=(frozen.constraints+frozen.extra)[0]
        old=frozen_fact.source.clone()
        snapshot=sd.learning_state(frozen)
        copied=(snapshot.constraints+snapshot.extra)[0]
        assert copied.source.data_ptr()!=frozen_fact.source.data_ptr()
        assert torch.equal(old,copied.source)
        second=e.step(torch.randn(8,4))
        assert second['causal_program']['decision_prequential'] is not None
        learned=second['causal_program']['successor_learning']
        assert learned['event']==actual and learned['target_excluded']
        assert actual in learned['evidence']
        # Original delayed event identity remains the exclusion target.
        assert e.successor_distribution_last['branches']==2*(e.joint_proposal_budget//2)
        assert all(torch.isfinite(torch.tensor(e.successor_distribution_last['values'])).flatten())
        assert first['world'].get('provider_calls',0)==second['world'].get('provider_calls',0)==0
    finally:a.close()
