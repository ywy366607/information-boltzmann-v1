"""Shared-mechanism numerical contracts, not trained synthetic abilities."""
from dataclasses import replace

import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence,Port,Variable,compose,log_score,mechanism,posterior
from scripts.ib_local.shared_relation_basis import SharedRelationBasis
from test_causal_relation_program import configured


def test_condition_does_not_change_with_action_and_silence_action_is_exact_zero():
    torch.manual_seed(83)
    basis=SharedRelationBasis(8)
    s=torch.randn(1,8).expand(9,-1)
    a=torch.cat((s[:1],torch.randn(8,8)),0)
    x=basis(s,a)
    assert x.shape==(9,4)
    torch.testing.assert_close(x[:,:2],x[:1,:2].expand(9,-1))
    assert torch.equal(x[0,2:],torch.zeros(2))
    assert basis(s,a,s).shape==(9,5)


def test_eight_facts_cannot_be_interpolated_by_condition_alone():
    torch.manual_seed(19)
    basis=SharedRelationBasis(8).double()
    x=basis(torch.randn(8,8,dtype=torch.float64),torch.randn(8,8,dtype=torch.float64))
    c,a=x[:,:2],x[:,2:]
    residual=a-c@torch.linalg.pinv(c)@a
    assert torch.linalg.matrix_rank(c)==2
    assert torch.linalg.matrix_rank(residual)==2
    assert torch.linalg.matrix_rank(x)==4


def test_shared_scale_is_differentiable_and_does_not_reset_to_unit_noise():
    x=torch.tensor([[1.,-1.],[1.,1.]]).repeat(4,1)
    y=torch.zeros(8,3)
    scale=torch.full((3,),.01,requires_grad=True)
    result=posterior(x,y,x[:1],.1,noise_prior=scale)
    torch.testing.assert_close(result['noise'],torch.full((3,),.002))
    (-log_score(result,torch.full((1,3),.1)).mean()).backward()
    assert scale.grad is not None and scale.grad.norm()>0
    assert not posterior(x,y,x[:1],.1)['noise'].requires_grad


def test_all_shared_roles_and_encoder_receive_real_composed_prediction_gradient():
    engine=configured();model=engine.model;m=model.memory
    model.relation_basis=SharedRelationBasis(model.encoder.out_features)
    s,a=m.source[:1].clone(),m.action[:1].clone()
    state=model.causal_relations.bind(model,s)
    path=compose(model,state,s,a,ridge=.1,reward_valid=torch.ones(m.capacity,dtype=torch.bool))
    labels=model.encoder(m.internal_next[:1])
    loss=sum(-log_score(path[name],labels).mean() for name in ('mother','decision','direct_decision'))
    loss.backward()
    assert model.encoder.weight.grad.norm()>0
    for name,value in model.relation_basis.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(),name
        assert value.grad.norm()>0,name
    assert path['mother']['x'].shape[1]==4
    assert path['decision']['x'].shape[1]==5


def test_cold_fact_is_excluded_by_identity_without_a_hot_slot():
    engine=configured();model=engine.model;m=model.memory
    model.relation_basis=SharedRelationBasis(model.encoder.out_features)
    s,a=m.source[:1],m.action[:1]
    state=model.causal_relations.bind(model,s)
    facts=tuple(Evidence(m.source[i:i+1],m.action[i:i+1],m.mother_next[i:i+1],
        Port.MOTHER,True,slot=-1,fact_id=f'episode:{i}') for i in range(3))
    state=replace(state,binding=Variable('q',Port.MOTHER),constraints=facts)
    result=mechanism(model,state,s,a,exclude_fact='episode:1')
    assert result['bound_evidence']==2
    assert result['x'].shape==(2,4)
    assert mechanism(model,state,s,a,legacy=True)['x'].shape==(3,2*model.encoder.out_features)


def test_shared_basis_roundtrip_keeps_predictions_and_noise():
    torch.manual_seed(91)
    a=SharedRelationBasis(8)
    b=SharedRelationBasis(8)
    b.load_state_dict(a.state_dict())
    source,action,mother=torch.randn(3,8),torch.randn(3,8),torch.randn(3,8)
    torch.testing.assert_close(a(source,action,mother),b(source,action,mother),rtol=0,atol=0)
    torch.testing.assert_close(a.noise(mother=True),b.noise(mother=True),rtol=0,atol=0)
