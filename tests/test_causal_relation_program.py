"""Numerical, gradient, provenance and integration contracts; no task training."""
import copy

import pytest
import torch

from scripts.ib_local.causal_relation_program import (
    Evidence, Port, MOTHER, TRANSPORT, VALUE, check_program, compose, imagine,
    infer_role, log_score, mechanism, posterior, real_evidence, revise, roles)
from scripts.ib_local.causal_relation_program import Call, Variable, PortValue, execute
from scripts.ib_local.navigation_causal_program import migrate_causal_program
from test_navigation_persistent_question import candidate


def configured():
    e = candidate()
    e.joint_question_update = e.mother_shadow_enabled = True
    m = e.model.memory
    m.mother_valid.copy_(m.internal_valid)
    m.mother_stamp.copy_(m.last_write)
    m.mother_next.copy_(m.internal_next)
    e.internal_gain_state['count'][0] = 3
    e.internal_pending = None
    migrate_causal_program(e)
    return e


def test_conjugate_dual_solve_matches_primal_and_noise_scale():
    torch.manual_seed(10)
    x, y, q = torch.randn(6,4,dtype=torch.float64), torch.randn(6,3,dtype=torch.float64), torch.randn(2,4,dtype=torch.float64)
    p = posterior(x,y,q,.3)
    a = x.T@x + .3*torch.eye(4,dtype=x.dtype)
    b = torch.linalg.solve(a,x.T@y)
    leverage = (q*torch.linalg.solve(a,q.T).T).sum(-1)
    beta = 1+.5*(y.square().sum(0) - ((x.T@y)*b).sum(0))
    torch.testing.assert_close(p['mean'],q@b)
    torch.testing.assert_close(p['leverage'],leverage)
    torch.testing.assert_close(p['noise'],beta/4)
    expected = torch.distributions.StudentT(8,loc=p['mean'],scale=p['scale2'].sqrt()).log_prob(q[:,:3]).mean(-1)
    torch.testing.assert_close(log_score(p,q[:,:3]),expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_same_query_evidence_update_has_the_grounded_revision_identity(dtype):
    """Map the review's affine identity to the ACTUAL posterior function.

    This is a conjugate-algebra unit check, not a learning experiment. Other
    contexts need not improve, and the production evidence budget is bounded.
    """
    generator = torch.Generator().manual_seed(151115)
    x = torch.randn(5, 4, generator=generator, dtype=dtype)
    y = torch.randn(5, 3, generator=generator, dtype=dtype)
    query = torch.tensor([[1., .3, -.2, .7]], dtype=dtype)
    observed = torch.tensor([[.8, -.4, 1.2]], dtype=dtype)
    ridge = .3
    old = posterior(x, y, query, ridge)
    new = posterior(torch.cat((x, query)), torch.cat((y, observed)), query, ridge)
    precision = x.T @ x + ridge * torch.eye(x.shape[1], dtype=dtype)
    leverage = (query @ torch.linalg.solve(precision, query.T)).squeeze()
    gain = leverage / (1 + leverage)
    assert 0 < gain < 1
    torch.testing.assert_close(new['mean'], old['mean'] + gain * (observed - old['mean']))
    torch.testing.assert_close(new['mean'] - observed,
                               (1 - gain) * (old['mean'] - observed))


def test_empty_evidence_is_finite_and_role_axes_do_not_mix():
    p = posterior(torch.zeros(0,4),torch.zeros(0,3),torch.randn(2,4),.1)
    assert torch.isfinite(log_score(p,torch.randn(2,3))).all()
    s,a=torch.randn(2,4),torch.randn(2,4)
    bound=roles(s,a)
    torch.testing.assert_close(bound[:,0],s)
    torch.testing.assert_close(bound[:,1],a-s)
    with pytest.raises(ValueError):
        check_program((TRANSPORT,VALUE),(Port.CONDITION,Port.ACTION))
    assert Port.VALUE in check_program((MOTHER,TRANSPORT,VALUE),(Port.CONDITION,Port.ACTION))


def test_actual_variable_binding_is_shared_and_type_checked():
    condition,action=Variable('c',Port.CONDITION),Variable('a',Port.ACTION)
    unknown=Variable('same-x',Port.MOTHER)
    later=Variable('later',Port.DECISION)
    calls=(Call(MOTHER,(condition,action),unknown),Call(TRANSPORT,(condition,action,unknown),later))
    generators={MOTHER.name:lambda s,a:PortValue(s.tensor+a.tensor,{}),
                TRANSPORT.name:lambda s,a,y:PortValue(y.tensor*2,{})}
    bindings={condition:PortValue(torch.tensor(1.),{}),action:PortValue(torch.tensor(2.),{}),
              unknown:PortValue(torch.tensor(7.),{'actual':True})}
    values,trace=execute(calls,bindings,generators)
    assert trace[0][1].tensor==3 and values[unknown].tensor==7 and values[later].tensor==14
    with pytest.raises(ValueError):
        execute((Call(MOTHER,(action,condition),unknown),),bindings,generators)


def test_same_episode_ports_share_one_budget_and_real_commit_U():
    e=configured(); model=e.model; m=model.memory
    state=model.causal_relations.bind(model,m.source[:1])
    fact=Evidence(m.source[:1],m.action[:1],m.mother_next[:1],Port.MOTHER,True,slot=0,stamp=7)
    revised=revise(state,fact)
    decision=Evidence(m.source[:1],m.action[:1],m.internal_next[:1],Port.DECISION,True,
                      slot=0,stamp=7,mother=m.mother_next[:1])
    both=revise(revised,decision)
    assert torch.equal(revised.valid,both.valid)
    assert int(both.valid.sum())+1==len(state.valid)


def test_real_and_shadow_U_are_same_calculation_without_mutating_facts():
    e=configured(); model=e.model; m=model.memory
    s,a,y=torch.randn(1,m.width),torch.randn(1,m.width),torch.randn(1,m.width)
    state=model.causal_relations.bind(model,s)
    before=copy.deepcopy(model.state_dict())
    real=revise(state,Evidence(s,a,y,Port.MOTHER,observed=True))
    shadow=revise(state,Evidence(s,a,y,Port.MOTHER,observed=False))
    r=mechanism(model,real,s,a); f=mechanism(model,shadow,s,a)
    torch.testing.assert_close(r['mean'],f['mean'])
    for name,tensor in before.items():
        if torch.is_tensor(tensor): assert torch.equal(tensor,model.state_dict()[name])
    with pytest.raises(ValueError): model.causal_relations.commit(model,state,shadow.extra[0])
    with pytest.raises(ValueError): revise(real,real.extra[0])


def test_stable_binding_commit_and_recycled_fact_rejection():
    e=configured(); model=e.model; m=model.memory
    s=m.source[:1].clone(); a=m.action[:1].clone()
    state=model.causal_relations.bind(model,s)
    again=model.causal_relations.bind(model,s)
    assert state.identity==again.identity
    observed=Evidence(s,a,m.mother_next[:1],Port.MOTHER,True,slot=0,stamp=int(m.last_write[0]))
    assert model.causal_relations.commit(model,state,observed)
    assert not model.causal_relations.commit(model,state,observed) # old mechanism version
    current=model.causal_relations.state(state.index)
    m.last_write[0]+=1
    assert 0 not in real_evidence(model,current,Port.MOTHER).tolist()
    assert not model.causal_relations.commit(model,current,observed)


def test_entire_composition_has_real_evidence_gradients_and_typed_transport():
    e=configured(); model=e.model; m=model.memory
    s,a=torch.randn(1,m.width),torch.randn(1,m.width)
    state=model.causal_relations.bind(model,s)
    p=compose(model,state,s,a,ridge=.1,reward_valid=torch.ones(m.capacity,dtype=torch.bool))
    assert p['mother']['mean'].shape==p['decision']['mean'].shape==(1,model.encoder.out_features)
    loss=p['mother']['mean'].square().mean()+p['decision']['mean'].square().mean()
    loss.backward()
    assert model.encoder.weight.grad.norm()>0
    assert model.long_predictor[0].weight.grad.norm()>0
    assert model.mother_from_boundary.weight.grad.norm()>0
    changed=compose(model,state,s,a,ridge=.1,reward_valid=torch.ones(m.capacity,dtype=torch.bool),mother=p['mother']['mean'].detach()+1)
    assert not torch.allclose(p['decision']['mean'],changed['decision']['mean'])


def test_shadow_hypotheses_and_inverse_binding_leave_live_state_unchanged():
    e=configured(); model=e.model; m=model.memory
    s=m.source[:1].clone(); a=m.action[:2].clone()
    state=model.causal_relations.bind(model,s)
    before=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    p=imagine(model,state,s.expand_as(a),a,ridge=.1,reward_valid=torch.ones(m.capacity,dtype=torch.bool))
    assert p['hypotheses'].shape==(2,2,model.encoder.out_features)
    assert (p['information']>=0).all()
    assert (p['quadrature_information']>=-1e-6).all()
    assert (p['quadrature_information']<=torch.log(torch.tensor(2.))+1e-5).all()
    inverse=infer_role(model,state,open_role=Port.ACTION,condition=s,action=None,candidates=a,
                       observed=p['path']['mother']['mean'][:1],ridge=.1)
    torch.testing.assert_close(inverse['log_weights'].exp().sum(),torch.tensor(1.))
    assert torch.equal(rng,torch.get_rng_state())
    for name,tensor in before.items():
        if torch.is_tensor(tensor): assert torch.equal(tensor,model.state_dict()[name])


def test_registry_roundtrip_keeps_identity_evidence_and_pending():
    e=configured(); m=e.model.memory
    state=e.model.causal_relations.bind(e.model,m.source[:1])
    other=configured(); other.model.load_state_dict(copy.deepcopy(e.model.state_dict()))
    assert other.model.causal_relations.state(state.index).identity==state.identity
    assert e.causal_program_gain_archive['internal']['count'][0]==3
    assert e.internal_gain_state['count'][0]==0
