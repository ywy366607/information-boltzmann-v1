"""Production-function numerical/whole-transition contracts, no ability task."""
import copy
from dataclasses import replace

import pytest
import torch

from scripts.ib_local.causal_relation_program import compose, real_evidence, Port, Evidence, revise, mechanism
from scripts.ib_local.generative_relations import marginal, PROGRAMS, EXTRA_FIELDS
from scripts.ib_local.navigation_generative_cycle import configure_generative_cycle
from test_joint_world_cycle import complete_interface_engine


def configured():
    e = complete_interface_engine()
    configure_generative_cycle(e)
    return e


def test_whole_migration_keeps_original_facts_parameters_rng_and_optimizer():
    e = complete_interface_engine()
    state = copy.deepcopy(e.model.state_dict())
    parameters = [id(p) for p in e.model.parameters()]
    rng = torch.get_rng_state().clone()
    configure_generative_cycle(e)
    assert torch.equal(rng, torch.get_rng_state())
    assert parameters == [id(p) for p in e.model.parameters()][:len(parameters)]
    for k, v in state.items():
        if torch.is_tensor(v):
            assert torch.equal(v, e.model.state_dict()[k])
    ids = {id(p) for group in e.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in e.model.parameters() if p.requires_grad)


def test_typed_program_weights_choose_executed_shared_operations_and_can_relearn():
    e = configured(); model, m = e.model, e.model.memory
    s = torch.randn(1, m.width); a = s+.1
    state = model.causal_relations.bind(model, s)
    kwargs = dict(ridge=.1, reward_valid=torch.ones(m.capacity,dtype=torch.bool))
    mediated = compose(model, replace(state,program_weights=s.new_tensor([1.,0.])),s,a,**kwargs)
    direct = compose(model, replace(state,program_weights=s.new_tensor([0.,1.])),s,a,**kwargs)
    torch.testing.assert_close(mediated['decision']['mean'],mediated['program_predictions'][0]['mean'])
    torch.testing.assert_close(direct['decision']['mean'],direct['direct_decision']['mean'])
    bank = model.causal_relations
    bank.observe_program(state,s.new_tensor([3.,-3.]))
    assert bank.state(state.index).program_weights[0] > .99
    for _ in range(bank.evidence_budget):
        bank.observe_program(state,s.new_tensor([-3.,3.]))
    assert bank.state(state.index).program_weights[1] > .99
    assert len(PROGRAMS) == len(bank.state(state.index).program_weights)


def test_evidence_score_contains_occam_term_and_is_order_invariant():
    torch.manual_seed(2)
    x,y = torch.randn(6,4,dtype=torch.float64),torch.randn(6,3,dtype=torch.float64)
    torch.testing.assert_close(marginal(x,y,.1),marginal(x.flip(0),y.flip(0),.1))
    assert torch.isfinite(marginal(x[:0],y[:0],.1))


def test_whole_cycle_runs_binding_internal_choice_words_real_U_and_delayed_credit():
    torch.set_num_threads(1)
    e = configured()
    row = e.step(torch.randn(8,4))
    g = row['generative_relation']
    assert g['query_role'] in ('mother','action','condition')
    assert g['internal_admission'] and len(g['internal_admission']) <= e.joint_proposal_budget
    assert row['causal_program']['evidence_committed']
    assert row['joint_language_search']['replay_max_error'] < 1e-3
    assert row['world_learning']['external_bytes_once'] == 128
    assert g['same_U'] and g['observed_response_bound']
    assert any(t is not None and t.abs().sum()>0 for t in e.prev['actor'][-1:])
    first_weights = e.model.binding_choice.weight.detach().clone()
    second = e.step(torch.randn(8,4))
    assert second['generative_relation']['program_evidence_updated']
    assert not torch.equal(first_weights, e.model.binding_choice.weight)


def test_complete_save_restore_next_whole_event_is_identical():
    torch.set_num_threads(1)
    from scripts.ib_local.navigation_checkpoint import snapshot,restore
    e = configured()
    e.step(torch.randn(8,4))
    saved = snapshot(e)
    a = e.step(torch.randn(8,4))
    weights = copy.deepcopy(e.model.state_dict())
    fresh = configured()
    restore(fresh,saved)
    b = fresh.step(torch.randn(8,4))
    assert a['choice'] == b['choice'] and a['utterance_hex'] == b['utterance_hex']
    assert a['generative_relation'] == b['generative_relation']
    assert a['metal_g'] == b['metal_g']
    for k,v in weights.items():
        if torch.is_tensor(v):
            assert torch.equal(v,fresh.model.state_dict()[k]),k


def test_disk_roundtrip_restores_program_evidence_and_family_without_reinitializing(tmp_path):
    from scripts.ib_local.causal_archive import attach_archive
    e = configured(); model=e.model; bank=model.causal_relations
    state=bank.bind(model,torch.randn(1,model.memory.width))
    bank.observe_program(state,torch.tensor([-2.,2.]))
    archive=attach_archive(e,tmp_path)
    try:
        expected={name:getattr(bank,name)[state.index].clone() for name in EXTRA_FIELDS}
        archive.sync()
        e.prev=e.internal_pending=None
        bank.identity[state.index]=0
        bank.program_owner[state.index]=0
        assert archive.warm_relation(state.identity)
        i=int(torch.nonzero(bank.identity==state.identity)[0])
        for name,value in expected.items():
            assert torch.equal(value,getattr(bank,name)[i]),name
    finally:
        archive.close()


def test_refutation_changes_scope_binding_and_same_U_matches_observed_shadow_and_commit():
    e=configured();model=e.model;m=model.memory;bank=model.causal_relations
    # Numerical scope test: collinear addresses with two incompatible output
    # relations. This is not a learned task or a capability score.
    with torch.no_grad():
        for i in range(4):
            m.occupied[i]=True;m.count[i]=1;m.last_write[i]=i+1
            m.mother_valid[i]=True;m.mother_stamp[i]=i+1
            m.source[i].zero_();m.source[i,0]=i+1
            m.action[i].copy_(m.source[i]);m.mother_next[i].zero_()
            m.mother_next[i,0]=-1000 if i<2 else 1000
    state=bank.bind(model,m.source[3:4])
    state=replace(state,valid=state.valid & (state.slots!=3))
    actual=Evidence(m.source[3:4],m.action[3:4],m.mother_next[3:4],Port.MOTHER,
                    True,slot=3,stamp=4)
    imagined=replace(actual,observed=False,encoded=True,outcome=model.encoder(actual.outcome))
    real=revise(state,actual,model=model)
    shadow=revise(state,imagined,model=model)
    assert real.revision_plan['operation']==shadow.revision_plan['operation']=='split'
    assert torch.equal(real.valid,shadow.valid)
    kwargs=dict(kind=Port.MOTHER,ridge=.1)
    a=mechanism(model,real,actual.source,actual.action,**kwargs)['mean']
    b=mechanism(model,shadow,actual.source,actual.action,**kwargs)['mean']
    torch.testing.assert_close(a,b)
    assert bank.commit(model,state,actual)
    child=bank.bind(model,m.source[:1]);parent=bank.bind(model,m.source[3:4])
    assert parent.identity==state.identity and child.identity!=parent.identity
    assert int(bank.parent[child.index])==parent.identity
    after=mechanism(model,parent,actual.source,actual.action,**kwargs)['mean']
    torch.testing.assert_close(a,after)


def test_imagined_known_response_is_bound_into_transport():
    e=configured();m=e.model.memory
    s=torch.randn(1,m.width);a=s+.2
    state=e.model.causal_relations.bind(e.model,s)
    weights=s.new_tensor([1.,0.])
    state=replace(state,program_weights=weights)
    kwargs=dict(ridge=.1,reward_valid=torch.ones(m.capacity,dtype=torch.bool))
    before=compose(e.model,state,s,a,**kwargs)
    response=before['mother']['mean'].detach()+1
    changed=compose(e.model,state,s,a,mother=response,**kwargs)
    assert not torch.allclose(before['decision']['mean'],changed['decision']['mean'])


def test_full_budget_compatible_relations_share_without_growing_or_dropping_new_fact():
    e=configured();model=e.model;m=model.memory;bank=model.causal_relations
    count=bank.evidence_budget
    with torch.no_grad():
        for module in (model.long_predictor,model.mother_from_boundary):
            for p in module.parameters():p.zero_()
        for i in range(2*count):
            m.occupied[i]=True;m.count[i]=1;m.last_write[i]=i+1
            m.mother_valid[i]=True;m.mother_stamp[i]=i+1
            m.source[i].zero_();m.source[i,0]=i+1
            m.action[i].copy_(m.source[i]);m.mother_next[i].zero_()
        for j in (0,1):
            bank.identity[j]=j+1;bank.generation[j]=1
            bank.condition[j].copy_(m.source[j*count])
            bank.slots[j].copy_(torch.arange(j*count,(j+1)*count))
            bank.stamps[j].copy_(m.last_write[bank.slots[j]])
            bank.valid[j].fill_(True)
        bank.next_identity.fill_(3)
    state=replace(bank.state(0),release=0)
    slot=2*count
    with torch.no_grad():
        m.occupied[slot]=True;m.count[slot]=1;m.last_write[slot]=slot+1
        m.mother_valid[slot]=True;m.mother_stamp[slot]=slot+1
        m.source[slot].copy_(m.source[0]);m.action[slot].copy_(m.source[0])
        m.mother_next[slot].zero_()
    evidence=Evidence(m.source[slot:slot+1],m.action[slot:slot+1],m.mother_next[slot:slot+1],
                      Port.MOTHER,True,slot=slot,stamp=slot+1)
    revised=revise(state,evidence,model=model)
    assert revised.revision_plan['operation']=='share'
    assert bank.commit(model,state,evidence)
    assert int(bank.valid[0].sum())==int(bank.valid[1].sum())==count
    assert torch.equal(bank.slots[0],bank.slots[1])
    assert slot in bank.slots[0].tolist()
    assert bank.identity[0]!=bank.identity[1] and bank.family[0]==bank.family[1]


def test_scoped_cold_search_and_new_binding_do_not_reuse_nonapplicable_pinned_identity(tmp_path):
    from scripts.ib_local.causal_archive import attach_archive
    e=configured();bank=e.model.causal_relations;m=e.model.memory
    source=torch.zeros(1,m.width);source[0,0]=1
    first=bank.bind(e.model,source)
    bank.scope_axes[first.index,0,0]=1
    bank.scope_bounds[first.index,0]=0
    bank.scope_signs[first.index,0]=1
    bank.scope_valid[first.index,0]=True
    bank._protected_identities={first.identity}
    archive=attach_archive(e,tmp_path)
    try:
        archive.sync()
        assert archive.best_relation(source) is None
        next_state=bank.bind(e.model,source)
        assert next_state.identity!=first.identity
        assert bank.identity[first.index]==first.identity
    finally:archive.close()
