"""Production state/gradient/restore contracts; no synthetic capability claims."""
import copy
from dataclasses import replace

import pytest
import torch

from scripts.ib_local.causal_archive import attach_archive
from scripts.ib_local.causal_relation_program import Evidence,Port,compose,mechanism,revise
from scripts.ib_local.generative_relations import dataset,marginal
from scripts.ib_local.joint_relation_update import (
    configure_joint_relation_update,overlap_consensus,register_use,relation_variables,share_variables)
from scripts.ib_local.navigation_generative_cycle import configure_generative_cycle
from scripts.ib_local.shared_relation_basis import SharedRelationBasis
from scripts.ib_local.shared_unknown_bindings import (
    _row,_with_constraints,commit,configure_shared_unknowns)
from test_joint_world_cycle import complete_interface_engine


def body(tmp_path,revision=2):
    torch.set_num_threads(1)
    e=complete_interface_engine();configure_generative_cycle(e)
    e.model.relation_basis=SharedRelationBasis(e.model.encoder.out_features)
    e.optimizer.add_param_group(dict(params=list(e.model.relation_basis.parameters()),lr=3e-4))
    e.params=[p for p in e.model.parameters() if p.requires_grad]
    e.groups=(*e.groups[:2],[p for n,p in e.model.named_parameters() if not n.startswith('body.')])
    a=attach_archive(e,tmp_path,relation_capacity=e.model.causal_relations.capacity)
    configure_shared_unknowns(e);configure_joint_relation_update(e,revision=revision)
    return e,a


@pytest.mark.parametrize('cap', [None, 6, 7, 8])
def test_private_reference_cap_restores_and_only_prunes_writes(cap):
    import json
    import sqlite3
    from types import SimpleNamespace as NS

    from scripts.ib_local.joint_relation_update import configure_reference_write_cap

    with sqlite3.connect(':memory:') as db:
        db.executescript('''CREATE TABLE relations(id INTEGER PRIMARY KEY, generation INTEGER);
            CREATE TABLE relation_variables(relation_id INTEGER, unknown_id TEXT, port TEXT,
            generation INTEGER, seen_version INTEGER, touched INTEGER,
            PRIMARY KEY(relation_id, unknown_id));''')
        db.executemany('INSERT INTO relations VALUES (?,1)', [(1,), (2,), (3,)])
        db.executemany('INSERT INTO relation_variables VALUES (?,?,?,?,?,?)',
                       [(rid, str(i), 'condition', 1, 0, i)
                        for rid in (1, 2, 3) for i in range(8)])
        archive = NS(db=db, sync=lambda: None)

        def engine():
            return NS(model=NS(joint_relation_update_schema=1, _causal_archive=archive,
                               causal_relations=NS(evidence_budget=8), memory=NS(clock=9)))

        config = {}
        configure_reference_write_cap(engine(), config, cap)
        restored = engine()
        config = json.loads(json.dumps(config))
        configure_reference_write_cap(restored, config)
        counts = lambda: db.execute('SELECT COUNT(*) FROM relation_variables '
                                   'GROUP BY relation_id ORDER BY relation_id').fetchall()
        assert counts() == [(8,), (8,), (8,)]
        if cap is not None:
            with pytest.raises(ValueError, match='cannot change'):
                configure_reference_write_cap(restored, config, 7 if cap == 8 else 8)
        register_use(restored, NS(identity=1, generation=1),
                     dict(identity='new', version=0, payload={'role': 'condition'}))
        assert counts() == [(cap or 8,), (8,), (8,)]
        share_variables(restored.model, 1, 2)
        assert counts() == [(cap or 8,), (cap or 8,), (8,)]
        assert restored.model.causal_relations.evidence_budget == 8
        assert db.execute('SELECT DISTINCT unknown_id FROM relation_variables '
                          'WHERE relation_id IN (1,2) ORDER BY unknown_id').fetchall() == [
                              (str(i),) for i in range(9 - (cap or 8), 8)] + [('new',)]


def test_evidence_score_generalized_prior_matches_sequential_prediction():
    from scripts.ib_local.causal_relation_program import posterior
    torch.manual_seed(5)
    x=torch.randn(7,4,dtype=torch.float64);y=torch.randn(7,3,dtype=torch.float64)
    prior=torch.tensor([.01,.2,.7],dtype=x.dtype)
    score=0
    for i in range(len(x)):
        d=posterior(x[:i],y[:i],x[i:i+1],.1,noise_prior=prior)
        score+=torch.distributions.StudentT(d['df'],d['mean'],d['scale2'].sqrt()).log_prob(y[i:i+1]).sum()
    torch.testing.assert_close(score,marginal(x,y,.1,noise_prior=prior))


def test_overlap_solve_descends_keeps_disagreement_and_propagates_gradients():
    a=torch.tensor([[0.,2.]],requires_grad=True);b=torch.tensor([[4.,-2.]],requires_grad=True)
    p=dict(mean=a,scale2=torch.ones_like(a),df=10.)
    q=dict(mean=b,scale2=torch.full_like(b,4),df=10.)
    out=overlap_consensus((p,q),a.new_tensor([.5,.5]))
    torch.testing.assert_close(out['mean'],.8*a+.2*b)
    assert bool((out['objective_after']<=out['objective_before']).all())
    assert bool((out['disagreement']>0).all())
    out['mean'].sum().backward();assert a.grad.norm()>0 and b.grad.norm()>0
    same=overlap_consensus((p,p),a.new_tensor([.5,.5]))
    torch.testing.assert_close(same['scale2'],p['scale2'])
    endpoint=overlap_consensus((p,q),a.new_tensor([0.,1.]))
    torch.testing.assert_close(endpoint['mean'],b)
    torch.testing.assert_close(endpoint['scale2'],q['scale2'])
    with pytest.raises(ValueError):overlap_consensus((p,q),a.new_tensor([0.,0.]))


def test_complete_cycle_uses_joint_U_and_coordinated_boundary(tmp_path):
    e,a=body(tmp_path)
    try:
        first=e.step(torch.randn(8,4));second=e.step(torch.randn(8,4))
        for r in (first,second):
            assert r['shared_unknown']['revision']['committed']
            assert r['joint_language_search']['replay_max_error']<1e-3
            assert r['generative_relation']['binding_replay_error']<1e-5
        state=e._program_question_state
        assert state.organization is not None
        xx,*_=dataset(e.model,state.organization)
        assert xx.shape[1]==4
        s=e.model.memory.source[:1];kwargs=dict(ridge=.1,reward_valid=e.reward_label_valid(wood=True))
        path=compose(e.model,state,s,s,**kwargs)
        assert 'overlap_coordination' in path
        assert bool((path['overlap_coordination']['objective_after']<=
                     path['overlap_coordination']['objective_before']+1e-5).all())
        assert a.db.execute('SELECT COUNT(*) FROM relation_variables').fetchone()[0]>0
        assert a.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:a.close()


@pytest.mark.parametrize('revision',[2,3,4])
def test_real_and_shadow_have_same_organization_and_never_write_imagined_edges(tmp_path,revision):
    e,a=body(tmp_path,revision)
    try:
        e.step(torch.randn(8,4))
        state=_with_constraints(a,e._program_question_state,
                                _row(a,e._shared_question_receipt['identity']))
        fact=next(f for f in state.constraints if f.kind==Port.MOTHER)
        real=Evidence(fact.source+.1,fact.action+.2,fact.outcome+.3,Port.MOTHER,
                      True,slot=987,stamp=123,fact_id='episode:123456')
        shadow=replace(real,observed=False,encoded=True,outcome=e.model.encoder(real.outcome))
        before=a.db.total_changes
        r=revise(state,real,model=e.model);s=revise(state,shadow,model=e.model)
        assert r.organization.revision_plan['operation']==s.organization.revision_plan['operation']
        assert torch.equal(r.organization.valid,s.organization.valid)
        torch.testing.assert_close(mechanism(e.model,r,fact.source,fact.action)['mean'],
                                   mechanism(e.model,s,fact.source,fact.action)['mean'])
        assert a.db.total_changes==before
    finally:a.close()


def test_same_variable_changes_two_relations_but_not_frozen_participation(tmp_path):
    e,a=body(tmp_path)
    try:
        first=e.step(torch.randn(8,4));state=e._program_question_state
        q=_row(a,first['shared_unknown']['identity']);bank=e.model.causal_relations
        free=int(torch.nonzero(bank.identity==0)[0]);rid=int(bank.next_identity)
        with torch.no_grad():
            bank.identity[free]=rid;bank.generation[free]=1;bank.next_identity+=1
            bank.condition[free].copy_(bank.condition[state.index])
        other=bank.state(free)
        register_use(e,state,q)
        # Only the accepted real sharing operation transfers the use to a
        # second scope; it must not invent an unrelated new unknown identity.
        share_variables(e.model,state.identity,other.identity)
        frozen=relation_variables(e.model,state)
        left=relation_variables(e.model,state);right=relation_variables(e.model,other)
        s=e.model.memory.source[:1].clone();act=e.model.memory.action[:1].clone()
        # Wait only for an actual event not already in this question's
        # support. Recommitting a duplicate is correctly prediction-idempotent.
        for _ in range(8):
            row=e.step(torch.randn(8,4))
            eid=row['shared_unknown']['revision']['observed_episode']
            known={v[0] for v in a.db.execute(
                'SELECT episode_id FROM unknown_evidence WHERE unknown_id=?',(q['identity'],))}
            if eid not in known:break
        else:pytest.fail('fixture produced no genuinely new evidence for the selected question')
        # Isolate THIS variable's intervention. Real sharing may have already
        # supplied the same event through another question, which should not
        # create a second numerical effect when the fact is deduplicated.
        with a.db:
            a.db.execute('DELETE FROM relation_variables WHERE relation_id IN (?,?) AND unknown_id<>?',
                         (state.identity,other.identity,q['identity']))
        # Hold the now-current encoder and every weight fixed across the
        # intervention below. The effect must come from the new constraint,
        # not from the intervening world's ordinary parameter update.
        before=mechanism(e.model,relation_variables(e.model,state),s,act)['mean'].detach().clone()
        # Use this genuinely observed event as a new constraint on the same
        # persistent question, without relabelling an inverse action as success.
        q=_row(a,q['identity'])
        e._program_question_state=_with_constraints(a,state,q)
        e._shared_question_receipt=dict(identity=q['identity'],version=q['version'])
        changed=commit(e,row);assert changed['committed']
        after_left=relation_variables(e.model,state);after_right=relation_variables(e.model,other)
        assert after_left.variable_uses[0][1]==after_right.variable_uses[0][1]==changed['version_after']
        assert frozen.variable_uses[0][1]<changed['version_after']
        x=mechanism(e.model,after_left,s,act)['mean'];y=mechanism(e.model,after_right,s,act)['mean']
        torch.testing.assert_close(x,y)
        assert not torch.allclose(x,before)
        stale=relation_variables(e.model,replace(other,generation=999))
        assert not stale.variable_uses
        assert len({f.fact_id for f in after_left.constraints})<=len(state.valid)
    finally:a.close()


@pytest.mark.parametrize('revision',[2,3,4])
def test_full_body_and_dependency_archive_restore_same_next_event(tmp_path,revision):
    import gzip
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
    from test_joint_world_cycle import assert_same_saved_values
    e,a=body(tmp_path/'live',revision);restored_archive=None
    try:
        e.step(torch.randn(8,4))
        config=dict(joint_relation_update={'schema':1,'implementation_revision':revision},shared_unknowns={'schema':1})
        save(e,tmp_path/'saved',1,config)
        with gzip.open(tmp_path/'saved/last.pt.gz','rb') as stream:
            state=torch.load(stream,map_location='cpu',weights_only=False)
        expected=e.step(torch.randn(8,4))
        model=copy.deepcopy(e.model.state_dict());optimizer=copy.deepcopy(e.optimizer.state_dict())
        other,unused=body(tmp_path/'fresh',revision);unused.close();del other.model._causal_archive
        restore(other,state)
        restored_archive=attach_archive(other,tmp_path/'restored',
            manifest=state['config']['causal_archive'],relation_capacity=a.budgets['relations'])
        configure_shared_unknowns(other);configure_joint_relation_update(other,revision=revision)
        actual=other.step(torch.randn(8,4))
        for name in ('choice','utterance_hex','shared_unknown','joint_relation_update','metal_g'):
            assert expected[name]==actual[name],name
        assert_same_saved_values(model,other.model.state_dict())
        assert_same_saved_values(optimizer,other.optimizer.state_dict())
        assert restored_archive.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:
        a.close()
        if restored_archive is not None:restored_archive.close()


def test_canonical_prediction_organization_and_cache_interventions_agree(tmp_path):
    e,a=body(tmp_path,3)
    try:
        for _ in range(2):e.step(torch.randn(8,4))
        state=e._program_question_state
        assert state.evidence_bound and state.organization.evidence_bound
        fact=next(f for f in state.constraints if f.kind==Port.MOTHER)
        d=mechanism(e.model,state,fact.source,fact.action)
        x,y,_,refs=dataset(e.model,state.organization)
        torch.testing.assert_close(x,d['x']);torch.testing.assert_close(y,d['y'])
        assert len(set(refs))<=len(state.valid)
        changed=replace(fact,outcome=fact.outcome+.2)
        replace_fact=lambda fs:tuple(changed if f is fact else f for f in fs)
        altered=replace(state,constraints=replace_fact(state.constraints),
            organization=replace(state.organization,constraints=replace_fact(state.organization.constraints)))
        nd=mechanism(e.model,altered,fact.source,fact.action)
        nx,ny,*_=dataset(e.model,altered.organization)
        torch.testing.assert_close(nx,nd['x']);torch.testing.assert_close(ny,nd['y'])
        assert not torch.allclose(d['y'],nd['y'])
        # An unrelated live cache value must affect neither canonical view.
        e.model.memory.mother_next.add_(.7)
        again=mechanism(e.model,state,fact.source,fact.action)
        _,again_y,*_=dataset(e.model,state.organization)
        torch.testing.assert_close(again['y'],d['y']);torch.testing.assert_close(again_y,y)
    finally:a.close()


def test_canonical_scope_retains_cold_fact_after_hot_eviction(tmp_path):
    e,a=body(tmp_path,3)
    try:
        e.step(torch.randn(8,4));bank=e.model.causal_relations
        state=bank.state(e._program_question_state.index)
        fact=next(f for f in state.constraints if f.kind==Port.MOTHER)
        eid=int(fact.fact_id.split(':')[1]);slot=fact.slot
        assert slot>=0
        before=mechanism(e.model,state,fact.source,fact.action)['mean'].detach().clone()
        with torch.no_grad():e.model.memory.occupied[slot]=False
        a.sync()
        assert a.db.execute('SELECT 1 FROM evidence WHERE relation_id=? AND episode_id=?',
                            (state.identity,eid)).fetchone()
        cold=bank.state(state.index)
        assert any(f.fact_id==fact.fact_id and f.slot==-1 for f in cold.constraints)
        torch.testing.assert_close(mechanism(e.model,cold,fact.source,fact.action)['mean'],before)
        assert a.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:a.close()


@pytest.mark.parametrize('revision',[3,4])
def test_recycled_hot_address_cannot_exclude_a_different_frozen_fact(tmp_path,revision):
    """Excluding today's label must not drop yesterday's frozen participant."""
    from scripts.ib_local.relation_evidence import mechanism_evidence
    e,a=body(tmp_path,revision)
    try:
        e.step(torch.randn(8,4))
        state=e.model.causal_relations.state(e._program_question_state.index)
        fact=next(f for f in state.constraints if f.kind==Port.MOTHER and f.slot>=0)
        before=mechanism_evidence(e.model,state,Port.MOTHER)
        # Simulate reuse of the finite cache address after this immutable
        # participation snapshot was taken; the old fact stays on SSD/in it.
        with torch.no_grad():e.model.memory.last_write[fact.slot]=fact.stamp+1
        after=mechanism_evidence(e.model,state,Port.MOTHER,exclude=fact.slot)
        expected=before if revision==4 else tuple(f for f in before if f.slot!=fact.slot)
        assert [f.fact_id for f in after]==[f.fact_id for f in expected]
        dist=mechanism(e.model,state,fact.source,fact.action,exclude=fact.slot)
        assert len(dist['x'])==len(expected)
    finally:a.close()


def test_qualified_exclusion_uses_fact_before_address_and_excludes_own_label(tmp_path):
    from scripts.ib_local.relation_evidence import mechanism_evidence
    e,a=body(tmp_path,4)
    try:
        e.step(torch.randn(8,4))
        state=e.model.causal_relations.state(e._program_question_state.index)
        fact=next(f for f in state.constraints if f.kind==Port.MOTHER and f.slot>=0)
        assert not mechanism_evidence(e.model,state,Port.MOTHER,exclude=fact.slot)
        # An explicit different ID takes precedence over a coincident address.
        assert mechanism_evidence(e.model,state,Port.MOTHER,exclude=fact.slot,
                                  exclude_fact='episode:999999')
        with torch.no_grad():e.model.memory.last_write[fact.slot]+=1
        assert not mechanism_evidence(e.model,state,Port.MOTHER,exclude_fact=fact.fact_id)
        assert len(mechanism(e.model,state,fact.source,fact.action,
                             exclude_fact=fact.fact_id)['x'])==0
    finally:a.close()


def test_delayed_fact_identity_migrates_valid_receipt_and_rejects_stale_fallback(tmp_path):
    from scripts.ib_local.relation_evidence import pending_fact_identity
    e,a=body(tmp_path,4)
    try:
        first=e.step(torch.randn(8,4));pending=e.internal_pending
        identity=pending['causal_program']['episode_identity']
        assert identity==first['causal_program']['committed_episode_identity']
        assert pending_fact_identity(e.model,pending)==identity
        legacy=copy.deepcopy(pending);legacy['causal_program'].pop('episode_identity')
        assert pending_fact_identity(e.model,legacy)==identity
        with torch.no_grad():e.model.memory.last_write[pending['slot']]+=1
        # The recorded receipt retains identity; a bare reused address cannot
        # retrospectively supply the missing identity of an old receipt.
        assert pending_fact_identity(e.model,pending)==identity
        with pytest.raises(ValueError,match='matching committed fact'):
            pending_fact_identity(e.model,legacy)
    finally:a.close()
