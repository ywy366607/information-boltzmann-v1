"""Production finite-U/latent-U and complete-cycle provenance contracts."""
from dataclasses import replace,asdict
import json

import pytest
import torch

from scripts.ib_local import relation_retention_use as use
from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local.causal_relation_program import Port,revise,mechanism,log_score
from scripts.ib_local.relation_evidence import mechanism_evidence,evidence_key
from scripts.ib_local.shared_unknown_bindings import _row,_with_constraints,_read_evidence
from scripts.ib_local.causal_use_value import configure as configure_value,queries
from scripts.ib_local.identified_event_credit import configure as configure_events
from test_joint_relation_update import body


def configured(tmp_path,schema=2):
    e,a=body(tmp_path,revision=4)
    e.autonomous_event_clock=0
    configure_value(e,'C2');configure_events(e,'identified');ev.configure(e)
    use.configure(e,'apply',schema=schema)
    return e,a


def test_lagged_moments_scoped_bounded_reversible_and_neutral():
    z=use.UseSnapshot('unknown:1',mode='apply')
    x=torch.tensor([[1.,0.],[0.,2.],[.2,.3]],dtype=torch.float64)
    ids=['episode:1','episode:2','episode:3']
    h,p=use.priorities(x,ids,z,.1)
    assert torch.equal(h,p)
    a=use.advance(z,ids,dict(zip(ids,[1.,-1.,.2])),'episode:4')
    assert z.moments == ()
    assert bool((use.biases(a,ids,h).abs()<1).all())
    assert use.advance(a,ids,dict(zip(ids,[1.,-1.,.2])),'episode:4') == a
    for i in range(5):
        a=use.advance(a,ids,{'episode:1':-1.},f'episode:{5+i}')
    assert use.biases(a,ids,h)[0] < 0
    order=[2,0,1]
    got=use.select(x,ids,a,.1)[1]['released']
    assert use.select(x[order],[ids[i] for i in order],a,.1)[1]['released']==got
    assert len(use.filtered(a,ids[:1]).moments)==1


def test_batched_use_matches_original_port_and_raw_latent_U(tmp_path):
    e,a=configured(tmp_path)
    try:
        rows=[e.step(torch.randn(8,4)) for _ in range(3)]
        state=e.internal_pending['causal_program']['state']
        incoming=_read_evidence(a,int(e.internal_pending['causal_program']['episode_identity'].split(':')[1]))[0]
        facts=mechanism_evidence(e.model,state,Port.MOTHER)
        assert len(facts)>0
        with torch.no_grad():
            encoded=tuple(ev.encode_fact(e.model,f) for f in facts)
            inc=ev.encode_fact(e.model,incoming)
            got=use.contributions(e.model,encoded,inc,e.relation_ridge)
            full=log_score(mechanism(e.model,state,incoming.source,incoming.action,ridge=e.relation_ridge),
                           e.model.encoder(incoming.outcome)).mean()
            for f in facts:
                d=mechanism(e.model,state,incoming.source,incoming.action,ridge=e.relation_ridge,
                            exclude_fact=f.fact_id)
                want=float(full-log_score(d,e.model.encoder(incoming.outcome)).mean())
                assert got[evidence_key(f)]==pytest.approx(want,abs=3e-5)
            raw=revise(state,incoming,model=e.model)
            ctx=ev.from_body(e,state,queries(e,state))
            latent=ev.revise(e.model,ctx,inc,ridge=e.relation_ridge)
            actual=ev.from_body(e,raw,queries(e,state))
            assert {f.identity for f in latent.facts}=={f.identity for f in actual.facts}
            assert latent.retention_use.scope==actual.retention_use.scope
            assert latent.retention_use.revision==actual.retention_use.revision
            for l,r in zip(latent.retention_use.moments,actual.retention_use.moments):
                assert l[0]==r[0] and l[3]==r[3]
                assert l[1:3]==pytest.approx(r[1:3],abs=3e-6)
        assert rows[-1]['causal_use']['finite_U']['verified']
        fresh=rows[-1]['causal_use']['finite_U']
        assert fresh['fresh_use']['scope'].startswith('unknown:')
        assert fresh['fresh_organization']['scope'].startswith('relation:')
        assert rows[-1]['causal_use']['reward_version']=='finite_relation_use_v5'
        assert rows[-1]['shared_unknown']['revision']['retention_use']['trace']['benefits']
        assert a.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:a.close()


def test_snapshot_commit_idempotency_stale_and_branch_sql_purity(tmp_path):
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        st=e._program_question_state
        scope=st.binding.identity
        old=use._load(e.model,scope)
        ids=[f.identity for f in ev.from_body(e,st,queries(e,st)).facts]
        new=use.advance(old,ids,{k:.01 for k in ids},'episode:independent-contract')
        with a.db:use.commit(e.model,old,new)
        use.commit(e.model,old,new)
        with pytest.raises(RuntimeError):
            use.commit(e.model,old,replace(new,last_event='different'))
        with torch.no_grad():
            bound=_with_constraints(a,st,_row(a,scope))
            assert bound.retention_use==new
            ctx=ev.from_body(e,bound,queries(e,bound))
            incoming=replace(ctx.facts[0],identity='event:-1:-1',phase_identity='mother:0:event:-1:-1')
            changes=a.db.total_changes
            result=ev.revise(e.model,ctx,incoming,ridge=e.relation_ridge)
            assert a.db.total_changes==changes
            assert result.retention_use.revision==new.revision+1
            assert use._load(e.model,scope)==new
        before=json.dumps(asdict(new),sort_keys=True)
        use.configure(e,'apply')
        assert json.dumps(asdict(use._load(e.model,scope)),sort_keys=True)==before
    finally:a.close()


def test_organization_uses_pre_event_snapshot_and_returns_next_moments(tmp_path,monkeypatch):
    """Bind the compound production U clock, not only the local selector."""
    from scripts.ib_local import generative_relations as gr
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        state=e._program_question_state.organization
        state=e.model.causal_relations.state(state.index)
        assert state is not None and state.evidence_bound
        fact=mechanism_evidence(e.model,state,Port.MOTHER)[0]
        incoming=replace(fact,observed=False,encoded=False,fact_id='event:clock:next',
                         outcome=fact.outcome+.2)
        old=replace(state.retention_use,revision=17,moments=())
        state=replace(state,retention_use=old,extra=())
        calls=[]
        def inspect(model,successor,event,*,selection_state=None):
            assert selection_state is state
            assert selection_state.retention_use is old
            assert successor.retention_use.revision==18
            assert successor.retention_use.moments
            calls.append(successor.retention_use)
            return successor
        monkeypatch.setattr(gr,'organize',inspect)
        updated=revise(state,incoming,model=e.model)
        assert len(calls)==1 and updated.retention_use==calls[0]
        assert old.revision==17 and old.moments==()
    finally:a.close()


@pytest.mark.parametrize('admitted',[True,False])
def test_split_shadow_resets_changed_scope_like_real_commit(tmp_path,monkeypatch,admitted):
    """Force only the split decision; execute the production support mapping."""
    from scripts.ib_local import generative_relations as gr
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        state=e.model.causal_relations.state(e._program_question_state.organization.index)
        fact=mechanism_evidence(e.model,state,Port.MOTHER)[0]
        facts=tuple(replace(fact,fact_id=f'episode:{900001+i}',source=fact.source+i*.1)
                    for i in range(3))
        incoming=replace(fact,observed=False,fact_id='event:split:incoming')
        moments=use.advance(state.retention_use,[f.fact_id for f in facts],
                            {f.fact_id:.2 for f in facts},incoming.fact_id)
        scope={k:v.clone() for k,v in state.scope.items()}
        scope['valid'].zero_()
        successor=replace(state,constraints=facts,extra=(incoming,),scope=scope,
                          retention_use=moments)
        def split_scores(model,st,x,y,raw,ridge,noise):
            return (x.new_zeros(()),x.new_tensor([100.]),
                    x.new_ones((1,model.encoder.out_features)),torch.arange(4,device=x.device))
        monkeypatch.setattr(gr,'organization_scores',split_scores)
        if not admitted:
            monkeypatch.setattr(gr,'split_destination',lambda model,state:None)
        updated=gr.organize(e.model,successor,incoming,selection_state=state)
        assert updated.revision_plan['operation']==('split' if admitted else 'retain')
        assert updated.retention_use.moments==(() if admitted else moments.moments)
        assert updated.retention_use.revision==moments.revision
        assert updated.scope['valid'].sum()==int(admitted)
        assert moments.moments  # caller/other branches are unchanged
    finally:a.close()


def test_legacy_clock_upgrade_versions_labels_and_preserves_body(tmp_path):
    e,a=configured(tmp_path,schema=1)
    try:
        assert e.settlement_kind=='finite_relation_use_v4'
        e.step(torch.randn(8,4))
        saved=list(a.db.execute('SELECT payload FROM unknown_retention ORDER BY unknown_id'))
        weight=e.model.encoder.weight.detach().clone()
        pending=e.internal_pending
        groups=len(e.optimizer.param_groups)
        e.causal_use_calibration=dict(count=3,cross=.2,second=.5)
        e.causal_use_value_evidence.seen.fill_(5)
        use.configure(e,'apply',schema=2)
        assert e.settlement_kind=='finite_relation_use_v5'
        assert e.model.relation_retention_use_schema==2
        assert e.relation_retention_use_migration['kind']=='finite_relation_use_v4'
        assert e.relation_retention_use_migration['calibration']['count']==3
        assert e.causal_use_calibration['count']==0
        assert not bool(e.reward_label_valid(wood=True).any())
        assert torch.equal(weight,e.model.encoder.weight)
        assert pending is e.internal_pending and groups==len(e.optimizer.param_groups)
        assert saved==list(a.db.execute('SELECT payload FROM unknown_retention ORDER BY unknown_id'))
        e.causal_use_value_evidence.seen.fill_(2)
        use.configure(e,'apply',schema=2)
        assert bool((e.causal_use_value_evidence.seen==2).all())
        with pytest.raises(ValueError):use.configure(e,'apply',schema=1)
        row=e.step(torch.randn(8,4))
        assert row['causal_use']['reward_version']=='finite_relation_use_v5'
        assert row['causal_use']['finite_U']['verified']
    finally:a.close()


@pytest.mark.parametrize('schema',[1,2])
@pytest.mark.parametrize('epischema',[1,2,3,'3_native','4_native','5_native','6_native','5_batch','5_residual','5_amortized','5_planner','5_competing'])
def test_versioned_clock_full_restore_keeps_next_loop(tmp_path,schema,epischema):
    import copy
    import gzip
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    from scripts.ib_local.causal_archive import attach_archive
    from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
    from scripts.ib_local.joint_relation_update import configure_joint_relation_update
    from test_joint_world_cycle import assert_same_saved_values
    residual = epischema == '5_residual'
    competing = epischema == '5_competing'
    planner = epischema in ('5_planner','5_competing')
    amortized = epischema in ('5_amortized','5_planner','5_competing')
    runtime = 'batch' if epischema in ('5_batch','5_residual','5_amortized','5_planner','5_competing') else 'serial'
    word_execution='native_anchor' if epischema in ('3_native','4_native','5_native','6_native','5_batch','5_residual','5_amortized','5_planner','5_competing') else 'historical'
    if isinstance(epischema,str):epischema=int(epischema[0])
    e,a=configured(tmp_path/'live',schema=schema)
    if epischema>=3:e.gain_units_schema=1
    ev.configure(e,schema=epischema,word_execution=word_execution)
    e.native_word_runtime = runtime
    e.model._preview_batch_enabled = runtime=='batch' or epischema>=6
    if residual:
        from scripts.ib_local.native_residual_quadrature import configure as configure_quadrature
        configure_quadrature(e, dict(schema=1, mode='residual', seed=191120))
    if amortized:
        from scripts.ib_local.native_amortized_decision import configure as configure_amortization
        specification = dict(schema=1, seed=201120)
        if planner:
            specification['planner'] = 'action_batch'
            specification['correction'] = 'decision_budget'
            if competing:
                specification['teacher_selection'] = 'competing_actions'
        configure_amortization(e, specification)
    restored=None
    try:
        e.step(torch.randn(8,4))
        config=dict(relation_retention_use=dict(schema=schema,mode='apply'),
                    epistemic_value=dict(schema=epischema,word_execution=word_execution),
                    native_word_runtime=dict(schema=2 if runtime=='batch' else 1,mode=runtime,batch_size=4))
        if residual:
            config['successor_quadrature'] = e.successor_quadrature
        if amortized:
            config['native_amortization'] = e.native_amortization
        save(e,tmp_path/'saved',1,config)
        with gzip.open(tmp_path/'saved/last.pt.gz','rb') as stream:
            stored=torch.load(stream,map_location='cpu',weights_only=False)
        expected=e.step(torch.randn(8,4))
        weights=copy.deepcopy(e.model.state_dict())
        optimizer=copy.deepcopy(e.optimizer.state_dict())
        other,unused=body(tmp_path/'fresh',revision=4)
        unused.close();del other.model._causal_archive
        restore(other,stored)
        restored=attach_archive(other,tmp_path/'restored',
            manifest=stored['config']['causal_archive'],relation_capacity=a.budgets['relations'])
        configure_shared_unknowns(other)
        configure_joint_relation_update(other,revision=4)
        configure_value(other,'C2');configure_events(other,'identified');ev.configure(other,schema=epischema,word_execution=stored['config']['epistemic_value']['word_execution'])
        other.native_word_runtime = stored['config']['native_word_runtime']['mode']
        other.model._preview_batch_enabled = stored['config']['native_word_runtime']['schema']>=2 or epischema>=6
        if residual:
            configure_quadrature(other, stored['config']['successor_quadrature'])
        if amortized:
            configure_amortization(other, stored['config']['native_amortization'])
        use.configure(other,'apply',schema=stored['config']['relation_retention_use']['schema'])
        actual=other.step(torch.randn(8,4))
        assert other.model.relation_retention_use_schema==schema
        for key in ('choice','utterance_hex','metal_g','causal_use','shared_unknown'):
            assert actual[key]==expected[key],key
        assert_same_saved_values(weights,other.model.state_dict())
        assert_same_saved_values(optimizer,other.optimizer.state_dict())
        if amortized:
            assert_same_saved_values(e.native_value_bank, other.native_value_bank)
        if planner:
            assert_same_saved_values(e.latent_action_bank, other.latent_action_bank)
    finally:
        a.close()
        if restored is not None:restored.close()
