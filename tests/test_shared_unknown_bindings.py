"""Numerical production contracts for persistent questions, not a learned task."""
from dataclasses import replace

import pytest
import torch

from scripts.ib_local.causal_archive import attach_archive
from scripts.ib_local.causal_relation_program import Evidence,Port,mechanism,revise
from scripts.ib_local.navigation_generative_cycle import configure_generative_cycle
from scripts.ib_local.shared_unknown_bindings import (
    _payload,_open,_row,_with_constraints,candidates,configure_shared_unknowns)
from test_joint_world_cycle import complete_interface_engine


def body(tmp_path):
    torch.set_num_threads(1)
    engine=complete_interface_engine()
    configure_generative_cycle(engine)
    archive=attach_archive(engine,tmp_path,relation_capacity=engine.model.causal_relations.capacity)
    configure_shared_unknowns(engine)
    return engine,archive


def test_same_target_does_not_merge_inverse_questions_in_different_conditions():
    s=torch.tensor([[1.,2.]]);a=s+.1;y=s+.2
    key,_=_payload(Port.ACTION,s,a,y,'episode:1')
    different,_=_payload(Port.ACTION,s+1,a,y,'episode:1')
    different_port,_=_payload(Port.CONDITION,s,a,y,'episode:1')
    irrelevant_filler,_=_payload(Port.ACTION,s,a+99,y,'episode:1')
    assert key!=different and key!=different_port and key==irrelevant_filler


def test_whole_cycle_uses_selected_variable_in_program_and_real_revision(tmp_path):
    e,archive=body(tmp_path)
    try:
        first=e.step(torch.randn(8,4))
        second=e.step(torch.randn(8,4))
        for row in (first,second):
            q=row['shared_unknown']
            assert q['identity']==row['causal_program']['roles']['unknown_identity']
            assert q['program_port']==row['generative_relation']['query_role']
            assert q['revision']['committed']
            assert q['revision']['version_after']>q['revision']['version_before']
            assert q['revision']['support']<=e.model.causal_relations.evidence_budget
            assert row['generative_relation']['binding_replay_error']<1e-5
            assert row['joint_language_search']['replay_max_error']<1e-3
        assert second['generative_relation']['program_evidence_updated']
        assert archive.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:archive.close()


def test_recalled_variable_keeps_premises_and_identity_across_relation_names(tmp_path):
    e,archive=body(tmp_path)
    try:
        row=e.step(torch.randn(8,4))
        identity=row['shared_unknown']['identity']
        q=_row(archive,identity)
        state=e._program_question_state
        signature,payload=_payload(Port(q['payload']['role']),
            torch.tensor(q['payload']['source'])[None],torch.tensor(q['payload']['action'])[None],
            None if q['payload']['target'] is None else torch.tensor(q['payload']['target'])[None],
            q['payload']['anchor'])
        other=replace(state,identity=state.identity+1000)
        found=_open(e,other,payload,signature)
        assert found['identity']==identity
        assert found['payload']==q['payload']
        # Resuming is proposed by the existing retained-episode read, rather
        # than hardcoding a speaking cadence or matching the current target.
        slot=row['episode_slot'];m=e.model.memory
        e._retained_question_receipt=dict(anchor=slot,stamp=int(m.last_write[slot]))
        proposals=candidates(e,other,m.source[slot:slot+1]+3,m.action[slot:slot+1],
                            m.mother_next[slot:slot+1],f'episode:{row["shared_unknown"]["revision"]["observed_episode"]}')
        recalled=[p for p in proposals if p['recalled']]
        assert recalled and recalled[0]['question']['identity']==identity
        assert recalled[0]['payload']['source']==q['payload']['source']
    finally:archive.close()


def test_real_constraints_change_prediction_and_current_encoder_gradient(tmp_path):
    e,archive=body(tmp_path)
    try:
        e.step(torch.randn(8,4));model=e.model
        q=_row(archive,e._shared_question_receipt['identity'])
        state=_with_constraints(archive,e._program_question_state,q)
        assert state.constraints
        evidence=next(x for x in state.constraints if x.kind==Port.MOTHER)
        source,action=evidence.source,evidence.action
        full=mechanism(model,state,source,action,ridge=.1)['mean']
        empty=mechanism(model,replace(state,constraints=()),source,action,ridge=.1)['mean']
        assert not torch.allclose(full,empty)
        grad=torch.autograd.grad(full.square().sum(),model.encoder.weight,retain_graph=True)[0]
        blind_grad=torch.autograd.grad(empty.square().sum(),model.encoder.weight)[0]
        assert torch.isfinite(grad).all() and (grad-blind_grad).abs().sum()>0
    finally:archive.close()


def test_shadow_U_and_observed_U_use_same_release_and_never_write_imagination(tmp_path):
    e,archive=body(tmp_path)
    try:
        e.step(torch.randn(8,4))
        state=_with_constraints(archive,e._program_question_state,
                                _row(archive,e._shared_question_receipt['identity']))
        old=state.constraints
        assert old
        sample=next(v for v in old if v.kind==Port.MOTHER)
        actual=Evidence(sample.source+.1,sample.action+.2,sample.outcome+.3,
                        Port.MOTHER,True,slot=987,stamp=123,fact_id='episode:123456')
        imagined=replace(actual,observed=False,encoded=True,
                         outcome=e.model.encoder(actual.outcome))
        before=archive.db.total_changes
        real=revise(state,actual,model=e.model)
        shadow=revise(state,imagined,model=e.model)
        assert [v.fact_id for v in real.constraints]==[v.fact_id for v in shadow.constraints]
        a=mechanism(e.model,real,sample.source,sample.action,ridge=.1)['mean']
        b=mechanism(e.model,shadow,sample.source,sample.action,ridge=.1)['mean']
        torch.testing.assert_close(a,b)
        assert archive.db.total_changes==before
    finally:archive.close()


def test_unknown_constraints_survive_cold_snapshot_and_protect_their_facts(tmp_path):
    e,archive=body(tmp_path/'live')
    try:
        row=e.step(torch.randn(8,4))
        identity=row['shared_unknown']['identity']
        manifest=archive.checkpoint(tmp_path/'saved')
        prior=[tuple(r) for r in archive.db.execute('SELECT * FROM unknown_evidence ORDER BY 1,2')]
        other=complete_interface_engine();configure_generative_cycle(other)
        # Clone the hot numeric state without copying SQLite runtime handles.
        other.model.load_state_dict(e.model.state_dict())
        other_archive=attach_archive(other,tmp_path/'restored',manifest=manifest,
                                     **{'episode_capacity':archive.budgets['episodes'],
                                        'relation_capacity':archive.budgets['relations'],
                                        'max_bytes':archive.budgets['max_bytes']})
        try:
            configure_shared_unknowns(other)
            assert _row(other_archive,identity)==_row(archive,identity)
            assert list(other_archive.db.execute('SELECT * FROM unknown_evidence ORDER BY 1,2'))==prior
            pinned=other_archive.db.execute('SELECT episode_id FROM unknown_evidence WHERE unknown_id=?',
                                             (identity,)).fetchall()
            with pytest.raises(Exception):
                other_archive.db.execute('DELETE FROM episodes WHERE id=?',pinned[0])
        finally:other_archive.close()
    finally:archive.close()


def test_entire_saved_body_resumes_the_same_next_question_action_and_update(tmp_path):
    import copy
    import gzip
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local.navigation_checkpoint import restore
    from test_joint_world_cycle import assert_same_saved_values
    e,archive=body(tmp_path/'work');other_archive=None
    try:
        e.step(torch.randn(8,4))
        save(e,tmp_path/'saved',1,dict(shared_unknowns={'schema':1}))
        with gzip.open(tmp_path/'saved/last.pt.gz','rb') as stream:
            saved=torch.load(stream,map_location='cpu',weights_only=False)
        expected=e.step(torch.randn(8,4))
        expected_model=copy.deepcopy(e.model.state_dict())
        expected_optimizer=copy.deepcopy(e.optimizer.state_dict())
        other=complete_interface_engine();configure_generative_cycle(other)
        restore(other,saved)
        other_archive=attach_archive(other,tmp_path/'other',manifest=saved['config']['causal_archive'],
                                     relation_capacity=archive.budgets['relations'])
        configure_shared_unknowns(other)
        actual=other.step(torch.randn(8,4))
        for name in ('choice','utterance_hex','shared_unknown','metal_g','generative_relation'):
            assert actual[name]==expected[name],name
        assert_same_saved_values(expected_model,other.model.state_dict())
        assert_same_saved_values(expected_optimizer,other.optimizer.state_dict())
    finally:
        archive.close()
        if other_archive is not None:other_archive.close()


def test_question_quota_releases_unpinned_questions_without_aliasing_old_ids(tmp_path):
    e,archive=body(tmp_path)
    try:
        row=e.step(torch.randn(8,4))
        pinned=row['shared_unknown']['identity']
        state=e._program_question_state
        source=e.model.memory.source[:1].clone()
        first=None
        for i in range(e.model.causal_relations.capacity+2):
            signature,payload=_payload(Port.MOTHER,source+i+1,source,None,'unobserved')
            q=_open(e,state,payload,signature)
            if first is None:first=q
        assert archive.db.execute('SELECT COUNT(*) FROM unknown_bindings').fetchone()[0] <= e.model.causal_relations.capacity
        assert _row(archive,pinned) is not None
        assert _row(archive,first['identity']) is None
        signature,payload=_payload(Port.MOTHER,source+1,source,None,'unobserved')
        reopened=_open(e,state,payload,signature)
        assert reopened['identity']!=first['identity']
        assert archive.db.execute('PRAGMA foreign_key_check').fetchall()==[]
    finally:archive.close()
