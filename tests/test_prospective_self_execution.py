"""Native operator and complete-update contracts, without capability training."""
import torch

from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local import navigation_generative_cycle as cycle
from scripts.ib_local.prospective_self_execution import execute,question_context
from scripts.ib_local.known_utterance_runtime import known_film
from scripts.ib_local.table_branch import table_branch
from test_relation_retention_use import configured


def test_shared_question_context_matches_actual_intention(monkeypatch,tmp_path):
    e,archive=configured(tmp_path)
    saved={};original=cycle.question_choices;old=e._value
    def choices(*a,**kw):
        result=original(*a,**kw);saved.update(pre=a[2][None].clone(),packet=result)
        return result
    def value(h,films):
        proposal=saved['packet']['proposals'][e._binding_choices[-1]['selected']]
        expected=question_context(e,saved['pre'],proposal)[0]
        torch.testing.assert_close(expected,e._question_turn_context,rtol=3e-5,atol=3e-6)
        saved['checked']=True
        return old(h,films)
    monkeypatch.setattr(cycle,'question_choices',choices);e._value=value
    try:
        e.step(torch.randn(8,4))
        assert saved['checked']
    finally:archive.close()


def test_native_anchor_exact_word_endpoint_and_private_scope(tmp_path):
    e,archive=configured(tmp_path)
    try:
        e.step(torch.randn(8,4));model=e.model;m=model.memory
        model._question_context=None;pre=e._read_source().clone()
        state=model.causal_relations.bind(model,pre)
        proposal=cycle.question_choices(e,state,pre[0])['proposals'][0]
        slot=int(torch.nonzero(m.occupied)[0]);m._retrieval_store.write(slot,b'a',merged=False,keep=True)
        context=question_context(e,pre,proposal)[0]
        model._question_context=context;post=e._read_source().clone()
        with table_branch(e.table) as branch:known_film(e,b'a',branch)
        actual=e._draft_actions.pop()[None].clone()
        sentinel={'real':object()};e._utterance_payload_cache=sentinel
        sql=archive.db.total_changes;rng=torch.get_rng_state().clone();decision=e.decision.state.clone()
        result,actions,_=execute(e,pre,proposal,[slot])
        torch.testing.assert_close(result,post,rtol=0,atol=3e-6)
        torch.testing.assert_close(actions,actual,rtol=0,atol=3e-6)
        assert e._utterance_payload_cache is sentinel
        assert archive.db.total_changes==sql and torch.equal(rng,torch.get_rng_state())
        assert torch.equal(e.decision.state,decision) and torch.equal(model._question_context,context)
    finally:archive.close()


def test_complete_native_self_execution_keeps_original_U_and_credit(tmp_path):
    e,archive=configured(tmp_path)
    try:
        e.gain_units_schema=1;e.step(torch.randn(8,4))
        ev.configure(e,schema=3,word_execution='native_anchor')
        row=e.step(torch.randn(8,4))
        assert row['causal_use']['finite_U']['verified']
        assert row['causal_use']['reward_version']=='finite_relation_use_v5'
        assert row['joint_language_search']['replay_max_error']<1e-3
        assert row['generative_relation']['binding_replay_error']<1e-5
        assert e.epistemic_word_execution=='native_anchor'
        assert any(c['word_execution']=='native_anchor' for d in e.prospective_decision_last for c in d['cases'])
    finally:archive.close()
