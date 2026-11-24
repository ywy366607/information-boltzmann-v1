"""Actual operator/complete-cycle contracts for the prospective decision path."""
from dataclasses import replace

import torch

from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local.prospective_decision import boundary, lift_boundary, preview_bind, value, terminal
from scripts.ib_local.causal_relation_program import Port, mechanism, revise, Evidence
from scripts.ib_local.navigation_generative_cycle import question_choices
from scripts.ib_local.navigation_persistent_question import retained_receipt
from test_relation_retention_use import configured


def test_raw_boundary_lift_and_native_address_are_read_only(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.step(torch.randn(8, 4))
        model=e.model;source=e._internal_source.reshape(1,-1)
        with torch.no_grad():
            z=model.encoder(source);anchor=source+torch.randn_like(source)*.1
            h=lift_boundary(model,anchor,z)
            torch.testing.assert_close(model.encoder(h),z,rtol=2e-5,atol=3e-6)
            torch.testing.assert_close(lift_boundary(model,anchor,model.encoder(anchor)),anchor,rtol=0,atol=0)
            before={n:t.clone() for n,t in model.causal_relations.named_buffers()}
            sql=archive.db.total_changes
            preview,detail=preview_bind(model,source)
            assert archive.db.total_changes==sql
            assert all(torch.equal(t,before[n]) for n,t in model.causal_relations.named_buffers())
            real=model.causal_relations.bind(model,source)
            assert (preview.identity,preview.generation)==(real.identity,real.generation)
            assert detail['matched']
            receipt=retained_receipt(model,source)
            actual=question_choices(e,real,source[0],retained_packet=receipt)
            packet=question_choices(e,preview,source[0],retained_packet=receipt)
            torch.testing.assert_close(packet['logits'],actual['logits'],rtol=0,atol=0)
            # Also bind a fresh address: same uniform program prior and seeded
            # conditional predictor, with no allocation on the shadow side.
            model.memory.threshold=2.
            fresh,detail=preview_bind(model,source)
            assert not detail['matched']
            native=model.causal_relations.bind(model,source)
            torch.testing.assert_close(fresh.program_weights,native.program_weights,rtol=0,atol=0)
            for st in (fresh,native):
                assert st.program_weights is not None
            torch.testing.assert_close(
                mechanism(model,fresh,source,source,ridge=e.relation_ridge)['mean'],
                mechanism(model,native,source,source,ridge=e.relation_ridge)['mean'],rtol=2e-5,atol=3e-6)
    finally:archive.close()


def test_prospective_transition_uses_raw_history_and_keeps_hypotheses_private(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.gain_units_schema=1
        for _ in range(2):e.step(torch.randn(8, 4))
        model=e.model;state=e._program_question_state
        source=e._internal_source.reshape(1,-1);action=e._internal_action[1:2]
        with torch.no_grad():
            p=e._momentum_decision_read[None]
            d=mechanism(model,state,source,action,ridge=e.relation_ridge,momentum=p)
            incoming=Evidence(source,action,d['mean'],Port.MOTHER,False,encoded=True)
            updated=revise(state,incoming,model=model)
            context=ev.from_body(e,state,())
            context=replace(context,queries=context.facts)
            reference=ev.scores(model,context,ridge=e.relation_ridge)
            with terminal():
                torch.testing.assert_close(ev.scores(model,context,ridge=e.relation_ridge),
                                           reference,rtol=3e-5,atol=3e-6)
            # Also cover released queries and the prior branch's encoded fact.
            after=ev.revise(model,context,ev.encode_fact(model,incoming),ridge=e.relation_ridge)
            torch.testing.assert_close(ev.batched_scores(model,after,ridge=e.relation_ridge),
                ev.scores(model,after,ridge=e.relation_ridge),rtol=3e-5,atol=3e-6)
            latent=mechanism(model,updated,source,action,kind=Port.DECISION,
                             mother=d['mean'],ridge=e.relation_ridge)['mean']
            sql=archive.db.total_changes;rng=torch.get_rng_state().clone()
            count=sum(p.numel() for p in model.parameters())
            h,diag=boundary(e,updated,source,action,d['mean'],latent)
            torch.testing.assert_close(model.encoder(h),latent,rtol=2e-5,atol=3e-6)
            result=value(e,updated,source,action,d['mean'],latent,p)
            assert result.shape==(1,) and bool(torch.isfinite(result).all())
            assert archive.db.total_changes==sql and torch.equal(torch.get_rng_state(),rng)
            assert count==sum(p.numel() for p in model.parameters())
            assert e.prospective_decision_last[0]['cases']
    finally:archive.close()


def test_schema3_complete_learning_keeps_same_payment_and_restore_marker(tmp_path):
    e,archive=configured(tmp_path)
    try:
        e.gain_units_schema=1
        e.step(torch.randn(8,4))
        ev.configure(e,schema=3)
        row=e.step(torch.randn(8,4))
        assert row['causal_use']['epistemic_value']['schema']==3
        assert row['causal_use']['finite_U']['verified']
        assert row['causal_use']['reward_version']=='finite_relation_use_v5'
        assert row['joint_language_search']['replay_max_error']<1e-3
        assert row['generative_relation']['binding_replay_error']<1e-5
        assert torch.isfinite(e._program_forecasts['epistemic_after_value']).all()
    finally:archive.close()
