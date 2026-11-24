"""The encoded evaluator must reproduce the actual finite-U port calculation."""
from dataclasses import replace

import torch

from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local.causal_use_value import configure, queries
from scripts.ib_local.causal_relation_program import mechanism
from scripts.ib_local.finite_use_value import expected, score_scope, score_state
from test_joint_relation_update import body


def test_encoded_context_reproduces_body_posterior_scores_and_finite_U(tmp_path):
    engine, archive = body(tmp_path, revision=4)
    try:
        configure(engine, 'C2')
        engine.autonomous_event_clock = 0
        for _ in range(2):
            engine.step(torch.randn(engine.model.encoder.out_features, 4))
        with torch.no_grad():
            state = score_state(engine._program_question_state)
            q = queries(engine, state)
            assert q
            ctx = ev.from_body(engine, state, q)
            src = engine._internal_source.reshape(1, -1).expand(3, -1)
            act = engine._internal_action
            s, a = engine.model.encoder(src), engine.model.encoder(act)
            raw = mechanism(engine.model, state, src, act, ridge=engine.relation_ridge)
            encoded = ev.response(engine.model, ctx, s, a, ridge=engine.relation_ridge)
            for key in ('mean', 'scale2', 'x', 'y', 'query'):
                torch.testing.assert_close(encoded[key], raw[key])
            torch.testing.assert_close(ev.scores(engine.model, ctx, ridge=engine.relation_ridge),
                                       score_scope(engine, state, q))
            actual = expected(engine, state, src, act, raw, q)
            result = ev.expected(engine.model, ctx, s, a, ridge=engine.relation_ridge)
            torch.testing.assert_close(result, actual, rtol=3e-5, atol=3e-6)
            reordered = replace(ctx, facts=ctx.facts[::-1])
            # Explicit canonicalization restores the exact permitted view.
            reordered = replace(reordered, facts=tuple(sorted(reordered.facts,key=lambda f:f.phase_identity)))
            torch.testing.assert_close(ev.expected(engine.model, reordered, s, a,
                ridge=engine.relation_ridge), result)
    finally:
        archive.close()


def test_complete_contextual_value_cycle_keeps_real_finite_U_and_delayed_learning(tmp_path):
    e, archive = body(tmp_path, revision=4)
    try:
        configure(e, 'C2')
        e.autonomous_event_clock = 0
        e.step(torch.randn(e.model.encoder.out_features,4))
        ev.configure(e)
        params = sum(p.numel() for p in e.model.parameters())
        row = e.step(torch.randn(e.model.encoder.out_features,4))
        assert row['causal_use']['reward_version'] == 'finite_targeted_use_v2'
        assert row['causal_use']['epistemic_value']['schema'] == 1
        assert row['generative_relation']['same_U']
        assert row['causal_use']['finite_U'].get('verified',True)
        assert sum(p.numel() for p in e.model.parameters()) == params
        assert 'epistemic_after_value' in e._program_forecasts
        assert torch.isfinite(e._program_forecasts['epistemic_after_value']).all()
        # Restoring the marker reattaches the route without clearing evidence.
        before = e.causal_use_value_evidence.seen.clone()
        ev.configure(e)
        assert torch.equal(before,e.causal_use_value_evidence.seen)
    finally:
        archive.close()
