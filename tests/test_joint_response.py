"""Numerical contracts of the actual response/finite-U production operators."""
from dataclasses import replace

import torch

from scripts.ib_local import epistemic_value_reference as ev, joint_response
from scripts.ib_local.causal_relation_program import posterior, log_score, response_hypotheses
from test_relation_retention_use import configured


def test_joint_response_marginals_positive_shared_actions_and_predictive_gradient():
    g = torch.Generator().manual_seed(23)
    x = torch.randn(6, 4, generator=g, dtype=torch.float64)
    y = torch.randn(6, 9, generator=g, dtype=torch.float64)
    query = torch.randn(1, 4, generator=g, dtype=torch.float64).expand(3, -1)
    prior = torch.ones(9, dtype=x.dtype, requires_grad=True)
    d = posterior(x, y, query, .1, noise_prior=prior)
    d['query'] = query; d['fact_identities'] = tuple(f'event:{i}' for i in range(6))
    before = log_score(d, y[:3]).detach().clone()
    joint = joint_response.distribution(d)
    torch.testing.assert_close(joint['scale_matrix'].diagonal(dim1=-2, dim2=-1), d['scale2'])
    assert torch.linalg.eigvalsh(joint['scale_matrix']).min() > 0
    directions = torch.randn(2, 4, 9, generator=g, dtype=x.dtype)
    obs = torch.randn(2, 8, 9, generator=g, dtype=x.dtype)
    rng = torch.get_rng_state().clone()
    means, points, geometry = joint_response.hypotheses(d, directions, obs)
    torch.testing.assert_close(points[0], points[1], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(joint_response.evidence(means, geometry, points[:, 0])).all()
    # Predictive joint training cannot silently change the PAYMENT distribution.
    torch.testing.assert_close(before, log_score(d, y[:3]), rtol=0, atol=0)
    (-log_score(joint, y[:3]).mean()).backward()
    assert prior.grad.norm() > 0


def test_multi_scope_payment_batch_matches_original_values_and_gradients(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.step(torch.randn(8, 4))
        state = e._program_question_state
        c = ev.from_body(e, state, ())
        from scripts.ib_local.shared_unknown_bindings import _read_evidence
        identity = e.internal_pending['causal_program']['episode_identity']
        fact = _read_evidence(archive, int(identity.split(':')[1]))[0]
        c = replace(c, facts=(ev.encode_fact(e.model, fact),))
        c = replace(c, queries=c.facts)
        assert c.queries
        contexts = [c, replace(c, facts=c.facts[:1]), replace(c, facts=()),
                    replace(c, queries=())]
        expected = [ev.batched_scores(e.model, ctx, ridge=e.relation_ridge) for ctx in contexts]
        actual = ev.batched_context_scores(e.model, contexts, ridge=e.relation_ridge)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-5)
        parameter = e.model.long_predictor[0].weight
        ga = torch.autograd.grad(sum(x.sum() for x in actual), parameter, retain_graph=True)[0]
        gb = torch.autograd.grad(sum(x.sum() for x in expected), parameter)[0]
        torch.testing.assert_close(ga, gb, atol=5e-5, rtol=5e-5)
    finally:
        archive.close()


def test_real_response_loop_same_joint_draws_and_actual_late_learning(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.gain_units_schema = 1
        ev.configure(e, schema=6, word_execution='native_anchor')
        first = e.step(torch.randn(8, 4))
        second = e.step(torch.randn(8, 4))
        assert second['causal_use']['finite_U']['verified']
        assert second['causal_use']['reward_version'] == 'finite_relation_use_v5'
        assert second['causal_program']['mother_joint_prequential'] is not None
        assert second['causal_program']['successor_learning']['target_excluded']
        assert e.successor_distribution_last['joint']['samples'] == 4096
        assert first['causal_use']['epistemic_value']['successor_routes']
        state = e._program_question_state
        context = ev.from_body(e, state, ())
        context = replace(context, queries=context.facts)
        source = e.model.encoder(e._internal_source[None])
        actions = e.model.encoder(e._internal_action)
        with torch.no_grad():
            e.model._preview_batch_enabled = False
            reference = ev.expected(e.model, context, source, actions, ridge=e.relation_ridge)
            e.model._preview_batch_enabled = True
            batch = ev.expected(e.model, context, source, actions, ridge=e.relation_ridge)
        torch.testing.assert_close(reference, batch, atol=5e-5, rtol=5e-5)
        from scripts.ib_local import finite_use_value as finite
        from scripts.ib_local.causal_relation_program import compose
        from scripts.ib_local.causal_use_value import queries
        raw = e._internal_source[None]; actions = e._internal_action
        with torch.no_grad():
            d = compose(e.model, state, raw.expand_as(actions), actions,
                ridge=e.relation_ridge, reward_valid=e.reward_label_valid(wood=True))['mother']
            facts = queries(e, state)
            e.model._preview_batch_enabled = False
            reference = finite.expected(e, state, raw.expand_as(actions), actions, d, facts)
            e.model._preview_batch_enabled = True
            batch = finite.expected(e, state, raw.expand_as(actions), actions, d, facts)
        torch.testing.assert_close(reference, batch, atol=5e-5, rtol=5e-5)
    finally:
        archive.close()
