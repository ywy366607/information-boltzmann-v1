"""Actual operator, evidence identity, gradient and native interface contracts."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, posterior, mechanism_hypotheses
from scripts.ib_local.identity_preview import observation_noise, ranked_action_facts, fact_identity


def test_identity_noise_survives_subset_permutation_and_preserves_rng():
    template = torch.zeros(4, 10, 7)
    before = torch.random.get_rng_state().clone()
    noise = observation_noise(('mother:1:episode:203', 'mother:1:episode:283'), template)
    subset = observation_noise(('mother:1:episode:999', 'mother:1:episode:203'), template)
    torch.testing.assert_close(noise[:, 0], subset[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(noise[:2], -noise[2:], rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert observation_noise((), template).shape == (4, 0, 7)


def test_native_hypothesis_and_derivative_equivariance():
    gen = torch.Generator().manual_seed(15)
    x, y, q = [torch.randn(*shape, generator=gen, dtype=torch.float64, requires_grad=True)
               for shape in ((8, 4), (8, 9), (3, 4))]
    w = torch.randn(4, 4, 9, generator=gen, dtype=torch.float64)
    noise = torch.randn(4, 10, 9, generator=gen, dtype=torch.float64)
    ids = tuple(f'mother:1:episode:{i}' for i in (203, 1133, 1616, 1919, 1977, 2216, 267, 283))
    perm = [0, 7, 6, 1, 2, 3, 4, 5]
    left = posterior(x, y, q, .1)
    left['fact_identities'] = ids
    right = posterior(x[perm], y[perm], q, .1)
    right['fact_identities'] = tuple(ids[i] for i in perm)
    a = mechanism_hypotheses(left, q, w, noise)
    b = mechanism_hypotheses(right, q, w, noise)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    ga = torch.autograd.grad(a.square().sum(), (x, y, q), retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), (x, y, q))
    for u, v in zip(ga, gb):
        torch.testing.assert_close(u, v, atol=1e-8, rtol=1e-8)


def test_content_rank_and_identity_ties_preserve_candidate_set():
    model = SimpleNamespace(encoder=torch.nn.Identity())
    facts = tuple(Evidence(torch.tensor([v]), torch.tensor([v]), torch.tensor([v]),
        Port.MOTHER, True, fact_id=f'episode:{i}')
        for i, v in ((3, [1., 0.]), (1, [1., 0.]), (2, [-1., 0.])))
    a = ranked_action_facts(model, torch.tensor([1., 0.]), facts, 2)
    b = ranked_action_facts(model, torch.tensor([1., 0.]), facts[::-1], 2)
    assert [fact_identity(e) for e in a] == [fact_identity(e) for e in b]
    assert [e.fact_id for e in a] == ['episode:1', 'episode:3']
    c = ranked_action_facts(model, torch.tensor([-1., 0.]), facts, 1)
    assert c[0].fact_id == 'episode:2'


def test_native_prediction_organization_and_preview_share_fact_view(tmp_path):
    from test_joint_relation_update import body
    from scripts.ib_local.causal_relation_program import mechanism
    from scripts.ib_local.generative_relations import dataset
    from scripts.ib_local.relation_evidence import mechanism_evidence, fact_reference
    engine, archive = body(tmp_path, revision=4)
    try:
        model = engine.model
        model.identity_preview_revision = 1
        h = model.memory.source[:1]
        state = model.causal_relations.bind(model, h)
        facts = mechanism_evidence(model, state, Port.MOTHER)
        state = replace(state, constraints=facts, evidence_bound=True)
        reverse = replace(state, constraints=facts[::-1])
        a = mechanism(model, state, h, h)
        b = mechanism(model, reverse, h, h)
        ox, oy, _, refs = dataset(model, reverse)
        assert refs == [fact_reference(e) for e in mechanism_evidence(model, state, Port.MOTHER)]
        for key in ('mean', 'x', 'y'):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        torch.testing.assert_close(a['x'], ox)
        torch.testing.assert_close(a['y'], oy)
        assert a['fact_identities'] == b['fact_identities']
    finally:
        archive.close()
