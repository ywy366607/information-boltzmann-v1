"""Batched actual/hypothetical fact operators retain all original derivatives."""
from types import SimpleNamespace
import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, relation_features, _bound_evidence_design
from scripts.ib_local.shared_relation_basis import SharedRelationBasis


@pytest.mark.parametrize('kind,transport', [(Port.MOTHER, False), (Port.DECISION, False), (Port.DECISION, True)])
@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fact_batch_matches_scalar_values_and_parameter_derivatives(kind, transport, mixed, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA batching parity')
    torch.manual_seed(18)
    def linear(a, b):
        return torch.nn.Linear(a, b, dtype=torch.float64, device=device)
    model = SimpleNamespace(encoder=linear(12, 8),
        long_predictor=torch.nn.Sequential(linear(16, 8), torch.nn.SiLU(), linear(8, 8)),
        mother_from_boundary=linear(8, 8), joint_relation_update_schema=1,
        relation_basis=SharedRelationBasis(8, device=device, dtype=torch.float64))
    rand = lambda size: torch.randn(1, size, dtype=torch.float64, device=device)
    facts = []
    for i in range(8):
        encoded = mixed and i % 3 == 1
        facts.append(Evidence(rand(12), rand(12), rand(8 if encoded else 12), kind,
            observed=not encoded, encoded=encoded, mother=rand(8 if encoded else 12)))
    expected_x, expected_y = [], []
    for e in facts:
        s, a = model.encoder(e.source), model.encoder(e.action)
        y = e.outcome if e.encoded else model.encoder(e.outcome)
        if transport:
            mother = e.mother if e.encoded else model.encoder(e.mother)
            x, y = relation_features(model, s, a, mother), y-mother
        else:
            x = relation_features(model, s, a)
            prior = model.long_predictor(torch.cat((s, a), -1))
            if kind == Port.MOTHER:
                prior = prior+model.mother_from_boundary(prior)
            y = y-prior
        expected_x.append(x); expected_y.append(y)
    expected = (torch.cat(expected_x), torch.cat(expected_y))
    actual = _bound_evidence_design(model, facts, kind, transport, False)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    parameters = [p for module in (model.encoder, model.long_predictor,
        model.mother_from_boundary, model.relation_basis) for p in module.parameters()]
    gradients = [torch.autograd.grad(sum(v.square().sum() for v in pair), parameters,
        allow_unused=True) for pair in (actual, expected)]
    for a, b in zip(*gradients):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
