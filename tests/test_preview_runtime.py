"""Execution parity, lifetime and evidence identity; no capability experiment."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.ib_local.causal_relation_program import Evidence, Port, RelationState
from scripts.ib_local.generative_relations import marginal, partition_evidence, organization_scores, leave_one_out_scores
from scripts.ib_local.causal_relation_program import posterior, log_score
from scripts.ib_local.preview_runtime import preview_scope, read


def test_preview_reads_expire_bypass_training_and_release_after_error():
    model = object()
    calls = []
    def compute():
        calls.append(1)
        return len(calls)
    with torch.no_grad(), preview_scope(model):
        assert read(model, 'x', 1, compute) == read(model, 'x', 1, compute) == 1
        with torch.enable_grad():
            assert read(model, 'x', 1, compute) == 2
    with torch.no_grad():
        assert read(model, 'x', 1, compute) == 3
        with pytest.raises(ValueError), preview_scope(model):
            read(model, 'x', 1, compute)
            raise ValueError('interrupt hypothetical branch')
        assert read(model, 'x', 1, compute) == 5


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('n', [4, 8])
def test_all_organization_scores_match_separate_solves_for_each_target(device, n):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA parity')
    torch.manual_seed(1616)
    rand = lambda *shape: torch.randn(*shape, dtype=torch.float64, device=device)
    model = SimpleNamespace(encoder=torch.nn.Linear(12, 8, dtype=torch.float64, device=device),
                            joint_relation_update_revision=3)
    raw, act = rand(n, 12), rand(n, 12)
    facts = tuple(Evidence(raw[i:i+1], act[i:i+1], rand(1, 11), Port.MOTHER,
                           True, fact_id=f'episode:{i+1}') for i in range(n))
    model.memory = SimpleNamespace(source=raw)
    state = RelationState(1, 1, 1, 0, torch.arange(n, device=device),
                          torch.ones(n, device=device), torch.ones(n, dtype=torch.bool, device=device),
                          constraints=facts, evidence_bound=True)
    x = rand(n, 5)
    noise = torch.rand(11, dtype=x.dtype, device=device)+.1
    with torch.no_grad(), preview_scope(model):
        for h in range(3):
            # Same geometry but independent outcome. Never reuse its target
            # solve, split score, or chosen U operation from the other branch.
            y = rand(n, 11)
            base, scores, _, order = organization_scores(model, state, x, y, raw, .2, noise)
            torch.testing.assert_close(base, marginal(x, y, .2, noise_prior=noise), atol=2e-10, rtol=2e-10)
            torch.testing.assert_close(scores, partition_evidence(x, y, order, .2, noise_prior=noise),
                                       atol=2e-10, rtol=2e-10)
        # In-place changed condition must invalidate a previous geometry key.
        raw.add_(.1*rand(n, 12))
        y = rand(n, 11)
        actual = organization_scores(model, state, x, y, raw, .2, noise)
    with torch.no_grad():
        expected = organization_scores(model, state, x, y, raw, .2, noise)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('n', [1, 4, 8])
@pytest.mark.parametrize('learned_noise', [False, True])
def test_batched_leave_one_out_keeps_values_and_all_derivatives(device, n, learned_noise):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA LOO parity')
    torch.manual_seed(1617)
    x = torch.randn(n, 5, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(n, 11, dtype=x.dtype, device=device, requires_grad=True)
    noise = (torch.rand(11, dtype=x.dtype, device=device)+.1).requires_grad_() if learned_noise else None
    indices = list(range(n))[::-1]
    actual = leave_one_out_scores(x, y, indices, .2, noise_prior=noise)
    expected = torch.stack([log_score(posterior(
        x[[j for j in range(n) if j != i]], y[[j for j in range(n) if j != i]],
        x[i:i+1], .2, noise_prior=noise), y[i:i+1]).sum() for i in indices])
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    params = [x, y]+([] if noise is None else [noise])
    a = torch.autograd.grad(actual.sum(), params)
    b = torch.autograd.grad(expected.sum(), params)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, atol=2e-9, rtol=2e-9)
