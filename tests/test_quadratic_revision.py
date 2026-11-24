"""Equivalence of actual solver branches; no synthetic ability training."""
import torch
import pytest

from scripts.ib_local.quadratic_revision import diagonal_gain, evidence_correction


def test_diagonal_gain_is_exact_old_gate_and_has_same_derivative():
    logits = torch.linspace(-40, 40, 31, requires_grad=True)
    new = diagonal_gain(logits)
    old = logits.sigmoid()
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    a = torch.autograd.grad(new.sum(), logits)[0]
    b = torch.autograd.grad(old.sum(), logits)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_full_solver_preserves_old_operations_and_all_gradients():
    torch.manual_seed(15)
    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)
    q = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    new = evidence_correction(x, y, q, .1)
    kernel = x@x.T+.1*torch.eye(len(x), dtype=x.dtype)
    coefficient = torch.linalg.solve(kernel, y)
    old = (kernel, coefficient, (q@x.T)@coefficient,
           torch.linalg.solve(kernel, x@q.T))
    for a, b in zip(new, old):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = torch.autograd.grad(sum(t.square().sum() for t in new), (x, y, q))
    b = torch.autograd.grad(sum(t.square().sum() for t in old), (x, y, q))
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, rtol=0, atol=0)


def test_diagonal_and_dense_solvers_have_the_same_quadratic_minimizer():
    torch.manual_seed(15)
    old = torch.randn(8, 5, dtype=torch.float64)
    proposal = torch.randn_like(old)
    logits = torch.linspace(-4, 4, 8, dtype=old.dtype)
    gain = diagonal_gain(logits)[:, None]
    diagonal = (1-gain)*old+gain*proposal
    new_precision = logits.exp()
    dense = torch.linalg.solve(torch.diag(1+new_precision),
                               old+new_precision[:, None]*proposal)
    torch.testing.assert_close(diagonal, dense, rtol=1e-12, atol=1e-12)


def test_shared_factor_preserves_all_solve_derivatives():
    torch.manual_seed(16)
    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)
    q = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    actual = evidence_correction(x, y, q, .1, return_factor=True)
    expected = evidence_correction(x, y, q, .1)
    for a, b in zip(actual[:4], expected):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    for u, v in zip(torch.autograd.grad(sum(t.square().sum() for t in actual[:4]), (x,y,q)),
                    torch.autograd.grad(sum(t.square().sum() for t in expected), (x,y,q))):
        torch.testing.assert_close(u, v, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('count', [0, 8])
def test_hypothesis_factor_reuse_keeps_values_and_gradients(device, count):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA posterior pullback')
    from scripts.ib_local.causal_relation_program import posterior, mechanism_hypotheses
    torch.manual_seed(23)
    x = torch.randn(count, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(count, 7, dtype=x.dtype, device=device, requires_grad=True)
    q = torch.randn(3, 4, dtype=x.dtype, device=device, requires_grad=True)
    directions = torch.randn(2, 4, 7, device=device, dtype=x.dtype)
    observations = torch.randn(2, count+2, 7, device=device, dtype=x.dtype)
    p = posterior(x, y, q, .1)
    a = mechanism_hypotheses(p, q, directions, observations)
    b = mechanism_hypotheses(dict(p, factor=None), q, directions, observations)
    torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    for u, v in zip(torch.autograd.grad(a.square().sum(), (x,y,q), retain_graph=True, allow_unused=True),
                    torch.autograd.grad(b.square().sum(), (x,y,q), allow_unused=True)):
        if u is None:
            assert v is None
        else:
            torch.testing.assert_close(u, v, rtol=2e-10, atol=2e-10)
