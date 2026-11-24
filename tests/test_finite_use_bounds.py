"""Density/payment bounds for the exact production finite-U score."""
import torch

from scripts.ib_local.causal_relation_program import posterior, log_score
from scripts.ib_local.finite_use_bounds import score_ceiling, payment_ceiling


def test_production_student_score_obeys_fixed_prior_evidence_ceiling():
    g = torch.Generator().manual_seed(11019)
    prior = torch.tensor([.01, .2, .5], dtype=torch.float64)
    ceiling = score_ceiling(prior, 9)
    for n in (0, 1, 4, 8, 9):
        x = torch.randn(n, 4, generator=g, dtype=prior.dtype)
        y = torch.randn(n, 3, generator=g, dtype=prior.dtype)
        q = torch.randn(7, 4, generator=g, dtype=prior.dtype)
        d = posterior(x, y, q, .1, noise_prior=prior)
        at_peak = log_score(d, d['mean'])
        random = log_score(d, torch.randn(7, 3, generator=g, dtype=prior.dtype))
        assert bool((random <= at_peak+1e-12).all())
        assert bool((at_peak <= ceiling+1e-12).all())
        before = random-.1
        assert float((random-before).mean()) <= float(payment_ceiling(before, prior, 9))
