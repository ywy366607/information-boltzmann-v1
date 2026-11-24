"""Bounds for the ORIGINAL marginal finite-U score at fixed coordinates.

Every posterior scale is at least beta_prior/(1+n/2). The Student-t density
is maximal at its mean. Maximizing this peak over the allowed evidence count
gives a certificate for fixed current queries, prior and encoder coordinates.
It is an optimistic ceiling, not an attainable performance target.
"""
import math

import torch


def score_ceiling(noise_prior, max_evidence):
    if max_evidence < 0 or not bool((noise_prior > 0).all()):
        raise ValueError('finite nonnegative evidence bound and positive prior required')
    n = torch.arange(max_evidence+1, device=noise_prior.device, dtype=noise_prior.dtype)
    df = 2+n
    alpha = 1+n/2
    peak = (torch.lgamma((df[:, None]+1)/2)-torch.lgamma(df[:, None]/2)
            -.5*torch.log(df[:, None]*math.pi*noise_prior[None]/alpha[:, None]))
    return peak.mean(-1).max()


def payment_ceiling(before_scores, noise_prior, max_evidence):
    if before_scores.numel() == 0:
        return before_scores.new_zeros(())
    return score_ceiling(noise_prior, max_evidence)-before_scores.mean()
