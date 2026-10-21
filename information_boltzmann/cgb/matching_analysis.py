"""Gauge and rank analysis of the developmental matching model."""
from __future__ import annotations

import numpy as np


def balanced_rank_embedding(code, affinity, rank):
    """Embed an identical interaction into a wider code and balance its gauge.

    C -> t C, A -> A/t^2 leaves C A C^T unchanged. At the minimum
    mean(C^2) = 2 mean(A^2). For the same active factors padded to a wider k,
    the minimized penalty scales as k^(-4/3), so lambda must scale as k^(4/3).
    This only concerns parameterization of the offline fit, not brain energy.
    """
    code, affinity = np.asarray(code,dtype=float), np.asarray(affinity,dtype=float)
    old_rank = code.shape[1]
    if affinity.shape != (old_rank,old_rank) or rank < old_rank:
        raise ValueError('Expected square affinity and a nondecreasing rank')
    c = np.zeros((len(code),rank)); a = np.zeros((rank,rank))
    c[:,:old_rank] = code; a[:old_rank,:old_rank] = affinity
    c2, a2 = np.mean(c*c), np.mean(a*a)
    if c2 <= 0 or a2 <= 0:
        raise ValueError('Gauge balancing requires a nonzero interaction')
    scale = (2*a2/c2)**(1/6)
    return c*scale, a/scale**2


def regularization_for_rank(value, source_rank, target_rank):
    return value*(target_rank/source_rank)**(4/3)
