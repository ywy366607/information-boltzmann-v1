"""Experimental structural-capacity constraint, separate from activity storage.

Mean trace(B B^T) is a declared propagation-capacity proxy, not measured ATP
consumption. This module does not alter MediumState or run a learning rule.
The active production trainer has not opted into this candidate.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def transport_capacity(factor: torch.Tensor) -> torch.Tensor:
    """Quadrature mean trace(A), one value per individual [B].

    factor has shape [B,X,Y,Z,3,3]. A uniform unit B costs three, independently
    of runtime grid resolution. Distinct batch members have separate budgets.
    """
    if factor.ndim != 6 or factor.shape[-2:] != (3, 3):
        raise ValueError('Expected factor [B,X,Y,Z,3,3]')
    return factor.square().sum((-2, -1)).mean((1, 2, 3))


class TransportCapacityBudget(nn.Module):
    """Radial projection onto a fixed per-individual capacity ball.

    Above budget, normalize B by sqrt(budget/cost); below budget, keep B.
    Positive scaling retains strict SPD whenever the input factor is invertible.
    The active-budget derivative couples allocation across sites, while leaving
    ratios and principal directions unchanged for a fixed raw material field.
    This is one resource constraint, not a second dissipative bath.
    """

    def __init__(self, budget: float):
        super().__init__()
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Capacity budget must be finite and positive')
        self.register_buffer('budget', torch.tensor(budget, dtype=torch.float64))

    def forward(self, factor: torch.Tensor) -> torch.Tensor:
        return factor * self.allocation_scale(factor)

    def allocation_scale(self, factor: torch.Tensor) -> torch.Tensor:
        """Scale from slow structure; apply local fast gains afterwards."""
        cost = transport_capacity(factor)
        torch._assert_async(torch.isfinite(cost).all(), 'Non-finite transport capacity')
        # Clamp the ratio before sqrt: zero-capacity input has finite derivatives.
        ratio = (cost / self.budget.to(factor)).clamp_min(1.)
        scale = ratio.rsqrt().reshape(-1, 1, 1, 1, 1, 1)
        return scale
