"""Yield on systematic surprise, not L2 reconstruction error.

Borrowed mechanism: ReLU(s − τ) is exact ∅ below threshold.
s is Bayesian U / VFE gap / F already computed on slices, scattered to
points. τ = softplus(W k + b). Write direction stays Deslice.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YieldGate(nn.Module):
    """Per-point yield: scale = ReLU(s − τ) / s. k is the point (GDN key)."""

    def __init__(self, d_x: int):
        super().__init__()
        self.to_tau = nn.Linear(int(d_x), 1)
        nn.init.zeros_(self.to_tau.weight)
        nn.init.constant_(self.to_tau.bias, -2.0)  # softplus(-2) ≈ 0.13

    def forward(
        self, s: torch.Tensor, k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """s [B,N] ≥ 0, k [B,N,d] → scale [B,N,1], tau [B,N,1], excess [B,N,1]."""
        tau = F.softplus(self.to_tau(k))
        s = s.reshape(k.shape[0], k.shape[1], 1).to(dtype=k.dtype).clamp_min(0.0)
        excess = F.relu(s - tau)
        scale = excess / s.clamp_min(1e-6)
        return scale, tau, excess
