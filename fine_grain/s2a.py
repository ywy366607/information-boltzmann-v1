"""S2a: amortized G — will another look change q(answer)?"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_cat(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) over last dim, [B]."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * (p.log() - q.log())).sum(dim=-1)


def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


class AnswerGazeHead(nn.Module):
    """Ĝ(h): P(another look would move the answer)."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        nn.init.constant_(self.net[-1].bias, 2.0)  # start "look again" so we don't collapse to F2 on day 0

    def forward(self, h_pool: torch.Tensor) -> torch.Tensor:
        return self.net(h_pool).squeeze(-1)
