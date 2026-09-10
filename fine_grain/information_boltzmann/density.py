"""Normalized conditional joint density via affine coupling and logistic noise."""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .state import PhaseState


class Coupling(nn.Module):
    def __init__(self, dim: int, hidden: int, parity: int):
        super().__init__()
        self.register_buffer("mask", ((torch.arange(dim) + parity) % 2).float())
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * dim))

    def coefficients(self, z: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        context = context.expand(*z.shape[:-1], -1)
        scale, shift = self.net(torch.cat((z * self.mask, context), -1)).chunk(2, -1)
        return 0.6 * scale.tanh() * (1 - self.mask), 0.5 * shift.tanh() * (1 - self.mask)

    def forward(self, z: Tensor, context: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
        scale, shift = self.coefficients(z, context)
        if inverse:
            return (z - shift) * (-scale).exp(), -scale.sum(-1)
        return z * scale.exp() + shift, scale.sum(-1)


class PhaseDensity:
    def __init__(self, operator: "InitialDensity", context: Tensor):
        self.operator, self.context = operator, context

    def sample(self, count: int, generator: torch.Generator | None = None) -> PhaseState:
        op = self.operator
        u = torch.rand(count, 2 * op.phase_dim, device=self.context.device,
                       dtype=self.context.dtype, generator=generator)
        u = u.clamp(torch.finfo(u.dtype).eps, 1 - torch.finfo(u.dtype).eps)
        base = op.base_scale * (u.log() - torch.log1p(-u))
        joint, _ = op.transform(base, self.context)
        return PhaseState(*joint.chunk(2, -1))

    def log_prob(self, x: Tensor, v: Tensor) -> Tensor:
        op = self.operator
        base, logdet = op.transform(torch.cat((x, v), -1), self.context, inverse=True)
        a = base / op.base_scale
        return (-F.softplus(a) - F.softplus(-a) - math.log(op.base_scale)).sum(-1) + logdet


class InitialDensity(nn.Module):
    def __init__(self, vocab: int, phase_dim: int, hidden: int, layers: int,
                 base_scale: float = 0.25):
        super().__init__()
        self.phase_dim, self.base_scale = phase_dim, base_scale
        self.embedding = nn.Embedding(vocab, hidden)
        self.encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(Coupling(2 * phase_dim, hidden, i % 2) for i in range(layers))

    def condition(self, tokens: Tensor, mask: Tensor | None = None) -> PhaseDensity:
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("A nonempty one-dimensional boot sequence is required")
        h = self.embedding(tokens)
        pos = torch.arange(len(tokens), device=h.device, dtype=h.dtype)[:, None]
        freq = torch.exp(-torch.arange(h.shape[-1], device=h.device, dtype=h.dtype) / h.shape[-1] * 8)
        h = self.encoder(h + torch.sin(pos * freq))
        weights = torch.ones(len(tokens), device=h.device, dtype=h.dtype) if mask is None else mask.to(h)
        if weights.sum() <= 0:
            raise ValueError("Boot sequence must contain an unmasked token")
        return PhaseDensity(self, (h * weights[:, None]).sum(0) / weights.sum())

    def transform(self, z: Tensor, context: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
        logdet = z.new_zeros(z.shape[:-1])
        layers = reversed(self.layers) if inverse else self.layers
        for layer in layers:
            z, change = layer(z, context, inverse)
            logdet = logdet + change
        return z, logdet
