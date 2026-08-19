"""Rectified / OT flow matching on the point field.

Same Native MoT graph. t is an extra condition on X (zero-init, off when
absent). Ports only change (x0, x1, prompt).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


def sample_t(n: int, device, kind: str = "uniform") -> torch.Tensor:
    """t ∈ (0,1). ``jit`` = official JiT logit-normal (P_mean=-0.8, P_std=0.8)."""
    k = str(kind).lower()
    if k in ("jit", "official"):
        return torch.sigmoid(torch.randn(n, device=device) * 0.8 - 0.8)
    if k in ("logit_normal", "logit", "rf"):
        return torch.sigmoid(torch.randn(n, device=device))
    return torch.rand(n, device=device)


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1-t) x0 + t x1. t: [B] or [B,1,1,1]."""
    tb = t.reshape(-1, *([1] * (x0.ndim - 1))).to(dtype=x0.dtype, device=x0.device)
    return (1.0 - tb) * x0 + tb * x1


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def v_from_x_pred(x_hat: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """JiT: v_θ = (x_θ − z_t) / (1 − t). t=1 is data."""
    tb = t.reshape(-1, *([1] * (z_t.ndim - 1))).to(dtype=z_t.dtype, device=z_t.device)
    return (x_hat - z_t) / (1.0 - tb).clamp(min=eps)


class TimeCondition(nn.Module):
    """t ∈ [0,1] → d_x. Last layer zero so t is a no-op at init."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        te = self.net(t.reshape(-1, 1).to(dtype=like.dtype, device=like.device))
        return te.unsqueeze(1)


class TimeMod(nn.Module):
    """Legacy raw-t FiLM. Prefer AdaLNZero(c) with sinusoidal t + text."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ab = self.net(t.reshape(-1, 1).to(dtype=x.dtype, device=x.device))
        scale, shift = ab.chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Official JiT/DiT: sinusoidal(t) → MLP. Live from step 0 (not zero-init)."""

    def __init__(self, d: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

    @staticmethod
    def sinusoid(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().reshape(-1, 1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoid(t, self.freq_dim).to(dtype=self.mlp[0].weight.dtype))


class AdaLNZero(nn.Module):
    """Official modulate: x ⊙ (1+scale(c)) + shift(c). Last linear zero = identity."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.net(c.to(dtype=x.dtype)).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@torch.no_grad()
def euler_integrate(step_fn, x0: torch.Tensor, n_steps: int = 8, clamp: bool = True) -> torch.Tensor:
    """x ← x + Δt v(x, t). step_fn(x, t[B]) → v."""
    return ode_integrate(step_fn, x0, n_steps=n_steps, method="euler", clamp=clamp)


@torch.no_grad()
def heun_integrate(step_fn, x0: torch.Tensor, n_steps: int = 8, clamp: bool = True) -> torch.Tensor:
    """JiT Heun (RK2). Last step is Euler — v = (x−z)/(1−t) blows up at t=1."""
    return ode_integrate(step_fn, x0, n_steps=n_steps, method="heun", clamp=clamp)


@torch.no_grad()
def ode_integrate(
    step_fn,
    x0: torch.Tensor,
    n_steps: int = 8,
    method: str = "heun",
    clamp: bool = True,
) -> torch.Tensor:
    """Integrate z' = v(z,t) from t=0 to t=1.

    Euler: one v eval per step.
    Heun (official JiT): predictor-corrector, 2 evals, last step Euler.
    """
    z = x0
    n = max(1, int(n_steps))
    B = x0.shape[0]
    times = torch.linspace(0.0, 1.0, n + 1, device=x0.device, dtype=x0.dtype)
    kind = str(method).lower()
    for i in range(n):
        t = times[i].expand(B)
        t_next = times[i + 1].expand(B)
        dt = (times[i + 1] - times[i]).to(dtype=x0.dtype)
        v0 = step_fn(z, t)
        if kind in ("heun", "rk2") and i < n - 1:
            z_eul = z + dt * v0
            if clamp:
                z_eul = z_eul.clamp(0.0, 1.0)
            v1 = step_fn(z_eul, t_next)
            z = z + dt * 0.5 * (v0 + v1)
        else:
            z = z + dt * v0
        if clamp:
            z = z.clamp(0.0, 1.0)
    return z.clamp(0.0, 1.0) if clamp else z
