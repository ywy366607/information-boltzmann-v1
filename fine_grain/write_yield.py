"""Write-side Yield: admission τ, not leak α.

GDN's Delta rule removes *predictable* redundancy:
  Δ_eff = write⊙v − S(erase⊙k)
A novel garbage key still has S(erase⊙k)≈0, so it writes if β>0.

Yield answers a different question: is the leftover innovation *worth*
occupying finite state?

  u = sign(Δ) ⊙ ReLU(|Δ| − τ)
  |Δ| ≤ τ  ⇒  u = 0   (refuse before memory)
  S ← α S + β u kᵀ     (α is retention; not the garbage switch)

α→1 keeps old signal; τ stops worthless writes. They are not one knob.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def yield_residual(delta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """u = sign(Δ) ⊙ ReLU(|Δ| − τ). Broadcast τ to delta."""
    t = tau.to(device=delta.device, dtype=delta.dtype)
    return delta.sign() * F.relu(delta.abs() - t)


def retain_and_write(
    x: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
    pi=1.0,
) -> torch.Tensor:
    """X ← α X + π u. α=1 is pure residual write (generation canvas)."""
    return float(alpha) * x + pi * delta
