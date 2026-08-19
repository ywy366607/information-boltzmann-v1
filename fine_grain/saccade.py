"""Next-look pixel mass from reconstruction residual.

Does not change SliceRead's Q. Unexplained pixels get higher mass in the
next pool. Full-res, no grid, no IG.
"""
from __future__ import annotations

import torch


def residual_map(X: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
    """Per-pixel reconstruction error [B, N]."""
    return (X - recon_x).pow(2).mean(dim=-1)


def residual_rel(X: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
    """mean_n r_n / mean(X²), shape [B]. Small ⇒ this look already explained the field."""
    r = residual_map(X, recon_x)
    denom = X.pow(2).mean(dim=(-1, -2)).clamp_min(1e-6)
    return r.mean(dim=-1) / denom


def residual_pixel_mass(
    X: torch.Tensor, recon_x: torch.Tensor, gain: float = 1.0,
) -> torch.Tensor:
    """π_n = 1 + gain * (r_n / mean(r) - 1), [B, N], mean ≈ 1."""
    r = residual_map(X, recon_x)
    r_bar = r.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return (1.0 + float(gain) * (r / r_bar - 1.0)).clamp_min(0.0)
