"""Training-only spatial measure for the existing Gaussian RGB likelihood."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def stratified_rgb_weights(target: torch.Tensor, foreground=None) -> torch.Tensor:
    """Equal mass for detail, its two-pixel neighbourhood, and the remainder.

    Explicit masks are supervision only. Without masks, use the strongest 20%
    of nontrivial finite-difference edges (absolute threshold .01). Empty strata
    are excluded; a constant image receives uniform weights. This deliberately
    changes the observation measure, not an unbiased estimate of uniform NLL.
    Returned [B,1,H,W] weights have spatial mean one and no autograd history.
    """
    if target.ndim != 4 or target.shape[1] != 3 or not torch.isfinite(target).all():
        raise ValueError("target must be finite [B,3,H,W]")
    if foreground is None:
        dx = F.pad((target[..., 1:] - target[..., :-1]).abs().mean(1, keepdim=True), (0, 1))
        dy = F.pad((target[..., 1:, :] - target[..., :-1, :]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
        strength = torch.maximum(dx, dy)
        threshold = torch.quantile(strength.flatten(1).float(), .8, dim=1).to(target.dtype)
        detail = (strength >= threshold[:, None, None, None]) & (strength > .01)
    else:
        if foreground.shape != (target.shape[0], *target.shape[-2:]):
            raise ValueError("foreground must be [B,H,W]")
        detail = foreground[:, None].to(device=target.device).bool()
    expanded = F.max_pool2d(detail.to(target.dtype), 5, stride=1, padding=2).bool()
    strata = torch.cat([detail, expanded & ~detail, ~expanded], dim=1).to(target.dtype)
    area = strata.sum((2, 3), keepdim=True)
    active = (area > 0).to(target.dtype).sum(1, keepdim=True)
    mass = (strata / area.clamp_min(1)).sum(1, keepdim=True) / active.clamp_min(1)
    return mass * target.shape[-1] * target.shape[-2]


def spatial_nll_mean(nll: torch.Tensor, weights=None) -> torch.Tensor:
    """Optional observation measure; absence preserves the original operation."""
    if weights is None:
        return nll.flatten(1).mean(1)
    if weights.shape != (nll.shape[0], 1, *nll.shape[-2:]):
        raise ValueError("RGB likelihood weights must be [B,1,H,W]")
    if weights.requires_grad:
        raise ValueError("observation weights must not be learned through the target")
    weights = weights.to(device=nll.device, dtype=nll.dtype)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("observation weights must be finite and nonnegative")
    if not torch.allclose(weights.mean((1, 2, 3)), weights.new_ones(nll.shape[0]), atol=1e-5, rtol=1e-5):
        raise ValueError("observation weights must have spatial mean one")
    return (nll * weights).flatten(1).mean(1)
