"""Shared language evaluation contracts for Information Boltzmann."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@torch.no_grad()
def field_energy_statistics(field: torch.Tensor) -> dict[str, float]:
    """Parseval DC/spatial split on a uniform periodic grid, without an FFT.

    Energy is site-averaged and summed over content channels. Each sample's
    spatial mean is computed separately; no batch averaging erases structure.
    """
    if field.ndim != 5:
        raise ValueError("Expected [B,X,Y,Z,D] kinetic field")
    center = field.mean(dim=(1, 2, 3), keepdim=True)
    total = 0.5 * field.square().sum(-1).mean()
    dc = 0.5 * center.square().sum(-1).mean()
    spatial = 0.5 * (field - center).square().sum(-1).mean()
    denominator = total.clamp_min(torch.finfo(field.dtype).tiny)
    values = torch.stack((total, dc, spatial, dc / denominator,
                          spatial / denominator)).cpu().tolist()
    return dict(zip(("field_energy", "dc_energy", "spatial_energy",
                     "dc_share", "spatial_share"), values))


@dataclass(frozen=True)
class WarmSiteSpec:
    """Fixed local contexts initialized from one mature-state regime."""

    site_starts: tuple[int, ...] = (8192, 12288, 16384, 20480)
    warm_in_tokens: int = 256
    score_tokens: int = 128

    def __post_init__(self) -> None:
        if not self.site_starts:
            raise ValueError("Warm-site evaluation requires at least one site")
        if any(start < 0 for start in self.site_starts):
            raise ValueError("site starts must be non-negative")
        if len(set(self.site_starts)) != len(self.site_starts):
            raise ValueError("site starts must be unique")
        if self.warm_in_tokens < 1:
            raise ValueError("Warm-site evaluation requires positive warm-in")
        if self.score_tokens < 1:
            raise ValueError("site score_tokens must be positive")

    @property
    def required_tokens_per_site(self) -> int:
        return self.warm_in_tokens + self.score_tokens + 1
