"""Shared language evaluation contracts for Information Boltzmann."""
from __future__ import annotations

from dataclasses import dataclass


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
