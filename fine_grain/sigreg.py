"""Official LeJEPA SIGReg: Epps–Pulley on random 1D sketches.

Port of galilai-group/lejepa:
  lejepa.univariate.EppsPulley
  lejepa.multivariate.SlicingUnivariateTest

Single-process: DDP all_reduce is a no-op (world_size=1).
Projection directions A are sampled under no_grad (same as upstream).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """T = N ∫ |φ_N(t) − φ_0(t)|² w(t) dt against N(0,1), t ≥ 0 + symmetry.

    Upstream: t_max=3, n_points=17, trapezoid, weights already include φ(t).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        self.n_points = int(n_points)
        t = torch.linspace(0, float(t_max), self.n_points, dtype=torch.float32)
        dt = float(t_max) / (self.n_points - 1)
        weights = torch.full((self.n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = t.square().mul(0.5).neg().exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, N, K) → (*, K)
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t.to(device=x.device, dtype=x.dtype)
        cos_mean = torch.cos(x_t).mean(dim=-3)
        sin_mean = torch.sin(x_t).mean(dim=-3)
        phi = self.phi.to(device=x.device, dtype=x.dtype)
        err = (cos_mean - phi).square() + sin_mean.square()
        return (err @ self.weights.to(device=x.device, dtype=x.dtype)) * N


class SlicingUnivariateTest(nn.Module):
    """x @ A, A = random unit directions. Official default sampler='gaussian'."""

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int = 1024,
        reduction: str = "mean",
        clip_value: Optional[float] = None,
    ):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = int(num_slices)
        self.reduction = reduction
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(int(seed))
        return self._generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, N, D)
        with torch.no_grad():
            seed = int(self.global_step.item())
            g = self._get_generator(x.device, seed)
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0).clamp_min(1e-8)
            self.global_step.add_(1)
        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, stats.new_zeros(()), stats)
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats


def official_sigreg(
    Z: torch.Tensor,
    num_slices: int = 1024,
    n_points: int = 17,
    t_max: float = 3.0,
) -> torch.Tensor:
    """LeJEPA SIGReg on a cloud of embeddings.

    Z: [N, D] or [B, M, D] (flattened to N=B·M samples in R^D).
    """
    if Z.ndim == 3:
        Z = Z.reshape(Z.shape[0] * Z.shape[1], Z.shape[2])
    if Z.ndim != 2:
        raise ValueError(f"SIGReg expects [N,D] or [B,M,D], got {tuple(Z.shape)}")
    test = SlicingUnivariateTest(
        EppsPulley(t_max=t_max, n_points=n_points),
        num_slices=num_slices,
        reduction="mean",
    ).to(device=Z.device)
    return test(Z)


# Module-level tester so random-direction RNG advances across steps (official).
_DEFAULT_TEST: Optional[SlicingUnivariateTest] = None
_DEFAULT_KEY: Optional[tuple] = None


def compute_sigreg_loss(
    Z: torch.Tensor,
    num_slices: int = 1024,
    n_points: int = 17,
    t_max: float = 3.0,
    **_ignored,
) -> Dict[str, torch.Tensor]:
    """Drop-in used by BayesianSurpriseGate. SIGReg is on *live* Z."""
    global _DEFAULT_TEST, _DEFAULT_KEY
    x = Z.reshape(-1, Z.shape[-1]) if Z.ndim == 3 else Z
    key = (int(num_slices), int(n_points), float(t_max), str(x.device), str(x.dtype))
    if _DEFAULT_TEST is None or _DEFAULT_KEY != key:
        _DEFAULT_TEST = SlicingUnivariateTest(
            EppsPulley(t_max=t_max, n_points=n_points),
            num_slices=num_slices,
            reduction="mean",
        ).to(device=x.device, dtype=x.dtype)
        _DEFAULT_KEY = key
    loss = _DEFAULT_TEST(x)
    return {
        "sigreg_total": loss,
        "kind": "epps_pulley_slicing",
        "num_slices": float(num_slices),
    }
