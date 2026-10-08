"""Causal physical waiting time, separate from numerical sampling resolution.

The caller supplies features measured BEFORE assimilating the next observation.
This module does not read tokens, targets, or advance/reset any physical state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class IntrinsicTimePolicy(nn.Module):
    """A small history-conditioned clock; zero initialization gives reference.

    ``features`` contains only pre-input read-aperture history supplied by the
    caller. A new token must not control the global clock through these inputs.
    ``max_duration`` is an explicit execution safety limit, not a physical law:
    exceeding it raises instead of silently clipping the physical trajectory.
    """

    def __init__(self, reference_duration: float, *, feature_dim: int = 4,
                 max_duration: float | None = None):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        reference = _positive(reference_duration, "reference_duration")
        if max_duration is not None:
            max_duration = _positive(max_duration, "max_duration")
            if max_duration < reference:
                raise ValueError("max_duration cannot exclude the initial reference")
        self.register_buffer("reference_duration", torch.tensor(reference, dtype=torch.float64))
        self.max_duration = max_duration
        self.head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.head.in_features:
            raise ValueError("Expected pre-input features [batch, feature_dim]")
        torch._assert_async(torch.isfinite(features).all(), "Nonfinite clock features")
        duration = self.reference_duration.to(features) * self.head(features).squeeze(-1).exp()
        torch._assert_async((torch.isfinite(duration) & (duration > 0)).all(),
                            "Intrinsic duration must remain positive and representable")
        if self.max_duration is not None:
            torch._assert_async((duration <= self.max_duration).all(),
                                "Intrinsic duration exceeds explicit execution budget")
        return duration


@dataclass(frozen=True)
class EvolutionSchedule:
    """Frozen loop counts; pass the original duration to interval_duration.

    Counts are chosen outside autograd. Every interval remains differentiable
    in physical duration. Reuse this object in checkpoint recomputation; a fixed
    schedule can also be planned against an explicit maximum duration before
    compilation. Both medium and observer clocks must advance once per interval.
    """

    observer_count: int
    solver_substeps: int
    planned_duration: float

    @classmethod
    def for_duration(cls, duration: float | torch.Tensor, *, solver_max_step: float,
                     observer_max_step: float, max_steps: int) -> "EvolutionSchedule":
        solver_step = _positive(solver_max_step, "solver_max_step")
        observer_step = _positive(observer_max_step, "observer_max_step")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be an explicit positive integer budget")
        # A shared count can cover a batch; distinct physical clocks remain the
        # caller's responsibility. Only loop selection uses detached values.
        if isinstance(duration, torch.Tensor):
            if (duration.numel() < 1 or not duration.is_floating_point()
                    or not bool(torch.isfinite(duration).all())
                    or not bool((duration >= 0).all())):
                raise ValueError("duration must contain finite nonnegative floating values")
            requested = float(duration.detach().max().cpu())
        else:
            requested = float(duration)
            if not math.isfinite(requested) or requested < 0:
                raise ValueError("duration must be finite and nonnegative")
        observers = max(1, math.ceil(requested / observer_step))
        solvers = max(1, math.ceil((requested / observers) / solver_step))
        if observers * solvers > max_steps:
            raise ValueError("Requested precision exceeds explicit solver-step budget")
        return cls(observers, solvers, requested)

    def interval_duration(self, duration: torch.Tensor) -> torch.Tensor:
        """Actual physical interval, without detaching or clipping duration."""
        if not duration.is_floating_point():
            raise ValueError("Physical duration must use a floating tensor")
        torch._assert_async((torch.isfinite(duration) & (duration >= 0)).all(),
                            "Finite nonnegative physical duration required")
        # Floating comparisons allow only rounding error, not extra physical time.
        tolerance = 8 * torch.finfo(duration.dtype).eps
        torch._assert_async((duration <= self.planned_duration * (1 + tolerance)).all(),
                            "Duration exceeds the frozen schedule; replan explicitly")
        return duration / self.observer_count


@dataclass(frozen=True)
class CharacteristicTimeScales:
    cell_crossing_time: torch.Tensor
    torus_crossing_time: torch.Tensor


def factor_characteristic_time(factor: torch.Tensor, *, cell_spacing: Sequence[float],
                               domain_lengths: Sequence[float] = (1., 1., 1.)) -> CharacteristicTimeScales:
    """Reference estimates from actual B and geometry, not a token-count rule.

    The cell scale is 1/(sqrt(2)*max ||diag(1/h)B||_F), a conservative
    local rotation-angle scale for the existing skew edge solver. The domain
    scale is domain diagonal / max singular speed(B). It is a nominal fastest
    traversal scale, NOT proof of arrival, controllability, or useful mixing:
    heterogeneous bottlenecks, dispersion, damping and sensor geometry matter.
    A zero factor returns infinite scales: no finite transport reference exists.
    Taking a global maximum defines one clock reference, not a fast activity
    feedback policy. Use frozen/calibrated factors when constructing a policy.
    """
    if factor.ndim < 2 or factor.shape[-2:] != (3, 3) or not factor.is_floating_point():
        raise ValueError("factor must be a floating tensor ending in [3,3]")
    if len(cell_spacing) != 3 or len(domain_lengths) != 3:
        raise ValueError("Three spacings and domain lengths required")
    spacing = factor.new_tensor([_positive(x, "cell_spacing") for x in cell_spacing])
    lengths = factor.new_tensor([_positive(x, "domain_lengths") for x in domain_lengths])
    torch._assert_async(torch.isfinite(factor).all(), "Finite transport factor required")
    resolved_rate = (factor / spacing[:, None]).square().sum((-2, -1)).sqrt().amax()
    speed = torch.linalg.matrix_norm(factor, ord=2).amax()
    return CharacteristicTimeScales((math.sqrt(2.) * resolved_rate).reciprocal(),
                                   lengths.norm() / speed)
