from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fine_grain.active_gdn2 import ActiveInferenceGDN2, ActiveInferenceState


@dataclass
class CascadeGDNState:
    """Persistent states for a geometric bank of memory time scales."""

    levels: Tuple[ActiveInferenceState, ...]


class CascadeActiveInferenceGDN2(nn.Module):
    """Reference multi-scale GDN state bank derived from GenerationCascadeMemory.

    One parameterized GDN kernel is shared across levels; only persistent state
    is replicated.  Each level applies an exponential retention factor with a
    geometric time constant before assimilating the same arrived event.  This
    is an explicit reference path, not a silent replacement for ActiveInferenceGDN2.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int,
        res: int,
        n_levels: int = 4,
        base_tau: float = 1.0,
        gamma: float = 4.0,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be positive")
        if base_tau <= 0.0 or gamma <= 1.0:
            raise ValueError("base_tau must be positive and gamma must exceed one")
        self.memory = ActiveInferenceGDN2(d_model=d_model, n_slots=n_slots, res=res)
        taus = base_tau * gamma ** torch.arange(n_levels, dtype=torch.float32)
        self.register_buffer("time_constants", taus, persistent=True)

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CascadeGDNState:
        return CascadeGDNState(tuple(
            self.memory.initial_state(batch, device, dtype)
            for _ in range(self.time_constants.numel())
        ))

    def decay_state(self, state: ActiveInferenceState, tau: torch.Tensor) -> ActiveInferenceState:
        """Apply one exponential retention step while preserving the mean."""
        factor = torch.exp(-1.0 / tau.to(state.natural).clamp_min(1e-6))
        natural = state.natural * factor
        precision = state.precision * factor
        return ActiveInferenceState(
            natural=natural,
            precision=precision,
            velocity_mean=state.velocity_mean,
            velocity_precision=state.velocity_precision,
            last_tokens=state.last_tokens,
            last_centers=state.last_centers,
            last_observation_precision=state.last_observation_precision,
        )

    def assimilate(
        self,
        state: CascadeGDNState,
        tokens: torch.Tensor,
        centers: torch.Tensor,
        observation_precision,
        update_motion: bool = False,
    ) -> CascadeGDNState:
        levels = []
        for tau, level in zip(self.time_constants, state.levels):
            decayed = self.decay_state(level, tau)
            levels.append(self.memory.assimilate(
                decayed, tokens, centers, observation_precision,
                update_motion=update_motion,
            ))
        return CascadeGDNState(tuple(levels))

    def query(
        self,
        state: CascadeGDNState,
        tokens: torch.Tensor,
        target_centers: torch.Tensor,
        horizon,
        action: Optional[torch.Tensor] = None,
        action_precision=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a precision-weighted mixture of all time scales."""
        means = []
        variances = []
        for level in state.levels:
            mean, logvar, _ = self.memory.query(
                level, tokens, target_centers, horizon,
                action=action, action_precision=action_precision,
            )
            means.append(mean)
            variances.append(logvar.exp())
        weights = (1.0 / self.time_constants).to(tokens).view(1, -1, 1, 1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        mean = (torch.stack(means, dim=1) * weights).sum(dim=1)
        variance = (torch.stack(variances, dim=1) * weights).sum(dim=1)
        return mean, variance.clamp_min(1e-6).log()
