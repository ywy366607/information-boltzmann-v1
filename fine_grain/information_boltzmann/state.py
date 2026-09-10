"""Equal-weight continuous phase samples, not velocity bins."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PhaseState:
    x: Tensor  # [particles, phase_dim]
    v: Tensor  # dx/dt between collisions
    time: float = 0.0

    def detach(self) -> "PhaseState":
        return PhaseState(self.x.detach(), self.v.detach(), self.time)

    def belief(self) -> tuple[Tensor, Tensor]:
        mean = self.x.mean(0)
        centered = self.x - mean
        return mean, centered.T @ centered / len(self.x)

    def moments(self) -> dict[str, Tensor]:
        return {
            "position_second_moment": self.x.square().sum(-1).mean(),
            "kinetic_energy": 0.5 * self.v.square().sum(-1).mean(),
            "cross_moment": (self.x * self.v).sum(-1).mean(),
            "position_variance": self.belief()[1].trace(),
            "velocity_variance": (self.v - self.v.mean(0)).square().sum(-1).mean(),
        }
