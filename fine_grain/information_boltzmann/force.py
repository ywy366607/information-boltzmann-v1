"""Observed-token drive inside a confined, damped phase transport."""
import math

import torch
from torch import Tensor, nn

from .state import PhaseState


class DataForce(nn.Module):
    def __init__(self, vocab: int, dim: int, hidden: int, kappa: float = 1.0,
                 gamma: float = 1.0, amplitude: float = 1.0, temperature: float = 0.0):
        super().__init__()
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        self.kappa, self.gamma, self.amplitude = kappa, gamma, amplitude
        self.temperature = temperature
        self.embedding = nn.Embedding(vocab, hidden)
        self.net = nn.Sequential(nn.Linear(hidden + dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def drive(self, x: Tensor, token: int, time: float) -> Tensor:
        h = self.embedding.weight[token].expand(len(x), -1)
        clock = x.new_tensor([math.sin(time), math.cos(time)]).expand(len(x), -1)
        return self.amplitude / math.sqrt(x.shape[-1]) * self.net(torch.cat((x, h, clock), -1)).tanh()

    def forward(self, x: Tensor, v: Tensor, token: int, time: float) -> Tensor:
        return -self.kappa * x - self.gamma * v + self.drive(x, token, time)

    def kick(self, x: Tensor, v: Tensor, token: int, time: float, duration: float,
             generator: torch.Generator | None = None, budget: dict | None = None) -> Tensor:
        # Exact Ornstein-Uhlenbeck velocity update for frozen x, drive and time.
        # When temperature > 0, adds Langevin thermal fluctuations with variance T * (1 - e^{-2 gamma h}).
        decay = math.exp(-self.gamma * duration)
        integral = -math.expm1(-self.gamma * duration) / self.gamma if self.gamma else duration
        drive = self.drive(x, token, time)
        c = -self.kappa * x + drive
        mean = decay * v + integral * c
        result = mean
        if self.temperature > 0.0 and duration > 0.0:
            noise_var = self.temperature * (-math.expm1(-2.0 * self.gamma * duration))
            noise_std = math.sqrt(max(0.0, noise_var))
            if generator is not None and generator.device.type != v.device.type:
                noise = torch.randn(v.shape, device=generator.device, dtype=v.dtype, generator=generator).to(v.device)
            else:
                noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
            result = mean + noise_std * noise
        if budget is not None:
            # Exact work along the deterministic frozen-force velocity path.
            if self.gamma > 0:
                terminal = c / self.gamma
                transient = v - terminal
                integral_v = terminal * duration + transient * integral
                integral_v2 = (terminal.square().sum(-1) * duration
                               + 2 * (terminal * transient).sum(-1) * integral
                               + transient.square().sum(-1)
                               * (-math.expm1(-2*self.gamma*duration))/(2*self.gamma))
            else:
                integral_v = v*duration + c*duration**2/2
                integral_v2 = v.square().sum(-1)*duration
            values = {
                "drive_work": (drive*integral_v).sum(-1).mean(),
                "trap_work": (-self.kappa*x*integral_v).sum(-1).mean(),
                "deterministic_damping_loss": self.gamma*integral_v2.mean(),
                # Endpoint OU fluctuation energy includes noise/damping interaction;
                # it is NOT the continuous-time raw heat injection gamma*d*T*h.
                "ou_fluctuation_energy": .5*(result.square()-mean.square()).sum(-1).mean(),
                "ou_expected_fluctuation_energy": v.new_tensor(.5*v.shape[-1]*self.temperature*(-math.expm1(-2*self.gamma*duration))),
            }
            for key, value in values.items():
                budget[key] = budget.get(key, 0.0) + float(value.detach())
        return result

    def transport(self, state: PhaseState, token: int, duration: float,
                  generator: torch.Generator | None = None, budget: dict | None = None) -> PhaseState:
        mid = state.time + duration / 2
        velocity = self.kick(state.x, state.v, token, mid, duration / 2, generator=generator, budget=budget)
        position = state.x + duration * velocity  # dx/dt=v on the drift segment
        if budget is not None:
            change = .5*self.kappa*(position.square()-state.x.square()).sum(-1).mean()
            budget["drift_potential_change"] = budget.get("drift_potential_change", 0.) + float(change.detach())
        velocity = self.kick(position, velocity, token, mid, duration / 2, generator=generator, budget=budget)
        return PhaseState(position, velocity, state.time + duration)
