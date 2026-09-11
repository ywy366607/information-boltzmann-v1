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
        self.gamma_mode = "fixed"
        self.gamma_field = None
        self.embedding = nn.Embedding(vocab, hidden)
        self.net = nn.Sequential(nn.Linear(hidden + dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def configure_gamma(self, mode="fixed", minimum=.03, maximum=2.):
        if mode not in ("fixed", "global", "local"):
            raise ValueError("Invalid gamma mode")
        if not 0 < minimum < self.gamma < maximum:
            if mode != "fixed":
                raise ValueError("Learned gamma initialization must be inside positive bounds")
        self.gamma_mode, self.gamma_bounds = mode, (minimum, maximum)
        if mode != "fixed":
            dim = self.net[-1].out_features
            self.gamma_field = nn.Linear(dim if mode == "local" else 1, 1)
            nn.init.zeros_(self.gamma_field.weight)
            fraction = (self.gamma-minimum)/(maximum-minimum)
            nn.init.constant_(self.gamma_field.bias, math.log(fraction/(1-fraction)))

    def damping(self, x):
        if self.gamma_field is None:
            return x.new_full((len(x),1),self.gamma)
        features = x if self.gamma_mode == "local" else x.new_zeros((len(x),1))
        low,high=self.gamma_bounds
        return low+(high-low)*self.gamma_field(features).sigmoid()


    def drive(self, x: Tensor, token: int, time: float) -> Tensor:
        h = self.embedding.weight[token].expand(len(x), -1)
        clock = x.new_tensor([math.sin(time), math.cos(time)]).expand(len(x), -1)
        return self.amplitude / math.sqrt(x.shape[-1]) * self.net(torch.cat((x, h, clock), -1)).tanh()

    def forward(self, x: Tensor, v: Tensor, token: int, time: float) -> Tensor:
        return -self.kappa * x - self.damping(x) * v + self.drive(x, token, time)

    def kick(self, x: Tensor, v: Tensor, token: int, time: float, duration: float,
             generator: torch.Generator | None = None, budget: dict | None = None) -> Tensor:
        # Exact Ornstein-Uhlenbeck velocity update for frozen x, drive and time.
        # When temperature > 0, adds Langevin thermal fluctuations with variance T * (1 - e^{-2 gamma h}).
        gamma = self.damping(x)
        decay = torch.exp(-gamma * duration)
        integral = torch.where(gamma > 0, -torch.expm1(-gamma*duration)/gamma.clamp_min(1e-30), gamma.new_full(gamma.shape,duration))
        drive = self.drive(x, token, time)
        c = -self.kappa * x + drive
        mean = decay * v + integral * c
        result = mean
        if self.temperature > 0.0 and duration > 0.0:
            noise_var = self.temperature * (-torch.expm1(-2.0 * gamma * duration))
            noise_std = noise_var.clamp_min(0).sqrt()
            if generator is not None and generator.device.type != v.device.type:
                noise = torch.randn(v.shape, device=generator.device, dtype=v.dtype, generator=generator).to(v.device)
            else:
                noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
            result = mean + noise_std * noise
        if budget is not None:
            # Exact work along the deterministic frozen-force velocity path.
            if self.gamma_field is not None or self.gamma > 0:
                terminal = c / gamma
                transient = v - terminal
                integral_v = terminal * duration + transient * integral
                integral_v2 = (terminal.square().sum(-1) * duration
                               + 2 * (terminal * transient).sum(-1) * integral.squeeze(-1)
                               + transient.square().sum(-1)
                               * (-torch.expm1(-2*gamma.squeeze(-1)*duration))/(2*gamma.squeeze(-1)))
            else:
                integral_v = v*duration + c*duration**2/2
                integral_v2 = v.square().sum(-1)*duration
            values = {
                "drive_work": (drive*integral_v).sum(-1).mean(),
                "trap_work": (-self.kappa*x*integral_v).sum(-1).mean(),
                "deterministic_damping_loss": (gamma.squeeze(-1)*integral_v2).mean(),
                # Endpoint OU fluctuation energy includes noise/damping interaction;
                # it is NOT the continuous-time raw heat injection gamma*d*T*h.
                "ou_fluctuation_energy": .5*(result.square()-mean.square()).sum(-1).mean(),
                "ou_expected_fluctuation_energy": .5*v.shape[-1]*self.temperature*(-torch.expm1(-2*gamma*duration)).mean(),
            }
            for key, value in values.items():
                budget[key] = budget.get(key, 0.0) + (value.detach() if getattr(budget, "defer_cpu", False) else float(value.detach()))
        return result

    def transport(self, state: PhaseState, token: int, duration: float,
                  generator: torch.Generator | None = None, budget: dict | None = None) -> PhaseState:
        mid = state.time + duration / 2
        velocity = self.kick(state.x, state.v, token, mid, duration / 2, generator=generator, budget=budget)
        position = state.x + duration * velocity  # dx/dt=v on the drift segment
        if budget is not None:
            change = .5*self.kappa*(position.square()-state.x.square()).sum(-1).mean()
            budget["drift_potential_change"] = budget.get("drift_potential_change", 0.) + (change.detach() if getattr(budget, "defer_cpu", False) else float(change.detach()))
        velocity = self.kick(position, velocity, token, mid, duration / 2, generator=generator, budget=budget)
        return PhaseState(position, velocity, state.time + duration)
