"""Minimal distribution -> phase dynamics -> distribution readout model."""
import math
import torch
from torch import Tensor, nn

from .collision import CollisionKernel
from .density import InitialDensity
from .force import DataForce
from .state import PhaseState


class InformationBoltzmann(nn.Module):
    def __init__(self, vocab_size: int = 260, phase_dim: int = 2, particles: int = 16,
                 hidden_dim: int = 48, flow_layers: int = 4, steps: int = 4,
                 event_interval: float = 1.0, kappa: float = 1.0, gamma: float = 1.0,
                 amplitude: float = 1.0, collision_rate: float = 1.0,
                 kernel_width: float = 1.0, temperature: float = 0.0,
                 adaptive_gamma: bool = False, gamma_lr: float = 0.02,
                 gamma_min: float = 0.01, gamma_max: float = 2.0,
                 scale_kappa_with_gamma: bool = True):
        super().__init__()
        if phase_dim < 2 or particles < 2 or steps < 1:
            raise ValueError("Require phase_dim>=2, particles>=2 and steps>=1")
        if min(kappa, kernel_width, event_interval) <= 0 or gamma < 0 or collision_rate < 0 or amplitude < 0 or temperature < 0:
            raise ValueError("Invalid force, time, collision or temperature parameters")
        self.particles, self.steps, self.event_interval = particles, steps, event_interval
        self.temperature = temperature
        self.adaptive_gamma = adaptive_gamma
        self.gamma_lr = gamma_lr
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.scale_kappa_with_gamma = scale_kappa_with_gamma
        self.vocab_size = vocab_size
        self.initial = InitialDensity(vocab_size, phase_dim, hidden_dim, flow_layers)
        self.force = DataForce(vocab_size, phase_dim, hidden_dim, kappa, gamma, amplitude, temperature=temperature)
        self.collision = CollisionKernel(phase_dim, hidden_dim, collision_rate, kernel_width)
        self.features = nn.Sequential(nn.Linear(2 * phase_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def initialize(self, boot_tokens: Tensor, generator: torch.Generator | None = None) -> PhaseState:
        return self.initial.condition(boot_tokens).sample(self.particles, generator)

    def decode(self, state: PhaseState) -> Tensor:
        return self.decoder(self.features(torch.cat((state.x, state.v), -1)).mean(0))

    def _advance_steps(self, state: PhaseState, observed_token: int,
                       generator: torch.Generator | None = None, budget: dict | None = None) -> tuple[PhaseState, Tensor, dict]:
        dt = self.event_interval / self.steps
        log_prob = state.x.new_zeros(())
        stats = {"candidates": 0, "accepted": 0, "cross_moment_change": 0.0, "pairs": []}
        for _ in range(self.steps):
            state = self.force.transport(state, observed_token, dt / 2, generator=generator, budget=budget)
            if budget is not None:
                before_energy = .5*state.v.square().sum(-1).mean()
            state, lp, report = self.collision(state, dt, generator)
            if budget is not None:
                delta = .5*state.v.square().sum(-1).mean() - before_energy
                budget["collision_energy_error"] = budget.get("collision_energy_error", 0.) + float(delta.detach())
            log_prob = log_prob + lp
            for key in ("candidates", "accepted", "cross_moment_change"):
                stats[key] += report[key]
            stats["pairs"].extend(report.get("pairs", []))
            state = self.force.transport(state, observed_token, dt / 2, generator=generator, budget=budget)
        return state, log_prob, stats

    def advance(self, state: PhaseState, observed_token: int,
                generator: torch.Generator | None = None) -> tuple[PhaseState, Tensor, dict]:
        if not 0 <= observed_token < self.vocab_size:
            raise ValueError("Token outside vocabulary")
        if self.adaptive_gamma:
            raise RuntimeError(
                "Unvalidated criticality controller disabled: calibrate full-phase "
                "conditional response first; use adaptive_gamma=False")
        return self._advance_steps(state, observed_token, generator)

    @classmethod
    def from_config(cls, config: dict) -> "InformationBoltzmann":
        m, d = config["model"], config["dynamics"]
        return cls(vocab_size=config["data"]["vocab_size"],
                   phase_dim=m["phase_dim"], particles=m["particles"],
                   hidden_dim=m["hidden_dim"], flow_layers=m["flow_layers"],
                   steps=d["steps"], event_interval=d["event_interval"],
                   kappa=d["kappa"], gamma=d["gamma"],
                   amplitude=d["acceleration_norm_bound"],
                   collision_rate=d["max_collision_rate"], kernel_width=d["kernel_width"],
                   temperature=d.get("temperature", 0.0),
                   adaptive_gamma=d.get("adaptive_gamma", False),
                   gamma_lr=d.get("gamma_lr", 0.02),
                   gamma_min=d.get("gamma_min", 0.01),
                   gamma_max=d.get("gamma_max", 2.0),
                   scale_kappa_with_gamma=d.get("scale_kappa_with_gamma", True))
