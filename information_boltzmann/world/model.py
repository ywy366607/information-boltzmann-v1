"""One differentiable Hamiltonian for motion, compliance and a room wave field.

This is an initialized reduced physical model, not a model fitted to real data.
Room modes are a Galerkin truncation. The scalar impedance interface is a
power-consistent approximation, not a resolved fluid/solid surface solver.
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Material:
    name: str
    color: str
    mass: float
    radius: float
    frequency_hz: float
    mode_mass: float
    mode_decay: float
    interface_stiffness: float
    contact_stiffness: float = 18000.0
    contact_damping: float = 3.0
    tangential_damping: float = 0.8
    shape_participation: float = 0.04

    def __post_init__(self):
        positive = (self.mass, self.radius, self.frequency_hz, self.mode_mass,
                    self.interface_stiffness, self.contact_stiffness)
        if min(positive) <= 0 or min(self.mode_decay, self.contact_damping,
                                    self.tangential_damping) < 0:
            raise ValueError("Material inertia/stiffness must be positive; loss nonnegative")
        if not 0 <= self.shape_participation <= 1:
            raise ValueError("shape_participation must be in [0, 1]")


DEFAULT_MATERIALS = (
    Material("柔顺材料 A", "#e9a35d", .12, .13, 110, .012, 12, 28),
    Material("弹性材料 B", "#67c7ba", .18, .12, 190, .016, 7, 36),
    Material("硬质材料 C", "#91a8ed", .15, .11, 280, .009, 4, 42),
)


@dataclass(frozen=True)
class WorldConfig:
    room: tuple[float, float, float] = (3.0, 2.4, 2.0)
    sample_rate: int = 8000
    cutoff_hz: float = 480.0
    sound_speed: float = 343.0
    air_density: float = 1.225
    air_decay: float = 1.2
    body_drag: float = .04
    gravity: float = 9.81

    def __post_init__(self):
        if min(self.room) <= 0 or self.sample_rate <= 0:
            raise ValueError("Positive room dimensions and sample rate required")
        if not 0 < self.cutoff_hz < self.sample_rate / 2:
            raise ValueError("Retained wave bandwidth must be below Nyquist")
        if min(self.sound_speed, self.air_density) <= 0:
            raise ValueError("Positive acoustic parameters required")
        if min(self.air_decay, self.body_drag, self.gravity) < 0:
            raise ValueError("Decay, drag and gravity must be nonnegative")


def room_modes(config: WorldConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized cosine modes; exclude the zero-pressure DC mode."""
    room = np.asarray(config.room)
    limits = np.floor(2 * room * config.cutoff_hz / config.sound_speed).astype(int)
    indices = np.asarray(list(product(*(range(n + 1) for n in limits))))
    wave_numbers = np.pi * indices / room
    omega = config.sound_speed * np.linalg.norm(wave_numbers, axis=-1)
    keep = (omega > 0) & (omega <= 2 * np.pi * config.cutoff_hz)
    indices, wave_numbers, omega = indices[keep], wave_numbers[keep], omega[keep]
    if len(omega) == 0:
        raise ValueError("cutoff_hz retains no nonconstant room modes")
    normalizer = np.prod(np.sqrt(np.where(indices > 0, 2., 1.) / room), axis=-1)
    order = np.argsort(omega)
    return wave_numbers[order].copy(), normalizer[order].copy(), omega[order].copy()


def basis_numpy(points: np.ndarray, wave_numbers: np.ndarray,
                normalizer: np.ndarray) -> np.ndarray:
    return np.prod(np.cos(np.asarray(points)[..., None, :] * wave_numbers), axis=-1) * normalizer


class GenerativeWorldLaw(nn.Module):
    """Positive material parameters and shared local potential.

    q: [objects, 3], eta: [objects], wave: [room_modes]. All gradients are from
    the SAME energy. The optimized CPU runtime must match these derivatives.
    J is canonical. Learnable log parameters are available for future system
    identification; no optimizer is run by the interactive world.
    """

    def __init__(self, config: WorldConfig = WorldConfig(),
                 materials: Sequence[Material] = DEFAULT_MATERIALS):
        super().__init__()
        self.config = config
        self.materials = tuple(materials)
        if not materials:
            raise ValueError("At least one material object required")
        for key in ("mass", "mode_mass", "frequency_hz", "interface_stiffness",
                    "contact_stiffness"):
            values = torch.tensor([getattr(m, key) for m in materials], dtype=torch.float64)
            setattr(self, "log_" + key, nn.Parameter(values.log()))
        for key in ("mode_decay", "contact_damping", "tangential_damping"):
            values = torch.tensor([getattr(m, key) for m in materials], dtype=torch.float64)
            setattr(self, "sqrt_" + key, nn.Parameter(values.sqrt()))
        self.sqrt_air_decay = nn.Parameter(torch.tensor(config.air_decay, dtype=torch.float64).sqrt())
        self.sqrt_body_drag = nn.Parameter(torch.tensor(config.body_drag, dtype=torch.float64).sqrt())
        wave_numbers, normalizer, omega = room_modes(config)
        self.register_buffer("wave_numbers", torch.from_numpy(wave_numbers))
        self.register_buffer("normalizer", torch.from_numpy(normalizer))
        self.register_buffer("omega_air", torch.from_numpy(omega))
        self.register_buffer("room", torch.tensor(config.room, dtype=torch.float64))
        self.register_buffer("radius", torch.tensor([m.radius for m in materials], dtype=torch.float64))
        self.register_buffer("shape", torch.tensor([m.shape_participation for m in materials], dtype=torch.float64))

    def basis(self, points: Tensor) -> Tensor:
        return torch.cos(points[..., None, :] * self.wave_numbers).prod(-1) * self.normalizer

    def potential(self, q: Tensor, eta: Tensor, wave: Tensor) -> Tensor:
        mass = self.log_mass.exp()
        modal_mass = self.log_mode_mass.exp()
        omega = 2 * torch.pi * self.log_frequency_hz.exp()
        energy = (mass * self.config.gravity * q[:, 2]).sum()
        energy = energy + .5 * (modal_mass * omega.square() * eta.square()).sum()
        energy = energy + .5 * self.config.air_density * (self.omega_air.square() * wave.square()).sum()
        displacement = self.basis(q) @ wave
        energy = energy + .5 * (self.log_interface_stiffness.exp() * (eta - displacement).square()).sum()
        radius = self.radius + self.shape * eta
        lower = torch.relu(radius[:, None] - q)
        upper = torch.relu(q + radius[:, None] - self.room)
        stiffness = self.log_contact_stiffness.exp()
        energy = energy + .5 * (stiffness[:, None] * (lower.square() + upper.square())).sum()
        for i in range(len(self.materials)):
            for j in range(i):
                penetration = torch.relu(radius[i] + radius[j] - torch.linalg.vector_norm(q[i] - q[j]))
                pair_stiffness = 2 * stiffness[i] * stiffness[j] / (stiffness[i] + stiffness[j])
                energy = energy + .5 * pair_stiffness * penetration.square()
        return energy

    def energy(self, q: Tensor, eta: Tensor, wave: Tensor,
               p: Tensor, modal_p: Tensor, wave_p: Tensor) -> Tensor:
        kinetic = .5 * (p.square() / self.log_mass.exp()[:, None]).sum()
        kinetic += .5 * (modal_p.square() / self.log_mode_mass.exp()).sum()
        kinetic += .5 * wave_p.square().sum() / self.config.air_density
        return kinetic + self.potential(q, eta, wave)

    def runtime_parameters(self) -> dict:
        """Export current weights without creating a second physical model."""
        result = {key: getattr(self, "log_" + key).detach().exp().cpu().numpy().copy()
                  for key in ("mass", "mode_mass", "frequency_hz",
                              "interface_stiffness", "contact_stiffness")}
        result.update({key: getattr(self, key).detach().cpu().numpy().copy()
                       for key in ("wave_numbers", "normalizer", "omega_air", "room", "radius", "shape")})
        result.update({key: getattr(self, "sqrt_" + key).detach().square().cpu().numpy().copy()
                       for key in ("mode_decay", "contact_damping", "tangential_damping", "air_decay", "body_drag")})
        return result

    def dissipation_power(self, q: Tensor, eta: Tensor, p: Tensor,
                          modal_p: Tensor, wave_p: Tensor) -> Tensor:
        """p^T M^-1 R M^-1 p >= 0, including shared contact velocities."""
        mass, modal_mass = self.log_mass.exp(), self.log_mode_mass.exp()
        velocity = p / mass[:, None]
        modal_velocity = modal_p / modal_mass
        power = self.sqrt_body_drag.square() * (p.square() / mass[:, None]).sum()
        power += (self.sqrt_mode_decay.square() * modal_p.square() / modal_mass).sum()
        power += self.sqrt_air_decay.square() * wave_p.square().sum() / self.config.air_density
        normal_gamma = self.sqrt_contact_damping.square()
        tangent_gamma = self.sqrt_tangential_damping.square()
        radius = self.radius + self.shape * eta
        for i in range(len(self.materials)):
            for axis in range(3):
                for side in range(2):
                    penetration = radius[i] - q[i, axis] if side == 0 else radius[i] + q[i, axis] - self.room[axis]
                    active = (penetration > 0).to(q.dtype)
                    sign = 1 if side == 0 else -1
                    v = sign * velocity[i, axis] - self.shape[i] * modal_velocity[i]
                    tangent = velocity[i].square().sum() - velocity[i, axis].square()
                    power += active * (normal_gamma[i] * v.square() + tangent_gamma[i] * tangent)
            for j in range(i):
                delta = q[i] - q[j]
                distance = torch.linalg.vector_norm(delta)
                normal = delta / distance.clamp_min(torch.finfo(q.dtype).eps)
                active = (radius[i] + radius[j] > distance).to(q.dtype)
                relative = velocity[i] - velocity[j]
                normal_v = (normal * relative).sum()
                v = normal_v - self.shape[i] * modal_velocity[i] - self.shape[j] * modal_velocity[j]
                tangent = (relative - normal_v * normal).square().sum()
                power += active * (.5 * (normal_gamma[i] + normal_gamma[j]) * v.square()
                                   + .5 * (tangent_gamma[i] + tangent_gamma[j]) * tangent)
        return power

    def manifest(self) -> dict:
        return {"config": asdict(self.config), "materials": [asdict(m) for m in self.materials],
                "modes": len(self.omega_air), "parameter_provenance": "physical_initialization_not_fitted",
                "interface": "reduced_scalar_displacement_impedance",
                "noise": "none", "agent": "none"}
