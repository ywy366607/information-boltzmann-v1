"""Fast executor of the differentiable world law and its energy ledger.

Conservative integration: kick / exact uncoupled Hamiltonian flow / kick.
Losses are exact momentum contractions with complementary heat accounting.
No contact callback creates sound. Room modes are persistent state.
"""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch

from .model import GenerativeWorldLaw, basis_numpy

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def interaction(q, eta, wave, mass, interface, stiffness, room, radius,
                shape, wave_numbers, normalizer, gravity):
    """Analytic -grad of the same potential as GenerativeWorldLaw."""
    bodies, modes = len(eta), len(wave)
    force = np.zeros_like(q)
    modal_force = np.zeros_like(eta)
    air_force = np.zeros_like(wave)
    contact_force = np.zeros_like(q)
    energies = np.zeros(3)  # gravity, contact, interface
    effective = radius + shape * eta
    for i in range(bodies):
        force[i, 2] -= mass[i] * gravity
        energies[0] += mass[i] * gravity * q[i, 2]
        field = 0.0
        gradient = np.zeros(3)
        weights = np.empty(modes)
        for k in range(modes):
            cx = np.cos(q[i, 0] * wave_numbers[k, 0])
            cy = np.cos(q[i, 1] * wave_numbers[k, 1])
            cz = np.cos(q[i, 2] * wave_numbers[k, 2])
            weights[k] = normalizer[k] * cx * cy * cz
            field += weights[k] * wave[k]
            gradient[0] -= normalizer[k] * wave_numbers[k, 0] * np.sin(q[i, 0] * wave_numbers[k, 0]) * cy * cz * wave[k]
            gradient[1] -= normalizer[k] * wave_numbers[k, 1] * np.sin(q[i, 1] * wave_numbers[k, 1]) * cx * cz * wave[k]
            gradient[2] -= normalizer[k] * wave_numbers[k, 2] * np.sin(q[i, 2] * wave_numbers[k, 2]) * cx * cy * wave[k]
        difference = eta[i] - field
        transfer = interface[i] * difference
        energies[2] += .5 * interface[i] * difference * difference
        modal_force[i] -= transfer
        force[i] += transfer * gradient
        air_force += transfer * weights
        for axis in range(3):
            low = max(0.0, effective[i] - q[i, axis])
            high = max(0.0, effective[i] + q[i, axis] - room[axis])
            value = stiffness[i] * (low - high)
            force[i, axis] += value
            contact_force[i, axis] += value
            modal_force[i] -= stiffness[i] * shape[i] * (low + high)
            energies[1] += .5 * stiffness[i] * (low * low + high * high)
        for j in range(i):
            delta = q[i] - q[j]
            distance = np.sqrt(np.dot(delta, delta))
            if distance < 1e-12:
                raise ValueError("Coincident body centers are a singular initial state")
            penetration = max(0., effective[i] + effective[j] - distance)
            pair_stiffness = 2 * stiffness[i] * stiffness[j] / (stiffness[i] + stiffness[j])
            contact = pair_stiffness * penetration
            normal = delta / distance
            force[i] += contact * normal
            force[j] -= contact * normal
            contact_force[i] += contact * normal
            contact_force[j] -= contact * normal
            modal_force[i] -= contact * shape[i]
            modal_force[j] -= contact * shape[j]
            energies[1] += .5 * pair_stiffness * penetration * penetration
    return force, modal_force, air_force, contact_force, energies


@njit(cache=True)
def dissipate(q, eta, p, modal_p, wave_p, mass, modal_mass, radius, shape,
              room, damping, tangent_damping, mode_decay, air_density,
              body_drag, air_decay, dt):
    heat = np.zeros(5)  # body, material, air, contact normal, friction
    before = .5 * np.sum(p * p / mass[:, None])
    p *= np.exp(-body_drag * dt)
    heat[0] = before - .5 * np.sum(p * p / mass[:, None])
    before = .5 * np.sum(modal_p * modal_p / modal_mass)
    modal_p *= np.exp(-mode_decay * dt)
    heat[1] = before - .5 * np.sum(modal_p * modal_p / modal_mass)
    before = .5 * np.sum(wave_p * wave_p) / air_density
    wave_p *= np.exp(-air_decay * dt)
    heat[2] = before - .5 * np.sum(wave_p * wave_p) / air_density
    effective = radius + shape * eta
    for i in range(len(mass)):
        for axis in range(3):
            for side in range(2):
                penetration = effective[i] - q[i, axis] if side == 0 else effective[i] + q[i, axis] - room[axis]
                if penetration <= 0:
                    continue
                sign = 1.0 if side == 0 else -1.0
                inverse_mass = 1.0 / mass[i] + shape[i] ** 2 / modal_mass[i]
                velocity = sign * p[i, axis] / mass[i] - shape[i] * modal_p[i] / modal_mass[i]
                retention = np.exp(-damping[i] * inverse_mass * dt)
                impulse = (1.0 - retention) * velocity / inverse_mass
                p[i, axis] -= sign * impulse
                modal_p[i] += shape[i] * impulse
                heat[3] += .5 * velocity ** 2 / inverse_mass * (1.0 - retention ** 2)
                for tangent in range(3):
                    if tangent != axis:
                        before = .5 * p[i, tangent] ** 2 / mass[i]
                        p[i, tangent] *= np.exp(-tangent_damping[i] * dt / mass[i])
                        heat[4] += before - .5 * p[i, tangent] ** 2 / mass[i]
        for j in range(i):
            delta = q[i] - q[j]
            distance = np.sqrt(np.dot(delta, delta))
            if distance >= effective[i] + effective[j] or distance < 1e-12:
                continue
            normal = delta / distance
            relative = p[i] / mass[i] - p[j] / mass[j]
            velocity = np.dot(normal, relative) - shape[i] * modal_p[i] / modal_mass[i] - shape[j] * modal_p[j] / modal_mass[j]
            inverse_mass = 1.0 / mass[i] + 1.0 / mass[j] + shape[i] ** 2 / modal_mass[i] + shape[j] ** 2 / modal_mass[j]
            gamma = .5 * (damping[i] + damping[j])
            retention = np.exp(-gamma * inverse_mass * dt)
            impulse = (1.0 - retention) * velocity / inverse_mass
            p[i] -= normal * impulse
            p[j] += normal * impulse
            modal_p[i] += shape[i] * impulse
            modal_p[j] += shape[j] * impulse
            heat[3] += .5 * velocity ** 2 / inverse_mass * (1.0 - retention ** 2)
            relative = p[i] / mass[i] - p[j] / mass[j]
            tangent = relative - np.dot(normal, relative) * normal
            inverse_mass = 1.0 / mass[i] + 1.0 / mass[j]
            gamma = .5 * (tangent_damping[i] + tangent_damping[j])
            retention = np.exp(-gamma * inverse_mass * dt)
            impulse_t = (1.0 - retention) * tangent / inverse_mass
            p[i] -= impulse_t
            p[j] += impulse_t
            heat[4] += .5 * np.dot(tangent, tangent) / inverse_mass * (1.0 - retention ** 2)
    return heat


@njit(cache=True)
def advance(q, eta, wave, p, modal_p, wave_p, heat, samples, dt,
            mass, modal_mass, omega_object, interface, stiffness, room,
            radius, shape, wave_numbers, normalizer, omega_air, mode_decay,
            damping, tangent_damping, air_density, gravity, body_drag,
            air_decay, pressure_readout):
    audio = np.empty((samples, 2))
    cosine_object = np.cos(omega_object * dt)
    sine_object = np.sin(omega_object * dt)
    cosine_air = np.cos(omega_air * dt)
    sine_air = np.sin(omega_air * dt)
    force, modal_force, air_force, contact, _ = interaction(
        q, eta, wave, mass, interface, stiffness, room, radius, shape,
        wave_numbers, normalizer, gravity)
    for n in range(samples):
        heat += dissipate(q, eta, p, modal_p, wave_p, mass, modal_mass,
                          radius, shape, room, damping, tangent_damping,
                          mode_decay, air_density, body_drag, air_decay, .5 * dt)
        p += .5 * dt * force
        modal_p += .5 * dt * modal_force
        wave_p += .5 * dt * air_force
        q += dt * p / mass[:, None]
        previous = eta.copy()
        eta[:] = cosine_object * eta + sine_object * modal_p / (modal_mass * omega_object)
        modal_p[:] = cosine_object * modal_p - sine_object * modal_mass * omega_object * previous
        previous_air = wave.copy()
        wave[:] = cosine_air * wave + sine_air * wave_p / (air_density * omega_air)
        wave_p[:] = cosine_air * wave_p - sine_air * air_density * omega_air * previous_air
        force, modal_force, air_force, contact, _ = interaction(
            q, eta, wave, mass, interface, stiffness, room, radius, shape,
            wave_numbers, normalizer, gravity)
        p += .5 * dt * force
        modal_p += .5 * dt * modal_force
        wave_p += .5 * dt * air_force
        heat += dissipate(q, eta, p, modal_p, wave_p, mass, modal_mass,
                          radius, shape, room, damping, tangent_damping,
                          mode_decay, air_density, body_drag, air_decay, .5 * dt)
        audio[n] = pressure_readout @ wave
    return audio, contact


@dataclass
class WorldState:
    q: np.ndarray
    eta: np.ndarray
    wave: np.ndarray
    p: np.ndarray
    modal_p: np.ndarray
    wave_p: np.ndarray


class PersistentWorld:
    """One evolving scene. Controls change momentum and account external work."""

    def __init__(self, law: Optional[GenerativeWorldLaw] = None,
                 positions: Optional[np.ndarray] = None):
        self.law = law if law is not None else GenerativeWorldLaw()
        self.config = self.law.config
        self.parameters = self.law.runtime_parameters()
        count = len(self.law.materials)
        if positions is None:
            room = np.asarray(self.config.room)
            positions = np.column_stack((np.linspace(.65 / 3, .75, count) * room[0],
                                         np.full(count, .4375 * room[1]),
                                         np.linspace(.325, .65, count) * room[2]))
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (count, 3) or not np.isfinite(positions).all():
            raise ValueError("positions must be finite [objects, 3]")
        if np.any(positions <= self.parameters["radius"][:, None]) or np.any(
                positions + self.parameters["radius"][:, None] >= self.parameters["room"]):
            raise ValueError("Initial bodies must fit inside the room")
        modes = len(self.law.omega_air)
        self.state = WorldState(positions.copy(), np.zeros(count), np.zeros(modes),
                                np.zeros((count, 3)), np.zeros(count), np.zeros(modes))
        self.time = 0.0
        self.world_id = str(uuid4())
        self.steps = 0
        self.heat = np.zeros(5)
        self.external_work = 0.0
        self.contact_force = np.zeros((count, 3))
        self.listener = np.array([1.5, .45, .9])
        self.listener_yaw = 0.0
        self.ear_separation = .18
        self.history = deque(maxlen=2000)
        self.initial_energy = self.energy()["total"]
        self.last_audio_peak = 0.0

    def ears(self) -> np.ndarray:
        lateral = np.array([np.cos(self.listener_yaw), np.sin(self.listener_yaw), 0.])
        return self.listener[None, :] + np.array([-.5, .5])[:, None] * self.ear_separation * lateral

    def set_listener(self, position, yaw: float = 0.0) -> None:
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(yaw):
            raise ValueError("Finite listener position and yaw required")
        if np.any(position < .15) or np.any(position > self.parameters["room"] - .15):
            raise ValueError("Listener must remain inside room")
        self.listener = position.copy()
        self.listener_yaw = float(yaw)

    def impulse(self, object_index: int, impulse) -> float:
        if not 0 <= object_index < len(self.law.materials):
            raise ValueError("Unknown object")
        impulse = np.asarray(impulse, dtype=float)
        if impulse.shape != (3,) or not np.isfinite(impulse).all():
            raise ValueError("Finite 3D impulse required")
        if np.linalg.norm(impulse) > 2.0:
            raise ValueError("Interactive actuator budget is 2 N s per operation")
        mass = self.parameters["mass"][object_index]
        before = .5 * np.dot(self.state.p[object_index], self.state.p[object_index]) / mass
        self.state.p[object_index] += impulse
        after = .5 * np.dot(self.state.p[object_index], self.state.p[object_index]) / mass
        work = after - before
        self.external_work += work
        return float(work)

    def interaction(self):
        s, p = self.state, self.parameters
        return interaction(s.q, s.eta, s.wave, p["mass"], p["interface_stiffness"],
                           p["contact_stiffness"], p["room"], p["radius"],
                           p["shape"], p["wave_numbers"], p["normalizer"], self.config.gravity)

    def energy(self) -> dict[str, float]:
        s, p = self.state, self.parameters
        parts = self.interaction()[-1]
        omega = 2 * np.pi * p["frequency_hz"]
        kinetic = .5 * np.sum(s.p ** 2 / p["mass"][:, None])
        material = .5 * np.sum(s.modal_p ** 2 / p["mode_mass"] + p["mode_mass"] * omega ** 2 * s.eta ** 2)
        air = .5 * np.sum(s.wave_p ** 2 / self.config.air_density + self.config.air_density * p["omega_air"] ** 2 * s.wave ** 2)
        return dict(body_kinetic=float(kinetic), gravity=float(parts[0]),
                    contact=float(parts[1]), interface=float(parts[2]),
                    vibration=float(material), air=float(air),
                    total=float(kinetic + material + air + parts.sum()))

    def step(self, samples: int = 256) -> np.ndarray:
        if samples <= 0 or samples > self.config.sample_rate * 2:
            raise ValueError("A block must have 1..2 seconds of samples")
        s, p, c = self.state, self.parameters, self.config
        basis = basis_numpy(self.ears(), p["wave_numbers"], p["normalizer"])
        pressure_readout = basis * (c.air_density * c.sound_speed * p["omega_air"])[None, :]
        audio, contact = advance(
            s.q, s.eta, s.wave, s.p, s.modal_p, s.wave_p, self.heat,
            samples, 1.0 / c.sample_rate, p["mass"], p["mode_mass"],
            2 * np.pi * p["frequency_hz"], p["interface_stiffness"],
            p["contact_stiffness"], p["room"], p["radius"], p["shape"],
            p["wave_numbers"], p["normalizer"], p["omega_air"],
            p["mode_decay"], p["contact_damping"], p["tangential_damping"],
            c.air_density, c.gravity, float(p["body_drag"]), float(p["air_decay"]), pressure_readout)
        self.steps += samples
        self.time = self.steps / c.sample_rate
        self.contact_force = contact
        if not all(np.isfinite(v).all() for v in (s.q, s.eta, s.wave, s.p, s.modal_p, s.wave_p, audio)):
            raise FloatingPointError("Non-finite world state; simulation stopped")
        if np.any(p["radius"] + p["shape"] * s.eta <= 0):
            raise FloatingPointError("Reduced body model exceeded its deformation domain")
        self.last_audio_peak = float(np.abs(audio).max())
        snapshot = self.snapshot()
        self.history.append({key: snapshot[key] for key in
                             ("time", "energy", "heat", "external_work", "balance_error", "audio_peak_pa", "contact_peak_n")})
        return audio

    def pressure(self, points: np.ndarray) -> np.ndarray:
        p, c = self.parameters, self.config
        return basis_numpy(points, p["wave_numbers"], p["normalizer"]) @ (
            c.air_density * c.sound_speed * p["omega_air"] * self.state.wave)

    def snapshot(self) -> dict:
        energy = self.energy()
        residual = energy["total"] + float(self.heat.sum()) - self.initial_energy - self.external_work
        return {"world_id": self.world_id, "time": self.time, "steps": self.steps, "room": list(self.config.room),
                "energy": energy, "heat": dict(zip(("drag", "material", "air", "contact", "friction"), self.heat.tolist())),
                "external_work": self.external_work, "initial_energy": self.initial_energy,
                "balance_error": float(residual), "audio_peak_pa": self.last_audio_peak,
                "contact_peak_n": float(np.linalg.norm(self.contact_force, axis=-1).max()),
                "listener": self.listener.tolist(), "listener_yaw": self.listener_yaw,
                "ears": self.ears().tolist(), "modes": len(self.state.wave),
                "sample_rate": self.config.sample_rate, "cutoff_hz": self.config.cutoff_hz,
                "objects": [{"id": i, "name": material.name, "color": material.color,
                             "position": self.state.q[i].tolist(),
                             "radius": float(material.radius + material.shape_participation * self.state.eta[i]),
                             "deformation_m": float(material.shape_participation * self.state.eta[i]),
                             "mode_displacement_m": float(self.state.eta[i]),
                             "velocity": (self.state.p[i] / self.parameters["mass"][i]).tolist(),
                             "touch_force_n": self.contact_force[i].tolist()}
                            for i, material in enumerate(self.law.materials)]}

    def save(self, path: Path) -> None:
        """Atomic physical-state checkpoint; includes heat, work and exact time."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self.state, name) for name in WorldState.__dataclass_fields__}
        arrays.update({"law_" + name: tensor.detach().cpu().numpy()
                       for name, tensor in self.law.state_dict().items()})
        metadata = dict(manifest=self.law.manifest(), world_id=self.world_id,
                        steps=self.steps, external_work=self.external_work,
                        initial_energy=self.initial_energy, listener=self.listener.tolist(),
                        listener_yaw=self.listener_yaw, history=list(self.history))
        arrays.update(heat=self.heat, metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PersistentWorld":
        from .model import Material, WorldConfig
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            manifest = metadata["manifest"]
            config = WorldConfig(**manifest["config"])
            law = GenerativeWorldLaw(config, [Material(**spec) for spec in manifest["materials"]])
            law.load_state_dict({name: torch.from_numpy(saved["law_" + name].copy())
                                 for name in law.state_dict()}, strict=True)
            world = cls(law)
            for name in WorldState.__dataclass_fields__:
                value = saved[name].copy()
                if value.shape != getattr(world.state, name).shape or not np.isfinite(value).all():
                    raise ValueError("Invalid checkpoint field: " + name)
                setattr(world.state, name, value)
            world.world_id = metadata["world_id"]
            world.steps = int(metadata["steps"])
            world.time = world.steps / config.sample_rate
            world.heat = saved["heat"].copy()
            world.external_work = float(metadata["external_work"])
            world.initial_energy = float(metadata["initial_energy"])
            world.set_listener(metadata["listener"], metadata["listener_yaw"])
            world.history.extend(metadata["history"])
            world.contact_force = world.interaction()[3]
            return world
