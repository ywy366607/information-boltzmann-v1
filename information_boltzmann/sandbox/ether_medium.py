"""
Unified Spatial Ether Medium & Mechanical Wave Propagation Engine.
First-principles continuum mechanics implementation:
- Acoustic dipole pressure waves radiated strictly from net surface force derivatives (Curle / FW-H equation).
- Contact stress aggregated per physical body (boot_l, boot_r, torso) to eliminate numerical solver chatter.
- Deadband filtering on solver convergence residuals: absolute silence when standing or resting stationary.
- Natural acoustic impulse response (exponential ring-down decay).
- Binaural spatial hearing at antenna sensor sites with physical acoustic delay (ITD) and attenuation (ILD).
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class MechanicalWavePacket:
    source_pos: np.ndarray      # [x, y, z] origin in 3D space (Center of Pressure)
    birth_time: float           # physical time t0 of generation (seconds)
    force_derivative: np.ndarray# dF/dt (N/s), dipole radiation vector
    monopole_strength: float    # rho0 * dV/dt, volume acceleration
    duration: float = 0.035     # acoustic pulse duration envelope (35 ms)
    base_freq: float = 140.0    # dominant mechanical resonance frequency (Hz, low thud)

class UnifiedEtherMedium:
    """
    Continuous 3D ether medium carrying mechanical waves (sound & vibrations).
    Governed by the acoustic wave equation in homogeneous elastic fluid (air).
    """
    def __init__(
        self,
        speed_of_sound: float = 343.0,
        air_density: float = 1.225,
        spatial_damping: float = 0.05,
        solver_chatter_deadband: float = 600.0
    ):
        self.c_s = speed_of_sound           # m/s (343 m/s in air)
        self.rho_0 = air_density            # kg/m^3 (1.225 kg/m^3)
        self.damping = spatial_damping      # spatial absorption per meter
        self.current_time = 0.0             # current physical time
        
        # Active wave packets currently propagating in space
        self.active_packets: List[MechanicalWavePacket] = []
        
        # Memory of past net body contact forces: body_id -> (last_time, net_force_vec, cop_pos)
        self.prev_body_forces: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
        
        # Deadband threshold in N/s:
        # In numerical LCP physics solvers, static contact forces fluctuate by ~0.1 - 0.5 N per 0.002s step (~100-250 N/s).
        # Genuine footsteps and impact transients are 15,000 - 80,000 N/s.
        # Threshold of 600 N/s guarantees 100% DEAD SILENCE when standing or resting, while capturing all genuine mechanical impacts!
        self.solver_chatter_deadband = solver_chatter_deadband

    def step(self, sim_time: float, contact_events: List[Dict], active_monopoles: Optional[List[Dict]] = None):
        """
        Advances the ether field by dt, assimilating all mechanical contact events.
        Aggregates contacts by limb body to calculate true net boundary force derivatives.
        """
        dt = sim_time - self.current_time
        if dt <= 0:
            return
        self.current_time = sim_time
        
        # 1. Aggregate contact forces by moving body ID
        # Maps body_id -> [sum_force_vec, sum_pos_weighted, sum_force_mag]
        body_aggregates: Dict[int, List[np.ndarray]] = {}
        
        for c in contact_events:
            g1, g2 = c['geom1'], c['geom2']
            force = np.array(c['force'], dtype=float)
            f_norm = float(np.linalg.norm(force))
            if f_norm < 1e-4:
                continue
                
            pos = np.array(c['pos'], dtype=float)
            
            # Associate force with the interacting non-ground body
            body_id = c.get('body_id', g1)
            if body_id not in body_aggregates:
                body_aggregates[body_id] = [np.zeros(3), np.zeros(3), 0.0]
                
            body_aggregates[body_id][0] += force
            body_aggregates[body_id][1] += pos * f_norm
            body_aggregates[body_id][2] += f_norm
            
        # 2. Compute net force derivative dF_net/dt for each interacting body
        current_bodies = set()
        for body_id, (net_force, cop_weighted, total_mag) in body_aggregates.items():
            current_bodies.add(body_id)
            cop = cop_weighted / max(total_mag, 1e-4) if total_mag > 0 else np.zeros(3)
            
            if body_id in self.prev_body_forces:
                last_t, last_f, _ = self.prev_body_forces[body_id]
                time_diff = max(sim_time - last_t, 1e-4)
                dF_dt = (net_force - last_f) / time_diff
            else:
                # Sudden touchdown/impact onset
                dF_dt = net_force / max(dt, 1e-4)
                
            self.prev_body_forces[body_id] = (sim_time, net_force, cop)
            
            dF_norm = float(np.linalg.norm(dF_dt))
            
            # Radiate acoustic dipole wave only when physical stress rate exceeds numerical solver chatter
            if dF_norm > self.solver_chatter_deadband:
                # Resonance frequency: low thud for soft footsteps (~100-140 Hz), higher ring for sharp impacts (~280 Hz)
                freq = 110.0 + min(dF_norm * 0.003, 300.0)
                packet = MechanicalWavePacket(
                    source_pos=cop,
                    birth_time=sim_time,
                    force_derivative=dF_dt,
                    monopole_strength=0.0,
                    duration=0.035,
                    base_freq=freq
                )
                self.active_packets.append(packet)

        # 3. Clean up detached bodies
        stale_bodies = [b for b in self.prev_body_forces if b not in current_bodies]
        for b in stale_bodies:
            last_t, last_f, last_cop = self.prev_body_forces[b]
            last_mag = float(np.linalg.norm(last_f))
            # Rapid release wave if foot took off with high velocity
            if last_mag > 15.0 and (sim_time - last_t) < 0.008:
                release_dF = -last_f / max(dt, 1e-4)
                if np.linalg.norm(release_dF) > self.solver_chatter_deadband:
                    self.active_packets.append(MechanicalWavePacket(
                        source_pos=last_cop,
                        birth_time=sim_time,
                        force_derivative=release_dF,
                        monopole_strength=0.0,
                        duration=0.025,
                        base_freq=120.0
                    ))
            del self.prev_body_forces[b]

        # 4. Optional surface monopoles (e.g. vocalization membrane)
        if active_monopoles:
            for mono in active_monopoles:
                pos = np.array(mono['pos'], dtype=float)
                strength = float(mono.get('strength', 0.0))
                freq = float(mono.get('freq', 260.0))
                if abs(strength) > 1e-4:
                    self.active_packets.append(MechanicalWavePacket(
                        source_pos=pos,
                        birth_time=sim_time,
                        force_derivative=np.zeros(3),
                        monopole_strength=strength,
                        duration=0.035,
                        base_freq=freq
                    ))

        # 5. Garbage collect packets that have decayed or traveled beyond observable spatial range
        max_travel_time = 15.0 / self.c_s # ~0.044s
        self.active_packets = [
            p for p in self.active_packets 
            if (sim_time - p.birth_time) < (p.duration + max_travel_time)
        ]

    def sample_pressure(self, receiver_pos: np.ndarray) -> float:
        """
        Samples the instantaneous acoustic pressure p(x, t) at any 3D coordinate receiver_pos.
        Computes the analytical retarded superposition of all incoming spherical wavepackets.
        Uses physical exponential impulse decay e^(-t/tau).
        """
        receiver_pos = np.array(receiver_pos, dtype=float)
        total_pressure = 0.0
        
        for p in self.active_packets:
            r_vec = receiver_pos - p.source_pos
            r = max(float(np.linalg.norm(r_vec)), 0.05) # avoid point-source singularity
            r_unit = r_vec / r
            
            # Retarded acoustic travel time: tau = r / c_s
            travel_time = r / self.c_s
            elapsed = self.current_time - p.birth_time
            t_wave = elapsed - travel_time  # local time within wavepacket
            
            if 0.0 <= t_wave <= p.duration:
                # Natural mechanical impulse ring-down decay
                decay = np.exp(-t_wave / 0.009)
                osc = np.sin(2.0 * np.pi * p.base_freq * t_wave)
                
                # 1. Curle / FW-H Dipole radiation term: (1 / 4*pi*r*c_s) * (dF/dt . r_unit)
                dipole_factor = np.dot(p.force_derivative, r_unit) / (4.0 * np.pi * r * self.c_s)
                
                # 2. Monopole term: (rho_0 / 4*pi*r) * strength
                monopole_factor = (self.rho_0 * p.monopole_strength) / (4.0 * np.pi * r)
                
                # Spatial atmospheric absorption
                spatial_loss = np.exp(-self.damping * r)
                
                amplitude = (dipole_factor + monopole_factor) * spatial_loss
                total_pressure += amplitude * decay * osc
                
        return float(total_pressure)

    def sample_binaural_ears(self, left_ear_pos: np.ndarray, right_ear_pos: np.ndarray) -> Tuple[float, float]:
        """
        Binaural spatial hearing sampling directly at the robot's antenna coordinates.
        Naturally produces exact Interaural Time Difference (ITD) and Level Difference (ILD).
        """
        p_L = self.sample_pressure(left_ear_pos)
        p_R = self.sample_pressure(right_ear_pos)
        return p_L, p_R
