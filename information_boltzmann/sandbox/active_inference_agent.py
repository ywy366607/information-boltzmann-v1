"""
First-Principles Continuous Active Inference & Variational Free Energy Engine for Bipedal Sandbox.

Theoretical Foundations:
1. Karl Friston (2010, 2013, 2017) Continuous Active Inference in Motor Control:
   - "Motor commands are an illusion. Predictions, not commands."
   - The central nervous system NEVER generates or stores "actions" or "motion libraries".
   - The brain outputs ONLY descending proprioceptive predictions mu_q (expected sensory state).
   - The spinal reflex arc (Ia stretch reflex) contracts muscle fibers purely to eliminate
     the discrepancy between expected proprioception mu_q and actual peripheral feedback q:
         tau = -Pi_p * (q - mu_q) - B_bath * q_dot
     where B_bath * q_dot is the thermodynamic dissipation into the Unified Cold Bath.

2. Information Boltzmann & Non-Equilibrium Steady State (NESS):
   - Epistemic Curiosity (exploration of unknown ground support & sound waves) drives
     continuous active sampling on the Variational Free Energy landscape:
         G = Risk - Salience
   - When Risk is low (upright stability), Epistemic Salience dominates, driving active, visible
     stepping exploration, contralateral arm swing, and tactile ground palpation.
   - When Risk surges (external caretaker push or heavy tilt), Pragmatic Homeostasis immediately
     dominates, suppressing swing and bracing both feet on the ground to dissipate impulse work.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional


class ContinuousEpistemicField:
    """
    Tracks continuous uncertainty and prospective information gain across:
    1. Tactile Ground Support & Compliance (Left Foot, Right Foot)
    2. Acoustic Ether Wave Environment (Antenna Ear Receptors)
    3. Spatial Horizon / Infotaxis (Ground Field Ahead)
    4. Postural Body Schema & Proprioceptive Limits
    """
    def __init__(self, bounds: float = 8.0, resolution: int = 80):
        self.bounds = bounds
        self.resolution = resolution
        self.spatial_grid = np.ones((resolution, resolution), dtype=np.float32)
        self.xs = np.linspace(-bounds, bounds, resolution)
        self.ys = np.linspace(-bounds, bounds, resolution)

        self.u_tactile_l = 0.8
        self.u_tactile_r = 0.8
        self.u_acoustic = 0.8
        self.u_vestibular = 0.1
        self.u_spatial = 0.7

    def assimilate_observations(
        self,
        dt: float,
        x: float,
        y: float,
        touch_l: float,
        touch_r: float,
        audio_amp: float,
        tilt_rad: float
    ):
        """
        Assimilates ongoing sensory evidence:
        - Tactile contact discharges uncertainty when the foot senses reaction force.
        - Prolonged stance or unweighted flight accumulates uncertainty.
        - Acoustic events discharge acoustic curiosity; silence accumulates curiosity.
        """
        # 1. Spatial Infotaxis decay around current location
        ix = int(np.clip((x + self.bounds) / (2.0 * self.bounds) * self.resolution, 0, self.resolution - 1))
        iy = int(np.clip((y + self.bounds) / (2.0 * self.bounds) * self.resolution, 0, self.resolution - 1))
        radius = 3
        i_min, i_max = max(0, ix - radius), min(self.resolution, ix + radius + 1)
        j_min, j_max = max(0, iy - radius), min(self.resolution, iy + radius + 1)
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                d2 = (self.xs[i] - x)**2 + (self.ys[j] - y)**2
                if d2 < 0.25:
                    self.spatial_grid[i, j] *= (1.0 - 0.08 * np.exp(-d2 / 0.15))

        # 2. Tactile Ground Uncertainty
        # Discharge upon solid ground contact; accumulate when foot is in the air or stationary
        if touch_l > 15.0:
            self.u_tactile_l = max(0.05, self.u_tactile_l - 2.5 * dt)
        else:
            self.u_tactile_l = min(1.0, self.u_tactile_l + 1.2 * dt)

        if touch_r > 15.0:
            self.u_tactile_r = max(0.05, self.u_tactile_r - 2.5 * dt)
        else:
            self.u_tactile_r = min(1.0, self.u_tactile_r + 1.2 * dt)

        # 3. Acoustic Ether Wave Uncertainty
        if audio_amp > 0.02:
            self.u_acoustic = max(0.05, self.u_acoustic - 3.0 * dt)
        else:
            self.u_acoustic = min(1.0, self.u_acoustic + 0.6 * dt)

        # 4. Vestibular Postural Uncertainty
        self.u_vestibular = float(np.clip(tilt_rad / 0.35, 0.0, 1.0))

        # 5. Spatial horizon prospective uncertainty
        self.u_spatial = float(np.mean(self.spatial_grid[ix:min(self.resolution, ix+4), iy:min(self.resolution, iy+4)]))


class ActiveInferenceAgent:
    """
    Continuous Active Inference Controller with Spontaneous Stepping & Push Dissipation.
    
    Maintains descending continuous proprioceptive expectations mu_q across all 18 actuators:
    - 6 DoF Left Leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    - 6 DoF Right Leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    - 3 DoF Left Arm: shoulder_pitch, shoulder_roll, elbow_pitch
    - 3 DoF Right Arm: shoulder_pitch, shoulder_roll, elbow_pitch
    """
    def __init__(self, dt: float = 0.002):
        self.dt = dt
        self.epistemic = ContinuousEpistemicField(bounds=8.0, resolution=80)

        # 1. Hierarchical Homeostatic & Postural Believed Attractors
        # Vital safety setpoints (phenotypic survival constraints)
        self.nominal_z = 0.64         # Nominal upright standing height (m)
        self.prior_z = self.nominal_z # Backward compatible alias
        self.prior_pitch = 0.0        # Neutral vertical orientation setpoint
        self.prior_roll = 0.0         # Neutral lateral orientation setpoint
        self.prior_torso_touch = 0.0  # Zero torso impact

        # Precision Weights (Inverse Variances on Predictions)
        self.pi_pitch = 40.0
        self.pi_pitch_rate = 6.0
        self.pi_roll = 45.0
        self.pi_roll_rate = 6.0
        self.pi_z = 25.0
        self.pi_torso = 60.0

        # Postural Mode State (Hierarchical Affordance)
        # Modes: 'BIPED_EXPLORE', 'CROUCH_PALPATE', 'GROUND_RECOVERY'
        self.postural_mode = 'BIPED_EXPLORE'
        self.expected_pitch = 0.0
        self.expected_z = self.nominal_z

        # Continuous Biological CPG & Coordination Dynamics
        self.phi = 0.0
        self.omega_base = 2.0 * math.pi * 0.78  # ~0.78 Hz natural biped stepping rhythm
        self.omega_0 = 3.91                     # Inverted pendulum capture frequency sqrt(g / z0)

        # Nominal joint resting setpoints
        self.q_nom_knee = 0.05
        self.q_nom_hip = 0.0
        self.q_nom_ankle = -0.033

        self.actuator_keys = [
            'motor_hip_yaw_l', 'motor_hip_roll_l', 'motor_hip_pitch_l',
            'motor_knee_l', 'motor_ankle_pitch_l', 'motor_ankle_roll_l',
            'motor_hip_yaw_r', 'motor_hip_roll_r', 'motor_hip_pitch_r',
            'motor_knee_r', 'motor_ankle_pitch_r', 'motor_ankle_roll_r',
            'motor_shoulder_pitch_l', 'motor_shoulder_roll_l', 'motor_elbow_l',
            'motor_shoulder_pitch_r', 'motor_shoulder_roll_r', 'motor_elbow_r'
        ]

        # Current continuous descending proprioceptive prophecy mu_q
        self.mu_q: Dict[str, float] = {k: 0.0 for k in self.actuator_keys}
        self.mu_q['motor_knee_l'] = self.q_nom_knee
        self.mu_q['motor_knee_r'] = self.q_nom_knee
        self.mu_q['motor_hip_pitch_l'] = self.q_nom_hip
        self.mu_q['motor_hip_pitch_r'] = self.q_nom_hip
        self.mu_q['motor_ankle_pitch_l'] = self.q_nom_ankle
        self.mu_q['motor_ankle_pitch_r'] = self.q_nom_ankle
        self.mu_q['motor_shoulder_roll_l'] = 0.12
        self.mu_q['motor_shoulder_roll_r'] = -0.12
        self.mu_q['motor_elbow_l'] = -0.25
        self.mu_q['motor_elbow_r'] = -0.25

        # Continuous Policy Variables (Expected Free Energy Attractors)
        self.v_cmd = 0.0         # Forward translation velocity intent (m/s)
        self.omega_cmd = 0.0     # Yaw angular velocity intent (rad/s)
        self.eta_v = 4.0         # Policy gradient rate for translation
        self.eta_omega = 8.0     # Policy gradient rate for turning
        self.sim_time = 0.0
        self.toy_satisfied_timer = 0.0

        # Telemetry metrics
        self.vfe = 0.0
        self.pragmatic_risk = 0.0
        self.epistemic_salience = 0.0
        self.tactile_curiosity = 0.0
        self.acoustic_curiosity = 0.0
        self.spatial_curiosity = 0.0

    def reset(self):
        """Resets the internal belief state to nominal homeostatic prior."""
        self.phi = 0.0
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.sim_time = 0.0
        self.toy_satisfied_timer = 0.0
        self.epistemic = ContinuousEpistemicField(bounds=8.0, resolution=80)
        self.postural_mode = 'BIPED_EXPLORE'
        for k in self.actuator_keys:
            self.mu_q[k] = 0.0
        self.mu_q['motor_knee_l'] = self.q_nom_knee
        self.mu_q['motor_knee_r'] = self.q_nom_knee
        self.mu_q['motor_hip_pitch_l'] = self.q_nom_hip
        self.mu_q['motor_hip_pitch_r'] = self.q_nom_hip
        self.mu_q['motor_ankle_pitch_l'] = self.q_nom_ankle
        self.mu_q['motor_ankle_pitch_r'] = self.q_nom_ankle
        self.mu_q['motor_shoulder_roll_l'] = 0.12
        self.mu_q['motor_shoulder_roll_r'] = -0.12
        self.mu_q['motor_elbow_l'] = -0.25
        self.mu_q['motor_elbow_r'] = -0.25

    def step(
        self,
        dt: float,
        pos: np.ndarray,
        pitch: float,
        pitch_rate: float,
        roll: float,
        roll_rate: float,
        touch_l: float,
        touch_r: float,
        touch_torso: float,
        audio_amp: float,
        is_fallen: bool = False,
        vel: Optional[np.ndarray] = None,
        audio_l: float = 0.0,
        audio_r: float = 0.0,
        yaw: float = 0.0,
        toy_pos: Optional[np.ndarray] = None,
        touched_toy: bool = False,
        touch_hand_l: float = 0.0,
        touch_hand_r: float = 0.0,
        rot_mat: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Pure First-Principles Continuous Active Inference Controller:
        1. Hierarchical Sensory Inference:
           - Vestibular tilt: pitch, roll, yaw, pitch_rate, roll_rate
           - Tactile stress: touch_l, touch_r, touch_torso, touched_toy
           - Spatial Ether Acoustics: audio_l, audio_r (interaural level/phase difference)
           - External Affordances: spatial toy beacon coordinates toy_pos
        2. Expected Free Energy (EFE) Policy Optimization:
           - Inferred Target Bearing: heading_err = atan2(dx, dy) - yaw
           - Epistemic Value:
             * Acoustic/Spatial Orientation: drives yaw turning rate omega to cancel heading_err.
             * Information Gain: drives forward translation v towards target when aligned.
           - Pragmatic Value (Survival / Homeostatic Bounds):
             * Penalizes walking when misaligned (alignment <= 0.4 -> v = 0, pure in-place turn).
             * Penalizes forward movement when at target (dist <= 0.30m -> v = 0, crouching/palpating).
             * Reflexive balance torques enforce upright phenotypic vertical attractor.
        3. Descending Proprioceptive Predictions (Biological Oscillator + Reflex Arcs):
           - Inverted pendulum cadence (1.60 Hz)
           - Dynamic phased yaw stepping: swing leg pivots outward in flight, stance leg provides reaction torque,
             enabling seamless full 360-degree in-place and continuous turning!
           - Contralateral arm swing for momentum balance; transitions to active bilateral palpation upon reaching target.
           - Flat-sole kinematic invariant keeps boot sole 100% parallel to the floor plane.
        """
        self.dt = dt
        self.sim_time += dt
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        v_y = float(vel[1]) if vel is not None else 0.0
        v_x = float(vel[0]) if vel is not None else 0.0
        tilt_rad = float(math.sqrt(pitch**2 + roll**2))

        # 1. Epistemic Field & Sensory Evidence
        self.epistemic.assimilate_observations(dt, x, y, touch_l, touch_r, audio_amp, tilt_rad)
        self.tactile_curiosity = float(0.5 * (self.epistemic.u_tactile_l + self.epistemic.u_tactile_r))
        self.acoustic_curiosity = float(self.epistemic.u_acoustic)
        self.spatial_curiosity = float(self.epistemic.u_spatial)

        # Track haptic contact & satisfaction
        hand_contact = (touch_hand_l > 1.0 or touch_hand_r > 1.0)
        if touched_toy or hand_contact:
            self.toy_satisfied_timer += dt
            self.tactile_curiosity = max(0.05, self.tactile_curiosity - 3.0 * dt)

        # 1. Instantaneous Inverted Pendulum Capture Point (Divergent Component of Motion, DCM)
        # First-principles Linear Inverted Pendulum Model (LIPM): xi = pos + vel / omega_0
        pitch_err = pitch - self.prior_pitch
        delta_xi_sagittal = self.nominal_z * math.sin(pitch_err) + (v_y + self.nominal_z * pitch_rate) / self.omega_0
        delta_xi_coronal = - self.nominal_z * math.sin(roll) + (v_x - self.nominal_z * roll_rate) / self.omega_0

        # 2. Target Inference & Egocentric Body-Frame Projection (First-Principles)
        has_toy = (toy_pos is not None)
        has_sound = (audio_amp > 0.02)

        if has_toy:
            delta_pos = np.array(toy_pos, dtype=np.float64) - np.array(pos, dtype=np.float64)
            if rot_mat is not None:
                # SE(3) projection into body frame:
                # Column 0: Body Right (+X), Column 1: Body Forward (+Y), Column 2: Body Up (+Z)
                local_pos = rot_mat.T @ delta_pos
                local_x = float(local_pos[0])
                local_y = float(local_pos[1])
            else:
                c_y = math.cos(yaw)
                s_y = math.sin(yaw)
                local_x = float(delta_pos[0] * c_y + delta_pos[1] * s_y)
                local_y = float(-delta_pos[0] * s_y + delta_pos[1] * c_y)

            dist_to_target = float(math.sqrt(local_x**2 + local_y**2))

            # Reset satisfaction timer if toy is far away or moved
            if dist_to_target > 0.45:
                self.toy_satisfied_timer = 0.0

            toy_satiated = (self.toy_satisfied_timer > 6.0)

            if (not toy_satiated) and dist_to_target < 2.5:
                raw_err = float(math.atan2(-local_x, local_y))
                # Prevent branch-cut oscillation at +/- pi when target is directly behind
                if abs(raw_err) > 2.8 and abs(self.omega_cmd) > 0.05:
                    if np.sign(raw_err) != np.sign(self.omega_cmd):
                        raw_err = - raw_err
                heading_err = raw_err
                has_active_goal = True
            else:
                heading_err = 0.0
                has_active_goal = False
        elif has_sound:
            binaural_diff = float(audio_l - audio_r)
            # Louder on left (binaural_diff > 0) -> turn left (+Z CCW)
            heading_err = float(np.clip(2.5 * binaural_diff, -math.pi, math.pi))
            dist_to_target = 1.0 / max(audio_amp, 0.05)
            has_active_goal = True
        else:
            heading_err = 0.0
            dist_to_target = 999.0
            has_active_goal = False

        # 3. Expected Free Energy (EFE) Policy Gradient
        if has_active_goal:
            alignment = math.cos(heading_err)  # 1.0 when aligned, -1.0 when behind
            target_omega = float(np.clip(1.6 * heading_err, -0.65, 0.65))
            if alignment > 0.40 and dist_to_target > 0.35:
                target_v = 0.09 * max(0.0, alignment) * min(1.0, dist_to_target / 0.60)
            else:
                target_v = 0.0
        else:
            alignment = 1.0
            target_omega = 0.0
            target_v = 0.0

        self.omega_cmd += self.eta_omega * dt * (target_omega - self.omega_cmd)
        self.v_cmd += self.eta_v * dt * (target_v - self.v_cmd)

        nav_drive = float(np.clip(self.v_cmd / 0.03 + abs(self.omega_cmd) / 0.20, 0.0, 1.0))
        # Dynamic Capture Point divergence beyond physical Base of Support (BoS):
        # Chunky boot length 0.30m provides BoS from -0.10m (heel) to +0.18m (toe).
        # Inside BoS: Tier 1 & 2 Ankle/Hip strategy restores equilibrium via res_hip/res_roll.
        # Outside BoS: Tier 3 Stepping strategy engages CPG to plant swing foot at Capture Point.
        bos_excess = max(0.0, delta_xi_sagittal - 0.18, -0.10 - delta_xi_sagittal)
        disturbance_drive = float(np.clip(bos_excess / 0.08, 0.0, 1.0))
        loco_drive = max(nav_drive, disturbance_drive)

        # Dynamic squat reflex: under forward surprise/VFE surge or forward disturbance,
        # flexing knees lowers CoM height z0, reducing tipping torque tau = m*g*z0*sin(theta).
        # When falling or stepping backward, knees must remain extended to reach behind CoM.
        if pitch < -0.02 or delta_xi_sagittal < -0.04:
            squat_reflex = 0.0
        else:
            squat_reflex = float(np.clip(0.50 * max(0.0, pitch - 0.04) + 0.20 * max(0.0, pitch_rate) + 0.25 * disturbance_drive, 0.0, 0.35))

        # 4. Hierarchical Mode Attractor
        if is_fallen or touch_torso > 10.0 or tilt_rad > math.radians(45.0):
            self.postural_mode = 'GROUND_RECOVERY'
            dynamic_state = "GROUND_RECOVERY"
        elif has_active_goal and dist_to_target <= 0.38:
            self.postural_mode = 'CROUCH_PALPATE'
            dynamic_state = "CROUCH_PALPATE"
        elif abs(self.omega_cmd) > 0.12 and self.v_cmd < 0.02 and disturbance_drive < 0.2:
            self.postural_mode = 'BIPED_EXPLORE'
            dynamic_state = "TURN_IN_PLACE"
        elif loco_drive > 0.05:
            self.postural_mode = 'BIPED_EXPLORE'
            dynamic_state = "REACTIVE_STEP" if disturbance_drive > nav_drive else "WALK_APPROACH"
        else:
            self.postural_mode = 'BIPED_EXPLORE'
            dynamic_state = "STAND_BALANCE"

        # Tier 1 Restoring torques for upright stance
        pitch_setpoint = 0.02 * (self.v_cmd / 0.08)
        p_err = pitch - pitch_setpoint
        res_hip = float(np.clip(- (1.6 * p_err + 0.20 * pitch_rate), -0.35, 0.35))
        res_roll = float(np.clip(-(1.6 * roll + 0.22 * roll_rate), -0.25, 0.25))

        target_mu = {}

        if self.postural_mode == 'GROUND_RECOVERY':
            # Active Crawl / Quadruped Ground Push-Up Self-Rescue
            target_mu['motor_shoulder_pitch_l'] = 1.30 # Push down against ground
            target_mu['motor_shoulder_pitch_r'] = 1.30
            target_mu['motor_shoulder_roll_l'] = 0.25
            target_mu['motor_shoulder_roll_r'] = -0.25
            target_mu['motor_elbow_l'] = -0.15 # Extend arms to push body up
            target_mu['motor_elbow_r'] = -0.15
            target_mu['motor_hip_yaw_l'] = 0.0
            target_mu['motor_hip_yaw_r'] = 0.0
            target_mu['motor_hip_roll_l'] = 0.0
            target_mu['motor_hip_roll_r'] = 0.0
            target_mu['motor_hip_pitch_l'] = 0.35 # Tuck knees under pelvis
            target_mu['motor_hip_pitch_r'] = 0.35
            target_mu['motor_knee_l'] = 0.85 # Stable low crawl
            target_mu['motor_knee_r'] = 0.85
            target_mu['motor_ankle_pitch_l'] = -0.35
            target_mu['motor_ankle_pitch_r'] = -0.35
            target_mu['motor_ankle_roll_l'] = 0.0
            target_mu['motor_ankle_roll_r'] = 0.0

        elif self.postural_mode == 'CROUCH_PALPATE':
            # Squat posture: knees bend to lower torso, soles flat on floor
            target_mu['motor_hip_yaw_l'] = 0.0
            target_mu['motor_hip_yaw_r'] = 0.0
            target_mu['motor_hip_roll_l'] = res_roll * 0.4
            target_mu['motor_hip_roll_r'] = res_roll * 0.4
            target_mu['motor_hip_pitch_l'] = float(np.clip(-0.32 + res_hip * 0.3, -0.60, 0.60))
            target_mu['motor_hip_pitch_r'] = float(np.clip(-0.32 + res_hip * 0.3, -0.60, 0.60))
            target_mu['motor_knee_l'] = 0.52
            target_mu['motor_knee_r'] = 0.52
            target_mu['motor_ankle_pitch_l'] = float(np.clip(-0.20 - res_hip * 0.1, -0.60, 0.60))
            target_mu['motor_ankle_pitch_r'] = float(np.clip(-0.20 - res_hip * 0.1, -0.60, 0.60))
            target_mu['motor_ankle_roll_l'] = - target_mu['motor_hip_roll_l']
            target_mu['motor_ankle_roll_r'] = - target_mu['motor_hip_roll_r']

        else:
            # BIPED_EXPLORE: Active Locomotion or Double-Support Stance
            if loco_drive > 0.02:
                cadence = 1.60 + 0.80 * disturbance_drive
                self.phi += dt * 2.0 * math.pi * cadence * loco_drive
                self.phi = self.phi % (2.0 * math.pi)

                # First-Principles Tactile Phase Entrainment:
                # Real mechanical contact forces entrain the neural oscillator:
                # When one foot firmly supports body weight (>25N) and the other has lifted (<10N):
                if touch_l > 25.0 and touch_r < 10.0 and (self.phi < math.pi):
                    self.phi = math.pi + 0.15
                elif touch_r > 25.0 and touch_l < 10.0 and (self.phi >= math.pi):
                    self.phi = 0.15

                s_cpg = math.sin(self.phi)
                c_cpg = math.cos(self.phi)

                s_l = max(0.0, s_cpg)
                s_r = max(0.0, -s_cpg)
                st_l = max(0.0, -s_cpg)
                st_r = max(0.0, s_cpg)

                sway = 0.035 * loco_drive * c_cpg

                # Dynamic Stride Length from Capture Point in World Coordinates
                stride_fwd = float(np.clip(self.v_cmd * 1.2 + 1.15 * delta_xi_sagittal + 0.40 * max(0.0, pitch), -0.50, 0.65))

                turn_dir = np.sign(self.omega_cmd) if abs(self.omega_cmd) > 1e-4 else 0.0
                turn_mag = min(1.0, abs(self.omega_cmd) / 0.60)
                yaw_step = 0.55 * turn_dir * turn_mag

                hip_yaw_l = yaw_step * s_l - 0.20 * turn_dir * st_l
                hip_yaw_r = yaw_step * s_r + 0.20 * turn_dir * st_r

                # Swing leg reaches uninhibited to stride_fwd
                # Stance leg provides compliant restoring torque res_hip
                hip_p_l = stride_fwd * s_l + res_hip * 0.4 * st_l
                hip_p_r = stride_fwd * s_r + res_hip * 0.4 * st_r

                # Dynamic knee lift and squat reflex:
                if stride_fwd < -0.04:
                    knee_lift = 0.08 * loco_drive
                else:
                    knee_lift = (0.42 + 0.08 * disturbance_drive) * loco_drive

                knee_l = self.q_nom_knee + squat_reflex + knee_lift * s_l
                knee_r = self.q_nom_knee + squat_reflex + knee_lift * s_r

                # Ankle pitch enforces horizontal flat sole invariant relative to ground plane
                ankle_p_l = - hip_p_l - (knee_l - self.q_nom_knee) - pitch * st_l - 0.08 * loco_drive * s_l
                ankle_p_r = - hip_p_r - (knee_r - self.q_nom_knee) - pitch * st_r - 0.08 * loco_drive * s_r

                hip_r_l = float(np.clip(sway + res_roll * 0.5, -0.30, 0.30))
                hip_r_r = float(np.clip(sway + res_roll * 0.5, -0.30, 0.30))
                ankle_r_l = float(np.clip(-hip_r_l, -0.25, 0.25))
                ankle_r_r = float(np.clip(-hip_r_r, -0.25, 0.25))
            else:
                # Quiescent double-support standing with flexible posture
                hip_yaw_l = 0.0
                hip_yaw_r = 0.0
                knee_l = self.q_nom_knee + squat_reflex
                knee_r = self.q_nom_knee + squat_reflex
                hip_p_l = res_hip * 0.4
                hip_p_r = res_hip * 0.4
                ankle_p_l = -0.05 - (knee_l - self.q_nom_knee) - res_hip * 0.2
                ankle_p_r = -0.05 - (knee_r - self.q_nom_knee) - res_hip * 0.2
                hip_r_l = res_roll * 0.5
                hip_r_r = res_roll * 0.5
                ankle_r_l = -hip_r_l
                ankle_r_r = -hip_r_r

            target_mu['motor_hip_yaw_l'] = hip_yaw_l
            target_mu['motor_hip_yaw_r'] = hip_yaw_r
            target_mu['motor_hip_roll_l'] = hip_r_l
            target_mu['motor_hip_roll_r'] = hip_r_r
            target_mu['motor_hip_pitch_l'] = float(np.clip(hip_p_l, -0.60, 0.60))
            target_mu['motor_hip_pitch_r'] = float(np.clip(hip_p_r, -0.60, 0.60))
            target_mu['motor_knee_l'] = float(np.clip(knee_l, 0.0, 1.20))
            target_mu['motor_knee_r'] = float(np.clip(knee_r, 0.0, 1.20))
            target_mu['motor_ankle_pitch_l'] = float(np.clip(ankle_p_l, -0.60, 0.60))
            target_mu['motor_ankle_pitch_r'] = float(np.clip(ankle_p_r, -0.60, 0.60))
            target_mu['motor_ankle_roll_l'] = ankle_r_l
            target_mu['motor_ankle_roll_r'] = ankle_r_r

        # =========================================================================
        # Active Inference Arm Motor Control:
        # F_arm = F_momentum + F_reflex + F_reach + F_affect
        # =========================================================================
        if self.postural_mode != 'GROUND_RECOVERY':
            # 1. Contralateral Locomotion Swing (Momentum Compensation)
            s_arm = math.sin(self.phi)
            loco_pitch_l = - 0.45 * loco_drive * s_arm
            loco_pitch_r = + 0.45 * loco_drive * s_arm
            loco_elbow_l = -0.25 - 0.40 * max(0.0, -s_arm) * loco_drive
            loco_elbow_r = -0.25 - 0.40 * max(0.0, s_arm) * loco_drive
            loco_roll_l = 0.12
            loco_roll_r = -0.12

            # 2. Vestibular Tightrope Balance & Protective Reflexes (Parachute & Push-Up)
            roll_reflex_l = float(np.clip(0.9 * max(0.0, -roll - 0.15 * roll_rate), 0.0, 0.50))
            roll_reflex_r = float(np.clip(-0.9 * max(0.0, roll + 0.15 * roll_rate), -0.50, 0.0))
            pitch_brace = float(np.clip(1.2 * max(0.0, pitch - 0.06), 0.0, 0.60))

            is_hand_ground = (touch_hand_l > 4.0 or touch_hand_r > 4.0)
            parachute_active = (pitch > math.radians(16.0) or disturbance_drive > 0.40)

            # 3. Active Epistemic Reaching & Palpation
            if has_active_goal and dist_to_target < 0.65:
                q_reach = float(np.clip(1.0 - (dist_to_target - 0.20) / 0.45, 0.0, 1.0))
            else:
                q_reach = 0.0
            reach_pitch = 0.70 * q_reach
            reach_elbow = -0.50 * q_reach
            reach_roll_l = -0.06 * q_reach
            reach_roll_r = +0.06 * q_reach

            # 4. Biological Breathing / Micro-sway & Affective Expression
            resp = 0.035 * math.sin(self.sim_time * 2.0 * math.pi * 0.28)

            if is_hand_ground:
                # Active Push-Up: Hands hit the ground, push upper body up and backward!
                arm_p_l = 1.35
                arm_p_r = 1.35
                arm_el_l = -0.15 # Extend elbow to shove off floor
                arm_el_r = -0.15
                arm_r_l = 0.20
                arm_r_r = -0.20
            elif parachute_active:
                # Protective Extension: Shoot arms forward and downward to break fall
                reach_ground = float(np.clip((pitch - math.radians(16.0)) / math.radians(18.0), 0.0, 1.0))
                arm_p_l = 0.40 + 0.85 * reach_ground
                arm_p_r = 0.40 + 0.85 * reach_ground
                arm_el_l = -0.25 + 0.15 * reach_ground # Extend forward
                arm_el_r = -0.25 + 0.15 * reach_ground
                arm_r_l = 0.14 + 0.12 * reach_ground
                arm_r_r = -0.14 - 0.12 * reach_ground
            elif self.postural_mode == 'CROUCH_PALPATE' or (touched_toy and has_active_goal):
                # Gentle rhythmic palpation and petting
                palp = 0.10 * math.sin(self.sim_time * 2.0 * math.pi * 1.5)
                arm_p_l = 0.68 + palp
                arm_p_r = 0.68 - palp
                arm_el_l = -0.72 + 0.5 * palp
                arm_el_r = -0.72 - 0.5 * palp
                arm_r_l = 0.06 + roll_reflex_l
                arm_r_r = -0.06 + roll_reflex_r
            elif loco_drive > 0.05:
                # Dynamic biped walking swing
                arm_p_l = loco_pitch_l + pitch_brace + resp + reach_pitch
                arm_p_r = loco_pitch_r + pitch_brace + resp + reach_pitch
                arm_el_l = loco_elbow_l + reach_elbow
                arm_el_r = loco_elbow_r + reach_elbow
                arm_r_l = loco_roll_l + roll_reflex_l + reach_roll_l
                arm_r_r = loco_roll_r + roll_reflex_r + reach_roll_r
            else:
                # Quiet standing presence with living respiratory sway & curious posture
                curious_lift = 0.22 * min(1.0, max(0.0, self.acoustic_curiosity - 0.3))
                arm_p_l = 0.05 + resp + pitch_brace + curious_lift + reach_pitch
                arm_p_r = 0.05 + resp + pitch_brace + curious_lift + reach_pitch
                arm_el_l = -0.22 - resp - curious_lift * 1.2 + reach_elbow
                arm_el_r = -0.22 - resp - curious_lift * 1.2 + reach_elbow
                arm_r_l = 0.12 + 0.3 * resp + roll_reflex_l + reach_roll_l
                arm_r_r = -0.12 - 0.3 * resp + roll_reflex_r + reach_roll_r

            target_mu['motor_shoulder_pitch_l'] = float(np.clip(arm_p_l, -0.80, 1.45))
            target_mu['motor_shoulder_pitch_r'] = float(np.clip(arm_p_r, -0.80, 1.45))
            target_mu['motor_shoulder_roll_l'] = float(np.clip(arm_r_l, 0.0, 0.75))
            target_mu['motor_shoulder_roll_r'] = float(np.clip(arm_r_r, -0.75, 0.0))
            target_mu['motor_elbow_l'] = float(np.clip(arm_el_l, -1.40, 0.15))
            target_mu['motor_elbow_r'] = float(np.clip(arm_el_r, -1.40, 0.15))

        # 5. Smooth Neuromuscular Proprioceptive Filtering
        alpha = 0.35
        for k in self.actuator_keys:
            self.mu_q[k] = (1.0 - alpha) * self.mu_q[k] + alpha * target_mu[k]

        # 6. Variational Free Energy & Pragmatic Risk Computation
        inst_vfe = (
            0.5 * self.pi_pitch * (pitch**2) +
            0.5 * self.pi_roll * (roll**2) +
            0.5 * self.pi_torso * (max(0.0, touch_torso)**2) +
            (0.5 * (heading_err**2) if has_active_goal else 0.0) +
            (0.2 * min(5.0, dist_to_target) if has_active_goal else 0.0)
        )
        self.pragmatic_risk = float(0.5 * self.pi_pitch * (pitch**2) + 0.5 * self.pi_roll * (roll**2))
        self.epistemic_salience = float(0.4 * self.tactile_curiosity + 0.3 * self.acoustic_curiosity + 0.3 * (1.0 / (1.0 + min(5.0, dist_to_target))))
        self.vfe = float(inst_vfe + 1.2 * self.epistemic_salience)

        return self._build_return_dict(dynamic_state=dynamic_state)

    def _build_return_dict(self, dynamic_state: str) -> Dict[str, float]:
        res = {k: float(v) for k, v in self.mu_q.items()}
        res['vfe'] = float(self.vfe)
        res['epistemic_gain'] = float(self.epistemic_salience)
        res['pragmatic_risk'] = float(self.pragmatic_risk)
        res['tactile_curiosity'] = float(self.tactile_curiosity)
        res['acoustic_curiosity'] = float(self.acoustic_curiosity)
        res['spatial_curiosity'] = float(self.spatial_curiosity)
        res['prediction_error_norm'] = float(self.pragmatic_risk)
        res['gait_state'] = dynamic_state
        res['postural_mode'] = self.postural_mode
        return res
