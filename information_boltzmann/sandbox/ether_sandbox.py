"""
The Complete Unified Spatial Ether Sandbox (MVP)
Integrates MuJoCo Multibody Dynamics, the First-Principles Ether Wave Medium,
and the Bipedal Active Locomotion Controller.
"""

import os
import time
import numpy as np
import mujoco

from information_boltzmann.sandbox.ether_medium import UnifiedEtherMedium
from information_boltzmann.sandbox.walking_cpg import BipedalCPGController

class EtherSandbox:
    def __init__(self, xml_path: str = None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "robot_pet.xml")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Ether medium carrying mechanical waves
        self.ether = UnifiedEtherMedium(speed_of_sound=343.0, air_density=1.225)
        
        # First-principles Active Inference & Variational Free Energy Engine
        # Motor actions emerge strictly from minimizing free energy F = Complexity - Accuracy
        from information_boltzmann.sandbox.active_inference_agent import ActiveInferenceAgent
        self.brain = ActiveInferenceAgent(dt=self.model.opt.timestep)
        
        # Facial screen expression system (discrete token actions)
        from information_boltzmann.sandbox.expression_system import ExpressionSystem, ExpressionToken
        self.expressions = ExpressionSystem(self.model)
        
        # Query site & geom IDs
        self.left_ear_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_ear_sensor")
        self.right_ear_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_ear_sensor")
        self.boot_l_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "boot_l")
        self.boot_r_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "boot_r")
        self.boot_l_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "boot_l_fore")
        self.boot_r_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "boot_r_fore")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_root")
        self.cam_eyes_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "first_person_eyes")
        self.toy_ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "toy_ball_joint")
        self.toy_ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "toy_ball_geom")
        self.hand_l_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hand_l_geom")
        self.hand_r_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hand_r_geom")
        
        # Actuator mapping
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self.actuator_map = {name: i for i, name in enumerate(self.actuator_names)}
        
        # Exploration steering generator (smooth wandering trajectory)
        self.steer_bias = 0.0
        self.explore_phase = 0.0
        
        # External disturbance / Caretaker interaction
        self.applied_force = np.zeros(3)
        self.applied_torque = np.zeros(3)
        self.applied_time_left = 0.0
        
        # Caretaker auto-recovery
        self.fallen_duration = 0.0
        
        # Reset state to ground-settled position
        self._apply_homeostatic_pose()
        self.dt = self.model.opt.timestep
        self.expressions.set_expression(self.data, ExpressionToken.DEFAULT)

    def _apply_homeostatic_pose(self):
        self.data.qpos[0:3] = [0.0, 0.0, 0.638]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        # Compliant leg resting angles for upright vertical stance
        self.data.qpos[17] = 0.0     # hip_pitch_l
        self.data.qpos[18] = 0.05    # knee_pitch_l
        self.data.qpos[19] = -0.033  # ankle_pitch_l
        self.data.qpos[23] = 0.0     # hip_pitch_r
        self.data.qpos[24] = 0.05    # knee_pitch_r
        self.data.qpos[25] = -0.033  # ankle_pitch_r
        # Arms resting pose
        self.data.qpos[10] = 0.08  # shoulder_roll_l
        self.data.qpos[11] = -0.25 # elbow_pitch_l
        self.data.qpos[13] = -0.08 # shoulder_roll_r
        self.data.qpos[14] = -0.25 # elbow_pitch_r
        for idx in [7, 8, 9, 12, 15, 16, 20, 21, 22, 26]:
            self.data.qpos[idx] = 0.0
            
        # Reset toy ball pose if present
        if self.model.nq > 27:
            self.data.qpos[27:30] = [0.0, 1.2, 0.10]
            self.data.qpos[30:34] = [1.0, 0.0, 0.0, 0.0]
            if self.data.qvel.shape[0] > 26:
                self.data.qvel[26:32] = 0.0
                
        mujoco.mj_forward(self.model, self.data)

    def throw_toy(self, x: float = 0.0, y: float = 1.2, z: float = 0.10, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        """Places or throws the interactive curiosity toy ball and generates acoustic pulse."""
        if self.model.nq > 27:
            self.data.qpos[27:30] = [x, y, max(z, 0.10)]
            self.data.qpos[30:34] = [1.0, 0.0, 0.0, 0.0]
            if self.data.qvel.shape[0] > 26:
                self.data.qvel[26:29] = [vx, vy, vz]
                self.data.qvel[29:32] = 0.0
            from information_boltzmann.sandbox.ether_medium import MechanicalWavePacket
            packet = MechanicalWavePacket(
                source_pos=np.array([x, y, max(z, 0.10)], dtype=np.float64),
                birth_time=float(self.data.time),
                force_derivative=np.array([0.0, 0.0, 25000.0], dtype=np.float64),
                monopole_strength=0.04,
                duration=0.045,
                base_freq=380.0
            )
            self.ether.active_packets.append(packet)
        if hasattr(self.brain, 'toy_satisfied_timer'):
            self.brain.toy_satisfied_timer = 0.0

    def help_up(self):
        """
        Caretaker physical interaction: Gently pulls the robot up back onto its feet.
        Restores upright orientation at compliant stance, zeroing velocities.
        """
        self._apply_homeostatic_pose()
        self.applied_force[:] = 0.0
        self.applied_time_left = 0.0
        self.brain.reset()
        from information_boltzmann.sandbox.expression_system import PrimaryEmotion, SubEmotion
        self.expressions.set_emotion_explicit(PrimaryEmotion.JOY, SubEmotion.CHEERFUL, hold_duration=2.5)

    def push(self, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0, duration: float = 0.12):
        """Applies an external push/disturbance force to the torso."""
        self.applied_force = np.array([fx, fy, fz], dtype=np.float64)
        self.applied_time_left = duration

    def step(self):
        """
        Advances the universe by 1 physical microstep dt:
        1. Step MuJoCo multibody dynamics.
        2. Extract ALL mechanical contact points and forces across the scene.
        3. Tactile sensing: boundary integration of contact stress on each limb.
        4. Radiate mechanical dipole waves into the continuous Ether Medium.
        5. Sample binaural spatial hearing at the left and right ear sites.
        6. Compute CPG motor targets from tactile and vestibular balance.
        """
        # 0. Apply external caretaker / environmental force if active
        if self.applied_time_left > 0:
            self.data.xfrc_applied[self.root_body_id, :3] = self.applied_force
            self.applied_time_left -= self.dt
        else:
            self.data.xfrc_applied[self.root_body_id, :] = 0.0

        # 1. Advance MuJoCo Physics
        mujoco.mj_step(self.model, self.data)
        
        # 2. Extract ALL mechanical contacts in the physical universe (First-Principles)
        contact_events = []
        touch_l_total = 0.0
        touch_r_total = 0.0
        touch_hand_l = 0.0
        touch_hand_r = 0.0
        touch_torso_total = 0.0
        touched_toy = False
        
        toy_pos = None
        if self.model.nq > 27:
            toy_pos = self.data.qpos[27:30].copy()
        
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            c_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, c_force)
            force_vec = con.frame.reshape(3, 3).T @ c_force[:3]
            force_mag = float(np.linalg.norm(force_vec))
            
            # Boundary integration of tactile contact stress by limb
            b_id = None
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]
            if body1 == self.boot_l_body_id or body2 == self.boot_l_body_id:
                touch_l_total += force_mag
                b_id = 1 # boot_l
                if con.geom1 == self.toy_ball_geom_id or con.geom2 == self.toy_ball_geom_id:
                    touched_toy = True
            elif body1 == self.boot_r_body_id or body2 == self.boot_r_body_id:
                touch_r_total += force_mag
                b_id = 2 # boot_r
                if con.geom1 == self.toy_ball_geom_id or con.geom2 == self.toy_ball_geom_id:
                    touched_toy = True
            elif con.geom1 == self.hand_l_geom_id or con.geom2 == self.hand_l_geom_id:
                touch_hand_l += force_mag
                b_id = 5 # hand_l
                if con.geom1 == self.toy_ball_geom_id or con.geom2 == self.toy_ball_geom_id:
                    touched_toy = True
            elif con.geom1 == self.hand_r_geom_id or con.geom2 == self.hand_r_geom_id:
                touch_hand_r += force_mag
                b_id = 6 # hand_r
                if con.geom1 == self.toy_ball_geom_id or con.geom2 == self.toy_ball_geom_id:
                    touched_toy = True
            elif (con.geom1 == self.torso_geom_id and con.geom2 == self.ground_geom_id) or \
                 (con.geom2 == self.torso_geom_id and con.geom1 == self.ground_geom_id):
                touch_torso_total += force_mag
                b_id = 3 # torso
            elif con.geom1 == self.toy_ball_geom_id or con.geom2 == self.toy_ball_geom_id:
                b_id = 4 # toy_ball
                
            if b_id is not None:
                contact_events.append({
                    'body_id': b_id,
                    'geom1': con.geom1,
                    'geom2': con.geom2,
                    'pos': con.pos.copy(),
                    'force': force_vec
                })
            
        # 3. Inject mechanical waves into the Ether Medium
        self.ether.step(sim_time=self.data.time, contact_events=contact_events)
        
        # 4. Sample sensory traces on the agent's boundary
        left_ear_pos = self.data.site_xpos[self.left_ear_id].copy()
        right_ear_pos = self.data.site_xpos[self.right_ear_id].copy()
        p_L, p_R = self.ether.sample_binaural_ears(left_ear_pos, right_ear_pos)
        audio_amplitude = max(abs(p_L), abs(p_R))
        
        # Torso pitch & roll orientation from rotation matrix
        R = self.data.xmat[self.root_body_id].reshape(3, 3)
        # Spine vector uz = R[:, 2] (from feet to head)
        # pitch: forward lean > 0 (leaning towards +Y)
        pitch_angle = float(np.arcsin(np.clip(R[1, 2], -1.0, 1.0)))
        # roll: right lean > 0 (leaning towards +X)
        roll_angle = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
        yaw_angle = float(np.arctan2(R[1, 0], R[0, 0]))
        # Note: In MuJoCo freejoint, d.qvel[3] has opposite sign of d(pitch)/dt
        pitch_rate = - float(self.data.qvel[3])
        roll_rate = float(self.data.qvel[4])
        
        robot_pos = self.data.qpos[:3].copy()
        robot_vel = self.data.qvel[:3].copy()
        forward_speed = float(np.linalg.norm(robot_vel[:2]))
        
        # If torso touched ground or heavy tilt (>45 deg), it fell down
        is_fallen = (touch_torso_total > 5.0) or (abs(np.degrees(pitch_angle)) > 45.0) or (abs(np.degrees(roll_angle)) > 45.0)
        
        # 5. First-Principles Active Inference & Variational Free Energy Engine
        # No ad-hoc action presets: actions emerge strictly from minimizing free energy
        ai_targets = self.brain.step(
            dt=self.dt,
            pos=robot_pos,
            pitch=pitch_angle,
            pitch_rate=pitch_rate,
            roll=roll_angle,
            roll_rate=roll_rate,
            touch_l=touch_l_total,
            touch_r=touch_r_total,
            touch_torso=touch_torso_total,
            audio_amp=audio_amplitude,
            is_fallen=is_fallen,
            vel=robot_vel,
            audio_l=p_L,
            audio_r=p_R,
            yaw=yaw_angle,
            toy_pos=toy_pos,
            touched_toy=touched_toy,
            touch_hand_l=touch_hand_l,
            touch_hand_r=touch_hand_r,
            rot_mat=R
        )
        
        for name, val in ai_targets.items():
            if name in self.actuator_map:
                self.data.ctrl[self.actuator_map[name]] = val
                
        # 6. Autonomous Screen Expression Token Selection
        # Automatic caretaker recovery: if fallen for > 1.2s, gently lift back up
        if is_fallen:
            self.fallen_duration += self.dt
            if self.fallen_duration > 1.2:
                self.help_up()
                self.fallen_duration = 0.0
                is_fallen = False
        else:
            self.fallen_duration = 0.0

        expr_token = self.expressions.select_autonomous_token(
            dt=self.dt,
            audio_amplitude=audio_amplitude,
            pitch_rad=pitch_angle,
            roll_rad=roll_angle,
            forward_speed=forward_speed,
            steering_rate=float(ai_targets.get('epistemic_gain', 0.0)),
            vfe=float(ai_targets.get('vfe', 0.0)),
            curiosity=float(ai_targets.get('epistemic_gain', 0.0)),
            risk=float(ai_targets.get('pragmatic_risk', 0.0)),
            is_fallen=is_fallen
        )
        
        return {
            'time': self.data.time,
            'pos': robot_pos,
            'vel': robot_vel,
            'speed': forward_speed,
            'touch_l': touch_l_total,
            'touch_r': touch_r_total,
            'touch_hand_l': touch_hand_l,
            'touch_hand_r': touch_hand_r,
            'touch_torso': touch_torso_total,
            'is_fallen': is_fallen,
            'pitch': pitch_angle,
            'roll': roll_angle,
            'yaw': yaw_angle,
            'toy_pos': toy_pos.tolist() if toy_pos is not None else [0.0, 1.2, 0.10],
            'touched_toy': touched_toy,
            'vfe': float(ai_targets.get('vfe', 0.0)),
            'epistemic_gain': float(ai_targets.get('epistemic_gain', 0.0)),
            'tactile_curiosity': float(ai_targets.get('tactile_curiosity', 0.0)),
            'acoustic_curiosity': float(ai_targets.get('acoustic_curiosity', 0.0)),
            'spatial_curiosity': float(ai_targets.get('spatial_curiosity', 0.0)),
            'pragmatic_risk': float(ai_targets.get('pragmatic_risk', 0.0)),
            'prediction_error': float(ai_targets.get('prediction_error_norm', 0.0)),
            'gait_state': str(ai_targets.get('gait_state', 'STAND')),
            'postural_mode': str(ai_targets.get('postural_mode', 'BIPED_EXPLORE')),
            'audio_l': p_L,
            'audio_r': p_R,
            'audio_amp': audio_amplitude,
            'expression_token': int(expr_token),
            'expression_label': self.expressions.current_label,
            'primary_emotion': self.expressions.primary_emotion.name,
            'sub_emotion': self.expressions.sub_emotion.value,
            'theme_style': self.expressions.theme_style.name,
            'valence': float(self.expressions.valence),
            'arousal': float(self.expressions.arousal),
            'num_contacts': self.data.ncon,
            'active_ether_waves': len(self.ether.active_packets)
        }
