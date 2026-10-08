"""
Embodied Tactile-Driven Bipedal Locomotion & Balance Controller.
Features:
1. Closed-loop active inference balance reflex (negative feedback on vestibular pitch and roll).
2. Tactile & vestibular-triggered step transitions.
3. Disturbance rejection reflex: aborts stepping into dual-stance recovery when perturbed.
4. Safe fall passive mode & seamless upright recovery without thread interruption.
"""

import math
import numpy as np

class BipedalCPGController:
    """
    Embodied balance and locomotion controller for the robot pet.
    """
    def __init__(
        self,
        step_duration: float = 0.38,
        stride: float = 0.13,
        sway: float = 0.06,
        step_freq: float = None,
        swing_amplitude: float = None
    ):
        if step_freq is not None and step_freq > 0:
            step_duration = 0.5 / step_freq
        if swing_amplitude is not None:
            stride = swing_amplitude
            
        # Vestibular balance gains (tuned for stable negative feedback)
        self.k_balance_pitch = 1.50
        self.d_balance_pitch = 0.12
        self.k_balance_roll = 1.20
        self.d_balance_roll = 0.12
        
        self.step_duration = step_duration
        self.stride = stride
        self.sway_amp = sway
        
        # State: 'STAND', 'STEP_L', 'STEP_R', 'FALLEN'
        self.state = 'STAND'
        self.state_time = 0.0
        self.step_phase = 0.0
        
        # Online adaptive balance state
        self.smoothed_loss = 0.0
        self.pitch_integral = 0.0

    def step(
        self,
        dt: float,
        touch_l: float,
        touch_r: float,
        pitch_angle: float,
        pitch_rate: float,
        roll_angle: float = 0.0,
        roll_rate: float = 0.0,
        steer: float = 0.0,
        learn: bool = True
    ) -> dict:
        """
        Advances the embodied controller by dt.
        Inputs:
          pitch_angle: radians, forward lean > 0
          pitch_rate: radians/sec
          roll_angle: radians, right lean > 0
          roll_rate: radians/sec
        """
        self.state_time += dt
        
        # Instantaneous balance loss
        inst_loss = float(pitch_angle**2 + roll_angle**2 + 0.05 * (pitch_rate**2 + roll_rate**2))
        self.smoothed_loss = 0.98 * self.smoothed_loss + 0.02 * inst_loss
        
        tilt_deg = float(np.degrees(np.sqrt(pitch_angle**2 + roll_angle**2)))
        
        # 1. Fall detection & passive protection
        if tilt_deg > 45.0:
            self.state = 'FALLEN'
            
        if self.state == 'FALLEN':
            if tilt_deg < 15.0:
                # Being helped up or uprighted
                self.state = 'STAND'
                self.state_time = 0.0
                self.pitch_integral = 0.0
            else:
                # Relaxed safe mode on the ground
                return {
                    'motor_hip_roll_l': 0.0, 'motor_hip_roll_r': 0.0,
                    'motor_hip_l': 0.0, 'motor_hip_r': 0.0,
                    'motor_knee_l': 0.0, 'motor_knee_r': 0.0,
                    'motor_shoulder_l': 0.0, 'motor_shoulder_r': 0.0,
                    'state': 'FALLEN',
                    'balance_loss': float(self.smoothed_loss),
                    'k_pitch': float(self.k_balance_pitch)
                }

        # 2. Active Balance Reflex computation
        if learn:
            self.pitch_integral = np.clip(self.pitch_integral + pitch_angle * dt, -0.20, 0.20)
            
        ctrl_pitch = np.clip(
            - (self.k_balance_pitch * pitch_angle + self.d_balance_pitch * pitch_rate + 0.15 * self.pitch_integral),
            -0.35, 0.35
        )
        ctrl_roll = np.clip(
            + (self.k_balance_roll * roll_angle + self.d_balance_roll * roll_rate),
            -0.35, 0.35
        )
        
        # 3. Reflex disturbance recovery: if perturbed during walking, abort step into dual stance
        if self.state in ['STEP_L', 'STEP_R'] and tilt_deg > 9.0:
            self.state = 'STAND'
            self.state_time = 0.0

        # 4. Gait State Machine
        if self.state == 'STAND':
            hip_l = ctrl_pitch
            hip_r = ctrl_pitch
            roll_l = ctrl_roll
            roll_r = ctrl_roll
            knee_l = 0.0
            knee_r = 0.0
            arm_l = 0.0
            arm_r = 0.0
            
            # Start walking once standing is stable for 0.35s
            if self.state_time > 0.35 and tilt_deg < 3.5:
                self.state = 'STEP_L'
                self.state_time = 0.0
                self.step_phase = 0.0
                
        elif self.state == 'STEP_L':
            # Weight transfer to right foot, swing left foot forward
            self.step_phase = min(1.0, self.state_time / self.step_duration)
            s_curve = np.sin(self.step_phase * np.pi)
            
            # Sway right
            roll_l = ctrl_roll + self.sway_amp * s_curve
            roll_r = ctrl_roll + self.sway_amp * s_curve
            
            stride_eff = self.stride * (1.0 + 0.4 * steer)
            hip_l = ctrl_pitch + stride_eff * s_curve
            hip_r = ctrl_pitch - 0.4 * stride_eff * s_curve
            
            # Ground clearance knee bend
            knee_l = 0.12 * s_curve
            knee_r = 0.0
            
            arm_l = -0.20 * s_curve
            arm_r = 0.20 * s_curve
            
            # Transition to right step on foot contact or step completion
            if (self.step_phase >= 0.70 and touch_l > 25.0) or (self.state_time >= self.step_duration):
                self.state = 'STEP_R'
                self.state_time = 0.0
                self.step_phase = 0.0
                
        elif self.state == 'STEP_R':
            # Weight transfer to left foot, swing right foot forward
            self.step_phase = min(1.0, self.state_time / self.step_duration)
            s_curve = np.sin(self.step_phase * np.pi)
            
            # Sway left
            roll_l = ctrl_roll - self.sway_amp * s_curve
            roll_r = ctrl_roll - self.sway_amp * s_curve
            
            stride_eff = self.stride * (1.0 - 0.4 * steer)
            hip_r = ctrl_pitch + stride_eff * s_curve
            hip_l = ctrl_pitch - 0.4 * stride_eff * s_curve
            
            knee_r = 0.12 * s_curve
            knee_l = 0.0
            
            arm_r = -0.20 * s_curve
            arm_l = 0.20 * s_curve
            
            # Transition to left step on foot contact or step completion
            if (self.step_phase >= 0.70 and touch_r > 25.0) or (self.state_time >= self.step_duration):
                self.state = 'STEP_L'
                self.state_time = 0.0
                self.step_phase = 0.0
                
        else:
            hip_l = ctrl_pitch
            hip_r = ctrl_pitch
            roll_l = ctrl_roll
            roll_r = ctrl_roll
            knee_l = 0.0
            knee_r = 0.0
            arm_l = 0.0
            arm_r = 0.0

        return {
            'motor_hip_roll_l': float(roll_l),
            'motor_hip_roll_r': float(roll_r),
            'motor_hip_l': float(hip_l),
            'motor_hip_r': float(hip_r),
            'motor_knee_l': float(knee_l),
            'motor_knee_r': float(knee_r),
            'motor_shoulder_l': float(arm_l),
            'motor_shoulder_r': float(arm_r),
            'state': self.state,
            'balance_loss': float(self.smoothed_loss),
            'k_pitch': float(self.k_balance_pitch)
        }
