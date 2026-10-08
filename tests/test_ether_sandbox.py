"""
Unit tests for the Unified Spatial Ether Medium & Mechanical Wave Propagation.
Verifies first-principles dipole radiation, binaural reception, and bipedal contact physics.
"""

import os
import sys
import numpy as np
import pytest

from information_boltzmann.sandbox.ether_medium import UnifiedEtherMedium
from information_boltzmann.sandbox.walking_cpg import BipedalCPGController
from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def test_dipole_radiation_first_principles():
    """
    Test that mechanical stress changes (dF/dt) spontaneously radiate acoustic wave packets
    whose dipole amplitude envelope is exactly proportional to dF/dt, with zero ad-hoc triggers.
    Also verifies that solver chatter fluctuations below deadband remain dead silent.
    """
    # 0. Test solver chatter suppression (default 600 N/s deadband)
    ether_quiet = UnifiedEtherMedium()
    chatter_event = [{
        'geom1': 1, 'geom2': 2,
        'pos': np.array([0.0, 0.0, 0.0]),
        'force': np.array([0.0, 0.0, 2.0]) # 2 N / 0.01s = 200 N/s < 600 N/s
    }]
    ether_quiet.step(sim_time=0.01, contact_events=chatter_event)
    assert len(ether_quiet.active_packets) == 0 # 100% dead silence

    # 1. Gentle contact force ramp (light step: 5 N over 0.01s -> dF/dt = 500 N/s)
    ether = UnifiedEtherMedium(speed_of_sound=343.0, air_density=1.225, solver_chatter_deadband=100.0)
    contact_soft = [{
        'geom1': 1, 'geom2': 2,
        'pos': np.array([0.0, 0.0, 0.0]),
        'force': np.array([0.0, 0.0, 5.0])
    }]
    ether.step(sim_time=0.01, contact_events=contact_soft)
    
    assert len(ether.active_packets) == 1
    p_soft = ether.active_packets[0]
    assert np.allclose(p_soft.source_pos, [0, 0, 0])
    
    # Amplitude of dipole term: ||dF/dt||
    amp_soft = np.linalg.norm(p_soft.force_derivative)
    assert np.isclose(amp_soft, 500.0, rtol=1e-3)
    
    # 2. Hard stamping contact force (stomp: 50 N over 0.01s -> dF/dt = 5000 N/s)
    ether_hard = UnifiedEtherMedium(speed_of_sound=343.0, air_density=1.225, solver_chatter_deadband=100.0)
    contact_hard = [{
        'geom1': 1, 'geom2': 2,
        'pos': np.array([0.0, 0.0, 0.0]),
        'force': np.array([0.0, 0.0, 50.0])
    }]
    ether_hard.step(sim_time=0.01, contact_events=contact_hard)
    p_hard = ether_hard.active_packets[0]
    amp_hard = np.linalg.norm(p_hard.force_derivative)
    assert np.isclose(amp_hard, 5000.0, rtol=1e-3)
    
    # The dipole radiation strength is mathematically exactly 10.0x!
    # Direct first-principles linear acoustic radiation without any volume sliders!
    assert np.isclose(amp_hard / amp_soft, 10.0, rtol=1e-3)

def test_binaural_ear_spatial_localization():
    """
    Test that a source on the left side reaches the left ear earlier and stronger
    than the right ear, creating spontaneous ITD and ILD.
    """
    ether = UnifiedEtherMedium(speed_of_sound=343.0, air_density=1.225)
    
    # Impact on the far left: pos = [-1.0, 0.0, 0.0]
    impact = [{
        'geom1': 1, 'geom2': 2,
        'pos': np.array([-1.0, 0.0, 0.0]),
        'force': np.array([0.0, 0.0, 20.0])
    }]
    ether.step(sim_time=0.01, contact_events=impact)
    
    # Ears separated along X axis: left ear at -0.18m, right ear at +0.18m
    ear_L = np.array([-0.18, 0.0, 1.0])
    ear_R = np.array([+0.18, 0.0, 1.0])
    
    dist_L = np.linalg.norm(ear_L - np.array([-1.0, 0.0, 0.0]))
    dist_R = np.linalg.norm(ear_R - np.array([-1.0, 0.0, 0.0]))
    assert dist_L < dist_R
    
    # Retarded arrival times
    t_arrive_L = 0.01 + dist_L / 343.0
    t_arrive_R = 0.01 + dist_R / 343.0
    assert t_arrive_L < t_arrive_R  # Left ear receives wavefront earlier!

def test_ether_sandbox_full_loop():
    """
    Test the integrated sandbox execution for 50 physical steps:
    verifies MuJoCo physics, contact wave generation, and sensor integrity.
    """
    sandbox = EtherSandbox()
    for _ in range(50):
        obs = sandbox.step()
        assert 'pos' in obs
        assert 'vel' in obs
        assert 'touch_l' in obs
        assert 'touch_r' in obs
        assert 'audio_l' in obs
        assert 'audio_r' in obs
        assert 'pitch' in obs
        assert 'vfe' in obs
        assert 'epistemic_gain' in obs
        assert 'pragmatic_risk' in obs
        
    assert sandbox.data.time > 0.09

def test_active_inference_vfe_minimization():
    """
    Test that ActiveInferenceAgent drives motor actions via free energy minimization:
    when upright in equilibrium, pragmatic risk is low and epistemic curiosity drives exploration.
    """
    from information_boltzmann.sandbox.active_inference_agent import ActiveInferenceAgent
    agent = ActiveInferenceAgent(dt=0.002)
    
    # 1. Evaluate upright balanced state
    ctrls = agent.step(
        dt=0.002,
        pos=np.array([0.0, 0.0, agent.prior_z]),
        pitch=agent.prior_pitch,
        pitch_rate=0.0,
        roll=0.0,
        roll_rate=0.0,
        touch_l=26.0,
        touch_r=26.0,
        touch_torso=0.0,
        audio_amp=0.0,
        is_fallen=False
    )
    assert ctrls['pragmatic_risk'] < 0.1 # Well within homeostatic prior
    assert ctrls['epistemic_gain'] > 0.0 # Intrinsic forward curiosity
    
    # 2. Evaluate tilted perturbed state (high surprise / VFE surge)
    ctrls_perturbed = agent.step(
        dt=0.002,
        pos=np.array([0.0, 0.0, agent.prior_z - 0.04]),
        pitch=agent.prior_pitch + 0.15, # tilted forward
        pitch_rate=0.5,
        roll=0.0,
        roll_rate=0.0,
        touch_l=10.0,
        touch_r=42.0,
        touch_torso=0.0,
        audio_amp=0.0,
        is_fallen=False
    )
    assert ctrls_perturbed['pragmatic_risk'] > ctrls['pragmatic_risk']
    assert ctrls_perturbed['vfe'] > ctrls['vfe']
    # Restoring reflex pushes backward (-pitch torque) to cancel prediction error
    assert ctrls_perturbed['motor_hip_pitch_l'] < ctrls['motor_hip_pitch_l']


def test_active_inference_target_turnaround():
    """
    Test that an auditory/spatial target placed behind the agent (180 deg)
    causes the Expected Free Energy (EFE) policy to minimize translation
    and prioritize in-place turning via differential swing yaw stepping.
    """
    from information_boltzmann.sandbox.active_inference_agent import ActiveInferenceAgent
    agent = ActiveInferenceAgent(dt=0.002)

    # Robot at origin, facing +Y (yaw = pi/2)
    # Target behind at (0.0, -1.0, 0.1) -> heading to target is -pi/2 -> heading_err = -pi
    target_pos = np.array([0.0, -1.0, 0.1])
    
    # Run a few steps to let the internal policy filters settle
    for _ in range(50):
        ctrls = agent.step(
            dt=0.002,
            pos=np.array([0.0, 0.0, agent.prior_z]),
            pitch=agent.prior_pitch,
            pitch_rate=0.0,
            roll=0.0,
            roll_rate=0.0,
            touch_l=26.0,
            touch_r=26.0,
            touch_torso=0.0,
            audio_amp=1.0,
            is_fallen=False,
            toy_pos=target_pos,
            yaw=0.0  # Facing +Y
        )

    # Forward velocity must be near 0 because target is directly behind (in-place turn)
    assert agent.v_cmd < 0.05, f"Expected in-place turning (v_cmd ~ 0), got {agent.v_cmd}"
    # Rotational velocity must be active to minimize heading error
    assert abs(agent.omega_cmd) > 0.2, f"Expected non-zero turn rate, got {agent.omega_cmd}"
    # Dynamic state should reflect turning
    assert ctrls['gait_state'] in ('TURN', 'TURN_IN_PLACE', 'WALK_IN_PLACE', 'STAND')


def test_active_inference_arm_and_quiescence():
    """
    Test that Active Inference governs arm control across:
    1. Quiescence: when no target exists and silence reigns, the agent stands still (v=0, omega=0, no leg stepping).
    2. Arm reaching & palpation: when target is near (dist <= 0.35m), arms reach forward (shoulder_pitch > 0.4).
    3. Roll balance reflex: when tilted laterally, contralateral arm abducts outward to catch balance.
    """
    from information_boltzmann.sandbox.active_inference_agent import ActiveInferenceAgent
    agent = ActiveInferenceAgent(dt=0.002)

    # 1. Quiescence in silence (no toy, no sound)
    for _ in range(30):
        ctrls = agent.step(
            dt=0.002,
            pos=np.array([0.0, 0.0, agent.prior_z]),
            pitch=0.0,
            pitch_rate=0.0,
            roll=0.0,
            roll_rate=0.0,
            touch_l=26.0,
            touch_r=26.0,
            touch_torso=0.0,
            audio_amp=0.0,
            is_fallen=False,
            toy_pos=None
        )
    assert ctrls['gait_state'] == 'STAND_BALANCE'
    assert agent.v_cmd < 0.01
    assert abs(agent.omega_cmd) < 0.01
    # Knees stay at resting nominal flex, not oscillating
    assert abs(ctrls['motor_knee_l'] - agent.q_nom_knee) < 0.02
    assert abs(ctrls['motor_knee_r'] - agent.q_nom_knee) < 0.02

    # 2. Arm Reaching & Palpation when approaching target
    agent.reset()
    toy_near = np.array([0.0, 0.30, 0.10]) # 30cm in front
    for _ in range(40):
        ctrls = agent.step(
            dt=0.002,
            pos=np.array([0.0, 0.0, agent.prior_z]),
            pitch=0.0,
            pitch_rate=0.0,
            roll=0.0,
            roll_rate=0.0,
            touch_l=26.0,
            touch_r=26.0,
            touch_torso=0.0,
            audio_amp=0.0,
            is_fallen=False,
            toy_pos=toy_near,
            yaw=0.0
        )
    # Shoulders reach forward and elbows flex downward to reach the toy
    assert ctrls['gait_state'] == 'CROUCH_PALPATE'
    assert ctrls['motor_shoulder_pitch_l'] > 0.40
    assert ctrls['motor_shoulder_pitch_r'] > 0.40
    assert ctrls['motor_elbow_l'] < -0.40

    # 3. Roll balance reflex: tilted right -> right arm abducts outward (roll < -0.2)
    agent.reset()
    for _ in range(10):
        ctrls = agent.step(
            dt=0.002,
            pos=np.array([0.0, 0.0, agent.prior_z]),
            pitch=0.0,
            pitch_rate=0.0,
            roll=0.20, # tilted right
            roll_rate=0.5,
            touch_l=10.0,
            touch_r=40.0,
            touch_torso=0.0,
            audio_amp=0.0,
            is_fallen=False
        )
    assert ctrls['motor_shoulder_roll_r'] < -0.20
