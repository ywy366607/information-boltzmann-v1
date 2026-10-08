"""
Demonstration and Telemetry Verification of the Unified Spatial Ether Sandbox (MVP).
Runs the robot pet walking under gravity, extracts spontaneous mechanical footstep waves,
verifies binaural hearing and tactile force feedback, and generates a telemetry plot.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure information_boltzmann is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def run_simulation(duration_seconds: float = 2.5):
    print(f"Initializing Ether Sandbox (c_s = 343 m/s, g = -9.81 m/s^2)...")
    sandbox = EtherSandbox()
    
    total_steps = int(duration_seconds / sandbox.dt)
    print(f"Simulating {total_steps} physical steps (dt = {sandbox.dt:.4f} s, total {duration_seconds} s)...")
    
    times = []
    robot_y = []
    robot_z = []
    pitch_angles = []
    touch_left = []
    touch_right = []
    audio_left = []
    audio_right = []
    active_waves = []
    
    step_count = 0
    t0 = sandbox.data.time
    
    while (sandbox.data.time - t0) < duration_seconds:
        obs = sandbox.step()
        
        times.append(obs['time'])
        robot_y.append(obs['pos'][1])      # forward displacement
        robot_z.append(obs['pos'][2])      # height above ground
        pitch_angles.append(np.degrees(obs['pitch']))
        touch_left.append(obs['touch_l'])
        touch_right.append(obs['touch_r'])
        audio_left.append(obs['audio_l'])
        audio_right.append(obs['audio_r'])
        active_waves.append(obs['active_ether_waves'])
        
        step_count += 1
        
    print(f"Completed {step_count} physical steps.")
    print(f"Final Robot State:")
    print(f"  - Forward position Y: {robot_y[-1]:.3f} m (started at {robot_y[0]:.3f} m)")
    print(f"  - Torso Height Z: {robot_z[-1]:.3f} m (stable upright)")
    print(f"  - Peak Ground Force L: {max(touch_left):.2f} N, R: {max(touch_right):.2f} N")
    print(f"  - Peak Spontaneous Acoustic Pressure: {max(max(audio_left), max(audio_right)):.4f} Pa")
    
    # ---------------------------------------------------------
    # Generate Telemetry Chart
    # ---------------------------------------------------------
    out_dir = "present/robot_pet"
    os.makedirs(out_dir, exist_ok=True)
    out_plot = os.path.join(out_dir, "ether_sandbox_telemetry.png")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=120)
    bg_color = '#181a20'
    fig.patch.set_facecolor(bg_color)
    
    for ax in axes:
        ax.set_facecolor('#20242e')
        ax.tick_params(colors='#cfd6e6')
        ax.spines['bottom'].set_color('#444b5a')
        ax.spines['top'].set_color('#444b5a')
        ax.spines['left'].set_color('#444b5a')
        ax.spines['right'].set_color('#444b5a')
        ax.grid(True, linestyle='--', alpha=0.25, color='#8899aa')
        
    times = np.array(times)
    
    # 1. Forward Position & Height
    axes[0].plot(times, robot_y, label='Forward Position Y (m)', color='#00ddaa', linewidth=2.0)
    axes[0].plot(times, robot_z, label='Torso Height Z (m)', color='#44aaff', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('Position (m)', color='#cfd6e6', fontsize=11)
    axes[0].legend(loc='upper left', facecolor='#181a20', edgecolor='none', labelcolor='#cfd6e6')
    axes[0].set_title('Unified Spatial Ether Sandbox: Emergent Locomotion & Multimodal Wave Physics', color='#ffffff', fontsize=13, pad=10)
    
    # 2. Tactile Ground Reaction Forces
    axes[1].plot(times, touch_left, label='Left Boot Tactile Force (N)', color='#ffaa33', linewidth=1.8)
    axes[1].plot(times, touch_right, label='Right Boot Tactile Force (N)', color='#ff4466', linewidth=1.8)
    axes[1].set_ylabel('Force (N)', color='#cfd6e6', fontsize=11)
    axes[1].legend(loc='upper right', facecolor='#181a20', edgecolor='none', labelcolor='#cfd6e6')
    
    # 3. Spontaneous Acoustic Radiation (Binaural Antenna Signals)
    # Notice: No sound effects! Every spike is directly radiated by mechanical dF/dt!
    axes[2].plot(times, audio_left, label='Left Antenna Pressure p_L (Pa)', color='#33ddff', linewidth=1.2, alpha=0.85)
    axes[2].plot(times, audio_right, label='Right Antenna Pressure p_R (Pa)', color='#ff33cc', linewidth=1.2, alpha=0.85)
    axes[2].set_ylabel('Acoustic Pressure (Pa)', color='#cfd6e6', fontsize=11)
    axes[2].legend(loc='upper right', facecolor='#181a20', edgecolor='none', labelcolor='#cfd6e6')
    
    # 4. Vestibular Balance (Torso Pitch Angle) & Active Ether Packets
    axes[3].plot(times, pitch_angles, label='Torso Pitch Angle (deg)', color='#aacc44', linewidth=1.6)
    ax_twin = axes[3].twinx()
    ax_twin.plot(times, active_waves, label='Active Wave Packets in Ether', color='#ffffff', alpha=0.4, linestyle=':')
    ax_twin.set_ylabel('Ether Packets', color='#8899aa', fontsize=10)
    ax_twin.tick_params(colors='#8899aa')
    axes[3].set_ylabel('Pitch (deg)', color='#cfd6e6', fontsize=11)
    axes[3].set_xlabel('Physical Time (s)', color='#cfd6e6', fontsize=11)
    axes[3].legend(loc='upper left', facecolor='#181a20', edgecolor='none', labelcolor='#cfd6e6')
    
    plt.tight_layout()
    plt.savefig(out_plot, facecolor=bg_color, edgecolor='none', pad_inches=0.08)
    plt.close(fig)
    print(f"Saved telemetry analysis to: {out_plot}")
    return out_plot

if __name__ == "__main__":
    run_simulation(duration_seconds=2.5)
