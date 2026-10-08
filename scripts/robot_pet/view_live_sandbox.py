"""
Interactive 3D Live Desktop Viewer for the Robot Pet Ether Sandbox.
Launches a native 60 FPS GPU-accelerated 3D window (using MuJoCo's native GLFW viewer).

Features:
- Real-time physics & Ether wave propagation.
- Autonomous walking, balance adaptation, arena exploration, and facial expressions.
- Mouse interactive controls:
  * Left Click + Drag: Orbit camera
  * Right Click + Drag: Pan camera
  * Scroll Wheel: Zoom in / out
  * Ctrl + Right Click Drag: Apply physical perturbation force to the robot!
- Real-time telemetry printed to the console.

Usage:
  d:\\conda_envs\\vox\\python.exe scripts/robot_pet/view_live_sandbox.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def run_live_viewer():
    print("=" * 72)
    print("🤖 Launching Robot Pet Ether Sandbox 3D Live Viewer...")
    print("Controls:")
    print("  - Left-Click + Drag:       Orbit camera around robot")
    print("  - Right-Click + Drag:      Pan camera")
    print("  - Scroll Wheel:            Zoom in / out")
    print("  - Ctrl + Right-Click Drag: Apply physical push force (Perturbation reflex)")
    print("  - Spacebar:                Pause / Resume physics")
    print("=" * 72)

    sandbox = EtherSandbox()
    dt = sandbox.dt # 0.002s

    with mujoco.viewer.launch_passive(sandbox.model, sandbox.data) as viewer:
        # Camera configuration
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -14.0
        viewer.cam.azimuth = -45.0
        
        step_idx = 0
        last_print_time = time.time()
        
        while viewer.is_running():
            step_start = time.perf_counter()
            
            # Step the full unified physical universe & Ether medium
            obs = sandbox.step()
            step_idx += 1
            
            # Smooth third-person camera tracking
            viewer.cam.lookat[0] = 0.92 * viewer.cam.lookat[0] + 0.08 * obs['pos'][0]
            viewer.cam.lookat[1] = 0.92 * viewer.cam.lookat[1] + 0.08 * obs['pos'][1]
            viewer.cam.lookat[2] = 0.92 * viewer.cam.lookat[2] + 0.08 * (obs['pos'][2] + 0.35)
            
            # Sync graphics at ~60 FPS (every 8 physical steps = 0.016s)
            if step_idx % 8 == 0:
                viewer.sync()
                
            # Print live telemetry to console once every 0.35 seconds
            now = time.time()
            if now - last_print_time > 0.35:
                last_print_time = now
                pos = obs['pos']
                speed = obs['speed']
                pitch_deg = np.degrees(obs['pitch'])
                roll_deg = np.degrees(obs['roll'])
                loss = obs['balance_loss']
                expr = obs['expression_label']
                audio = obs['audio_amp']
                contacts = obs['num_contacts']
                waves = obs['active_ether_waves']
                
                status_line = (
                    f"⏱️ {obs['time']:5.2f}s | "
                    f"📍 Pos: ({pos[0]:+5.2f}, {pos[1]:+5.2f})m | "
                    f"⚡ Spd: {speed:4.2f}m/s | "
                    f"📐 Tilt: {pitch_deg:+4.1f}° | "
                    f"📉 Loss: {loss:6.4f} | "
                    f"🎭 Expr: [{expr}] | "
                    f"🌊 Ether: {audio:5.1f}Pa ({waves} waves)"
                )
                print(status_line, end="\r", flush=True)
                
            # Real-time pacing
            elapsed = time.perf_counter() - step_start
            time_to_wait = dt - elapsed
            if time_to_wait > 0.0005:
                time.sleep(time_to_wait)

    print("\nViewer closed. Session ended.")

if __name__ == "__main__":
    run_live_viewer()
