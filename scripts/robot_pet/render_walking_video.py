"""
Render a 3D animated GIF of the robot pet walking and exploring in the Ether Sandbox.
Uses MuJoCo's offscreen renderer to capture smooth 20 FPS video with camera tracking.
"""

import os
import sys
import numpy as np
from PIL import Image
import mujoco

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def render_walking_video(output_gif: str = "present/robot_pet/walking_live.gif", duration_sec: float = 3.2, fps: int = 20):
    print(f"Creating Ether Sandbox for visual recording...")
    sandbox = EtherSandbox()
    
    # 540x540 offscreen renderer
    renderer = mujoco.Renderer(sandbox.model, height=540, width=540)
    
    # Camera setup: smooth third-person tracking camera
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.4
    camera.elevation = -18.0
    camera.azimuth = 145.0
    
    dt = sandbox.dt
    total_steps = int(duration_sec / dt)
    steps_per_frame = int(1.0 / (fps * dt))
    
    pil_frames = []
    print(f"Simulating & Rendering {duration_sec}s of locomotion ({fps} FPS, {total_steps} steps)...")
    
    frame_count = 0
    for step in range(total_steps):
        obs = sandbox.step()
        
        if step % steps_per_frame == 0:
            # Camera tracks robot smoothly
            camera.lookat = [obs['pos'][0], obs['pos'][1] + 0.15, obs['pos'][2] + 0.1]
            
            renderer.update_scene(sandbox.data, camera=camera)
            pixels = renderer.render()
            pil_frames.append(Image.fromarray(pixels))
            frame_count += 1
            if frame_count % 15 == 0:
                print(f"  Captured Frame {frame_count}/{(int(duration_sec * fps))} (y={obs['pos'][1]:.2f}m)...")
                
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    print(f"Saving animated GIF ({len(pil_frames)} frames) to {output_gif}...")
    duration_ms = int(1000.0 / fps)
    pil_frames[0].save(
        output_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    print(f"Animated GIF saved successfully: {output_gif}")
    return output_gif

if __name__ == "__main__":
    render_walking_video()
