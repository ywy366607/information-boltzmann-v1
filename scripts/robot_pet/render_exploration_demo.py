"""
Render an animated GIF with Real-Time HUD Telemetry of the Robot Pet exploring the Ether Sandbox.
Visualizes:
1. Full 3D bipedal locomotion and arena exploration.
2. Dynamic screen facial expression token actions (Default ||, Happy ^^, Surprised OO, etc.).
3. Real-time Ether Medium binaural acoustic pressure and tactile ground reactions.
4. Active inference balance learning loss curve.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def draw_hud(frame: Image.Image, obs: dict, step: int, total_steps: int) -> Image.Image:
    """Overlays a clean cyberpunk / robotics telemetry HUD onto the rendered frame."""
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    
    # 1. Top HUD Bar (dark semi-transparent banner)
    top_bar = Image.new("RGBA", (w, 54), (15, 20, 30, 200))
    frame.paste(top_bar, (0, 0), top_bar)
    
    # 2. Bottom HUD Bar
    bot_bar = Image.new("RGBA", (w, 60), (15, 20, 30, 200))
    frame.paste(bot_bar, (0, h - 60), bot_bar)
    
    # Text info
    time_str = f"T: {obs['time']:4.2f}s"
    pos_str = f"POS: ({obs['pos'][0]:+4.2f}, {obs['pos'][1]:+4.2f})m"
    spd_str = f"SPD: {obs['speed']:4.2f}m/s"
    
    expr_label = obs['expression_label']
    loss_val = obs['balance_loss']
    
    # Color badge for expression
    badge_colors = {
        "||": (80, 160, 240),
        "^^": (50, 220, 120),
        "--": (180, 180, 180),
        "OO": (255, 90, 80),
        "?o": (240, 200, 60)
    }
    badge_col = (100, 200, 255)
    for key, col in badge_colors.items():
        if key in expr_label:
            badge_col = col
            break
            
    # Draw top labels
    draw.text((12, 8), "ROBOT PET | ETHER SANDBOX", fill=(200, 220, 255))
    draw.text((12, 28), f"{time_str}  {pos_str}  {spd_str}", fill=(160, 180, 200))
    
    # Expression pill badge
    expr_text = f"EXPR: {expr_label}"
    draw.rounded_rectangle([(w - 180, 10), (w - 12, 42)], radius=6, fill=(30, 40, 55), outline=badge_col, width=2)
    draw.text((w - 168, 18), expr_text, fill=badge_col)
    
    # Bottom HUD
    # Left: Ether Binaural Acoustic Waves
    p_L = obs['audio_l']
    p_R = obs['audio_r']
    waves_count = obs['active_ether_waves']
    draw.text((12, h - 52), f"ETHER ACOUSTIC DIPOLE WAVES ({waves_count} packets)", fill=(120, 200, 255))
    draw.text((12, h - 32), f"Mic L: {p_L:+6.1f} Pa  |  Mic R: {p_R:+6.1f} Pa", fill=(180, 210, 240))
    
    # Right: Balance Loss & Tactile
    tilt_deg = np.degrees(obs['pitch'])
    draw.text((w - 240, h - 52), f"BALANCE ADAPTATION", fill=(255, 200, 100))
    draw.text((w - 240, h - 32), f"Loss: {loss_val:6.4f}  |  Tilt: {tilt_deg:+4.1f}°", fill=(220, 220, 180))
    
    return frame

def render_exploration_demo(
    output_gif: str = "present/robot_pet/exploration_live.gif",
    duration_sec: float = 3.6,
    fps: int = 20
):
    print("=" * 72)
    print(f"🎬 Initializing Ether Sandbox Exploration Video Generator...")
    print(f"   Target: {output_gif} ({duration_sec}s @ {fps} FPS)")
    print("=" * 72)
    
    sandbox = EtherSandbox()
    
    # 540x540 offscreen renderer
    renderer = mujoco.Renderer(sandbox.model, height=540, width=540)
    
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.4
    camera.elevation = -14.0
    camera.azimuth = -45.0
    
    dt = sandbox.dt
    total_steps = int(duration_sec / dt)
    steps_per_frame = int(1.0 / (fps * dt))
    
    frames = []
    cam_lookat = np.array([0.0, 0.0, 0.74])
    
    print(f"Simulating physical universe & rendering HUD frames ({total_steps} microsteps)...")
    
    for step in range(total_steps):
        obs = sandbox.step()
        
        if step % steps_per_frame == 0:
            target_lookat = np.array([obs['pos'][0], obs['pos'][1], obs['pos'][2] + 0.35])
            cam_lookat = 0.90 * cam_lookat + 0.10 * target_lookat
            camera.lookat = cam_lookat.tolist()
            
            renderer.update_scene(sandbox.data, camera=camera)
            raw_rgb = renderer.render()
            pil_img = Image.fromarray(raw_rgb)
            
            # Overlay HUD
            hud_img = draw_hud(pil_img, obs, step, total_steps)
            frames.append(hud_img)
            
            frame_idx = len(frames)
            if frame_idx % 15 == 0 or frame_idx == int(duration_sec * fps):
                print(f"  Captured Frame {frame_idx:2d}/{(int(duration_sec * fps))} | "
                      f"Pos: ({obs['pos'][0]:+4.2f}, {obs['pos'][1]:+4.2f})m | "
                      f"Expr: [{obs['expression_label']}] | Loss: {obs['balance_loss']:.4f}")
                
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    print(f"\nCompressing & saving {len(frames)} frames into animated GIF: {output_gif}...")
    
    duration_ms = int(1000.0 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    print(f"✅ Exploration GIF saved successfully: {output_gif}")
    return output_gif

if __name__ == "__main__":
    render_exploration_demo()
