"""
Real-Time HTML5 Streaming Server for Robot Pet Ether Sandbox.
Delivers:
1. Real-time Video Stream (MJPEG 25-30 FPS) with switchable Third-Person and First-Person Eyes Camera.
2. Real-time Binaural Stereo Audio Stream (WebSocket -> Web Audio API) carrying mechanical wave radiation.
3. Interactive Caretaker Actions ("扶它起来" / Help Up, "推一下" / Push, "摸摸头" / Pet).
4. Continuous embodied life-long learning telemetry and screen expression diagnostics.

Usage:
  d:\\conda_envs\\vox\\python.exe scripts/robot_pet/web_stream_server.py --port 8080
"""

import os
import sys
import time
import io
import math
import json
import threading
import argparse
import numpy as np
from PIL import Image
import mujoco

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from information_boltzmann.sandbox.ether_sandbox import EtherSandbox
from information_boltzmann.sandbox.expression_system import ExpressionToken

# Physical Universe Streamer will be initialized via lifespan

class PhysicalUniverseStreamer:
    """Manages continuous physics, Ether wave propagation, rendering, and audio capture."""
    def __init__(self):
        print("⚡ Initializing Continuous Physical Sandbox & Ether Medium...")
        self.sandbox = EtherSandbox()
        self.dt = self.sandbox.dt # 0.002s (500 Hz)
        
        # Renderers (Third-person Free Camera & First-person Screen Eyes Camera)
        self.width = 540
        self.height = 420
        
        # Camera 1: Third-person smooth tracking camera
        self.cam_third = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam_third)
        self.cam_third.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_third.distance = 2.4
        self.cam_third.elevation = -14.0
        self.cam_third.azimuth = -45.0
        self.cam_lookat = np.array([0.0, 0.0, 0.74])
        
        # Active view mode: 'third_person' or 'first_person'
        self.view_mode = "third_person"
        
        # Latest frame JPEG bytes
        self.latest_jpeg = b""
        self.frame_lock = threading.Lock()
        
        # Audio buffer: 8000 Hz stereo PCM (Float32)
        # 500 Hz physics -> 16 audio samples per physics step
        self.audio_subsamples = 16
        self.audio_sample_rate = 500 * self.audio_subsamples # 8000 Hz
        self.audio_queue = []
        self.audio_lock = threading.Lock()
        
        # Latest telemetry
        self.latest_obs = {}
        self.telemetry_lock = threading.Lock()
        
        # Life timer
        self.start_wall_time = time.time()
        self.running = True
        
        # Start physics and rendering daemon threads
        self.sim_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.sim_thread.start()
        
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

    def _physics_loop(self):
        """High-frequency continuous 500 Hz physics & Ether wave propagation loop."""
        p_L_prev = 0.0
        p_R_prev = 0.0
        
        while self.running:
            step_start = time.perf_counter()
            try:
                obs = self.sandbox.step()
            except Exception as e:
                import traceback
                print(f"⚠️ Exception in physics step: {e}")
                traceback.print_exc()
                time.sleep(0.01)
                continue
            
            with self.telemetry_lock:
                self.latest_obs = obs
                
            # Synthesize 8000 Hz stereo audio by linear interpolation of acoustic pressure
            p_L = obs['audio_l']
            p_R = obs['audio_r']
            
            # Sub-sample buffer for continuous acoustic wave rendering
            # Normalization scale: 50 Pa peak ~ 0.8 audio gain
            scale = 0.8 / 50.0
            t_interp = np.linspace(0.0, 1.0, self.audio_subsamples, endpoint=False)
            sub_L = (p_L_prev + t_interp * (p_L - p_L_prev)) * scale
            sub_R = (p_R_prev + t_interp * (p_R - p_R_prev)) * scale
            p_L_prev = p_L
            p_R_prev = p_R
            
            # Clamp to prevent clipping [-1.0, 1.0]
            sub_L = np.clip(sub_L, -1.0, 1.0)
            sub_R = np.clip(sub_R, -1.0, 1.0)
            
            # Interleave stereo [L0, R0, L1, R1, ...]
            stereo = np.empty((self.audio_subsamples * 2,), dtype=np.float32)
            stereo[0::2] = sub_L
            stereo[1::2] = sub_R
            
            with self.audio_lock:
                self.audio_queue.append(stereo.tobytes())
                # Keep max 400ms buffer to prevent memory lag
                if len(self.audio_queue) > 200:
                    self.audio_queue.pop(0)
                    
            # Maintain 500 Hz pacing (0.002s)
            elapsed = time.perf_counter() - step_start
            wait_s = self.dt - elapsed
            if wait_s > 0.0003:
                time.sleep(wait_s)

    def _render_loop(self):
        """25-30 FPS rendering loop for video stream."""
        # Initialize Renderer in this worker thread to bind WGL OpenGL context properly
        renderer = mujoco.Renderer(self.sandbox.model, height=self.height, width=self.width)
        target_fps = 25.0
        frame_interval = 1.0 / target_fps
        
        while self.running:
            t0 = time.perf_counter()
            
            with self.telemetry_lock:
                pos = self.latest_obs.get('pos', np.array([0.0, 0.0, 0.74]))
                
            # Upload dynamic digital visor texture to GPU VRAM
            self.sandbox.expressions.upload_to_renderer(renderer._mjr_context)
            
            if self.view_mode == "third_person":
                target_lookat = np.array([pos[0], pos[1], pos[2] + 0.35])
                self.cam_lookat = 0.90 * self.cam_lookat + 0.10 * target_lookat
                self.cam_third.lookat = self.cam_lookat.tolist()
                renderer.update_scene(self.sandbox.data, camera=self.cam_third)
            else:
                # First-person screen visor eye camera
                renderer.update_scene(self.sandbox.data, camera=self.sandbox.cam_eyes_id)
                
            rgb = renderer.render()
            pil_img = Image.fromarray(rgb)
            
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=82)
            jpeg_bytes = buf.getvalue()
            
            with self.frame_lock:
                self.latest_jpeg = jpeg_bytes
                
            elapsed = time.perf_counter() - t0
            rem = frame_interval - elapsed
            if rem > 0.002:
                time.sleep(rem)

    def get_latest_jpeg(self) -> bytes:
        with self.frame_lock:
            return self.latest_jpeg

    def pop_audio_chunks(self) -> bytes:
        with self.audio_lock:
            if not self.audio_queue:
                return b""
            chunks = b"".join(self.audio_queue)
            self.audio_queue.clear()
            return chunks

    def help_up(self):
        """Caretaker lifts robot back to feet."""
        self.sandbox.help_up()

    def push(self, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0, duration: float = 0.15):
        """Caretaker applies push."""
        self.sandbox.push(fx=fx, fy=fy, fz=fz, duration=duration)

    def throw_toy(self, x: float = 0.0, y: float = 1.2, z: float = 0.15, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        """Caretaker throws or places the interactive curiosity ball."""
        self.sandbox.throw_toy(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    def chirp_toy(self):
        """Generates an acoustic chirp wave packet from the curiosity ball."""
        toy_pos = self.sandbox.data.qpos[27:30].copy() if self.sandbox.model.nq > 27 else np.array([0.0, 1.2, 0.10])
        from information_boltzmann.sandbox.ether_medium import MechanicalWavePacket
        packet = MechanicalWavePacket(
            source_pos=toy_pos,
            birth_time=float(self.sandbox.data.time),
            force_derivative=np.array([0.0, 0.0, 35000.0], dtype=np.float64),
            monopole_strength=0.06,
            duration=0.05,
            base_freq=440.0
        )
        self.sandbox.ether.active_packets.append(packet)

    def toggle_view(self) -> str:
        if self.view_mode == "third_person":
            self.view_mode = "first_person"
        else:
            self.view_mode = "third_person"
        return self.view_mode

from contextlib import asynccontextmanager

streamer = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global streamer
    streamer = PhysicalUniverseStreamer()
    yield
    if streamer:
        streamer.running = False

app = FastAPI(title="Robot Pet Ether Life Stream", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

@app.get("/video_feed")
def video_feed():
    def frame_generator():
        while True:
            if streamer is not None:
                jpeg = streamer.get_latest_jpeg()
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.035)
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if streamer is not None:
                audio_bytes = streamer.pop_audio_chunks()
                if audio_bytes:
                    await websocket.send_bytes(audio_bytes)
            await asyncio_sleep(0.03)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if streamer is not None:
                with streamer.telemetry_lock:
                    obs = streamer.latest_obs.copy()
                if obs:
                    telemetry_payload = {
                        "time": float(obs.get("time", 0.0)),
                        "pos_x": float(obs["pos"][0]),
                        "pos_y": float(obs["pos"][1]),
                        "pos_z": float(obs["pos"][2]),
                        "speed": float(obs.get("speed", 0.0)),
                        "pitch_deg": float(np.degrees(obs.get("pitch", 0.0))),
                        "roll_deg": float(np.degrees(obs.get("roll", 0.0))),
                        "touch_l": float(obs.get("touch_l", 0.0)),
                        "touch_r": float(obs.get("touch_r", 0.0)),
                        "touch_torso": float(obs.get("touch_torso", 0.0)),
                        "is_fallen": bool(obs.get("is_fallen", False)),
                        "audio_l": float(obs.get("audio_l", 0.0)),
                        "audio_r": float(obs.get("audio_r", 0.0)),
                        "audio_amp": float(obs.get("audio_amp", 0.0)),
                        "expression": str(obs.get("expression_label", "||")),
                        "primary_emotion": str(obs.get("primary_emotion", "JOY")),
                        "sub_emotion": str(obs.get("sub_emotion", "CHEERFUL")),
                        "theme_style": str(obs.get("theme_style", "CYBER_CHIBI")),
                        "valence": float(obs.get("valence", 0.65)),
                        "arousal": float(obs.get("arousal", 0.40)),
                        "vfe": float(obs.get("vfe", 0.0)),
                        "epistemic_gain": float(obs.get("epistemic_gain", 0.0)),
                        "tactile_curiosity": float(obs.get("tactile_curiosity", 0.0)),
                        "acoustic_curiosity": float(obs.get("acoustic_curiosity", 0.0)),
                        "spatial_curiosity": float(obs.get("spatial_curiosity", 0.0)),
                        "pragmatic_risk": float(obs.get("pragmatic_risk", 0.0)),
                        "prediction_error": float(obs.get("prediction_error", 0.0)),
                        "gait_state": str(obs.get("gait_state", "STAND")),
                        "postural_mode": str(obs.get("postural_mode", "BIPED_EXPLORE")),
                        "toy_x": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[0]),
                        "toy_y": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[1]),
                        "toy_z": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[2]),
                        "touched_toy": bool(obs.get("touched_toy", False)),
                        "waves": int(obs.get("active_ether_waves", 0)),
                        "view_mode": streamer.view_mode
                    }
                    await websocket.send_text(json.dumps(telemetry_payload))
            await asyncio_sleep(0.05) # 20 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

import asyncio
async def asyncio_sleep(secs):
    await asyncio.sleep(secs)

@app.get("/api/telemetry")
def get_telemetry():
    if streamer is None:
        return {"status": "starting"}
    with streamer.telemetry_lock:
        obs = streamer.latest_obs.copy()
    if not obs:
        return {"status": "waiting_first_frame"}
    return {
        "time": float(obs.get("time", 0.0)),
        "x": float(obs.get("pos", [0, 0, 0])[0]),
        "y": float(obs.get("pos", [0, 0, 0])[1]),
        "z": float(obs.get("pos", [0, 0, 0])[2]),
        "yaw": float(obs.get("yaw", 0.0)),
        "pitch": float(obs.get("pitch", 0.0)),
        "roll": float(obs.get("roll", 0.0)),
        "vfe": float(obs.get("vfe", 0.0)),
        "epistemic_gain": float(obs.get("epistemic_gain", 0.0)),
        "pragmatic_risk": float(obs.get("pragmatic_risk", 0.0)),
        "gait_state": str(obs.get("gait_state", "UNKNOWN")),
        "emotion": str(obs.get("current_emotion", "calm")),
        "toy_x": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[0]),
        "toy_y": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[1]),
        "toy_z": float(obs.get("toy_pos", [0.0, 1.2, 0.10])[2]),
        "touched_toy": bool(obs.get("touched_toy", False)),
        "waves": int(obs.get("active_ether_waves", 0)),
        "view_mode": streamer.view_mode
    }

@app.post("/api/action/throw_toy")
async def api_throw_toy(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    is_relative = bool(payload.get("relative", False))
    x = float(payload.get("x", 0.0))
    y = float(payload.get("y", 1.2))
    z = float(payload.get("z", 0.15))
    vx = float(payload.get("vx", 0.0))
    vy = float(payload.get("vy", 0.0))
    vz = float(payload.get("vz", 0.0))
    if streamer is not None:
        if is_relative:
            with streamer.telemetry_lock:
                obs = streamer.latest_obs.copy()
            robot_pos = obs.get("pos", [0.0, 0.0, 0.6])
            robot_yaw = float(obs.get("yaw", 0.0))
            # In world coordinates: Body forward is [-sin(yaw), cos(yaw)], Body right is [cos(yaw), sin(yaw)]
            c_y = math.cos(robot_yaw)
            s_y = math.sin(robot_yaw)
            world_x = float(robot_pos[0] + x * c_y - y * s_y)
            world_y = float(robot_pos[1] + x * s_y + y * c_y)
            streamer.throw_toy(x=world_x, y=world_y, z=z, vx=vx, vy=vy, vz=vz)
            return {"status": "ok", "toy_pos": [world_x, world_y, z]}
        else:
            streamer.throw_toy(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    return {"status": "ok", "toy_pos": [x, y, z]}

@app.post("/api/action/chirp_toy")
def api_chirp_toy():
    if streamer is not None:
        streamer.chirp_toy()
    return {"status": "ok", "message": "Radiated acoustic chirp from toy ball"}

@app.post("/api/action/help_up")
def api_help_up():
    if streamer is not None:
        streamer.help_up()
    return {"status": "ok", "message": "Robot helped up to standing posture"}

@app.post("/api/action/push")
async def api_push(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    fx = float(payload.get("fx", 0.0))
    fy = float(payload.get("fy", 0.0))
    duration = float(payload.get("duration", 0.12))
    if streamer is not None:
        streamer.push(fx=fx, fy=fy, duration=duration)
    return {"status": "ok", "pushed": [fx, fy]}

@app.post("/api/action/pet")
def api_pet():
    if streamer is not None:
        # Gentle pet on head generates small mechanical vibration into Ether
        streamer.push(fx=0.0, fy=0.0, fz=-15.0)
        from information_boltzmann.sandbox.expression_system import PrimaryEmotion, SubEmotion
        streamer.sandbox.expressions.set_emotion_explicit(PrimaryEmotion.JOY, SubEmotion.AFFECTIONATE, hold_duration=3.0)
    return {"status": "ok", "message": "Petted on head with affectionate hearts"}

@app.post("/api/action/set_theme")
async def api_set_theme(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    theme_idx = int(payload.get("theme", 0))
    if streamer is not None:
        from information_boltzmann.sandbox.expression_system import ThemeStyle
        streamer.sandbox.expressions.set_theme(ThemeStyle(theme_idx))
    return {"status": "ok", "theme": theme_idx}

@app.post("/api/action/set_emotion")
async def api_set_emotion(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    p = int(payload.get("primary", 1))
    s = payload.get("sub", None)
    hold = float(payload.get("hold_duration", 4.0))
    if streamer is not None:
        from information_boltzmann.sandbox.expression_system import PrimaryEmotion, SubEmotion
        sub_enum = SubEmotion(s) if s else None
        streamer.sandbox.expressions.set_emotion_explicit(PrimaryEmotion(p), sub_enum, hold_duration=hold)
    return {"status": "ok", "primary": p, "sub": s}

@app.post("/api/action/resume_auto")
def api_resume_auto():
    if streamer is not None:
        streamer.sandbox.expressions.manual_override = False
        streamer.sandbox.expressions._needs_upload = True
    return {"status": "ok", "message": "Autonomous emotion generation resumed"}

@app.post("/api/action/toggle_view")
def api_toggle_view():
    if streamer is not None:
        mode = streamer.toggle_view()
        return {"status": "ok", "view_mode": mode}
    return {"status": "error"}

HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>以太物理世界 — 机器人伴侣生命流</title>
  <style>
    :root {
      --bg: #0c0f14;
      --card-bg: rgba(22, 27, 36, 0.85);
      --border: rgba(60, 80, 110, 0.4);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.35);
      --success: #34d399;
      --danger: #f87171;
      --warning: #fbbf24;
      --text: #e2e8f0;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }
    header {
      background: rgba(15, 20, 28, 0.95);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      backdrop-filter: blur(10px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: 0.5px;
    }
    .brand-tag {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      border: 1px solid var(--accent);
    }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--success);
    }
    .pulse-dot {
      width: 10px;
      height: 10px;
      background: var(--success);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--success);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    main {
      flex: 1;
      padding: 20px;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 20px;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
    }
    .viewport-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    }
    .viewport-header {
      padding: 12px 18px;
      background: rgba(18, 23, 32, 0.9);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .screen-container {
      position: relative;
      background: #000;
      width: 100%;
      height: 480px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .screen-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .overlay-badge {
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(12, 16, 24, 0.85);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 8px;
      backdrop-filter: blur(8px);
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .expr-pill {
      font-weight: 700;
      color: var(--accent);
    }
    .fallen-alert {
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(220, 38, 38, 0.9);
      color: #fff;
      padding: 10px 24px;
      border-radius: 30px;
      font-weight: 600;
      font-size: 0.95rem;
      display: none;
      align-items: center;
      gap: 10px;
      box-shadow: 0 0 20px rgba(220, 38, 38, 0.6);
      animation: bounce 1s infinite alternate;
    }
    @keyframes bounce {
      0% { transform: translate(-50%, 0); }
      100% { transform: translate(-50%, -6px); }
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      backdrop-filter: blur(10px);
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 14px;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn-group {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .btn {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn:hover {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent);
      transform: translateY(-2px);
    }
    .btn-primary {
      background: linear-gradient(135deg, #0284c7, #0369a1);
      border-color: var(--accent);
      color: #fff;
      grid-column: span 2;
      padding: 14px;
      font-size: 1rem;
      box-shadow: 0 4px 15px rgba(2, 132, 199, 0.35);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, #0369a1, #075985);
      box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
    .btn-audio {
      background: rgba(16, 185, 129, 0.2);
      border-color: var(--success);
      color: var(--success);
      grid-column: span 2;
    }
    .btn-audio.active {
      background: var(--success);
      color: #0c0f14;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 0.85rem;
    }
    .stat-box {
      background: rgba(15, 20, 30, 0.6);
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stat-label {
      color: var(--muted);
      font-size: 0.75rem;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
    }
    .meter-container {
      margin-top: 10px;
    }
    .meter-bar {
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }
    .meter-fill {
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.1s linear;
    }
    .audio-visualizer {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      background: rgba(15, 20, 30, 0.6);
      padding: 10px;
      border-radius: 8px;
    }
    .vu-label { font-size: 0.75rem; color: var(--muted); width: 45px; }
    .vu-meter {
      flex: 1;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 5px;
      overflow: hidden;
    }
    .vu-fill-l { height: 100%; width: 0%; background: linear-gradient(90deg, #38bdf8, #818cf8); transition: width 0.08s ease; }
    .vu-fill-r { height: 100%; width: 0%; background: linear-gradient(90deg, #34d399, #38bdf8); transition: width 0.08s ease; }
    
    /* Theme Selector Styles */
    .theme-selector {
      display: flex;
      gap: 6px;
      align-items: center;
      background: rgba(15, 20, 30, 0.6);
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .theme-btn {
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 4px 9px;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .theme-btn:hover {
      color: #fff;
      border-color: var(--accent);
    }
    .theme-btn.active {
      background: rgba(56, 189, 248, 0.25);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }

    /* Emotional Hierarchy & Affect Matrix */
    .emotion-section {
      margin-top: 10px;
    }
    .emotion-header {
      font-size: 0.78rem;
      font-weight: 700;
      margin-bottom: 5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .chip-container {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 8px;
    }
    .emotion-chip {
      background: rgba(25, 33, 46, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--text);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .emotion-chip:hover {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .emotion-chip.active {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.35), rgba(129, 140, 248, 0.35));
      border-color: var(--accent);
      color: #fff;
      font-weight: 700;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
    }
    .affect-bar-wrapper {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 5px;
      font-size: 0.72rem;
    }
    .affect-bar {
      flex: 1;
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }
    .affect-fill {
      height: 100%;
      width: 50%;
      background: var(--accent);
      transition: width 0.15s ease, background 0.2s ease;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>🤖 伴侣机器人 · 以太物理环境</span>
      <span class="brand-tag">CONTINUOUS STREAM</span>
    </div>
    <div class="status-badge">
      <div class="pulse-dot"></div>
      <span id="lifeTimer">生存时间: 00:00:00</span>
    </div>
  </header>

  <main>
    <!-- Viewport Panel -->
    <section class="viewport-card">
      <div class="viewport-header" style="flex-wrap:wrap; gap:10px;">
        <div style="display:flex; align-items:center; gap:12px;">
          <span style="font-weight:600; font-size:0.9rem;">实时以太物理流 (25 FPS MJPEG)</span>
          <button class="btn" style="padding:4px 10px; font-size:0.78rem;" onclick="toggleView()">
            📷 <span id="viewBtnText">切换为主观第一视角</span>
          </button>
        </div>

        <!-- Multi-Theme Style Switcher -->
        <div class="theme-selector">
          <span style="font-size:0.75rem; color:var(--muted); margin-right:2px;">🎨 目镜风格:</span>
          <button class="theme-btn active" id="btnTheme0" onclick="selectTheme(0)">🎀 萌系赛博</button>
          <button class="theme-btn" id="btnTheme1" onclick="selectTheme(1)">🛰️ 科幻目镜</button>
          <button class="theme-btn" id="btnTheme2" onclick="selectTheme(2)">👾 复古像素</button>
          <button class="theme-btn" id="btnTheme3" onclick="selectTheme(3)">✨ 极简流光</button>
        </div>
      </div>

      <div class="screen-container">
        <img class="screen-img" src="/video_feed" alt="Robot Pet Physical Stream" id="videoStream">
        
        <div class="overlay-badge">
          <span>数字目镜表情:</span>
          <span class="expr-pill" id="exprBadge">(◠‿◠) 欣喜开朗</span>
        </div>

        <div class="fallen-alert" id="fallenAlert">
          ⚠️ 机器人摔倒了！点击右侧「扶它起来」拉它一把
        </div>
      </div>
    </section>

    <!-- Sidebar Controls & Telemetry -->
    <aside class="sidebar">
      <!-- Caretaker Interaction -->
      <div class="card">
        <div class="card-title">🤝 实体交互与推力微调滑块</div>
        
        <div style="margin-bottom: 12px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:6px;">
            <span>推力强度调节</span>
            <span id="pushForceVal" style="color:var(--accent); font-weight:bold;">20 N</span>
          </div>
          <input type="range" id="pushForceSlider" min="5" max="70" value="20" step="5" style="width:100%; accent-color:var(--accent); cursor:pointer;" oninput="updatePushVal(this.value)">
          <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--muted); margin-top:2px;">
            <span>5N (轻触)</span>
            <span>20N (适度测试)</span>
            <span>70N (强力推搡)</span>
          </div>
        </div>

        <div class="btn-group" style="grid-template-columns: 1fr 1fr; gap: 8px;">
          <button class="btn" onclick="actionPushDir(0, -1)">
            ⬆️ 往前推 (后仰恢复)
          </button>
          <button class="btn" onclick="actionPushDir(0, 1)">
            ⬇️ 往后推 (前倾恢复)
          </button>
          <button class="btn" onclick="actionPushDir(-1, 0)">
            ⬅️ 往左推
          </button>
          <button class="btn" onclick="actionPushDir(1, 0)">
            ➡️ 往右推
          </button>
          <button class="btn btn-primary" onclick="actionHelpUp()">
            🆙 扶它起来 (拉起站立)
          </button>
          <button class="btn" onclick="actionPet()">
            🖐️ 摸摸头 (激发亲昵爱心)
          </button>
        </div>
      </div>

      <!-- Interactive Curiosity Target: Radiant Toy Ball -->
      <div class="card">
        <div class="card-title" style="justify-content:space-between;">
          <span>🎾 好奇目标物互动 (物理以太信标)</span>
          <span class="badge" id="badgeToyStatus" style="background:rgba(251,191,36,0.2); color:#fbbf24; border-color:#fbbf24;">距离: 1.20m</span>
        </div>
        <p style="font-size:0.75rem; color:var(--muted); margin-bottom:10px;">
          在环境中投掷带声波与物理碰撞属性的橙色互动球。自主智能体将自发产生听觉趋向性 (Phonotaxis) 与空间信息增益探索 (Infotaxis)，迈步走向球体并蹲下触碰探究。
        </p>
        <div class="btn-group" style="grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
          <button class="btn" onclick="actionThrowToyRel(0.0, 1.2)" title="在机器人当前正前方1.2米放置小球">
            ⬆️ 投在身前
          </button>
          <button class="btn" onclick="actionThrowToyRel(-1.0, 0.0)" title="在机器人当前左侧1.0米放置小球">
            ⬅️ 投在左侧
          </button>
          <button class="btn" onclick="actionThrowToyRel(1.0, 0.0)" title="在机器人当前右侧1.0米放置小球">
            ➡️ 投在右侧
          </button>
          <button class="btn" onclick="actionThrowToyRel(-0.8, 1.0)" title="在机器人当前左前方放置小球">
            ↖️ 投向左前
          </button>
          <button class="btn" onclick="actionThrowToyRel(0.8, 1.0)" title="在机器人当前右前方放置小球">
            ↗️ 投向右前
          </button>
          <button class="btn" onclick="actionThrowToyRel(0.0, -1.2)" title="在机器人当前正后方1.2米放置小球">
            ⬇️ 投在身后
          </button>
        </div>
        <div style="margin-top: 8px;">
          <button class="btn btn-primary" style="width: 100%;" onclick="actionChirpToy()">
            🔊 敲击球体发声 (声波诱导)
          </button>
        </div>
      </div>

      <!-- Hierarchical Affect & Emotion Matrix -->
      <div class="card">
        <div class="card-title" style="justify-content:space-between;">
          <span>🎭 六大主情绪与层级分支情感空间</span>
          <button class="btn" style="padding:3px 8px; font-size:0.72rem;" onclick="actionResumeAuto()" title="恢复主动推理自由能自主涌现">🤖 自主涌现</button>
        </div>

        <!-- Affective Gauges (Valence & Arousal) -->
        <div style="background:rgba(15,20,30,0.6); padding:8px 10px; border-radius:8px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.05);">
          <div class="affect-bar-wrapper">
            <span style="width:68px; color:var(--muted);">效价 Valence:</span>
            <div class="affect-bar"><div class="affect-fill" id="meterValence" style="width:82%; background:#34d399;"></div></div>
            <span style="width:36px; text-align:right; font-weight:bold;" id="valValence">+0.65</span>
          </div>
          <div class="affect-bar-wrapper" style="margin-top:4px;">
            <span style="width:68px; color:var(--muted);">唤醒 Arousal:</span>
            <div class="affect-bar"><div class="affect-fill" id="meterArousal" style="width:40%; background:#38bdf8;"></div></div>
            <span style="width:36px; text-align:right; font-weight:bold;" id="valArousal">0.40</span>
          </div>
          <div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center; font-size:0.78rem;">
            <span style="color:var(--muted);">活跃主情绪 / 分支:</span>
            <span id="statPrimaryEmotion" style="font-weight:700; color:var(--accent);">💖 喜悦 (JOY) · 欣喜开朗</span>
          </div>
        </div>

        <!-- 6 Primary Emotion Groups with Interactive Sub-Branch Chips -->
        <div class="emotion-section">
          <div class="emotion-header" style="color:#f472b6;">💖 1. 喜悦 / 快乐 (JOY - 自由能稳态最小化)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-CHEERFUL" onclick="actionSetEmotion(1, 'CHEERFUL')">(^^) 欣喜开朗</button>
            <button class="emotion-chip" id="chip-ECSTATIC" onclick="actionSetEmotion(1, 'ECSTATIC')">(&gt; &lt;) 狂喜兴奋</button>
            <button class="emotion-chip" id="chip-CONTENT" onclick="actionSetEmotion(1, 'CONTENT')">(˘‿˘) 惬意满足</button>
            <button class="emotion-chip" id="chip-AFFECTIONATE" onclick="actionSetEmotion(1, 'AFFECTIONATE')">(♡ ♡) 喜爱亲昵</button>
            <button class="emotion-chip" id="chip-PLAYFUL" onclick="actionSetEmotion(1, 'PLAYFUL')">(^_-) 顽皮嬉戏</button>
          </div>

          <div class="emotion-header" style="color:#38bdf8;">🔍 2. 好奇 / 探索 (CURIOSITY - 认知信息增益驱动)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-INQUISITIVE" onclick="actionSetEmotion(2, 'INQUISITIVE')">(•ิ_•ิ)? 探究好奇</button>
            <button class="emotion-chip" id="chip-FOCUS_SCAN" onclick="actionSetEmotion(2, 'FOCUS_SCAN')">[⊙_⊙] 专注扫描</button>
            <button class="emotion-chip" id="chip-AWE_WONDER" onclick="actionSetEmotion(2, 'AWE_WONDER')">(★_★) 惊奇赞叹</button>
            <button class="emotion-chip" id="chip-EUREKA" onclick="actionSetEmotion(2, 'EUREKA')">(!_!) 灵光一闪</button>
          </div>

          <div class="emotion-header" style="color:#fbbf24;">⚡ 3. 惊讶 / 错愕 (SURPRISE - 预测误差冲击/惊诧)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-STARTLED" onclick="actionSetEmotion(3, 'STARTLED')">(O_O) 大吃一惊</button>
            <button class="emotion-chip" id="chip-BLANK_DOTS" onclick="actionSetEmotion(3, 'BLANK_DOTS')">(·_·) 呆愣错愕</button>
            <button class="emotion-chip" id="chip-PUZZLED" onclick="actionSetEmotion(3, 'PUZZLED')">(?_o) 困惑迷茫</button>
          </div>

          <div class="emotion-header" style="color:#818cf8;">💧 4. 悲伤 / 沮丧 (SADNESS - 持续误差累积/跌倒)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-POUTY" onclick="actionSetEmotion(4, 'POUTY')">(｡•́︿•̀｡) 委屈失落</button>
            <button class="emotion-chip" id="chip-DIZZY_FALLEN" onclick="actionSetEmotion(4, 'DIZZY_FALLEN')">(@_@) 跌倒眩晕</button>
            <button class="emotion-chip" id="chip-SLEEPY" onclick="actionSetEmotion(4, 'SLEEPY')">(-_-) 困倦疲乏</button>
            <button class="emotion-chip" id="chip-WEEPING" onclick="actionSetEmotion(4, 'WEEPING')">(T_T) 心碎悲伤</button>
          </div>

          <div class="emotion-header" style="color:#fb923c;">⚠️ 5. 恐惧 / 警惕 (FEAR - 稳态丧失风险激增)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-PANICKED" onclick="actionSetEmotion(5, 'PANICKED')">(&gt;_&lt;) 慌张惊恐</button>
            <button class="emotion-chip" id="chip-ALERT_GUARD" onclick="actionSetEmotion(5, 'ALERT_GUARD')">(°ロ°) 高度戒备</button>
            <button class="emotion-chip" id="chip-TIMID" onclick="actionSetEmotion(5, 'TIMID')">(•.•) 怯懦畏缩</button>
          </div>

          <div class="emotion-header" style="color:#f87171;">🔥 6. 愤怒 / 抵触 (ANGER - 外部受阻/逆力顽抗)</div>
          <div class="chip-container">
            <button class="emotion-chip" id="chip-ANNOYED" onclick="actionSetEmotion(6, 'ANNOYED')">(¬_¬) 恼怒不爽</button>
            <button class="emotion-chip" id="chip-RESISTING" onclick="actionSetEmotion(6, 'RESISTING')">(｀へ´) 坚韧顽抗</button>
            <button class="emotion-chip" id="chip-OVERHEATED" onclick="actionSetEmotion(6, 'OVERHEATED')">(♨_♨) 过载暴躁</button>
          </div>
        </div>
      </div>

      <!-- Spatial Binaural Audio -->
      <div class="card">
        <div class="card-title">🎧 耳部双声道以太声场 (第一性原理偶极子波)</div>
        <button class="btn btn-audio" id="btnAudio" onclick="toggleAudio()">
          🔊 开启耳机空间立体声听觉
        </button>

        <div class="audio-visualizer">
          <span class="vu-label">左耳 L:</span>
          <div class="vu-meter"><div class="vu-fill-l" id="vuL"></div></div>
          <span style="font-size:0.75rem; width:45px; text-align:right;" id="valL">0 Pa</span>
        </div>
        <div class="audio-visualizer" style="margin-top:6px;">
          <span class="vu-label">右耳 R:</span>
          <div class="vu-meter"><div class="vu-fill-r" id="vuR"></div></div>
          <span style="font-size:0.75rem; width:45px; text-align:right;" id="valR">0 Pa</span>
        </div>
      </div>

      <!-- Embodied Telemetry -->
      <div class="card">
        <div class="card-title">📊 纯连续主动推理与非平衡稳态 (Continuous Active Inference & NESS)</div>
        <div class="telemetry-grid">
          <div class="stat-box">
            <div class="stat-label">空间位置 (X, Y)</div>
            <div class="stat-value" id="statPos">(+0.00, +0.00)m</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">连续涌现动态 (Continuous Dynamics)</div>
            <div class="stat-value" id="statGait" style="font-size:0.75rem; color:var(--accent);">STAND</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">变分自由能 VFE (F)</div>
            <div class="stat-value" id="statVfe" style="color:var(--warning);">0.00</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">综合好奇心 (Salience)</div>
            <div class="stat-value" id="statCuriosity" style="color:var(--success);">0.000</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">🦶 触觉好奇心 (地面踩踏)</div>
            <div class="stat-value" id="statTouchCuriosity" style="font-size:0.95rem; color:#38bdf8;">0.000</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">🔊 声学好奇心 (以太回声)</div>
            <div class="stat-value" id="statAcousticCuriosity" style="font-size:0.95rem; color:#f472b6;">0.000</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">俯仰 / 侧倾姿态</div>
            <div class="stat-value" id="statTilt">0.0° / 0.0°</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">稳态偏离风险 Risk</div>
            <div class="stat-value" id="statRisk">0.000</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">🎾 好奇球坐标 (X, Y)</div>
            <div class="stat-value" id="statToyPos" style="font-size:0.95rem; color:#fbbf24;">(0.00, 1.20)</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">🎯 目标距离 / 状态</div>
            <div class="stat-value" id="statToyDist" style="font-size:0.95rem; color:#34d399;">1.20m (探索中)</div>
          </div>
        </div>

        <div class="meter-container">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--muted);">
            <span>左脚底触觉接地力</span>
            <span id="valTouchL">0 N</span>
          </div>
          <div class="meter-bar"><div class="meter-fill" id="meterTouchL"></div></div>
        </div>

        <div class="meter-container">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--muted);">
            <span>右脚底触觉接地力</span>
            <span id="valTouchR">0 N</span>
          </div>
          <div class="meter-bar"><div class="meter-fill" id="meterTouchR" style="background:var(--success);"></div></div>
        </div>
      </div>
    </aside>
  </main>

  <script>
    // Audio Context for binaural sound streaming
    let audioCtx = null;
    let audioWs = null;
    let nextAudioTime = 0;
    let isAudioPlaying = false;

    // Start / Toggle Binaural Audio
    async function toggleAudio() {
      const btn = document.getElementById('btnAudio');
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 8000 });
      }
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      if (!isAudioPlaying) {
        connectAudioWs();
        isAudioPlaying = true;
        btn.classList.add('active');
        btn.innerHTML = '🔊 耳部立体声听觉已开启 (正在播放脚步与振动)';
      } else {
        if (audioWs) audioWs.close();
        isAudioPlaying = false;
        btn.classList.remove('active');
        btn.innerHTML = '🔇 开启耳机空间立体声听觉';
      }
    }

    function connectAudioWs() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      audioWs = new WebSocket(`${proto}//${window.location.host}/ws/audio`);
      audioWs.binaryType = 'arraybuffer';

      audioWs.onmessage = (event) => {
        if (!audioCtx) return;
        const floatData = new Float32Array(event.data);
        const numFrames = floatData.length / 2;
        if (numFrames === 0) return;

        // Split interleaved stereo into 2 channels
        const audioBuffer = audioCtx.createBuffer(2, numFrames, 8000);
        const chanL = audioBuffer.getChannelData(0);
        const chanR = audioBuffer.getChannelData(1);

        for (let i = 0; i < numFrames; i++) {
          chanL[i] = floatData[i * 2];
          chanR[i] = floatData[i * 2 + 1];
        }

        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.destination);

        const currentTime = audioCtx.currentTime;
        if (nextAudioTime < currentTime) {
          nextAudioTime = currentTime + 0.02;
        }
        source.start(nextAudioTime);
        nextAudioTime += audioBuffer.duration;
      };

      audioWs.onclose = () => {
        if (isAudioPlaying) setTimeout(connectAudioWs, 1000);
      };
    }

    // Telemetry WebSocket
    let telemetryWs = null;
    function connectTelemetryWs() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      telemetryWs = new WebSocket(`${proto}//${window.location.host}/ws/telemetry`);

      telemetryWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
      };

      telemetryWs.onclose = () => {
        setTimeout(connectTelemetryWs, 1000);
      };
    }
    connectTelemetryWs();

    // UI Updates
    const startTime = Date.now();
    function updateLifeTimer() {
      const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
      const h = String(Math.floor(elapsedSec / 3600)).padStart(2, '0');
      const m = String(Math.floor((elapsedSec % 3600) / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      document.getElementById('lifeTimer').innerText = `生存时间: ${h}:${m}:${s}`;
    }
    setInterval(updateLifeTimer, 1000);

    function updateUI(d) {
      document.getElementById('statPos').innerText = `(${d.pos_x >= 0 ? '+' : ''}${d.pos_x.toFixed(2)}, ${d.pos_y >= 0 ? '+' : ''}${d.pos_y.toFixed(2)})m`;
      const pMode = d.postural_mode || 'BIPED_EXPLORE';
      const gState = d.gait_state || 'STAND';
      document.getElementById('statGait').innerText = `${pMode} · ${gState}`;
      document.getElementById('statVfe').innerText = (d.vfe || 0).toFixed(2);
      document.getElementById('statCuriosity').innerText = (d.epistemic_gain || 0).toFixed(3);
      document.getElementById('statTouchCuriosity').innerText = (d.tactile_curiosity || 0).toFixed(3);
      document.getElementById('statAcousticCuriosity').innerText = (d.acoustic_curiosity || 0).toFixed(3);
      document.getElementById('statTilt').innerText = `${d.pitch_deg.toFixed(1)}° / ${d.roll_deg.toFixed(1)}°`;
      document.getElementById('statRisk').innerText = (d.pragmatic_risk || 0).toFixed(3);

      document.getElementById('exprBadge').innerText = d.expression;

      // Update Affective Space (Valence & Arousal)
      const val = d.valence !== undefined ? d.valence : 0.0;
      const ar = d.arousal !== undefined ? d.arousal : 0.0;
      document.getElementById('valValence').innerText = (val >= 0 ? '+' : '') + val.toFixed(2);
      document.getElementById('valArousal').innerText = ar.toFixed(2);
      
      const valPct = Math.max(0, Math.min(100, ((val + 1.0) / 2.0) * 100));
      const meterVal = document.getElementById('meterValence');
      if (meterVal) {
        meterVal.style.width = valPct + '%';
        if (val >= 0.2) meterVal.style.background = '#34d399';
        else if (val <= -0.2) meterVal.style.background = '#f87171';
        else meterVal.style.background = '#fbbf24';
      }

      const arPct = Math.max(0, Math.min(100, ar * 100));
      const meterAr = document.getElementById('meterArousal');
      if (meterAr) meterAr.style.width = arPct + '%';

      // Update Primary Emotion & Sub-branch text
      const pri = d.primary_emotion || 'JOY';
      const sub = d.sub_emotion || 'CHEERFUL';
      const priLabels = {
        'JOY': '💖 喜悦 (JOY)',
        'CURIOSITY': '🔍 好奇 (CURIOSITY)',
        'SURPRISE': '⚡ 惊讶 (SURPRISE)',
        'SADNESS': '💧 悲伤 (SADNESS)',
        'FEAR': '⚠️ 恐惧 (FEAR)',
        'ANGER': '🔥 愤怒 (ANGER)',
        'NEUTRAL': '⚖️ 平静 (NEUTRAL)'
      };
      const statPri = document.getElementById('statPrimaryEmotion');
      if (statPri) statPri.innerText = `${priLabels[pri] || pri} · ${sub}`;

      // Update theme active button
      const th = d.theme_style || 'CYBER_CHIBI';
      const thIdxMap = { 'CYBER_CHIBI': 0, 'SCI_FI_HUD': 1, 'RETRO_PIXEL': 2, 'MINIMALIST': 3 };
      const curThIdx = thIdxMap[th] !== undefined ? thIdxMap[th] : 0;
      for (let i = 0; i < 4; i++) {
        const btn = document.getElementById('btnTheme' + i);
        if (btn) {
          if (i === curThIdx) btn.classList.add('active');
          else btn.classList.remove('active');
        }
      }

      // Highlight active emotion chip
      document.querySelectorAll('.emotion-chip').forEach(c => c.classList.remove('active'));
      const activeChip = document.getElementById('chip-' + sub);
      if (activeChip) activeChip.classList.add('active');

      // Fallen Alert
      const alert = document.getElementById('fallenAlert');
      if (d.is_fallen) {
        alert.style.display = 'flex';
        alert.innerText = '⚠️ 机器人摔倒了（稳态自由能偏离超标）！1.2 秒后监护者将自动扶起，亦可手动拉它一把';
      } else {
        alert.style.display = 'none';
      }

      // Tactile Bars (normalized to 300N)
      const touchL_pct = Math.min(100, (d.touch_l / 300.0) * 100);
      const touchR_pct = Math.min(100, (d.touch_r / 300.0) * 100);
      document.getElementById('meterTouchL').style.width = `${touchL_pct}%`;
      document.getElementById('meterTouchR').style.width = `${touchR_pct}%`;
      document.getElementById('valTouchL').innerText = `${d.touch_l.toFixed(0)} N`;
      document.getElementById('valTouchR').innerText = `${d.touch_r.toFixed(0)} N`;

      // Audio VU Meters (normalized to 50 Pa)
      const vuL_pct = Math.min(100, (Math.abs(d.audio_l) / 50.0) * 100);
      const vuR_pct = Math.min(100, (Math.abs(d.audio_r) / 50.0) * 100);
      document.getElementById('vuL').style.width = `${vuL_pct}%`;
      document.getElementById('vuR').style.width = `${vuR_pct}%`;
      document.getElementById('valL').innerText = `${d.audio_l.toFixed(1)} Pa`;
      document.getElementById('valR').innerText = `${d.audio_r.toFixed(1)} Pa`;

      // Curiosity Toy Telemetry
      if (d.toy_x !== undefined) {
        const statToyPos = document.getElementById('statToyPos');
        if (statToyPos) statToyPos.innerText = `(${d.toy_x.toFixed(2)}, ${d.toy_y.toFixed(2)})`;
        const dx = d.toy_x - d.pos_x;
        const dy = d.toy_y - d.pos_y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const touched = d.touched_toy ? "已触碰! 🎉" : "探索中";
        const statToyDist = document.getElementById('statToyDist');
        if (statToyDist) statToyDist.innerText = `${dist.toFixed(2)}m (${touched})`;
        const badgeToy = document.getElementById('badgeToyStatus');
        if (badgeToy) {
          badgeToy.innerText = `距球: ${dist.toFixed(2)}m`;
          if (d.touched_toy) {
            badgeToy.style.color = '#34d399';
            badgeToy.style.borderColor = '#34d399';
            badgeToy.innerText = '已触碰! 🎉';
          }
        }
      }
    }

    // Caretaker Action APIs
    let currentPushForce = 20;
    function updatePushVal(val) {
      currentPushForce = parseFloat(val);
      document.getElementById('pushForceVal').innerText = `${currentPushForce} N`;
    }

    async function actionHelpUp() {
      await fetch('/api/action/help_up', { method: 'POST' });
    }

    async function actionThrowToy(x, y) {
      await fetch('/api/action/throw_toy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: x, y: y, z: 0.15, relative: false, vx: 0.0, vy: 0.1 })
      });
    }

    async function actionThrowToyRel(relX, relY) {
      await fetch('/api/action/throw_toy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: relX, y: relY, z: 0.15, relative: true, vx: 0.0, vy: 0.0 })
      });
    }

    async function actionChirpToy() {
      await fetch('/api/action/chirp_toy', { method: 'POST' });
    }

    async function actionPushDir(dx, dy) {
      await fetch('/api/action/push', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fx: dx * currentPushForce, fy: dy * currentPushForce, duration: 0.12 })
      });
    }

    async function actionPet() {
      await fetch('/api/action/pet', { method: 'POST' });
    }

    async function selectTheme(themeIdx) {
      await fetch('/api/action/set_theme', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: themeIdx })
      });
      for (let i = 0; i < 4; i++) {
        const btn = document.getElementById('btnTheme' + i);
        if (btn) {
          if (i === themeIdx) btn.classList.add('active');
          else btn.classList.remove('active');
        }
      }
    }

    async function actionSetEmotion(primary, sub) {
      await fetch('/api/action/set_emotion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ primary: primary, sub: sub, hold_duration: 5.0 })
      });
      document.querySelectorAll('.emotion-chip').forEach(c => c.classList.remove('active'));
      const chip = document.getElementById('chip-' + sub);
      if (chip) chip.classList.add('active');
    }

    async function actionResumeAuto() {
      await fetch('/api/action/resume_auto', { method: 'POST' });
      document.querySelectorAll('.emotion-chip').forEach(c => c.classList.remove('active'));
    }

    async function toggleView() {
      const res = await fetch('/api/action/toggle_view', { method: 'POST' });
      const data = await res.json();
      const btnText = document.getElementById('viewBtnText');
      if (data.view_mode === 'first_person') {
        btnText.innerText = '切换为伴侣第三视角';
      } else {
        btnText.innerText = '切换为主观第一视角';
      }
    }
  </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Robot Pet Ether Life Stream Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port")
    args = parser.parse_args()

    print("=" * 72)
    print(f"🌟 Starting Robot Pet HTML5 Real-Time Streaming Server on http://localhost:{args.port}")
    print(f"   - Video Feed:  http://localhost:{args.port}/video_feed")
    print(f"   - Web UI:      http://localhost:{args.port}/")
    print(f"   - Binaural:    ws://localhost:{args.port}/ws/audio (8000 Hz Stereo)")
    print(f"   - Telemetry:   ws://localhost:{args.port}/ws/telemetry (20 Hz JSON)")
    print("=" * 72)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
