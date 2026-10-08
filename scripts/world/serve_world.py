"""Observe a persistent shared-law world. No agent or trained brain is loaded."""

import argparse
import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.world import GenerativeWorldLaw, PersistentWorld, WorldConfig
from information_boltzmann.world.model import basis_numpy


class WorldHost:
    def __init__(self, config: WorldConfig, state_file: Path, fresh: bool = False):
        self.config = config
        self.lock = threading.Lock()
        self.commands = queue.SimpleQueue()
        self.audio = deque(maxlen=120)
        self.sequence = 0
        self.state_file = state_file
        self.world = PersistentWorld.load(state_file) if state_file.exists() and not fresh else PersistentWorld(GenerativeWorldLaw(config))
        self.config = self.world.config
        config = self.config
        self.run_id = self.world.world_id
        self.paused = False
        self.status = "正在编译共同生成律"
        self.failure = None
        self.latest = self.world.snapshot()
        self.latest.update(history=list(self.world.history), run_id=self.run_id)
        x, y = np.meshgrid(np.linspace(.03, config.room[0] - .03, 24),
                           np.linspace(.03, config.room[1] - .03, 18))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .35)))
        parameters = self.world.parameters
        self.plane_readout = basis_numpy(points, parameters["wave_numbers"], parameters["normalizer"]) * (
            config.air_density * config.sound_speed * parameters["omega_air"])[None, :]
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        try:
            self.world.step(1)  # JIT once; the physical sample is retained.
            self.status = "世界持续演化中"
            samples = 256
            last_save = time.perf_counter()
            while self.running:
                start = time.perf_counter()
                while not self.commands.empty():
                    command = self.commands.get()
                    action = command["action"]
                    if action == "impulse":
                        self.world.impulse(int(command["object"]), command["impulse"])
                    elif action == "observer":
                        self.world.set_listener(command["position"], command.get("yaw", 0))
                    elif action == "pause":
                        self.paused = bool(command["paused"])
                        self.status = "世界已暂停" if self.paused else "世界持续演化中"
                    elif action == "restart":
                        self.world = PersistentWorld(GenerativeWorldLaw(self.config))
                        self.run_id = self.world.world_id
                        with self.lock:
                            self.audio.clear()
                            self.latest = self.world.snapshot()
                            self.latest.update(history=[], run_id=self.run_id, paused=self.paused)
                if self.paused:
                    if time.perf_counter() - last_save >= 2:
                        self.world.save(self.state_file)
                        last_save = time.perf_counter()
                    with self.lock:
                        self.latest["paused"] = True
                    time.sleep(.025)
                    continue
                audio = self.world.step(samples)
                elapsed = time.perf_counter() - start
                snapshot = self.world.snapshot()
                snapshot.update(run_id=self.run_id, paused=False,
                                status=self.status, wall_compute_seconds=elapsed,
                                realtime_factor=(samples / self.config.sample_rate) / max(elapsed, 1e-9),
                                manifest=self.world.law.manifest())
                snapshot["history"] = list(self.world.history)[-500:]
                # Actual field on an observation plane, not decorative rings.
                snapshot["wave_plane"] = {"shape": [18, 24], "height": .35,
                                          "pressure_pa": (self.plane_readout @ self.world.state.wave).tolist()}
                snapshot["audio_preview"] = audio[::4].tolist()
                with self.lock:
                    self.latest = snapshot
                    self.sequence += 1
                    self.audio.append((self.sequence, audio.astype("<f4").tobytes()))
                if time.perf_counter() - last_save >= 2:
                    self.world.save(self.state_file)
                    last_save = time.perf_counter()
                time.sleep(max(0, samples / self.config.sample_rate - (time.perf_counter() - start)))
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.failure = str(error)
            self.status = "世界停止：请检查数值诊断"


def make_handler(host: WorldHost):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if not self.path.startswith("/api/"):
                super().log_message(format, *args)

        def send(self, body, content_type="application/json", code=200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send((ROOT / "present/world/index.html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self.send(b"", "image/x-icon", 204)
            elif parsed.path == "/api/state":
                with host.lock:
                    state = dict(host.latest)
                    state.update(status=host.status, failure=host.failure)
                self.send(json.dumps(state, ensure_ascii=False, allow_nan=False).encode())
            elif parsed.path == "/api/audio":
                try:
                    after = int(parse_qs(parsed.query).get("after", [0])[0])
                except ValueError:
                    self.send(b'{"error":"invalid cursor"}', code=400)
                    return
                with host.lock:
                    chunks = [data for seq, data in host.audio if seq > after]
                    sequence = host.sequence
                    run = host.run_id
                self.send(b"".join(chunks[-16:]), "application/octet-stream", extra={
                    "X-Audio-Sequence": sequence, "X-Sample-Rate": host.config.sample_rate,
                    "X-World-Run": run})
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 8192:
                    raise ValueError("Invalid request length")
                command = json.loads(self.rfile.read(size))
                if self.path == "/api/impulse":
                    index = int(command["object"])
                    value = np.asarray(command["impulse"], dtype=float)
                    if not 0 <= index < len(host.world.law.materials) or value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) > 2:
                        raise ValueError("Invalid body or actuator impulse")
                    command["action"] = "impulse"
                elif self.path == "/api/observer":
                    position = np.asarray(command["position"], dtype=float)
                    yaw = float(command.get("yaw", 0))
                    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(yaw) or np.any(position < .15) or np.any(position > np.asarray(host.config.room) - .15):
                        raise ValueError("Invalid observer pose")
                    command["action"] = "observer"
                elif self.path == "/api/pause":
                    command = {"action": "pause", "paused": bool(command["paused"])}
                elif self.path == "/api/restart":
                    command = {"action": "restart"}
                else:
                    raise ValueError("Unknown action")
                host.commands.put(command)
                self.send(b'{"queued":true}')
            except (ValueError, KeyError, TypeError) as error:
                self.send(json.dumps({"error": str(error)}).encode(), code=400)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--cutoff-hz", type=float, default=480)
    parser.add_argument("--state-file", type=Path, default=ROOT / "results/generative_world_live/world_state.npz")
    parser.add_argument("--fresh", action="store_true", help="Explicitly establish a new initial world")
    args = parser.parse_args()
    host = WorldHost(WorldConfig(sample_rate=args.sample_rate, cutoff_hz=args.cutoff_hz), args.state_file, args.fresh)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(host))
    print(f"WORLD http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        host.running = False
        host.thread.join(timeout=5)
        host.world.save(host.state_file)
        server.server_close()


if __name__ == "__main__":
    main()
