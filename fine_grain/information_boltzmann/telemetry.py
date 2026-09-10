"""Real-time training telemetry server for continuous phase-space monitoring."""
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from socketserver import ThreadingMixIn
import threading
import time


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class TelemetryHub:
    def __init__(self, history_len: int = 150, live_file: Path | None = None):
        self.lock = threading.Lock()
        self.current_frame: dict = {
            "status": "idle",
            "event": 0,
            "token_id": 0,
            "token_char": "-",
            "ce": 0.0,
            "mean_nll": 0.0,
            "perplexity": 0.0,
            "lr": 0.0,
            "particles": {"x": [], "v": []},
            "collision_pairs": [],
            "metrics": {
                "var_x": 0.0,
                "var_v": 0.0,
                "a_eff": 0.0,
                "energy": 0.0,
                "ke": 0.0,
                "t_eff": 0.0,
                "gamma": 1.0,
            },
            "top5": [],
        }
        self.history = deque(maxlen=history_len)
        self.live_file = live_file or Path("results/live_training_state.json")
        self.listeners: list[threading.Event] = []
        self.server: HTTPServer | None = None
        self.server_thread: threading.Thread | None = None

    def push_frame(self, frame: dict) -> None:
        with self.lock:
            self.current_frame = frame
            self.history.append({
                "event": frame["event"],
                "ce": frame["ce"],
                "mean_nll": frame["mean_nll"],
                "a_eff": frame["metrics"]["a_eff"],
                "energy": frame["metrics"]["energy"],
                "var_x": frame["metrics"]["var_x"],
                "var_v": frame["metrics"]["var_v"],
                "t_eff": frame["metrics"]["t_eff"],
            })
            for evt in self.listeners:
                evt.set()

        # Atomically write live file to both target and fixed root path
        try:
            self.live_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.live_file.with_suffix(".tmp")
            payload = json.dumps({"current": self.current_frame, "history": list(self.history)})
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.live_file)

            # Also write to results/live_training_state.json for local file viewing
            root_live = Path("results/live_training_state.json")
            root_live.parent.mkdir(parents=True, exist_ok=True)
            root_tmp = root_live.with_suffix(".tmp")
            root_tmp.write_text(payload, encoding="utf-8")
            root_tmp.replace(root_live)
        except Exception:
            pass

    def start_server(self, port: int = 8080, html_path: Path | None = None) -> None:
        hub = self
        html_file = html_path or Path("results/phase_space_monitor.html")

        class TelemetryHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # suppress standard access logs to keep terminal clean

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    if html_file.exists():
                        self.wfile.write(html_file.read_bytes())
                    else:
                        self.wfile.write(b"<h1>Monitor HTML not generated yet</h1>")
                elif self.path == "/api/state":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with hub.lock:
                        payload = json.dumps({"current": hub.current_frame, "history": list(hub.history)})
                    self.wfile.write(payload.encode("utf-8"))
                elif self.path == "/api/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    evt = threading.Event()
                    with hub.lock:
                        hub.listeners.append(evt)

                    try:
                        while True:
                            evt.wait(timeout=1.0)
                            evt.clear()
                            with hub.lock:
                                data = json.dumps(hub.current_frame)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except (ConnectionError, BrokenPipeError):
                        pass
                    finally:
                        with hub.lock:
                            if evt in hub.listeners:
                                hub.listeners.remove(evt)
                else:
                    self.send_response(404)
                    self.end_headers()

        self.server = ThreadedHTTPServer(("", port), TelemetryHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"\n>>> Live Training Telemetry Monitor running at http://localhost:{port}/ <<<\n", flush=True)

    def stop_server(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server = None
