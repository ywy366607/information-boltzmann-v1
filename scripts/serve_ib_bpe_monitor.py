"""Read-only monitor; no CUDA imports or access to the training process."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--port', type=int, default=8080)
    a = p.parse_args()
    html = Path(__file__).resolve().parents[1] / 'present/ib_bpe_live.html'
    hardware = {}

    def sample_gpu():
        while True:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                fields = result.stdout.strip().splitlines()[0].split(',')
                hardware.update(dict(zip(('utilization_percent', 'used_mib', 'total_mib', 'temperature_c', 'power_w'), fields)), wall_time=time.time())
                with (a.run / 'hardware.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(hardware) + '\n')
            except Exception:
                pass
            time.sleep(3)

    threading.Thread(target=sample_gpu, daemon=True).start()

    def read(name, default):
        path = a.run / name
        return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] == '/api/state':
                progress = read('progress.json', {})
                history = []
                path = a.run / 'metrics.jsonl'
                if path.exists():
                    for line in path.read_text(encoding='utf-8').splitlines():
                        try:
                            history.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Last append may still be in progress.
                history = list({(r.get('kind'), r.get('step')): r for r in history}.values())
                data = dict(config=read('config.json', {}), progress=progress, current=read('live_state.json', None), history=history, hardware=hardware.copy())
                body = json.dumps(data).encode()
                mime = 'application/json'
            elif self.path.split('?')[0] == '/':
                body, mime = html.read_bytes(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
