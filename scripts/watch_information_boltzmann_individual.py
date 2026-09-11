"""Read-only sampled observation wrapper around the unchanged lifetime trainer.

The model, optimizer, RNG and persistent state are not modified by sampling.
Use the ordinary lifetime arguments, including --resume, after --watch-port.
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_information_boltzmann_lifetime as lifetime


class LiveView:
    def __init__(self, output, port):
        self.output = output
        self.lock = threading.Lock()
        self.last_sample = 0.
        self.current = None
        self.history = deque(maxlen=1200)
        self.finished = False
        viewer = self
        html = Path(__file__).resolve().parents[1]/'present/ib_individual_live.html'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.split('?')[0] == '/api/state':
                    with viewer.lock:
                        result = {'current': viewer.current, 'history': list(viewer.history),
                                  'finished': viewer.finished}
                    try:
                        result['progress'] = json.loads((output/'progress.json').read_text())
                    except (OSError, ValueError):
                        result['progress'] = None
                    data, mime = json.dumps(result).encode(), 'application/json'
                elif self.path.split('?')[0] in ('/', '/index.html'):
                    data, mime = html.read_bytes(), 'text/html; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.queue = __import__('queue').Queue(maxsize=4)
        threading.Thread(target=self.writer, daemon=True).start()

    def writer(self):
        with (self.output/'motion.jsonl').open('a', encoding='utf-8', buffering=1) as handle:
            while True:
                frame = self.queue.get()
                handle.write(json.dumps(frame, allow_nan=False)+'\n')

    @torch.no_grad()
    def observe(self, runner, ce):
        now = time.monotonic()
        if now-self.last_sample < .25:
            return
        self.last_sample = now
        state = runner.state
        gamma = runner.model.force.damping(state.x)
        frame = {'event': runner.events, 'update': runner.updates, 'phase_time': state.time,
                 'wall_time': time.time(), 'ce': ce, 'nll': runner.total_nll/runner.events,
                 'energy': float(lifetime.energy(runner.model, state)),
                 'gamma_mean': float(gamma.mean()), 'gamma_std': float(gamma.std(unbiased=False)),
                 'x': state.x.detach().cpu().tolist(), 'v': state.v.detach().cpu().tolist(),
                 'gamma': gamma.flatten().cpu().tolist(),
                 'accepted_collisions': runner.accepted}
        with self.lock:
            self.current = frame
            self.history.append({k: frame[k] for k in ('event', 'ce', 'nll', 'energy', 'gamma_mean', 'gamma_std')})
        try:
            self.queue.put_nowait(frame)
        except __import__('queue').Full:
            pass


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--watch-port', type=int, default=8080)
    options, remaining = p.parse_known_args()
    output = Path(remaining[remaining.index('--output')+1]).resolve()
    if not output.is_dir():
        raise ValueError('Attach this observer by resuming an existing individual')
    view = LiveView(output, options.watch_port)
    original = lifetime.StreamRunner.observe

    def observed(self, token, *, external=True):
        value = original(self, token, external=external)
        if self.optimizer is not None:
            view.observe(self, value)
        return value

    lifetime.StreamRunner.observe = observed
    lifetime.write_json(output/'observer.json', {
        'wrapper': str(Path(__file__).resolve()), 'sha256': lifetime.digest(__file__),
        'mode': 'read_only_observation_after_observe', 'max_hz': 4,
        'port': options.watch_port, 'changes_model_parameters_state_or_rng': False})
    sys.argv = [str(Path(lifetime.__file__).resolve()), *remaining]
    try:
        lifetime.main()
    finally:
        view.finished = True
    # Keep the final recorded movement accessible after training completes.
    threading.Event().wait()


if __name__ == '__main__':
    main()
