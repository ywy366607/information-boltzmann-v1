"""Local read-only cosmic medium dashboard. No Torch/CUDA or checkpoint reads."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time

import numpy as np


class DashboardDataStore:
    """Read one registered run per response; never borrow another run's data."""

    def __init__(self, fallback_run, registry=None):
        self.fallback_run = Path(fallback_run).resolve()
        self.registry = Path(registry).resolve() if registry else None
        self.cache = {}

    def selected_run(self):
        if self.registry is None or not self.registry.exists():
            return self.fallback_run.name, self.fallback_run
        registration = json.loads(self.registry.read_text(encoding='utf-8-sig'))
        identity = registration['active_run']
        entry = registration['runs'][identity]
        path = Path(entry['path'])
        if not path.is_absolute():
            path = self.registry.parent / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f'Registered run directory is missing: {path}')
        return identity, path

    @staticmethod
    def read_json(run, name, default):
        try:
            return json.loads((run / name).read_text(encoding='utf-8-sig'))
        except (OSError, json.JSONDecodeError):
            return default

    def records(self, run, name, count=400):
        path = run / name
        key = (str(path), count)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if key in self.cache and self.cache[key][0] == stamp:
                return self.cache[key][1]
            with path.open('rb') as handle:
                handle.seek(0, 2)
                start = max(0, handle.tell() - 2_000_000)
                handle.seek(start)
                if start:
                    handle.readline()
                lines = handle.read().decode('utf-8-sig').splitlines()
            rows = []
            for line in lines[-count:]:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.cache[key] = (stamp, rows)
            return rows
        except OSError:
            return []

    def state(self):
        identity, run = self.selected_run()
        evaluations = self.records(run, 'lifelong_evaluation.jsonl', 1)
        constructor = self.read_json(run, 'config.json', {}).get('constructor', {})
        spatial = material_tensor_view(self.read_json(run, 'spatial.json', None), constructor)
        if spatial is not None:
            spatial['channels'] = constructor.get('channels')
        return {'run': run.name, 'run_id': identity, 'run_path': str(run),
                'server_time': time.time(),
                'progress': self.read_json(run, 'progress.json', {}),
                'spatial': spatial, 'history': self.records(run, 'metrics.jsonl'),
                'evaluation': evaluations[-1] if evaluations else None}


def material_tensor_view(snapshot, constructor=None):
    """Actual reported propagation tensor and optional legacy aperture display.

    This is a display-only reconstruction of B=diag(c)T; no dynamics are run.
    """
    if snapshot is None:
        return None
    result = dict(snapshot)
    if constructor and 'read_footprint_by_probe' not in result:
        # Exact compact aperture formula for pre-telemetry-extension snapshots.
        # Geometry is read-only; never synthesize attention or temporal history.
        radius = constructor.get('read_port_radius')
        if radius is not None:
            coords = np.asarray(snapshot['coordinates'], dtype=np.float64)
            centers = np.asarray(snapshot['read_coords'], dtype=np.float64)
            delta = (coords[None] - centers[:, None] + .5) % 1. - .5
            footprint = np.maximum(1. - (delta / np.asarray(radius)) ** 2, 0.) ** 2
            footprint = footprint.prod(-1)
            result['read_footprint_by_probe'] = (footprint / np.maximum(
                footprint.sum(-1, keepdims=True), np.finfo(float).tiny)).tolist()
    speed = np.asarray(snapshot['speed'], dtype=np.float64)
    shear = np.asarray(snapshot.get('shear') or np.zeros_like(speed), dtype=np.float64)
    if snapshot.get('effective_transport_factor') is not None:
        factor = np.asarray(snapshot['effective_transport_factor'], dtype=np.float64)
    else:
        factor = np.broadcast_to(np.eye(3), (len(speed), 3, 3)).copy()
        factor[:, 1, 0] = shear[:, 0]
        factor[:, 2, 0] = shear[:, 1]
        factor[:, 2, 1] = shear[:, 2]
        factor *= speed[:, :, None]
    tensor = factor @ factor.transpose(0, 2, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    result['material_tensor_eigenvalues'] = eigenvalues.tolist()
    result['material_principal_axis'] = eigenvectors[:, :, -1].tolist()
    result['tensor_scope'] = ('actual propagation factor with STP and budget' if
                              snapshot.get('effective_transport_factor') is not None else
                              'legacy material and conduction; excludes transient STP and speed_reference')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8085)
    parser.add_argument('--registry', type=Path,
                        help='Hot-read active-run registry; defaults to the 8085 dashboard registry')
    args = parser.parse_args()
    run = args.run.resolve()
    page = Path(__file__).resolve().parents[2] / 'present/medium_cosmos_live.html'
    registry = args.registry or (page.with_name('medium_dashboard_runs.json')
                                  if args.port == 8085 else None)
    store = DashboardDataStore(run, registry)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] == '/api/state':
                try:
                    data = store.state()
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.send_error(503, f'Run registry unavailable: {error}')
                    return
                body = json.dumps(data, allow_nan=False).encode('utf-8')
                mime = 'application/json; charset=utf-8'
            elif self.path.split('?')[0] == '/':
                body, mime = page.read_bytes(), 'text/html; charset=utf-8'
            elif self.path.split('?')[0] == '/medium_flow.js':
                body, mime = page.with_name('medium_flow.js').read_bytes(), 'text/javascript; charset=utf-8'
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

    print(f'Medium cosmos: http://127.0.0.1:{args.port}/ -> {store.selected_run()[1]}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
