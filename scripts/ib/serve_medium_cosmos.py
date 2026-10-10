"""Local read-only cosmic medium dashboard. No Torch/CUDA or checkpoint reads."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
import urllib.parse

import numpy as np


class DashboardDataStore:
    """Read one registered run per response; never borrow another run's data."""

    def __init__(self, fallback_run, registry=None, comparison_report=None):
        self.fallback_run = Path(fallback_run).resolve()
        self.registry = Path(registry).resolve() if registry else None
        self.comparison_report = Path(comparison_report) if comparison_report else (
            Path(__file__).resolve().parents[2] /
            'results/published/medium_vs_fly_matched_50k_20261009.json')
        self.cache = {}

    def selected_run(self, override=None):
        if self.registry is None or not self.registry.exists():
            return self.fallback_run.name, self.fallback_run
        registration = json.loads(self.registry.read_text(encoding='utf-8-sig'))
        runs_dict = registration.get('runs', {})
        if override and override in runs_dict:
            identity = override
        else:
            identity = registration['active_run']
        entry = runs_dict[identity]
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

    def cached_json(self, path):
        key = ('json', str(path))
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if key not in self.cache or self.cache[key][0] != stamp:
                self.cache[key] = (stamp, json.loads(path.read_text(encoding='utf-8-sig')))
            return self.cache[key][1]
        except (OSError, json.JSONDecodeError):
            return {}

    def comparison(self, run):
        """Match live B intervals to the audited fly lineage without GPU work.

        A report explicitly authorizes the run and prior. Both dataset hashes,
        validation cursor, target count and fresh exposure must also agree.
        A revisit scores are deliberately excluded: each learner replays its own A.
        """
        root = self.comparison_report.parents[2]
        report = self.cached_json(self.comparison_report)
        if report.get('medium_run') != run.name or not report.get('prior_hash_equal'):
            return None
        config = self.cached_json(run / 'config.json')
        if config.get('prior_counts_sha256') != report.get('prior_hash'):
            return None
        dataset = config.get('data_sha256', {})
        reference_rows = []
        for name_key, stage in [('fly_same_individual_50k_stage', 'S4'),
                                ('fly_champion_later_stage', 'S≤14')]:
            name = report.get(name_key)
            if not name:
                continue
            reference = root / 'results' / name
            files = self.cached_json(reference / 'config.json').get('dataset_manifest', {}).get('files', {})
            if not all(dataset.get(key) and dataset[key] == files.get(key)
                       for key in ('train.npy', 'validation.npy')):
                continue
            reference_rows.extend((row, stage) for row in
                                  self.records(reference, 'lifelong_evaluation.jsonl', 1000))
        pairs = []
        for row in self.records(run, 'lifelong_evaluation.jsonl', 1000):
            curve, prior = row.get('B_curve', []), row.get('B_prior_curve', [])
            if not curve or len(curve) != len(prior):
                continue
            fresh = row.get('fresh_training_tokens', 0)
            candidates = [(other, stage) for other, stage in reference_rows
                          if other.get('fresh_validation_cursor') == row.get('fresh_validation_cursor')
                          and len(other.get('B_curve', [])) == len(curve)
                          and abs(other.get('bptt_train_tokens', -10000) - fresh) <= 128]
            if not candidates:
                continue
            other, stage = min(candidates, key=lambda item: abs(item[0]['bptt_train_tokens'] - fresh))
            medium_nll = float(np.mean(curve))
            fly_nll = float(np.mean(other['B_curve']))
            pairs.append({'fresh_tokens': fresh, 'fly_fresh_tokens': other['bptt_train_tokens'],
                          'medium_nll': medium_nll, 'fly_nll': fly_nll,
                          'reference_nll': float(np.mean(prior)),
                          'advantage': fly_nll - medium_nll, 'fly_stage': stage,
                          'medium_updates': row.get('optimizer_updates'),
                          'fly_updates': other.get('bptt_optimizer_updates'),
                          'target_start': row['fresh_validation_cursor'] - len(curve),
                          'target_end': row['fresh_validation_cursor'],
                          'targets': len(curve)})
        return {'pairs': pairs, 'parameters': report.get('trainable_parameters'),
                'matched_evaluations': len(pairs),
                'medium_wins': sum(pair['advantage'] > 0 for pair in pairs),
                'mean_advantage': float(np.mean([pair['advantage'] for pair in pairs])) if pairs else None,
                'speed_at_50k': report.get('speed'),
                'latest': pairs[-1] if pairs else None}

    def state(self, run_override=None):
        identity, run = self.selected_run(run_override)
        evaluations = self.records(run, 'lifelong_evaluation.jsonl', 1)
        constructor = self.read_json(run, 'config.json', {}).get('constructor', {})
        spatial = material_tensor_view(self.read_json(run, 'spatial.json', None), constructor)
        if spatial is not None:
            spatial['channels'] = constructor.get('channels')
        runs_info = []
        if self.registry and self.registry.exists():
            try:
                reg = json.loads(self.registry.read_text(encoding='utf-8-sig'))
                runs_info = [{'id': k, 'label': v.get('label', k)} for k, v in reg.get('runs', {}).items()]
            except Exception:
                pass
        return {'run': run.name, 'run_id': identity, 'run_path': str(run),
                'runs': runs_info,
                'server_time': time.time(),
                'progress': self.read_json(run, 'progress.json', {}),
                'spatial': spatial, 'history': self.records(run, 'metrics.jsonl'),
                'evaluation': evaluations[-1] if evaluations else None,
                'material_view': self.cached_json(run / 'material_view.json') or None,
                'comparison': self.comparison(run)}


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
    result['effective_row_speed'] = np.linalg.norm(factor, axis=-1).tolist()
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
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == '/api/state':
                query = urllib.parse.parse_qs(parsed.query)
                run_req = query.get('run', [None])[0]
                try:
                    data = store.state(run_override=run_req)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.send_error(503, f'Run registry unavailable: {error}')
                    return
                body = json.dumps(data, allow_nan=False).encode('utf-8')
                mime = 'application/json; charset=utf-8'
            elif self.path.split('?')[0] == '/':
                body, mime = page.read_bytes(), 'text/html; charset=utf-8'
            elif self.path.split('?')[0] == '/medium_flow.js':
                body, mime = page.with_name('medium_flow.js').read_bytes(), 'text/javascript; charset=utf-8'
            elif self.path.split('?')[0] == '/medium_structure.js':
                body, mime = page.with_name('medium_structure.js').read_bytes(), 'text/javascript; charset=utf-8'
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
