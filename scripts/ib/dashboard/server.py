"""Fly harness dashboard: a local, read-mostly view of the living individual.

Serves the page (static/index.html) and JSON endpoints read from the runs' own files. Nothing here touches training:
- E:/fly_stream_overflow/<run>/status: the chunk ends and the living causal gates;
- scores_cN.jsonl: every window's pre-update scores, the unigram on the same words, per-word dopamine, and in the
  mother world the speakers and the harness monitor;
- mother.jsonl: the conversation;
- ai_trace.jsonl: the acts;
- cell_health_cN.json: firing per class.

The only writes are the harness configuration (data/harness/*.json): the mother and the parenting plan, read by the
trainer at the start of a chunk once wired.

Run:  python scripts/ib/dashboard/server.py [--port 8765]   then open http://127.0.0.1:8765
Bound to 127.0.0.1 only.
"""
import argparse
import glob
import json
import os
import re
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get('FLY_RUNS', 'E:/fly_stream_overflow'))
STATIC = Path(__file__).resolve().parent / 'static'
HARNESS = ROOT / 'data' / 'harness'
GRAPH = ROOT / 'data' / 'malecns_v1' / 'fly_reservoir_coba_r2.npz'
ANNOTATIONS = ROOT / 'data' / 'malecns_v1' / 'body-annotations.feather'
BRAIN_CACHE = HARNESS / 'brain_points.json'
KEEP_CLASSES = ('Kenyon_Cell', 'MBON', 'DAN', 'ALPN', 'ALLN', 'olfactory', 'gustatory', 'CX')

NUM = r'([-+]?\d*\.?\d+(?:e[-+]?\d+)?)'
GATE = re.compile(r'^(\d\d:\d\d) gate c(\d+) \(living\): \(1\) brain contribution ' + NUM + r' \+- ' + NUM
                  + r' \[min ' + NUM + r' med ' + NUM + r' max ' + NUM + r'\].*?\(brain ([\d.]+), noedges ([\d.]+), unigram ([\d.]+)'
                  + r'.*?\(2\) surprise~phasic DA (\S+) next (\S+) DA cells (\S+).*?\(L\) latched (\d+), worth (.*)$')
CHUNK = re.compile(r'^(\d\d:\d\d) (?:born: )?chunk (\d+) done: \[ACTIVE EVAL (\d+)\] fresh B NLL=([\d.]+); A revisit=([\d.]+)')
AG = re.compile(r'^(\d\d:\d\d) gate c(\d+) AG: brain fresh B NLL=([\d.]+);.*?noedges fresh B NLL=([\d.]+)')
ACTS = re.compile(r'^(\d\d:\d\d) gate c(\d+) acts \(living\): (\{.*\})$')
_lock = threading.Lock()


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def runs():
    out = []
    for d in RUNS.iterdir():
        files = [d / 'status', d / 'mother.jsonl', d / 'scores.jsonl'] + sorted(d.glob('scores_c*.jsonl'))[-1:] if d.is_dir() else []
        files = [f for f in files if f.exists()]
        if files:
            out.append({'name': d.name, 'mtime': max(f.stat().st_mtime for f in files),
                        'mother': (d / 'mother.jsonl').exists(), 'acts': (d / 'ai_trace.jsonl').exists()})
    return sorted(out, key=lambda r: -r['mtime'])


def parse_status(run):
    path = RUNS / run / 'status'
    rep = {'chunks': [], 'gates': [], 'ag': [], 'acts': [], 'events': []}
    if not path.exists():
        return rep
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if m := GATE.match(line):
            worth = re.match(NUM + r' \+- ' + NUM, m.group(15))
            rep['gates'].append({'time': m.group(1), 'c': int(m.group(2)), 'mean': _f(m.group(3)), 'se': _f(m.group(4)),
                                 'min': _f(m.group(5)), 'median': _f(m.group(6)), 'max': _f(m.group(7)),
                                 'brain': _f(m.group(8)), 'noedges': _f(m.group(9)), 'unigram': _f(m.group(10)),
                                 'da': _f(m.group(11)), 'da_next': _f(m.group(12)), 'da_cells': _f(m.group(13)),
                                 'latched': int(m.group(14)),
                                 'latch_worth': _f(worth.group(1)) if worth else None,
                                 'latch_se': _f(worth.group(2)) if worth else None})
        elif m := CHUNK.match(line):
            rep['chunks'].append({'time': m.group(1), 'c': int(m.group(2)), 'words': int(m.group(3)),
                                  'fresh_b': _f(m.group(4)), 'revisit': _f(m.group(5))})
        elif m := AG.match(line):
            rep['ag'].append({'time': m.group(1), 'c': int(m.group(2)), 'brain': _f(m.group(3)), 'noedges': _f(m.group(4))})
        elif m := ACTS.match(line):
            try:
                d = json.loads(m.group(3).replace('NaN', 'null'))
            except json.JSONDecodeError:
                d = {}
            rep['acts'].append({'time': m.group(1), 'c': int(m.group(2)), **d})
        elif line:
            rep['events'].append(line)
    rep['events'] = rep['events'][-40:]
    return rep


def windows(run, last=400):
    """Per window: mean score, unigram, mean phasic dopamine, speakers; the latest `last` windows over all chunks."""
    rows = []
    files = sorted(glob.glob(str(RUNS / run / 'scores_c*.jsonl')), key=lambda p: int(re.findall(r'c(\d+)', p)[-1]))
    files += [str(RUNS / run / 'scores.jsonl')] if (RUNS / run / 'scores.jsonl').exists() else []
    for f in files:
        for line in open(f, encoding='utf-8'):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            da = r.get('word_da') or []
            sp = r.get('speakers') or []
            rows.append({'trained': r['trained'], 'nll': float(np.mean(r['scores'])), 'unigram': float(np.mean(r['unigram'])),
                         'phasic': float(np.mean([x[0] for x in da])) if da else None,
                         'fly_words': sum(1 for s in sp if s == 'fly') if sp else None,
                         'monitor': r.get('monitor')})
    rows.sort(key=lambda r: r['trained'])
    return rows[-last:]


def mother(run, last=200):
    path = RUNS / run / 'mother.jsonl'
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()[-last:]
    return [json.loads(l) for l in lines if l.strip()]


def health(run):
    files = sorted(glob.glob(str(RUNS / run / 'cell_health_c*.json')), key=lambda p: int(re.findall(r'c(\d+)', p)[-1]))
    if not files:
        return {}
    d = json.load(open(files[-1], encoding='utf-8'))
    groups = {k[6:]: v for k, v in d.get('groups', {}).items() if k.startswith('class:')}
    return {'file': Path(files[-1]).name, 'whole_brain': d.get('whole_brain'), 'flags': d.get('flags'),
            'classes': {k: {'mean_hz': v['rate_min_median_mean_max'][2] * 1000.0, 'silent': v['silent_frac'],
                            'n': v['n'], 'follows_text': v.get('follows_text')} for k, v in groups.items()}}


def brain_points(n_rest=24000, seed=0):
    """Every cell of the key classes plus a random sample of the rest, coordinates normalised to [-1, 1]."""
    with _lock:
        if BRAIN_CACHE.exists():
            return json.loads(BRAIN_CACHE.read_text(encoding='utf-8'))
        import pandas as pd
        g = np.load(GRAPH, allow_pickle=True)
        cls = pd.read_feather(ANNOTATIONS).set_index('bodyId').reindex(g['neuron_body_ids'])['class'].astype(str).values
        xyz = g['coords_um'].astype(np.float64)
        keep = np.isin(cls, KEEP_CLASSES)
        rest = np.flatnonzero(~keep)
        rng = np.random.default_rng(seed)
        idx = np.concatenate([np.flatnonzero(keep), rng.choice(rest, min(n_rest, rest.size), replace=False)])
        c = xyz[idx]
        centre, scale = (c.max(0) + c.min(0)) / 2, (c.max(0) - c.min(0)).max() / 2
        c = (c - centre) / scale
        names = sorted(set(cls[idx]))
        code = {n: i for i, n in enumerate(names)}
        rep = {'classes': names, 'n_total': int(len(cls)),
               'points': np.round(c, 4).tolist(), 'class': [code[x] for x in cls[idx]]}
        HARNESS.mkdir(parents=True, exist_ok=True)
        BRAIN_CACHE.write_text(json.dumps(rep), encoding='utf-8')
        return rep


DEFAULT_CONFIG = {
    'mother': {'backend': 'gemini', 'model': 'gemini-2.5-flash', 'temperature': 0.9, 'max_reply_tokens': 80,
               'history_turns': 20, 'fly_temperature': 1.0, 'max_fly_words': 16,
               'persona': 'A warm, patient mother talking with a very young child who is just learning to speak. '
                          'She answers in one or two short, simple sentences, responds to what the child said, '
                          'asks simple questions, names things, and sometimes starts a new topic.'},
    'parenting': {'phases': [{'name': 'listening', 'words': 40960, 'world': 'text', 'note': 'read to the newborn'},
                             {'name': 'babbling', 'words': 20480, 'world': 'mother', 'note': 'she answers any sound'}],
                  'topics': ['animals', 'food', 'family', 'the weather', 'colours', 'the garden'],
                  'revisit_every_words': 8192, 'switch_topic_every_turns': 12},
    'harness': {'alarm_repeat_bigrams': 0.5, 'alarm_identical_turns': 3, 'alarm_brain_contribution_le': 0.0,
                'check_every_words': 4096},
}


def config(name, body=None):
    HARNESS.mkdir(parents=True, exist_ok=True)
    path = HARNESS / f'{name}.json'
    if body is not None:
        json.loads(body)                                  # must be valid JSON
        path.write_text(body, encoding='utf-8')
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        run = (q.get('run') or [''])[0]
        if run and ('/' in run or '\\' in run or '..' in run):
            return self._json({'error': 'bad run'}, 400)
        try:
            if u.path == '/api/runs':
                return self._json(runs())
            if u.path == '/api/status':
                return self._json(parse_status(run))
            if u.path == '/api/windows':
                return self._json(windows(run, int((q.get('last') or ['400'])[0])))
            if u.path == '/api/mother':
                return self._json(mother(run))
            if u.path == '/api/health':
                return self._json(health(run))
            if u.path == '/api/brain':
                return self._json(brain_points())
            if u.path == '/api/config':
                return self._json(config('raising'))
        except Exception as exc:                        # a half-written file: report, never crash the server
            return self._json({'error': repr(exc)}, 500)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != '/api/config':
            return self._json({'error': 'not found'}, 404)
        body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8')
        try:
            return self._json(config('raising', body))
        except json.JSONDecodeError as exc:
            return self._json({'error': f'invalid JSON: {exc}'}, 400)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    print(f'Fly harness dashboard on http://127.0.0.1:{a.port}  (runs: {RUNS})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
