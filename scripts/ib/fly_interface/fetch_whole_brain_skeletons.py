"""Fetch every traced cell's supplied skeleton and store it compactly, resumably.

Source: the public maleCNS v1.0 SWC objects (the same coarse skeletons already
used for KC, APL, DAN and sensory geometry). Every node and root is kept; only
the storage changes: per shard of cells, float32 coordinates (nm) and radius,
int32 parent index local to the cell (-1 = root). The text is never stored.
Cells without a supplied skeleton are recorded, never invented.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pandas as pd
import requests

URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc'
_local = threading.local()


def fetch(body):
    session = getattr(_local, 'session', None)
    if session is None:
        session = _local.session = requests.Session()
    for attempt in range(5):
        try:
            r = session.get(URL.format(body), timeout=(10, 60))
            if r.status_code == 404:
                return body, None
            r.raise_for_status()
            if r.content.lstrip().startswith(b'<'):
                raise ValueError('invalid payload')
            return body, r.content
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def parse(text):
    table = pd.read_csv(io.BytesIO(text), sep=r'\s+', comment='#', header=None,
                        names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'], engine='c')
    ids = table['id'].to_numpy(np.int64)
    order = np.argsort(ids)
    local = np.full(len(ids), -1, np.int64)
    parent_ids = table['parent'].to_numpy(np.int64)
    pos = np.searchsorted(ids[order], parent_ids)
    pos = np.minimum(pos, len(ids) - 1)
    valid = (parent_ids >= 0) & (ids[order][pos] == parent_ids)
    if np.any((parent_ids >= 0) & ~valid):
        raise ValueError('SWC parent missing')
    local[valid] = order[pos[valid]]
    xyz = table[['x', 'y', 'z']].to_numpy(np.float32)
    radius = table['r'].to_numpy(np.float32)
    return xyz, radius, local.astype(np.int32)


def main(args):
    graph = np.load(args.graph, allow_pickle=False)
    bodies = graph['neuron_body_ids'].astype(np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    shards = [bodies[i:i + args.shard] for i in range(0, len(bodies), args.shard)]
    manifest_path = args.output / 'manifest.json'
    summary = {'dataset': 'male-cns:v1.0', 'source': URL.format('{body}'), 'cells': int(len(bodies)),
               'shard_size': args.shard, 'shards': len(shards)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, chunk in enumerate(shards):
            path = args.output / f'skeletons_{index:04d}.npz'
            if path.exists():
                continue
            if shutil.disk_usage(args.output).free < args.reserve_gb * 1024 ** 3:
                raise OSError('Disk reserve reached; skeleton cache retained')
            started = time.perf_counter()
            xyz, radius, parent, counts, missing = [], [], [], [], []
            for body, text in pool.map(fetch, chunk.tolist()):
                if text is None:
                    missing.append(body)
                    counts.append(0)
                    continue
                a, r, p = parse(text)
                xyz.append(a); radius.append(r); parent.append(p); counts.append(len(r))
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary, body_ids=chunk, node_count=np.asarray(counts, np.int32),
                     xyz_nm=np.concatenate(xyz) if xyz else np.zeros((0, 3), np.float32),
                     radius_nm=np.concatenate(radius) if radius else np.zeros(0, np.float32),
                     parent=np.concatenate(parent) if parent else np.zeros(0, np.int32),
                     missing=np.asarray(missing, np.int64))
            temporary.replace(path)
            print(json.dumps({'shard': index, 'of': len(shards), 'nodes': int(sum(counts)),
                              'missing': len(missing), 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    totals = {'nodes': 0, 'missing': 0, 'bytes': 0}
    for index in range(len(shards)):
        path = args.output / f'skeletons_{index:04d}.npz'
        with np.load(path) as z:
            totals['nodes'] += int(z['node_count'].sum())
            totals['missing'] += int(len(z['missing']))
        totals['bytes'] += path.stat().st_size
    summary.update(totals, status='complete', at=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--output', type=Path, default=Path('F:/fly_stream_overflow/whole_brain_skeletons_v1'))
    p.add_argument('--shard', type=int, default=1024)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--reserve-gb', type=float, default=3.0)
    main(p.parse_args())
