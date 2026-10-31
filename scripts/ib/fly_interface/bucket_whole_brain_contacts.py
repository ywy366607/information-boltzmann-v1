"""Stream every traced synapse contact of maleCNS v1.0 and bucket it by target skeleton shard.

Reads the public syn-partners (minconf 0.5, traced-only) Arrow IPC table one
record batch at a time over HTTP range requests; the 2.8 GB table is never
stored. Each contact keeps its pre/post cell (model order) and both
coordinates (8 nm voxels). Rows with a cell outside the model are counted.
Output: per post-shard raw int32 rows [post_cell, pre_cell, xp, yp, zp, xq, yq, zq]
(p = post point, q = pre point). Resumable by completed batch count.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.ipc as ipc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from http_range_file import HttpRangeFile

URL = ('https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
       'syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather')
COLUMNS = ['body_pre', 'body_post', 'x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post']


def main(args):
    graph = np.load(args.graph, allow_pickle=False)
    ids = graph['neuron_body_ids'].astype(np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'batches_done': 0, 'rows': 0, 'kept': 0, 'outside_model': 0, 'self_contacts': 0}
    source = (open(args.source, 'rb') if args.source else HttpRangeFile(URL))
    reader = ipc.open_file(source)
    total = reader.num_record_batches
    shards = (len(ids) + args.shard - 1) // args.shard
    if state['batches_done'] == 0:
        for s in range(shards):
            (args.output / f'post_{s:04d}.i32').unlink(missing_ok=True)
    else:
        # truncate any partial append beyond the last completed batch
        sizes = state.get('sizes', {})
        for s in range(shards):
            path = args.output / f'post_{s:04d}.i32'
            if path.exists():
                with open(path, 'r+b') as f:
                    f.truncate(int(sizes.get(str(s), 0)))
    handles = {s: open(args.output / f'post_{s:04d}.i32', 'ab') for s in range(shards)}

    def lookup(body):
        pos = np.minimum(np.searchsorted(sorted_ids, body), len(ids) - 1)
        hit = sorted_ids[pos] == body
        return np.where(hit, order[pos], -1)

    started = time.perf_counter()
    try:
        for b in range(state['batches_done'], total):
            batch = reader.get_batch(b)
            cols = {c: batch.column(batch.schema.get_field_index(c)).to_numpy(zero_copy_only=False) for c in COLUMNS}
            pre, post = lookup(cols['body_pre']), lookup(cols['body_post'])
            keep = (pre >= 0) & (post >= 0)
            state['outside_model'] += int((~keep).sum())
            selfc = keep & (pre == post)
            state['self_contacts'] += int(selfc.sum())
            keep &= ~selfc
            rows = np.stack([post[keep], pre[keep], cols['x_post'][keep], cols['y_post'][keep], cols['z_post'][keep],
                             cols['x_pre'][keep], cols['y_pre'][keep], cols['z_pre'][keep]], 1).astype(np.int32)
            shard = rows[:, 0] // args.shard
            sort = np.argsort(shard, kind='stable')
            rows, shard = rows[sort], shard[sort]
            cuts = np.flatnonzero(np.diff(shard)) + 1
            for part in np.split(np.arange(len(rows)), cuts):
                if len(part):
                    handles[int(shard[part[0]])].write(rows[part].tobytes())
            state['rows'] += int(batch.num_rows)
            state['kept'] += int(keep.sum())
            state['batches_done'] = b + 1
            if (b + 1) % 20 == 0 or b + 1 == total:
                for h in handles.values():
                    h.flush()
                state['sizes'] = {str(s): (args.output / f'post_{s:04d}.i32').stat().st_size for s in range(shards)}
                tmp = state_path.with_suffix('.tmp')
                tmp.write_text(json.dumps(state))
                tmp.replace(state_path)
                print(json.dumps({'batch': b + 1, 'of': total, 'kept': state['kept'],
                                  'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        for h in handles.values():
            h.close()
    state.update(status='complete', at=datetime.now(timezone.utc).isoformat(), source=URL, shard_size=args.shard,
                 columns=['post_cell', 'pre_cell', 'x_post', 'y_post', 'z_post', 'x_pre', 'y_pre', 'z_pre'],
                 units='8 nm voxels; cells in model neuron order')
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps({k: v for k, v in state.items() if k != 'sizes'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--output', type=Path, default=Path('E:/fly_stream_overflow/whole_brain_contacts_buckets'))
    p.add_argument('--shard', type=int, default=1024)
    p.add_argument('--source', type=Path, default=None, help='local copy of the syn-partners feather')
    main(p.parse_args())
