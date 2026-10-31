"""Project every bucketed synapse point onto the supplied skeleton of its own cell.

stage post: per target shard, each contact's postsynaptic point -> nearest
node of the target cell; rows are re-bucketed by the presynaptic cell's shard.
stage pre:  per source shard, each presynaptic point -> nearest node of the
source cell; one release site per distinct T-bar point (polyadic synapses
share a site). Nearest-node search never crosses cells (cells are separated
by a 1 m offset in a per-shard KD-tree). Distances are kept as evidence.
Contacts of cells without a supplied skeleton keep node -1 (recorded).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

VOXEL_NM = 8.0
SEPARATION = 1e9  # nm between cells inside one KD-tree


def load_shard(skeletons, index):
    with np.load(skeletons / f'skeletons_{index:04d}.npz') as z:
        counts = z['node_count'].astype(np.int64)
        xyz = z['xyz_nm'].astype(np.float64)
        bodies = z['body_ids']
    owner = np.repeat(np.arange(len(counts)), counts)
    return xyz, owner, counts, bodies


def project(xyz, owner, cell_local, points_vox):
    shifted = xyz.copy()
    shifted[:, 0] += owner * SEPARATION
    tree = cKDTree(shifted, balanced_tree=False, compact_nodes=False)
    query = points_vox.astype(np.float64) * VOXEL_NM
    query[:, 0] += cell_local * SEPARATION
    dist, node = tree.query(query, workers=-1)
    return node.astype(np.int64), dist.astype(np.float32)


def stage_post(args):
    out_dir = args.work / 'pre_buckets'
    out_dir.mkdir(exist_ok=True)
    handles = {}
    done_path = args.work / 'post_done.json'
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    shards = sorted(int(p.stem.split('_')[1]) for p in args.buckets.glob('post_*.i32'))
    if done:
        sizes = json.loads((args.work / 'pre_sizes.json').read_text())
        for name, size in sizes.items():
            with open(out_dir / name, 'r+b') as f:
                f.truncate(size)
    for s in shards:
        if s in done:
            continue
        started = time.perf_counter()
        rows = np.fromfile(args.buckets / f'post_{s:04d}.i32', dtype=np.int32).reshape(-1, 8)
        xyz, owner, counts, _ = load_shard(args.skeletons, s)
        cell_local = rows[:, 0].astype(np.int64) - s * args.shard
        has = counts[cell_local] > 0
        node = np.full(len(rows), -1, np.int64)
        dist = np.full(len(rows), np.nan, np.float32)
        if has.any():
            node[has], dist[has] = project(xyz, owner, cell_local[has], rows[has, 2:5])
            if np.any(owner[node[has]] != cell_local[has]):
                raise ValueError('Projection crossed cells')
        out = np.empty((len(rows), 7), np.int32)
        out[:, 0] = rows[:, 1]            # pre cell
        out[:, 1:4] = rows[:, 5:8]        # pre point
        out[:, 4] = rows[:, 0]            # post cell
        out[:, 5] = node                  # post node, local to its shard
        out[:, 6] = dist.view(np.int32)   # post projection distance (float32 bits)
        pre_shard = out[:, 0] // args.shard
        order = np.argsort(pre_shard, kind='stable')
        out, pre_shard = out[order], pre_shard[order]
        for part in np.split(np.arange(len(out)), np.flatnonzero(np.diff(pre_shard)) + 1):
            if len(part):
                name = f'pre_{int(pre_shard[part[0]]):04d}.i32'
                with open(out_dir / name, 'ab') as f:
                    f.write(out[part].tobytes())
        done.add(s)
        sizes = {p.name: p.stat().st_size for p in out_dir.glob('pre_*.i32')}
        (args.work / 'pre_sizes.json').write_text(json.dumps(sizes))
        done_path.write_text(json.dumps(sorted(done)))
        if args.delete_inputs:
            (args.buckets / f'post_{s:04d}.i32').unlink()
        finite = dist[np.isfinite(dist)]
        print(json.dumps({'stage': 'post', 'shard': s, 'contacts': int(len(rows)), 'no_skeleton': int((~has).sum()),
                          'dist_nm_median': float(np.median(finite)) if finite.size else None,
                          'dist_nm_p99': float(np.percentile(finite, 99)) if finite.size else None,
                          'seconds': round(time.perf_counter() - started, 1)}), flush=True)


def stage_pre(args):
    in_dir = args.work / 'pre_buckets'
    out_dir = args.work / 'contacts'
    out_dir.mkdir(exist_ok=True)
    for path in sorted(in_dir.glob('pre_*.i32')):
        s = int(path.stem.split('_')[1])
        target = out_dir / f'contacts_{s:04d}.npz'
        if target.exists():
            continue
        started = time.perf_counter()
        rows = np.fromfile(path, dtype=np.int32).reshape(-1, 7)
        xyz, owner, counts, _ = load_shard(args.skeletons, s)
        cell_local = rows[:, 0].astype(np.int64) - s * args.shard
        # one release site per distinct T-bar point of a cell
        key = np.ascontiguousarray(rows[:, :4]).view(np.dtype((np.void, 16))).reshape(-1)
        unique, site_of = np.unique(key, return_inverse=True)
        first = np.zeros(len(unique), np.int64)
        first[site_of[::-1]] = np.arange(len(rows))[::-1]
        site_cell_local = cell_local[first]
        has = counts[site_cell_local] > 0
        site_node = np.full(len(unique), -1, np.int64)
        site_dist = np.full(len(unique), np.nan, np.float32)
        if has.any():
            site_node[has], site_dist[has] = project(xyz, owner, site_cell_local[has], rows[first[has], 1:4])
        np.savez(target.with_suffix('.tmp.npz'),
                 pre_cell=rows[:, 0], post_cell=rows[:, 4], site=site_of.astype(np.int32),
                 post_node_local=rows[:, 5], post_dist_nm=rows[:, 6].view(np.float32),
                 site_cell=rows[first, 0], site_node_local=site_node.astype(np.int32),
                 site_dist_nm=site_dist, site_point_vox=rows[first, 1:4])
        target.with_suffix('.tmp.npz').replace(target)
        if args.delete_inputs:
            path.unlink()
        finite = site_dist[np.isfinite(site_dist)]
        print(json.dumps({'stage': 'pre', 'shard': s, 'contacts': int(len(rows)), 'release_sites': int(len(unique)),
                          'dist_nm_median': float(np.median(finite)) if finite.size else None,
                          'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    (args.work / 'contacts' / 'done.json').write_text(json.dumps({'at': datetime.now(timezone.utc).isoformat()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage', choices=('post', 'pre'), required=True)
    p.add_argument('--buckets', type=Path, default=Path('E:/fly_stream_overflow/whole_brain_contacts_buckets'))
    p.add_argument('--skeletons', type=Path, default=Path('F:/fly_stream_overflow/whole_brain_skeletons_v1'))
    p.add_argument('--work', type=Path, default=Path('F:/fly_stream_overflow/whole_brain_contacts_v1'))
    p.add_argument('--shard', type=int, default=1024)
    p.add_argument('--delete-inputs', action='store_true')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    stage_post(a) if a.stage == 'post' else stage_pre(a)
