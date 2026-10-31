"""Link a complete sensory endpoint set in bounded, restartable batches.

This builds anatomical execution operands, not receptor-expression evidence.
Original pair identities, all native SWC nodes/roots, presynaptic polyads and
individual postsynaptic contacts remain distinct. No soma current is added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from compile_native_connectome_program import checksum


def positions(keys, query):
    """Index exact sorted keys; missing keys return -1 without clipping aliases."""
    at = np.searchsorted(keys, query)
    found = at < len(keys)
    if len(keys):
        found &= keys[np.minimum(at, len(keys) - 1)] == query
    else:
        found[:] = False
    return np.where(found, at, -1)


def geometry(path):
    swc = np.atleast_2d(np.loadtxt(path))
    if swc.shape[1] != 7 or not np.isfinite(swc).all():
        raise ValueError('Invalid native SWC columns/values')
    node = swc[:, 0].astype(np.int64)
    parent_id = swc[:, 6].astype(np.int64)
    if not np.array_equal(node, swc[:, 0]) or len(np.unique(node)) != len(node):
        raise ValueError('Nonintegral or duplicate SWC node identity')
    order = np.argsort(node)
    child = np.flatnonzero(parent_id >= 0)
    parent_slot = positions(node[order], parent_id[child])
    if np.any(parent_slot < 0):
        raise ValueError('SWC parent absent from supplied morphology')
    parent = order[parent_slot]
    xyz, radius = swc[:, 2:5] * .008, swc[:, 5] * .008
    length = np.linalg.norm(xyz[child] - xyz[parent], axis=1)
    if np.any(length <= 0) or np.any(radius <= 0):
        raise ValueError('Nonpositive native segment length/radius; no invented repair')
    # Every parent chain must terminate at an actual root. Disconnected trees
    # remain disconnected; a virtual common soma is never introduced.
    ancestors = np.full(len(node), -1, dtype=np.int64)
    ancestors[child] = parent
    settled = np.zeros(len(node), dtype=np.uint8)
    for start in range(len(node)):
        if settled[start]:
            continue
        trail, here = [], start
        while here >= 0 and settled[here] == 0:
            settled[here] = 1
            trail.append(here)
            here = int(ancestors[here])
        if here >= 0 and settled[here] == 1:
            raise ValueError('Cycle in supplied SWC')
        settled[trail] = 2
    volume = np.pi * length * (radius[child] ** 2 + radius[child] * radius[parent]
                                + radius[parent] ** 2) / 3.
    capacity = np.bincount(np.r_[child, parent], weights=np.r_[volume / 2, volume / 2],
                           minlength=len(node))
    isolated = capacity == 0
    capacity[isolated] = 4 * np.pi * radius[isolated] ** 3 / 3
    axial = np.pi * radius[child] * radius[parent] / length
    return node, xyz, radius, capacity, child, parent, axial


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def link_batch(index, bodies, args, ids, sorted_ids, id_order, pair_keys, pair_order,
               pairs, output_keys, output_rows, output_delay):
    output = args.output / f'geometry_{index:04d}.npz'
    note = output.with_suffix('.json')
    if output.exists() and note.exists():
        record = json.loads(note.read_text(encoding='utf-8'))
        if record['sha256'] != checksum(output):
            raise ValueError('Cached geometry digest mismatch')
        return record
    while True:
        absent = [b for b in bodies if not (args.morphology / f'{b}.swc').exists()]
        if not absent:
            break
        state = json.loads((args.morphology / 'manifest.json').read_text(encoding='utf-8'))
        if not args.wait_for_morphology or state['status'] != 'fetching':
            raise ValueError(f'Native morphology missing for batch {index}: {len(absent)} bodies')
        # Download jobs run independently; this can be interrupted/restarted.
        time.sleep(3)
    incoming = pd.read_feather(args.incoming / f'contacts_{index:04d}.feather')
    outgoing = pd.read_feather(args.outgoing / f'contacts_{index:04d}.feather')

    def cell_index(values):
        slot = positions(sorted_ids, values)
        return np.where(slot >= 0, id_order[np.maximum(slot, 0)], -1)

    a = cell_index(incoming.body_pre.to_numpy(dtype=np.int64))
    b = cell_index(incoming.body_post.to_numpy(dtype=np.int64))
    match = positions(pair_keys, a * len(ids) + b)
    valid = (a >= 0) & (b >= 0) & (match >= 0)
    incoming = incoming.loc[valid].reset_index(drop=True)
    incoming_pair = pair_order[match[valid]]
    a = cell_index(outgoing.body_pre.to_numpy(dtype=np.int64))
    b = cell_index(outgoing.body_post.to_numpy(dtype=np.int64))
    match = positions(output_keys, a * len(ids) + b)
    valid = (a >= 0) & (b >= 0) & (match >= 0)
    outgoing = outgoing.loc[valid].reset_index(drop=True)
    outgoing_match = match[valid]
    # Cell batches are nonoverlapping. Within a batch DISTINCT (s,t) is
    # requested upstream; verify physical identities rather than drop repeats.
    contact_columns = ['body_pre', 'body_post', 'x_pre', 'y_pre', 'z_pre',
                       'x_post', 'y_post', 'z_post']
    if incoming.duplicated(contact_columns).any() or outgoing.duplicated(contact_columns).any():
        raise ValueError('Duplicate physical contact identities')
    in_groups = incoming.groupby('body_post', sort=False).indices
    out_groups = outgoing.groupby('body_pre', sort=False).indices
    pieces = {k: [] for k in ('node_ids', 'xyz_um', 'radius_um', 'volume', 'owner',
                              'u', 'v', 'g', 'site_ids', 'site_node', 'site_owner')}
    in_node = np.empty(len(incoming), dtype=np.int32)
    out_site = np.empty(len(outgoing), dtype=np.int32)
    in_distance = np.empty(len(incoming), dtype=np.float32)
    site_distances = []
    node_offset = site_offset = 0
    roots = 0
    for owner, body in enumerate(bodies):
        node, xyz, radius, volume, u, v, g = geometry(args.morphology / f'{body}.swc')
        spatial = cKDTree(xyz)
        pieces['node_ids'].append(np.c_[np.full(len(node), body, dtype=np.int64), node])
        pieces['xyz_um'].append(xyz)
        pieces['radius_um'].append(radius)
        pieces['volume'].append(volume)
        pieces['owner'].append(np.full(len(node), owner, dtype=np.int32))
        pieces['u'].append((u + node_offset).astype(np.int32))
        pieces['v'].append((v + node_offset).astype(np.int32))
        pieces['g'].append(g)
        indices = in_groups.get(body, np.empty(0, dtype=np.int64))
        if len(indices):
            distance, at = spatial.query(incoming.iloc[indices][['x_post', 'y_post', 'z_post']].to_numpy() * .008)
            in_node[indices] = at + node_offset
            in_distance[indices] = distance
        indices = out_groups.get(body, np.empty(0, dtype=np.int64))
        coords = outgoing.iloc[indices][['x_pre', 'y_pre', 'z_pre']].to_numpy(dtype=np.int64)
        sites, inverse = np.unique(coords, axis=0, return_inverse=True)
        distance, at = spatial.query(sites * .008)
        pieces['site_ids'].append(np.c_[np.full(len(sites), body, dtype=np.int64), sites])
        pieces['site_node'].append((at + node_offset).astype(np.int32))
        pieces['site_owner'].append(np.full(len(sites), owner, dtype=np.int32))
        site_distances.append(distance)
        out_site[indices] = inverse + site_offset
        node_offset += len(node)
        site_offset += len(sites)
        roots += len(node) - len(u)
    packed = {name: np.concatenate(value) for name, value in pieces.items()}
    packed.update(cell_body_ids=np.asarray(bodies, dtype=np.int64),
                  cell_index=cell_index(np.asarray(bodies, dtype=np.int64)),
                  incoming_pair=incoming_pair.astype(np.int32), incoming_node=in_node,
                  incoming_source=cell_index(incoming.body_pre.to_numpy(dtype=np.int64)).astype(np.int32),
                  incoming_source_site_ids=incoming[['body_pre', 'x_pre', 'y_pre', 'z_pre']].to_numpy(dtype=np.int64),
                  incoming_delay=pairs.original_delay.to_numpy()[incoming_pair],
                  incoming_projection_um=in_distance,
                  outgoing_master_edge=output_rows[outgoing_match].astype(np.int32),
                  outgoing_site=out_site, outgoing_delay=output_delay[outgoing_match],
                  outgoing_post=cell_index(outgoing.body_post.to_numpy(dtype=np.int64)).astype(np.int32),
                  outgoing_target_site_ids=outgoing[['body_post', 'x_post', 'y_post', 'z_post']].to_numpy(dtype=np.int64),
                  site_projection_um=np.concatenate(site_distances),
                  incoming_confidence=incoming[['confidence_pre', 'confidence_post']].to_numpy(dtype=np.float32),
                  outgoing_confidence=outgoing[['confidence_pre', 'confidence_post']].to_numpy(dtype=np.float32))
    # Per-contact counts are deliberately not normalized into an average
    # gain. The kinetic binder must declare resource sharing and pair scaling.
    temporary = output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **packed)
    temporary.replace(output)
    record = {'batch': index, 'cells': len(bodies), 'nodes': node_offset, 'roots': roots,
              'sites': site_offset, 'incoming_contacts': len(incoming),
              'incoming_pair_rows': sorted(map(int, np.unique(incoming_pair))),
              'outgoing_contacts': len(outgoing),
              'outgoing_pair_count': len(np.unique(packed['outgoing_master_edge'])),
              'incoming_projection_max_um': float(in_distance.max(initial=0)),
              'site_projection_max_um': float(packed['site_projection_um'].max(initial=0)),
              'sha256': checksum(output), 'uncompressed_array_bytes': sum(v.nbytes for v in packed.values())}
    atomic_json(note, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'incoming_pair_rows'}), flush=True)
    return record


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
                 for folder in (args.incoming, args.outgoing)]
    if any(m['status'] != 'complete' or m.get('errors') for m in manifests):
        raise ValueError('Both contact acquisitions must be complete')
    if manifests[0]['bodies'] != manifests[1]['bodies'] or manifests[0]['batch_size'] != manifests[1]['batch_size']:
        raise ValueError('Incoming/outgoing endpoint batches differ')
    pairs = pd.read_feather(args.pairs).rename(columns={'delay_tier': 'original_delay'})
    with np.load(args.master, allow_pickle=False) as master:
        ids = master['neuron_body_ids']
        pre, post, lag = master['edge_pre'], master['edge_post'], master['edge_delay']
    id_order = np.argsort(ids)
    sorted_ids = ids[id_order]
    pre_at = positions(sorted_ids, pairs.body_pre.to_numpy(dtype=np.int64))
    post_at = positions(sorted_ids, pairs.body_post.to_numpy(dtype=np.int64))
    if np.any(pre_at < 0) or np.any(post_at < 0):
        raise ValueError('Original return pair includes an absent cell')
    pair_key = id_order[pre_at] * len(ids) + id_order[post_at]
    pair_order = np.argsort(pair_key)
    pair_keys = pair_key[pair_order]
    if len(np.unique(pair_keys)) != len(pairs):
        raise ValueError('Duplicate original return pair identity')
    targets = id_order[positions(sorted_ids, np.asarray(manifests[0]['bodies'], dtype=np.int64))]
    selected = np.flatnonzero(np.isin(pre, targets))
    output_key = pre[selected].astype(np.int64) * len(ids) + post[selected]
    output_order = np.argsort(output_key)
    output_rows = selected[output_order]
    output_keys = output_key[output_order]
    output_delay = lag[output_rows]
    del pre, post, lag, selected, output_order, output_key
    identity = {'version': 1, 'dataset': 'male-cns:v1.0', 'master_sha256': checksum(args.master),
                'pairs_sha256': checksum(args.pairs), 'required_pairs': len(pairs),
                'bodies': manifests[0]['bodies'], 'batch_size': manifests[0]['batch_size']}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(previous[k] != value for k, value in identity.items()):
            raise ValueError('Incompatible immutable anatomical linker provenance')
    completed = []
    seen = np.zeros(len(pairs), dtype=bool)

    def save(status):
        missing = np.flatnonzero(~seen)
        record = {**identity, 'at': datetime.now(timezone.utc).isoformat(), 'status': status,
                  'completed_batches': len(completed), 'covered_pairs': int(seen.sum()),
                  'unlocalized_pair_rows': missing.tolist(),
                  'totals': {k: sum(r[k] for r in completed) for k in
                             ('cells', 'nodes', 'roots', 'sites', 'incoming_contacts', 'outgoing_contacts',
                              'outgoing_pair_count', 'uncompressed_array_bytes')},
                  'runtime_binding_complete': False, 'runtime_changed': False, 'checkpoint_changed': False,
                  'confidence_policy': 'All actual contact confidence values retained without threshold.',
                  'missing_execution_evidence': ['receptor/effector identities and locations',
                      'local electrical vs chemical action and release coupling',
                      'source release routing and complete-state runtime integration'],
                  'scope': 'Complete local anatomical operands; projection onto supplied morphology is recorded, not exact ultrastructural membrane segmentation. No soma fallback, pooling, new receptor-expression claim or active training.'}
        atomic_json(manifest_path, record)
    save('linking')
    size = identity['batch_size']
    for index, start in enumerate(range(0, len(identity['bodies']), size)):
        record = link_batch(index, identity['bodies'][start:start + size], args, ids, sorted_ids,
                            id_order, pair_keys, pair_order, pairs, output_keys, output_rows, output_delay)
        completed.append(record)
        seen[record['incoming_pair_rows']] = True
        save('linking')
    if not seen.all():
        pairs.iloc[np.flatnonzero(~seen)].to_feather(args.output / 'unlocalized_pairs.feather')
    save('complete' if seen.all() else 'incomplete_contact_coverage')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--master', type=Path, default=Path('data/malecns_v1/fly_reservoir_delayed.npz'))
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--incoming', type=Path, required=True)
    p.add_argument('--outgoing', type=Path, required=True)
    p.add_argument('--morphology', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-for-morphology', action='store_true')
    main(p.parse_args())
