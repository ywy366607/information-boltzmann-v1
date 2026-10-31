"""Locate original KC axo-axonal contacts; keep fast somatic weights untouched."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from build_native_kc_terminals import tree


def records(value):
    return np.ascontiguousarray(value, dtype=np.int64).view(
        np.dtype([(str(i), '<i8') for i in range(value.shape[1])])).reshape(-1)


def build(kc_geometry, raw, graph_path, output):
    if output.exists():
        raise ValueError('Do not overwrite an anatomical cache')
    with np.load(kc_geometry, allow_pickle=False) as z:
        data = {key: z[key] for key in ('kc_index', 'node_ids', 'owner', 'site_ids',
                 'fast_edge', 'fast_pre', 'fast_post', 'fast_delay')}
    with np.load(graph_path, allow_pickle=False) as z:
        bodies = z['neuron_body_ids']
    kc = data['kc_index']
    body_index = {int(bodies[i]): int(i) for i in kc}
    selected = np.isin(data['fast_post'], kc)
    edge, first = np.unique(data['fast_edge'][selected], return_index=True)
    pre, post, lag = [data['fast_' + k][selected][first] for k in ('pre', 'post', 'delay')]
    n = len(bodies)
    order = np.argsort(pre * n + post)
    keys, edge, lag = (pre * n + post)[order], edge[order], lag[order]
    site_key = records(data['site_ids'])
    site_order = np.argsort(site_key)
    sorted_sites = site_key[site_order]
    offsets = np.r_[0, np.cumsum(np.bincount(data['owner'], minlength=len(kc)))]
    target_tree = {}
    for slot, index in enumerate(kc):
        body = int(bodies[index])
        identity, xyz, *_ = tree(raw / f'{body}.swc')
        start, end = offsets[slot:slot + 2]
        if not np.array_equal(identity, data['node_ids'][start:end, 1]):
            raise ValueError('Recipient morphology identity mismatch')
        target_tree[body] = (cKDTree(xyz), int(start))
    parts = {name: [] for name in ('edge', 'site', 'node', 'delay', 'count')}
    total, maximum_distance = 0, 0.
    for batch, path in enumerate(sorted(raw.glob('kc_outgoing_*.feather'))):
        table = pd.read_feather(path)
        table = table[~table.post_in_calyx & table.body_post.isin(body_index)].copy()
        if not len(table):
            continue
        query = table.body_pre.map(body_index).to_numpy() * n + table.body_post.map(body_index).to_numpy()
        at = np.searchsorted(keys, query)
        valid = at < len(keys)
        valid &= keys[np.minimum(at, len(keys) - 1)] == query
        table, at = table.loc[valid].reset_index(drop=True), at[valid]
        if not len(table):
            continue
        source = records(table[['body_pre', 'x_pre', 'y_pre', 'z_pre']].to_numpy())
        loc = np.searchsorted(sorted_sites, source)
        if np.any(loc >= len(sorted_sites)) or not np.array_equal(sorted_sites[loc], source):
            raise ValueError('KC lateral source has no existing release site')
        node = np.empty(len(table), dtype=np.int64)
        for body, indices in table.groupby('body_post', sort=False).indices.items():
            spatial, start = target_tree[int(body)]
            distance, local = spatial.query(table.iloc[indices][['x_post','y_post','z_post']].to_numpy() * .008)
            node[indices] = local + start
            maximum_distance = max(maximum_distance, float(distance.max(initial=0)))
        grouped = pd.DataFrame({'edge': edge[at], 'site': site_order[loc], 'node': node,
                                'delay': lag[at]}).groupby(['edge', 'site', 'node', 'delay'], sort=True).size().reset_index(name='count')
        for key in parts:
            parts[key].append(grouped[key].to_numpy(dtype=np.int64))
        total += len(table)
        print(json.dumps({'batches': batch + 1, 'contacts': total}), flush=True)
    packed = {key: np.concatenate(value) for key, value in parts.items()}
    packed['kc_index'] = kc
    if int(packed['count'].sum()) != total:
        raise ValueError('Axonal contact count was not conserved')
    # Check every axon-only connection previously pinned out of the soma.
    dend = pd.read_feather(graph_path.parent / 'kc_kc_dendritic_synapses.feather')
    # The calyx table is sparse: axon-only connections are ABSENT, not rows
    # with a literal zero. Testing only explicit zeros would certify nothing.
    dend = dend[(dend.dendritic_synapses > 0) & dend.body_pre.isin(body_index) & dend.body_post.isin(body_index)]
    calyx_key = dend.body_pre.map(body_index).to_numpy() * n + dend.body_post.map(body_index).to_numpy()
    expected = np.unique(edge[~np.isin(keys, calyx_key)])
    missing = np.setdiff1d(expected, packed['edge'])
    if len(missing):
        raise ValueError(f'Missing {len(missing)} pinned axonal connections; do not claim complete coverage')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    manifest = {'dataset': 'male-cns:v1.0', 'version': 1,
        'kc_geometry_sha256': hashlib.sha256(kc_geometry.read_bytes()).hexdigest(),
        'contacts': total, 'grouped_contact_routes': len(packed['edge']),
        'edges': len(np.unique(packed['edge'])), 'pinned_axonal_edges_covered': len(expected),
        'recipient_nodes': len(np.unique(packed['node'])), 'maximum_projection_um': maximum_distance,
        'scope': 'All existing non-calyx KC-to-KC contact positions projected onto the unchanged supplied morphology; no receptor localization measured',
        'npz_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--kc-geometry', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.kc_geometry, args.raw, args.graph, args.output)
