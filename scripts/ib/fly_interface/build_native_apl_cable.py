"""Keep every supplied APL tree node and map actual contacts to cable volumes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BODIES = (10540, 10977)


def build(graph_path: Path, spatial: Path, output: Path):
    graph = np.load(graph_path, allow_pickle=False)
    ids = graph['neuron_body_ids']
    lookup = {int(body): i for i, body in enumerate(ids)}
    apl = np.asarray([lookup[b] for b in BODIES])
    annotation = pd.read_feather(graph_path.parent / 'body-annotations.feather').set_index('bodyId').reindex(ids)
    kc = annotation['class'].fillna('').to_numpy() == 'Kenyon_Cell'
    arrays, trees, node_offsets, node_owner, area_parts = {}, {}, {}, [], []
    node_ids, edge_u, edge_v, edge_g, roots = [], [], [], [], []
    offset = 0
    for slot, body in enumerate(BODIES):
        swc = np.loadtxt(spatial / f'{body}.swc')
        local_ids = swc[:, 0].astype(np.int64)
        if len(set(local_ids)) != len(local_ids) or np.any(swc[:, 5] <= 0):
            raise ValueError('Invalid actual SWC identities/radii')
        order = {int(n): i for i, n in enumerate(local_ids)}
        child = np.flatnonzero(swc[:, 6] >= 0)
        parent = np.asarray([order[int(n)] for n in swc[child, 6]])
        xyz, radius = swc[:, 2:5] * .008, swc[:, 5] * .008
        length = np.linalg.norm(xyz[child] - xyz[parent], axis=1)
        if np.any(length <= 0):
            raise ValueError('Zero-length supplied cable segment')
        surface = np.pi * (radius[child] + radius[parent]) * np.hypot(length, radius[child] - radius[parent])
        area = np.bincount(np.r_[child, parent], weights=np.r_[surface / 2, surface / 2], minlength=len(swc))
        # Isolated supplied nodes are retained as their own closed compartments.
        isolated = area == 0
        area[isolated] = 4 * np.pi * radius[isolated] ** 2
        node_offsets[body] = offset
        trees[body] = cKDTree(xyz)
        node_ids.append(np.column_stack((np.full(len(swc), body), local_ids)))
        node_owner.append(np.full(len(swc), slot))
        area_parts.append(area)
        edge_u.append(child + offset)
        edge_v.append(parent + offset)
        edge_g.append(np.pi * radius[child] * radius[parent] / length)
        roots.append(int((swc[:, 6] < 0).sum()))
        offset += len(swc)
    arrays.update(node_ids=np.concatenate(node_ids), node_owner=np.concatenate(node_owner),
                  area=np.concatenate(area_parts), cable_u=np.concatenate(edge_u),
                  cable_v=np.concatenate(edge_v), cable_g=np.concatenate(edge_g), apl_index=apl)
    edge_lookup = {}
    for channel in ('e', 'i'):
        pre = graph[f'edge_pre_{channel}'].astype(np.int64)
        post = graph[f'edge_post_{channel}'].astype(np.int64)
        selected = np.flatnonzero(np.isin(pre, apl) | np.isin(post, apl))
        delays = graph[f'edge_delay_{channel}'][selected]
        arrays[f'verify_{channel}_edge'] = selected
        arrays[f'verify_{channel}_pre'] = pre[selected]
        arrays[f'verify_{channel}_post'] = post[selected]
        for index, p, q, lag in zip(selected, pre[selected], post[selected], delays):
            key = (int(ids[p]), int(ids[q]))
            if key in edge_lookup:
                raise ValueError('Duplicate native pair; route must retain distinct edges')
            edge_lookup[key] = (channel, int(index), int(p), int(q), int(lag))
        del pre, post
    records, projection = [], []
    routes = {key: [] for key in ('in_e', 'in_i', 'out_i', 'out_mb')}
    for body in BODIES:
        for direction in ('incoming', 'outgoing'):
            table = pd.read_feather(spatial / f'{body}_{direction}.feather')
            other = 'body_pre' if direction == 'incoming' else 'body_post'
            valid = table[other].isin(lookup)
            table = table[valid].copy()
            endpoint = 'post' if direction == 'incoming' else 'pre'
            distance, node = trees[body].query(table[[f'{a}_{endpoint}' for a in 'xyz']].to_numpy() * .008)
            table['node'] = node + node_offsets[body]
            projection.extend(distance.tolist())
            # Preserve local APL self/cross-input release, rather than routing it
            # back through a pooled soma packet.
            table['pre_node'] = -1
            if direction == 'incoming':
                for other_body in BODIES:
                    mask = table.body_pre == other_body
                    if mask.any():
                        _, source_node = trees[other_body].query(table.loc[mask, ['x_pre', 'y_pre', 'z_pre']].to_numpy() * .008)
                        table.loc[mask, 'pre_node'] = source_node + node_offsets[other_body]
            counts = table.groupby(['body_pre', 'body_post', 'node', 'pre_node', 'post_in_calyx']).size().reset_index(name='contacts')
            groups = list(counts.groupby(['body_pre', 'body_post']))
            for (pre_body, post_body), group in groups:
                spec = edge_lookup.get((int(pre_body), int(post_body)))
                if spec is None:
                    continue  # Contact belongs to an edge outside this model's scaffold.
                channel, edge, pre, post, delay = spec
                if direction == 'incoming':
                    route = 'in_' + channel
                    sets = [(route, group)]
                else:
                    if channel != 'i':
                        raise ValueError('APL must keep its native inhibitory sign')
                    sets = [('out_i', group[group.post_in_calyx] if kc[post] else group)]
                    if kc[post]:
                        sets.append(('out_mb', group[~group.post_in_calyx]))
                for route, subset in sets:
                    if subset.empty:
                        continue
                    denom = int(subset.contacts.sum())
                    for row in subset.itertuples():
                        routes[route].append((edge, pre, post, delay, int(row.node), int(row.pre_node), row.contacts / denom))
            records.append({'body': body, 'direction': direction, 'modeled_contact_rows': len(table)})
    columns = ('edge', 'pre', 'post', 'delay', 'node', 'pre_node', 'fraction')
    for route, rows in routes.items():
        values = np.asarray(rows, dtype=np.float64)
        for i, col in enumerate(columns):
            arrays[f'{route}_{col}'] = values[:, i].astype(np.float64 if col == 'fraction' else np.int64)
    metadata = {'version': 1, 'dataset': 'male-cns:v1.0', 'bodies': BODIES,
        'nodes': offset, 'cable_segments': len(arrays['cable_u']), 'supplied_roots': roots,
        'no_invented_root_bridges': True, 'no_branch_or_roi_merging': True,
        'nearest_node_projection_um': {'median': float(np.median(projection)), 'p99': float(np.quantile(projection, .99)), 'max': float(np.max(projection))},
        'routes': {key: len(rows) for key, rows in routes.items()}, 'contacts': records,
        'source_manifest_sha256': hashlib.sha256((spatial / 'apl_geometry_manifest.json').read_bytes()).hexdigest(),
        'discretization': 'one finite volume per supplied SWC node, full forest; individual synapses assigned to nearest native node',
        'kc_scope': 'KC axonal inhibition still one output per KC; KC bouton cable is not reconstructed'}
    np.savez_compressed(output, **arrays)
    metadata['npz_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    parser.add_argument('--spatial', type=Path, default=Path('data/malecns_v1/spatial'))
    parser.add_argument('--output', type=Path, default=Path('data/malecns_v1/spatial/apl_cable_native_v1.npz'))
    args = parser.parse_args()
    build(args.graph, args.spatial, args.output)
