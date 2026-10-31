"""Locate existing DAN-to-KC synapses on the supplied, unpooled KC trees."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from build_native_kc_terminals import tree


def fetch_batch(bodies, targets, path):
    if path.exists():
        return pd.read_feather(path)
    token = os.environ.get('NEUPRINT_TOKEN') or os.environ['NEUPRINT_APPLICATION_CREDENTIALS']
    query = f'''MATCH (n:Neuron) WHERE n.bodyId IN {bodies}
WITH n MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)-[:SynapsesTo]->(t:Synapse)
WHERE s.confidence >= 0.5 AND t.confidence >= 0.5 AND t.bodyId IN {targets}
WITH DISTINCT s,t RETURN s.bodyId AS body_pre, t.bodyId AS body_post,
s.location.x AS x_pre,s.location.y AS y_pre,s.location.z AS z_pre,
t.location.x AS x_post,t.location.y AS y_post,t.location.z AS z_post'''
    for attempt in range(4):
        try:
            response = requests.post('https://neuprint.janelia.org/api/custom/custom',
                headers={'Authorization': 'Bearer ' + token},
                json={'dataset': 'male-cns:v1.0', 'cypher': query}, timeout=240)
            response.raise_for_status()
            result = response.json()
            table = pd.DataFrame(result['data'], columns=result['columns'])
            if table.isna().any().any() or table.duplicated().any():
                raise ValueError('Ambiguous DAN-KC physical contact identities')
            table.to_feather(path)
            return table
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def build(graph_path, geometry_path, raw, output):
    graph = np.load(graph_path, allow_pickle=False)
    ids = graph['neuron_body_ids']
    with np.load(geometry_path, allow_pickle=False) as data:
        kc, owner, node_ids = data['kc_index'], data['owner'], data['node_ids']
    edges = np.flatnonzero(np.isin(graph['dan_edge_post'], kc))
    pre, post = graph['dan_edge_pre'][edges], graph['dan_edge_post'][edges]
    bodies = sorted(map(int, ids[np.unique(pre)]))
    targets = list(map(int, ids[kc]))
    output.mkdir(parents=True, exist_ok=True)
    batches = [bodies[i:i+16] for i in range(0, len(bodies), 16)]
    tables = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(fetch_batch, b, targets, output/f'dan_kc_{j:03d}.feather') for j,b in enumerate(batches)]
        for future in as_completed(jobs):
            tables.append(future.result())
            print(json.dumps({'batches': len(tables), 'total': len(jobs)}), flush=True)
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated().any():
        raise ValueError('Duplicate contacts across source batches')
    pair = {(int(ids[a]),int(ids[b])):int(e) for e,a,b in zip(edges,pre,post)}
    table['edge'] = [pair.get((int(a),int(b)), -1) for a,b in zip(table.body_pre,table.body_post)]
    excluded = int((table.edge < 0).sum())
    table = table[table.edge >= 0].copy()
    if set(table.edge) != set(map(int,edges)):
        raise ValueError('Existing DAN-KC graph edge lacks spatial contacts')
    node = np.empty(len(table), dtype=np.int64)
    distances = []
    offsets = np.r_[0,np.cumsum(np.bincount(owner,minlength=len(kc)))]
    groups = table.groupby('body_post',sort=False).indices
    for slot,index in enumerate(kc):
        positions = groups.get(int(ids[index]))
        if positions is None:
            continue
        identity, xyz, *_ = tree(raw/f'{int(ids[index])}.swc')
        loc = np.arange(offsets[slot],offsets[slot+1])
        if not np.array_equal(identity, node_ids[loc,1]):
            raise ValueError('KC tree order differs from active geometry')
        distance, nearest = cKDTree(xyz).query(table.iloc[positions][['x_post','y_post','z_post']].to_numpy()*.008)
        node[positions] = loc[nearest]
        distances.extend(distance.tolist())
    table['node'] = node
    edge_counts = table.groupby('edge').size().to_dict()
    grouped = table.groupby(['edge','node'], sort=True).size().reset_index(name='count')
    edge = grouped.edge.to_numpy(dtype=np.int64)
    fraction = np.asarray([c/edge_counts[e] for c,e in zip(grouped['count'],edge)])
    arrays = dict(edge=edge, pre=graph['dan_edge_pre'][edge].astype(np.int64),
        post=graph['dan_edge_post'][edge].astype(np.int64), node=grouped.node.to_numpy(dtype=np.int64),
        delay=graph['dan_edge_delay'][edge].astype(np.int64), fraction=fraction,
        kc_index=kc, node_ids=node_ids)
    packed = output/'kc_dopamine_native_v1.npz'
    np.savez_compressed(packed, **arrays)
    report = {'dataset': 'male-cns:v1.0', 'confidence': .5,
        'graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        'kc_geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        'sha256': hashlib.sha256(packed.read_bytes()).hexdigest(),
        'graph_edges': len(edges), 'physical_contacts': len(table),
        'source_cells': len(np.unique(pre)), 'target_kcs': len(np.unique(post)),
        'located_source_target_nodes': len(edge), 'excluded_not_in_active_graph': excluded,
        'projection_distance_um_quantiles': np.quantile(distances,[0,.5,.95,1]).tolist(),
        'scope': 'Actual DAN-KC synapses mapped to every supplied KC tree; no ROI pooling or invented links. Fraction sums to1 per existing graph edge; source local DAN axonal physiology remains existing model.',
        'raw_files': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.glob('dan_kc_*.feather'))]}
    packed.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.graph, args.geometry, args.raw, args.output)
