"""Fetch full native APL trees and individual synaptic contacts, without credentials on disk."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import requests


DATASET = 'male-cns:v1.0'
SERVER = 'https://neuprint.janelia.org'
BODIES = (10540, 10977)


def fetch_contacts(body: int, direction: str, output: Path) -> dict:
    path = output / f'{body}_{direction}.feather'
    if path.exists():
        return {'file': path.name, 'rows': len(pd.read_feather(path)), 'reused': True}
    token = os.environ.get('NEUPRINT_TOKEN') or os.environ['NEUPRINT_APPLICATION_CREDENTIALS']
    if direction == 'incoming':
        pattern = '(n)-[:Contains]->(:SynapseSet)-[:Contains]->(t:Synapse)<-[:SynapsesTo]-(s:Synapse)'
    else:
        pattern = '(n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)-[:SynapsesTo]->(t:Synapse)'
    query = f"""
MATCH (n:Neuron) WHERE n.bodyId = {body}
WITH n MATCH {pattern}
WHERE s.confidence >= 0.5 AND t.confidence >= 0.5
WITH DISTINCT s, t
RETURN s.bodyId AS body_pre, t.bodyId AS body_post,
       s.location.x AS x_pre, s.location.y AS y_pre, s.location.z AS z_pre,
       t.location.x AS x_post, t.location.y AS y_post, t.location.z AS z_post,
       coalesce(t.`CA(L)`, false) OR coalesce(t.`CA(R)`, false)
       OR coalesce(t.`ACA(L)`, false) OR coalesce(t.`ACA(R)`, false) AS post_in_calyx
"""
    response = requests.post(SERVER + '/api/custom/custom',
        headers={'Authorization': 'Bearer ' + token},
        json={'cypher': query, 'dataset': DATASET}, timeout=240)
    response.raise_for_status()
    value = response.json()
    table = pd.DataFrame(value['data'], columns=value['columns'])
    if table.empty or table.isna().any().any():
        raise ValueError(f'Missing actual synapse identities/coordinates: {body} {direction}')
    if table.duplicated().any():
        raise ValueError('Duplicate physical synaptic partner rows; resolve source query')
    table.to_feather(path)
    return {'file': path.name, 'rows': len(table), 'response_bytes': len(response.content)}


def fetch_tree(body: int, output: Path) -> dict:
    path = output / f'{body}.swc'
    url = ('https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/'
           f'skeletons-malecns/skeletons-swc/{body}.swc')
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    return {'file': path.name, 'url': url, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/malecns_v1/spatial'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for body in BODIES:
            jobs.append(pool.submit(fetch_tree, body, args.output))
            for direction in ('incoming', 'outgoing'):
                jobs.append(pool.submit(fetch_contacts, body, direction, args.output))
        records = [job.result() for job in jobs]
    for record in records:
        record['sha256'] = hashlib.sha256((args.output / record['file']).read_bytes()).hexdigest()
    report = {'dataset': DATASET, 'at': datetime.now(timezone.utc).isoformat(),
        'bodies': BODIES, 'coordinate_unit_um': 0.008,
        'synapse_confidence_threshold': 0.5, 'records': records,
        'source': 'https://male-cns.janelia.org/download/',
        'scope': 'full supplied SWC, individual contacts; no branch/ROI merging or invented roots'}
    (args.output / 'apl_geometry_manifest.json').write_text(
        json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
