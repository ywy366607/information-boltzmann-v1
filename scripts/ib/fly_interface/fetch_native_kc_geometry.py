"""Cache native KC trees/output sites and DPM contacts, without stored credentials."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import requests

from fetch_native_apl_geometry import fetch_contacts, fetch_tree


def retry(function, *args):
    for attempt in range(5):
        try:
            return function(*args)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


def outgoing(bodies, output, index):
    path = output / f'kc_outgoing_{index:03d}.feather'
    if path.exists():
        return {'file': path.name, 'reused': True}
    token = os.environ.get('NEUPRINT_TOKEN') or os.environ['NEUPRINT_APPLICATION_CREDENTIALS']
    query = f"""
MATCH (n:Neuron) WHERE n.bodyId IN {list(map(int, bodies))}
WITH n MATCH (n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)-[:SynapsesTo]->(t:Synapse)
WHERE s.confidence >= 0.5 AND t.confidence >= 0.5
WITH DISTINCT s,t
RETURN s.bodyId AS body_pre, t.bodyId AS body_post,
       s.location.x AS x_pre, s.location.y AS y_pre, s.location.z AS z_pre,
       t.location.x AS x_post, t.location.y AS y_post, t.location.z AS z_post,
       coalesce(t.`CA(L)`, false) OR coalesce(t.`CA(R)`, false)
       OR coalesce(t.`ACA(L)`, false) OR coalesce(t.`ACA(R)`, false) AS post_in_calyx
"""
    response = requests.post('https://neuprint.janelia.org/api/custom/custom',
        headers={'Authorization': 'Bearer '+token},
        json={'dataset': 'male-cns:v1.0', 'cypher': query}, timeout=240)
    response.raise_for_status()
    data = response.json()
    table = pd.DataFrame(data['data'], columns=data['columns'])
    if table.isna().any().any() or table.duplicated().any():
        raise ValueError('Ambiguous/missing native KC output contact identity')
    table.to_feather(path)
    return {'file': path.name, 'rows': len(table)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--annotations', type=Path, default=Path('data/malecns_v1/body-annotations.feather'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotation = pd.read_feather(args.annotations, columns=['bodyId','class','type'])
    bodies = annotation.loc[annotation['class']=='Kenyon_Cell','bodyId'].astype(int).tolist()
    (args.output/'kc_bodies.json').write_text(json.dumps(bodies), encoding='utf-8')
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(retry, fetch_tree, body, args.output) for body in bodies]
        for index, job in enumerate(as_completed(jobs), 1):
            records.append(job.result())
            if index % 256 == 0:
                print(json.dumps({'trees_saved':index,'total':len(bodies)}), flush=True)
    batches = [bodies[left:left+64] for left in range(0,len(bodies),64)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(retry, outgoing, b, args.output, i) for i,b in enumerate(batches)]
        jobs += [pool.submit(retry, fetch_contacts, body, 'outgoing', args.output)
                 for body in annotation.loc[annotation['type']=='DPM','bodyId'].astype(int)]
        for job in as_completed(jobs):
            item = job.result(); records.append(item)
            print(json.dumps(item), flush=True)
    for item in records:
        item['sha256'] = hashlib.sha256((args.output/item['file']).read_bytes()).hexdigest()
    manifest = {'dataset':'male-cns:v1.0','coordinate_unit_um':.008,
        'synapse_confidence_threshold':.5,'kc_bodies':bodies,'records':records,
        'source':'https://male-cns.janelia.org/download/',
        'scope':'all supplied KC trees and individual output synapse partner coordinates; no ROI or branch pooling'}
    (args.output/'kc_geometry_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':'fetched','trees':len(bodies),'files':len(records)}),flush=True)


if __name__=='__main__':
    main()
