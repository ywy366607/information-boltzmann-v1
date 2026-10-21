"""Inventory sensory-exit identity coverage before generative E2 rewiring.

No graph is changed. Counts describe the source graph, not learned edge efficacy
or calibrated cell-pair geometry. Work is bounded to one fast-edge tier at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def main(args):
    with np.load(args.strength,allow_pickle=False) as strength:
        types=strength['types'].astype(str)
    with np.load(args.spatial,allow_pickle=False) as spatial:
        if not np.array_equal(types,spatial['types'].astype(str)):
            raise ValueError('Strength/spatial identity ordering differs')
        spatial_types=types[np.asarray(spatial['known'],dtype=bool)]
    with np.load(args.occurrence,allow_pickle=False) as occurrence:
        if not np.array_equal(types,occurrence['types'].astype(str)):
            raise ValueError('Strength/occurrence identity ordering differs')
    graph=np.load(args.graph,allow_pickle=False)
    ann=pd.read_feather(args.graph.parent/'body-annotations.feather',
                        columns=['bodyId','class','type']).drop_duplicates('bodyId')
    ann=ann.set_index('bodyId').reindex(graph['neuron_body_ids'])
    cls,typ=ann['class'].astype(str).to_numpy(),ann['type'].astype(str).to_numpy()
    with np.load(args.graph.parent/'sensory_partitions.npz') as parts:
        indices=np.concatenate([parts[k] for k in ('visual_idx','chemo_idx','mechano_idx')])
    # Exact historical receptor_lang selection, including smell, excluding taste.
    indices=indices[~np.isin(cls[indices],['gustatory','chemosensory'])]
    sensory=np.zeros(len(typ),dtype=bool);sensory[indices]=True
    known=np.isin(typ,types);spatial_known=np.isin(typ,spatial_types)
    rows={}
    for tag in ('e','i'):
        pre,post=graph['edge_pre_'+tag],graph['edge_post_'+tag]
        take=sensory[pre]
        pre,post=pre[take],post[take]
        identity=known[pre]&known[post]
        geometry=spatial_known[pre]&spatial_known[post]
        def missing(cells):
            names,counts=np.unique(typ[cells[~known[cells]]],return_counts=True)
            order=np.argsort(-counts,kind='stable')[:20]
            return {str(names[i]):int(counts[i]) for i in order}
        rows[tag]={'edges':len(pre),'identity_covered_edges':int(identity.sum()),
                   'both_spatial_types_known_edges':int(geometry.sum()),
                   'missing_pre_types_by_edge_count':missing(pre),
                   'missing_post_types_by_edge_count':missing(post)}
    total=sum(row['edges'] for row in rows.values())
    covered=sum(row['identity_covered_edges'] for row in rows.values())
    spatial_count=sum(row['both_spatial_types_known_edges'] for row in rows.values())
    report={'scope':'Source-graph sensory-exit coverage inventory; not generated rewiring or geometry/function acceptance',
            'sensory_cells':int(sensory.sum()),'identity_covered_sensory_cells':int((known&sensory).sum()),
            'strength_types':len(types),'spatial_known_types':len(spatial_types),
            'edges':rows,'total_edges':total,'identity_covered_fraction':covered/total,
            'both_spatial_types_known_fraction':spatial_count/total,
            'identity_order_exact':True,
            'missing': ['Cell-pair geometry/compartment/contact calibration',
                        'Types outside these fitted identities',
                        'Efficacy of learned zero/pinned edges differs from raw graph counts'],
            'sha256':{str(p):digest(p) for p in (args.graph,args.strength,args.spatial,args.occurrence,Path(__file__))}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('graph','strength','spatial','occurrence','out'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
