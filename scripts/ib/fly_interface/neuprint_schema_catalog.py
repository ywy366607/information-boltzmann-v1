"""Read MaleCNS schema/coverage with user-owned credentials; never save tokens."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests

SERVER='https://neuprint.janelia.org'


def user_token():
    for name in ('NEUPRINT_TOKEN','NEUPRINT_APPLICATION_CREDENTIALS'):
        value=os.environ.get(name)
        if value:
            return value
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
            for name in ('NEUPRINT_TOKEN','NEUPRINT_APPLICATION_CREDENTIALS'):
                try:
                    value=winreg.QueryValueEx(key,name)[0]
                except FileNotFoundError:
                    continue
                if value:
                    return value
    raise RuntimeError('No neuPrint credential in process/user environment')


def query(cypher,dataset):
    response=requests.post(SERVER+'/api/custom/custom',
        headers={'Authorization':'Bearer '+user_token()},
        json={'dataset':dataset,'cypher':cypher},timeout=(10,55))
    if not response.ok:
        # Do not dump response/request objects or authorization headers.
        raise RuntimeError(f'neuPrint query HTTP {response.status_code}')
    result=response.json()
    return [dict(zip(result['columns'],row)) for row in result['data']]


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'catalog.json').exists():
        raise ValueError('Use a fresh timestamped catalog; preserve provenance')
    jobs={
      'metadata':'''MATCH (m:Meta) RETURN m.dataset AS dataset,m.uuid AS uuid,
        m.lastDatabaseEdit AS lastDatabaseEdit,m.neuronProperties AS neuronProperties,
        m.primaryRois AS primaryRois,m.roiHierarchy AS roiHierarchy,
        m.voxelSize AS voxelSize,m.voxelUnits AS voxelUnits,
        m.totalPreCount AS totalPreCount,m.totalPostCount AS totalPostCount''',
      'neuron_coverage':'''MATCH (n:Neuron) RETURN count(n) AS neurons,count(n.type) AS typed,
        count(n.class) AS classified,count(n.consensusNt) AS consensusNt,
        count(n.receptorType) AS receptorType,count(n.birthtime) AS birthtime,
        count(n.trumanHl) AS trumanHl,count(n.hemibrainType) AS hemibrainType''',
      'cell_families':'''MATCH (n:Neuron) RETURN n.superclass AS superclass,n.class AS class,
        n.consensusNt AS consensusNt,count(n) AS count''',
      'neuron_ids':'''MATCH (n:Neuron) RETURN n.bodyId AS bodyId ORDER BY bodyId''',
      'connection_totals':'''MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
        RETURN count(e) AS neuron_pair_edges,sum(e.weight) AS connection_weight_sum''',
    }
    result={}
    # Two server queries at a time, not one huge synapse-coordinate download.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={name:pool.submit(query,q,args.dataset) for name,q in jobs.items()}
        for name,future in futures.items():
            result[name]=future.result()
            print(json.dumps({'query':name,'rows':len(result[name])}),flush=True)
    remote=np.array([row['bodyId'] for row in result.pop('neuron_ids')],dtype=np.int64)
    with np.load(args.graph,allow_pickle=False) as packed:
        local=packed['neuron_body_ids'].astype(np.int64)
        fast_edges=sum(packed[name].size for name in ('edge_pre_e','edge_pre_i'))
        dopamine_edges=packed['dan_edge_pre'].size if 'dan_edge_pre' in packed else 0
    np.save(args.output/'remote_neuron_ids.npy',remote)
    missing=np.setdiff1d(remote,local)
    extra=np.setdiff1d(local,remote)
    comparison={'graph':str(args.graph.resolve()),'local_neurons':len(local),
        'local_fast_edges':int(fast_edges),'local_separate_dopamine_edges':int(dopamine_edges),
        'local_stored_edge_channels_total':int(fast_edges+dopamine_edges),'remote_neurons':len(remote),
        'missing_remote_neurons':missing.tolist(),'extra_local_neurons':extra.tolist(),
        'all_remote_neuron_ids_present':len(missing)==0 and len(extra)==0,
        'ordered_remote_id_sha256':hashlib.sha256(remote.tobytes()).hexdigest(),
        'edge_scope':'Counts only; edge identity/weight equality not established by equal counts'}
    meta=result['metadata'][0]
    properties=meta.get('neuronProperties')
    if isinstance(properties,str):
        properties=json.loads(properties)
    meta['neuronProperties']=properties
    report={'created_at':datetime.now(timezone.utc).isoformat(),'server':SERVER,
        'dataset':args.dataset,'source':'Live authenticated read-only queries; credential never saved',
        'queries':jobs,'remote':result,'local_comparison':comparison,
        'scope':'Data/schema inventory for whole-graph mechanism compilation, not new biological mechanisms or functional validation'}
    (args.output/'catalog.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'coverage':result['neuron_coverage'],'connections':result['connection_totals'],
                      'comparison':{k:v for k,v in comparison.items() if not isinstance(v,list)},
                      'neuron_properties':properties},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',default='male-cns:v1.0')
    p.add_argument('--graph',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
