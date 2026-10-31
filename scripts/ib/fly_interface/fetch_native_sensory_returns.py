"""Resumable batched acquisition of all missing sensory-return contact locations.

This only fetches anatomical data. It never changes the brain, starts training,
or substitutes body-level current for an unresolved local contact mechanism.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests

from neuprint_schema_catalog import user_token


def fetch_batch(index,bodies,output,dataset,direction='incoming'):
    path=output/f'contacts_{index:04d}.feather'
    if path.exists():
        return {'batch':index,'rows':len(pd.read_feather(path,columns=['body_post'])),'reused':True}
    pattern=('(n)-[:Contains]->(:SynapseSet)-[:Contains]->(t:Synapse)<-[:SynapsesTo]-(s:Synapse)'
             if direction=='incoming' else
             '(n)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)-[:SynapsesTo]->(t:Synapse)')
    q=f'''MATCH (n:Neuron) WHERE n.bodyId IN {bodies}
    WITH n MATCH {pattern}
    WITH DISTINCT s,t
    RETURN s.bodyId AS body_pre,t.bodyId AS body_post,
      s.location.x AS x_pre,s.location.y AS y_pre,s.location.z AS z_pre,
      t.location.x AS x_post,t.location.y AS y_post,t.location.z AS z_post,
      s.confidence AS confidence_pre,t.confidence AS confidence_post'''
    for attempt in range(3):
        try:
            response=requests.post('https://neuprint.janelia.org/api/custom/custom',
                headers={'Authorization':'Bearer '+user_token()},
                json={'dataset':dataset,'cypher':q},timeout=(10,55))
            if not response.ok:
                raise RuntimeError(f'neuPrint HTTP {response.status_code}')
            data=response.json()
            table=pd.DataFrame(data['data'],columns=data['columns'])
            mandatory=['body_pre','body_post','x_pre','y_pre','z_pre','x_post','y_post','z_post']
            if table[mandatory].isna().any().any():
                raise ValueError('Missing contact identities/positions')
            temporary=path.with_suffix('.tmp')
            table.to_feather(temporary)
            temporary.replace(path)
            return {'batch':index,'rows':len(table),'reused':False}
        except (requests.RequestException,RuntimeError):
            if attempt==2:
                raise
            time.sleep(attempt+1)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_feather(args.pairs,columns=['body_pre','body_post'])
    bodies=sorted(map(int,pairs.body_post.unique()))
    batches=[bodies[i:i+args.batch_size] for i in range(0,len(bodies),args.batch_size)]
    manifest_path=args.output/'manifest.json'
    identity={'dataset':args.dataset,'bodies':bodies,'batch_size':args.batch_size,'direction':args.direction,
        'pair_source':str(args.pairs.resolve())}
    if manifest_path.exists():
        before=json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(before.get(k, 'incoming' if k == 'direction' else None)!=v
               for k,v in identity.items()):
            raise ValueError('Incompatible resumable contact acquisition')
    def save(status,done,errors):
        data={**identity,'at':datetime.now(timezone.utc).isoformat(),'status':status,
            'batches':len(batches),'completed':done,'errors':errors,
            'scope':'All confidence values retained; final linker selects original pairs and records position/receptor evidence. No active wiring or training.'}
        temporary=manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        temporary.replace(manifest_path)
    done=[];errors=[];pending={};cursor=0
    save('fetching',done,errors)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while cursor<len(batches) or pending:
            while cursor<len(batches) and len(pending)<args.workers and not errors:
                pending[pool.submit(fetch_batch,cursor,batches[cursor],args.output,args.dataset,args.direction)]=cursor
                cursor+=1
            if not pending:
                break
            finished,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in finished:
                index=pending.pop(future)
                try:
                    result=future.result();done.append(result)
                    print(json.dumps(result),flush=True)
                except Exception as error:
                    # No request objects, headers or response bodies in logs.
                    errors.append({'batch':index,'error_type':type(error).__name__})
                    print(json.dumps(errors[-1]),flush=True)
                save('fetching' if not errors else 'stopped_on_error',done,errors)
    save('complete' if len(done)==len(batches) and not errors else 'incomplete',done,errors)
    if errors:
        raise RuntimeError('Acquisition incomplete; cached batches preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--pairs',type=Path,default=Path('F:/fly_stream_overflow/malecns_sensory_return_catalog_20261019/sensory_return_edges.feather'))
    p.add_argument('--dataset',default='male-cns:v1.0')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--direction',choices=['incoming','outgoing'],default='incoming')
    args=p.parse_args()
    if args.batch_size<1 or args.workers not in range(1,5):
        p.error('Positive batch size and 1-4 workers required')
    main(args)
