"""Fetch supplied native morphology for a whole endpoint set, resumably."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import requests

_local=threading.local()


def fetch(body,output):
    path=output/f'{body}.swc'
    if path.exists() and path.stat().st_size:
        return body,path.stat().st_size,True
    session=getattr(_local,'session',None)
    if session is None:
        session=_local.session=requests.Session()
    url=f'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body}.swc'
    for attempt in range(3):
        try:
            response=session.get(url,timeout=(10,45))
            if not response.ok:
                raise RuntimeError(f'Morphology HTTP {response.status_code}')
            data=response.content
            if not data or data.lstrip().startswith(b'<'):
                raise ValueError('Invalid morphology payload')
            temporary=path.with_suffix('.tmp')
            temporary.write_bytes(data);temporary.replace(path)
            return body,len(data),False
        except (requests.RequestException,RuntimeError):
            if attempt==2:
                raise
            time.sleep(attempt+1)


def main(args):
    bodies=sorted(map(int,pd.read_feather(args.table,columns=[args.column])[args.column].unique()))
    args.output.mkdir(parents=True,exist_ok=True)
    identity={'dataset':'male-cns:v1.0','bodies':bodies,'source':'Supplied Google Cloud SWC objects, unchanged nodes/roots'}
    manifest=args.output/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous['bodies']!=bodies:
            raise ValueError('Different morphology request in this directory')
    errors=[];done=0;total=0;reused=0
    def save(status):
        record={**identity,'at':datetime.now(timezone.utc).isoformat(),'completed':done,
            'bytes':total,'reused':reused,'errors':errors,'status':status}
        temp=manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');temp.replace(manifest)
    save('fetching')
    # Bounded groups avoid submitting 15k pending tasks, and stop before
    # sacrificing disk reserve for complete learning states.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0,len(bodies),args.workers*4):
            if shutil.disk_usage(args.output).free<3*1024**3:
                errors.append({'error_type':'DiskReserve'});break
            futures={pool.submit(fetch,b,args.output):b for b in bodies[start:start+args.workers*4]}
            for future in as_completed(futures):
                try:
                    body,size,was_reused=future.result();done+=1;total+=size;reused+=int(was_reused)
                except Exception as error:
                    errors.append({'body':futures[future],'error_type':type(error).__name__})
            save('fetching')
            print(json.dumps({'completed':done,'requested':len(bodies),'bytes':total,'errors':len(errors)}),flush=True)
            if errors:
                break
    save('complete' if done==len(bodies) and not errors else 'incomplete')
    if errors:
        raise RuntimeError('Morphology batch incomplete; cache retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--column',default='body_post')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    args=p.parse_args()
    if args.workers not in range(1,17):
        p.error('Use 1-16 IO workers')
    main(args)
