"""Audit actual typed program path compositions without asserting filled faces."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np


def load(handle, obj):
    r=obj['tensor_ref'];handle.seek(r['offset']);raw=zlib.decompress(handle.read(r['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('tensor checksum mismatch')
    return np.broadcast_to(np.frombuffer(raw,dtype=r['dtype']).reshape(r['stored_shape']),r['shape']).astype(float)


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    events=[]
    # Only completed archives: gzip footer required. Never consume active writer.
    for segment,d in enumerate(sorted(a.work.glob('collect_observations*'))):
        try:
            with gzip.open(d/'operations.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(x) for x in f]
        except EOFError:
            continue
        selected={}
        for r in rows:
            data=r['data']
            if r['kind']=='event' and data.get('autonomous_event',{}).get('kind')=='external':
                role=(data.get('causal_program') or {}).get('roles') or {}
                selected[r['event']]={'relation':role.get('identity'),'generation':role.get('generation'),
                    'version':role.get('version'),'unknown':role.get('unknown_identity'),
                    'choice':data.get('choice'),'cycle':r.get('external_cycle')}
        by_event=collections.defaultdict(list)
        for r in rows:
            if r['kind']=='typed_execution' and r['event'] in selected:by_event[r['event']].append(r)
        with (d/'tensors.bin').open('rb') as handle:
            for event,info in selected.items():
                comparisons=[]
                for row in by_event[event]:
                    outputs={v['operation']:v for v in row['data'].get('alternative_outputs',[])}
                    if not {'ambient_transport','body_transition'} <= outputs.keys():continue
                    x,y=outputs['ambient_transport'],outputs['body_transition']
                    if x['identity']!=y['identity'] or x['identity']!=f"{info['relation']}:decision":continue
                    vx,vy=load(handle,x['tensor']),load(handle,y['tensor']);norm=float(np.linalg.norm(vx-vy));den=float(np.sqrt((np.square(vx).sum()+np.square(vy).sum())/2))
                    comparisons.append({'difference_l2':norm,'relative_difference':norm/den if den else None})
                events.append(dict(segment=segment,event=event,**info,matching_executions=len(comparisons),comparisons=comparisons))
    rel=[v['relative_difference'] for e in events for v in e['comparisons'] if v['relative_difference'] is not None]
    out={'external_events':len(events),'events_with_matching_path_execution':sum(e['matching_executions']>0 for e in events),
         'matching_executions':len(rel),'median_relative_difference':float(np.median(rel)) if rel else None,
         'events':events,'note':'Same typed endpoint is not proof that paths commute or that a higher face exists. Includes selected-identity program calls that may be previews; no claim each execution was committed.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='events'},ensure_ascii=False))


if __name__=='__main__':main()
