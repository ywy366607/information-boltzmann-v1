"""Track fixed earlier head-flow modes across actual later event boundaries."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--training-quiet-samples',type=int,default=None,
        help='freeze the previously chosen chronological training prefix when new observations arrive')
    a=p.parse_args()
    quiet=[]; bounds={}; kinds={}; receipts={}
    for seg,d in enumerate(sorted(a.work.glob('collect_observations*'))):
        with gzip.open(d/'operations.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                if row['kind']=='event':
                    key=(seg,int(row['event']));data=row['data']
                    kinds[key]=data.get('autonomous_event',{}).get('kind')
                    receipts[key]=data
        for path in sorted(d.glob('physical_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                for i,kind in enumerate(z['kind']):
                    key=(seg,int(z['event'][i]));flux=z['edge_flux'][i].astype(float)
                    if kind=='quiet_after':quiet.append((key,flux))
                    if kind in ('event_before','event_after'):
                        bounds.setdefault(key,{})[str(kind)]=flux
    split=len(quiet)//2 if a.training_quiet_samples is None else a.training_quiet_samples
    if not 2<=split<len(quiet):raise ValueError('training prefix must leave held-out observations')
    training=np.stack([v for _,v in quiet[:split]])
    modes=np.stack([np.linalg.svd(training[:,i]-training[:,i].mean(0),full_matrices=False)[2][0] for i in range(training.shape[1])])
    # Graph-only Hodge projection on the true 8-head / 28-potential-edge
    # graph; no assumed higher-order faces.
    ii,jj=np.triu_indices(8,1); incidence=np.zeros((8,len(ii)))
    incidence[ii,np.arange(len(ii))]=1;incidence[jj,np.arange(len(ii))]=-1
    cycle=np.eye(len(ii))-incidence.T@np.linalg.pinv(incidence@incidence.T)@incidence
    mode_cycle_fraction=np.square(modes@cycle).sum(-1)/np.square(modes).sum(-1)
    cut=quiet[split-1][0]; events=[]
    for key in sorted(bounds):
        if key<=cut or set(bounds[key])!={'event_before','event_after'}:continue
        kind=kinds.get(key);delta=bounds[key]['event_after']-bounds[key]['event_before']
        energy=np.square(delta).sum(-1); coeff=(delta*modes).sum(-1)
        frac=np.divide(np.square(coeff),energy,out=np.zeros_like(energy),where=energy>1e-30)
        valid=energy>1e-30; data=receipts.get(key,{})
        events.append({'clock':key,'kind':kind,'flow_delta_l2':float(np.linalg.norm(delta)),
            'projection_l2':float(np.linalg.norm(coeff)),
            'equal_layer_projection_fraction':float(frac[valid].mean()) if valid.any() else None,
            'per_layer_projection':coeff.tolist(),
            'relation':(data.get('shared_unknown') or {}).get('relation_identity'),
            'choice':data.get('choice')})
    groups={}
    for kind in sorted(set(e['kind'] for e in events if e['kind'])):
        subset=[e for e in events if e['kind']==kind]; active=[e for e in subset if e['equal_layer_projection_fraction'] is not None]
        groups[kind]={'events':len(subset),'nonzero_flow_events':len(active),
            'flow_delta_l2_median':float(np.median([e['flow_delta_l2'] for e in subset])),
            'equal_layer_projection_fraction_median':float(np.median([e['equal_layer_projection_fraction'] for e in active])) if active else None}
    out={'training_samples':split,'training_end':cut,'groups':groups,'events':events,
         'mode_graph_cycle_fraction_by_layer':mode_cycle_fraction.tolist(),
         'note':'Fixed first-half per-layer flow modes. Internal and external event durations, changes and sampling differ. Event-before/after describes the full real event, not an intervention or evidence of memory.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':main()
