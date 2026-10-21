"""Joint real-tree audit on the fixed spatial candidate's known cell types."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.tree_geometry import chain_geometry


def main(args):
    start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=True)
    z=np.load(args.data/'morphology.npz',allow_pickle=False)
    state=json.loads((args.run/'development_state.json').read_text(encoding='utf-8'))
    selected=np.unique(state['arrays']['cell_type'])
    valid=np.isfinite(z['soma']).all(1)&np.isin(z['side'],['L','R'])&(z['length']>0)
    tc,body,side=z['cell_type'][valid],z['body'][valid],z['side'][valid]
    rng=np.random.default_rng(20261022);parts=np.full(len(body),'train',dtype='U5')
    for t in np.unique(tc):
        idx=np.flatnonzero(tc==t);idx=idx[rng.permutation(len(idx))]
        hold=max(1,int(len(idx)*.1)) if len(idx)>=3 else 0
        parts[idx[:hold]]='val';parts[idx[hold:2*hold]]='test'
    meta={int(b):(int(t),str(s),str(p)) for b,t,s,p in zip(body,tc,side,parts) if t in selected}
    cache=args.out/'tree_records.npz'
    if cache.exists():
        q=np.load(cache);records={k:q[k] for k in q.files}
    else:
        rows=[]
        for si,path in enumerate(sorted(args.skeletons.glob('skeletons_*.npz'))):
            with np.load(path) as q:
                ids,counts,x,p=q['body_ids'],q['node_count'],q['xyz_nm'],q['parent']
            offset=0
            for b,n in zip(ids,counts):
                n=int(n);b=int(b);info=meta.get(b)
                if info:
                    a=x[offset:offset+n].astype(float)*.008;pa=p[offset:offset+n].astype(int)
                    pa[(pa>=n)|(pa<0)]=-1
                    g=chain_geometry(a,pa);chains=g['chains']
                    rows.append((b,*info,g['length'],g['branches'],g['tips'],g['roots'],len(chains),
                                 float(chains.sum()),float((chains**2).sum()),
                                 float(np.median(chains)) if len(chains) else 0.,
                                 float(np.median(g['primary'])) if len(g['primary']) else 0.,
                                 float(g['chord'].sum()/max(chains.sum(),1e-30))))
                offset+=n
            del x,p
            if si%24==0: print(json.dumps({'stage':'real_tree','shards':si+1,'cells':len(rows),'seconds':time.perf_counter()-start}),flush=True)
        names=['body','cell_type','side','split','length','branches','tips','roots','n_chains','sum_chain','sum_chain_sq','median_chain','primary','chord_arc']
        records={name:np.asarray([r[j] for r in rows]) for j,name in enumerate(names)}
        np.savez_compressed(cache,**records)
    nt=len(z['types']);train=records['split']=='train'
    # Gamma arclength distribution from pooled non-root chains of training bodies.
    counts=np.bincount(records['cell_type'][train],minlength=nt)
    def mean_field(key):
        return np.bincount(records['cell_type'][train],weights=records[key][train],minlength=nt)/np.maximum(counts,1)
    sums={key:np.bincount(records['cell_type'][train],weights=records[key][train],minlength=nt) for key in ('sum_chain','sum_chain_sq','n_chains')}
    mean=sums['sum_chain']/np.maximum(sums['n_chains'],1)
    variance=sums['sum_chain_sq']/np.maximum(sums['n_chains'],1)-mean**2
    shape=np.divide(mean**2,np.maximum(variance,1e-8),out=np.ones(nt),where=mean>0)
    import torch
    source=np.load(args.work/'e1_types3000.npz',allow_pickle=True)
    if not np.array_equal(source['names'].astype(str),z['types']): raise ValueError('Identity ordering changed')
    mask=(torch.rand((nt,nt),generator=torch.Generator().manual_seed(1))>=.1).numpy()
    w=source['W']*mask
    output_share=(w.sum(1)+.5)/(w.sum(1)+w.sum(0)+1)
    # Continuous local shape distribution; never retain individual tree edges.
    np.savez(args.out/'tree_program.npz',types=z['types'],known=counts>0,
             gamma_shape=np.clip(shape,.05,100),primary_length=mean_field('primary'),
             root_count=np.maximum(1,np.rint(mean_field('roots'))).astype(int),
             output_share=output_share,training_counts=counts)
    report={'scope':'Real compressed tree-chain statistics, fixed 48 known types, held-out bodies',
            'selected_types':selected.tolist(),'real_cells':len(records['body']),
            'split_cells':{k:int(np.sum(records['split']==k)) for k in ('train','val','test')},
            'train_type_counts':{str(t):int(counts[t]) for t in selected},
            'test_types':int(len(np.unique(records['cell_type'][records['split']=='test']))),
            'seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'extraction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--skeletons',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    main(p.parse_args())
