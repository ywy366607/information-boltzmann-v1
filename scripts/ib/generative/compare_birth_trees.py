"""Joint tree/space/contact comparison; no test-driven parameter search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.development import encounter
from information_boltzmann.cgb.tree_geometry import chain_geometry


def generated(folder):
    state=json.loads((folder/'development_state.json').read_text(encoding='utf-8'))
    seg={k:np.asarray(v) for k,v in state['segments'].items()}
    tc=np.asarray(state['arrays']['cell_type']);soma=np.asarray(state['arrays']['soma'])
    side=np.asarray(state['spatial_plan']['side'])
    records=[]
    for c in range(len(tc)):
        ids=np.flatnonzero(seg['cell']==c);p=seg['parent'][ids];ok=p>=0
        local=np.full(len(p),len(ids),int);local[ok]=np.searchsorted(ids,p[ok])
        x=np.concatenate([seg['end'][ids],soma[c,None]])
        g=chain_geometry(x,np.r_[local,-1]);chains=g['chains']
        length=np.linalg.norm(seg['end'][ids]-seg['start'][ids],axis=1)
        centers=(seg['end'][ids]+seg['start'][ids])/2
        mean=(centers*length[:,None]).sum(0)/length.sum()
        d=centers-mean;cov=(d.T*length)@d/length.sum()
        records.append(dict(cell_type=int(tc[c]),side=str(side[c]),length=g['length'],branches=g['branches'],
            tips=g['tips'],roots=g['roots'],median_chain=float(np.median(chains)),
            chord_arc=float(g['chord'].sum()/max(chains.sum(),1e-30)),centroid=mean,covariance=cov,
            output_share=float(length[seg['side'][ids]==0].sum()/length.sum())))
    return records,seg,tc


def contact_stats(seg,tc,args):
    types,codes=np.unique(tc,return_inverse=True);nt=len(types)
    membership=sp.csr_matrix((np.ones(len(tc)),(codes,np.arange(len(tc)))),shape=(nt,len(tc)))
    counts=np.bincount(codes)
    sizes=np.load(args.work/'e1_types3000.npz',allow_pickle=True)['sizes']
    result={}
    for scale in (4,2):
        e=encounter(seg,len(tc),scale,polarized=True)
        predicted=(membership@e@membership.T).toarray()/(counts[:,None]*counts[None,:])
        actual=np.load(args.work/f'e1c_overlap_trainlabels_{scale}um.npy',mmap_mode='r')[np.ix_(types,types)]
        actual=actual/(sizes[types,None]*sizes[None,types])
        mask=~np.eye(nt,dtype=bool)
        a,b=actual[mask],predicted[mask]
        floor=float(np.min(a[a>0])/2)
        loga,logb=np.log(np.maximum(a,floor)),np.log(np.maximum(b,floor))
        result[str(scale)]={'type_pairs':len(a),'real_positive':int((a>0).sum()),'generated_positive':int((b>0).sum()),
            'positive_contact_recall':float(np.sum((a>0)&(b>0))/max(1,np.sum(a>0))),
            'positive_contact_precision':float(np.sum((a>0)&(b>0))/max(1,np.sum(b>0))),
            'log_spearman':float(spearmanr(loga,logb).statistic),
            'log_rmse':float(np.sqrt(np.mean((loga-logb)**2))),
            'scope':'Training-label type-averaged polarity reference vs four generated cells/type; descriptive transfer diagnostic, not independent occurrence validation'}
    return result


def main(args):
    real=np.load(args.audit/'tree_records.npz');morph=np.load(args.data/'morphology.npz')
    tests=real['split']=='test';types=np.unique(real['cell_type'][tests])
    body_index={int(b):i for i,b in enumerate(morph['body'])}
    report={'scope':'Joint description on fixed held-out bodies; no new model tuning after report',
            'test_cells':int(tests.sum()),'test_types':len(types),'models':{}}
    for folder in args.runs:
        gen,seg,tc=generated(folder);metrics={}
        for name in ('branches','tips','median_chain','chord_arc','length'):
            errors=[];ratios=[]
            for t in types:
                target=real[name][tests&(real['cell_type']==t)].astype(float)
                prediction=np.array([r[name] for r in gen if r['cell_type']==t])
                if name in ('branches','tips'):
                    actual_length=real['length'][tests&(real['cell_type']==t)]
                    target=(target+1)/actual_length
                    prediction=np.array([(r[name]+1)/r['length'] for r in gen if r['cell_type']==t])
                expected_log=np.log(np.maximum(prediction,1e-12)).mean()
                errors.append(float(np.mean((np.log(np.maximum(target,1e-12))-expected_log)**2)))
                ratios.append(float(np.exp(expected_log-np.log(np.maximum(target,1e-12)).mean())))
            metrics[name]={'type_equal_log_rmse':float(np.sqrt(np.mean(errors))),
                           'median_generated_to_real_ratio':float(np.median(ratios))}
        centroid_errors=[];cov_errors=[]
        for t in types:
            errs=[];cs=[]
            for j in np.flatnonzero(tests&(real['cell_type']==t)):
                candidates=[r for r in gen if r['cell_type']==t and r['side']==real['side'][j]]
                if not candidates: continue
                idx=body_index[int(real['body'][j])]
                c=np.mean([r['centroid'] for r in candidates],axis=0)
                covariance=np.mean([r['covariance'] for r in candidates],axis=0)
                errs.append(np.linalg.norm(c-morph['centroid'][idx]))
                # Frobenius relative moment error, with each type equal weight.
                cs.append(np.linalg.norm(covariance-morph['covariance'][idx])/max(np.linalg.norm(morph['covariance'][idx]),1e-12))
            if errs: centroid_errors.append(np.mean(errs));cov_errors.append(np.mean(cs))
        report['models'][folder.name]={'tree':metrics,'centroid_mean_error_um_type_equal':float(np.mean(centroid_errors)),
            'covariance_relative_error_type_equal':float(np.mean(cov_errors)),
            'branch_quantiles':np.quantile([r['branches'] for r in gen],[0,.5,1]).tolist(),
            'completed':json.loads((folder/'report.json').read_text()).get('tree_growth_completed'),
            'contact':contact_stats(seg,tc,args)}
        print(json.dumps({folder.name:report['models'][folder.name]},indent=2),flush=True)
    (args.audit/'joint_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--runs',nargs='+',type=Path,required=True)
    main(p.parse_args())
