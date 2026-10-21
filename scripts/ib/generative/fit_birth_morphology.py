"""Fit morphology descriptors on type-disjoint real morphology splits."""
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
from information_boltzmann.cgb.morphology import identity_features,symmetric_log,MorphologyProgram
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator

FAMILIES={'soma':slice(0,3),'centroid_offset':slice(3,6),'length':slice(6,7),
          'log_covariance':slice(7,13),'branch_density':slice(13,14)}


def mse(a,b):
    return float(np.mean((a-b)**2))


def main(args):
    start=time.perf_counter()
    z=np.load(args.data/'morphology.npz',allow_pickle=False)
    generator=TypeStrengthGenerator.load(args.generator)
    if not np.array_equal(z['types'],generator.types): raise ValueError('Type order changed')
    valid=np.isfinite(z['soma']).all(1)&(z['length']>0)
    tc,side=z['cell_type'][valid],z['side'][valid]
    y=np.column_stack([z['soma'][valid],z['centroid'][valid]-z['soma'][valid],np.log(z['length'][valid]),
                       symmetric_log(z['covariance'][valid]),np.log((z['branches'][valid]+1)/z['length'][valid])])
    # Equal weight per type-side program; no populous class can dominate.
    key=tc*3+np.where(side=='L',0,np.where(side=='R',1,2))
    keys,inverse=np.unique(key,return_inverse=True)
    counts=np.bincount(inverse)
    grouped=np.column_stack([np.bincount(inverse,weights=y[:,j])/counts for j in range(y.shape[1])])
    group_type=keys//3
    group_side=np.array(['L','R','other'])[keys%3]
    u=np.random.default_rng(20261021).random(len(generator.types))
    split={'train':np.flatnonzero(u[group_type]>=.2),'val':np.flatnonzero(u[group_type]<.1),
           'test':np.flatnonzero((u[group_type]>=.1)&(u[group_type]<.2))}
    tr,va,te=split['train'],split['val'],split['test']
    x=identity_features(generator.values['code'],group_type,group_side)
    xm,xs=x[tr].mean(0),x[tr].std(0)
    xm[0],xs[0]=0,1
    xs=np.maximum(xs,1e-8)
    x=(x-xm)/xs
    ym,ys=grouped[tr].mean(0),np.maximum(grouped[tr].std(0),1e-8)
    yz=(grouped-ym)/ys
    baseline_weight=np.linalg.lstsq(x[tr,:3],yz[tr],rcond=None)[0]
    baseline=x[:,:3]@baseline_weight
    grid=[]
    gram=x[tr].T@x[tr]/len(tr)
    rhs=x[tr].T@yz[tr]/len(tr)
    penalty=np.eye(x.shape[1]);penalty[0,0]=0
    for lam in np.logspace(-4,4,17):
        weight=np.linalg.solve(gram+lam*penalty,rhs)
        predicted=x@weight
        grid.append({'lambda':float(lam),'train_mse':mse(predicted[tr],yz[tr]),
                     'val_mse':mse(predicted[va],yz[va]),'weight':weight})
    best=min(grid,key=lambda row:row['val_mse'])
    weight=best['weight'].copy()
    pred=x@weight
    deploy_family={}
    for name,sl in FAMILIES.items():
        supported=mse(pred[va,sl],yz[va,sl])<mse(baseline[va,sl],yz[va,sl])
        deploy_family[name]='identity' if supported else 'side_baseline'
        if not supported:
            weight[:,sl]=0
            weight[:3,sl]=baseline_weight[:,sl]
    # Selection and family fallback locked before looking at test scores.
    lock={'lambda':best['lambda'],'family_selection':deploy_family,
          'type_split_sha256':{k:hashlib.sha256(np.unique(group_type[idx]).tobytes()).hexdigest() for k,idx in split.items()}}
    (args.data/'morphology_selection_locked.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    fitted=x@weight
    families={}
    for name,sl in FAMILIES.items():
        a,b=mse(fitted[te,sl],yz[te,sl]),mse(baseline[te,sl],yz[te,sl])
        families[name]={'test_mse':a,'side_baseline_test_mse':b,'gain':1-a/b,
                        'deployed':deploy_family[name]}
    # Intratype seed variation, estimated only from training types.
    train_cells=u[tc]>=.2
    residual=y[train_cells,:3]-grouped[inverse[train_cells],:3]
    soma_noise=(residual.T@residual)/max(1,len(residual))
    path=args.data/'morphology_program.npz'
    np.savez(path,code=generator.values['code'],types=generator.types,weight=weight,
             feature_mean=xm,feature_scale=xs,target_mean=ym,target_scale=ys,soma_noise_covariance=soma_noise)
    program=MorphologyProgram.load(path)
    decoded=program.describe(group_type,group_side)
    expected=(fitted*ys+ym)[:,:3]
    error=float(np.max(np.abs(decoded['soma_mean']-expected)))
    assert error<1e-9
    report={'status':'completed','scope':'Type-disjoint morphology descriptor decoding within one maleCNS animal',
            'cell_count':int(valid.sum()),'group_count':len(keys),'known_types':int(len(np.unique(group_type))),
            'splits':{k:{'groups':len(v),'types':len(np.unique(group_type[v]))} for k,v in split.items()},
            'selected_lambda':best['lambda'],'families':families,'test_mse':mse(fitted[te],yz[te]),
            'baseline_test_mse':mse(baseline[te],yz[te]),
            'gain':1-mse(fitted[te],yz[te])/mse(baseline[te],yz[te]),
            'decoder_parameters':weight.size,'grid':[{k:v for k,v in row.items() if k!='weight'} for row in grid],
            'deployment_error':error,'seconds':time.perf_counter()-start,
            'fit_type_codes_fixed':True,'shape_and_wiring_validated':False,
            'program':str(path),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.data/'morphology_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='grid'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--generator',type=Path,required=True)
    main(p.parse_args())
