"""Convex cell-pair probability calibration with a locked type-pair split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import rankdata
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.occurrence import OccurrenceProgram
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator


def metrics(score,target):
    p=expit(score);target=np.asarray(target)
    positives=int(target.sum());negatives=len(target)-positives
    auc=(rankdata(score)[target>0].sum()-positives*(positives+1)/2)/(positives*negatives)
    bins=[]
    for idx in np.array_split(np.argsort(p),10):
        bins.append({'n':len(idx),'predicted':float(p[idx].mean()),'observed':float(target[idx].mean())})
    return {'n':len(target),'positives':positives,'log_loss':float(np.mean(np.logaddexp(0,score)-target*score)),
            'brier':float(np.mean((p-target)**2)),'auc':float(auc),'calibration_deciles':bins}


def main(args):
    start=time.perf_counter()
    z=np.load(args.data/'occurrence_panel.npz',allow_pickle=False)
    strength=TypeStrengthGenerator.load(args.generator)
    code=strength.values['code'].astype(float)
    code/=np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-30)
    types=z['cell_type'];n=len(types)
    pre,post=np.where(~np.eye(n,dtype=bool))
    target=(z['weights'][pre,post]>0).astype(float)
    # Exact same generator and shape as E1/V3d; all cells sharing a type pair
    # share its split. No direct held-out pair target entered the fixed code.
    u=torch.rand((len(code),len(code)),generator=torch.Generator().manual_seed(1)).numpy()[types[pre],types[post]]
    split={'train':np.flatnonzero(u>=.1),'val':np.flatnonzero(u<.05),
           'test':np.flatnonzero((u>=.05)&(u<.1))}
    features=[];floors=[];overlaps=[]
    for scale in (4,2):
        e=sp.load_npz(args.data/f'cell_overlap_{scale}um.npz').toarray()[pre,post]
        positive=e[split['train']];positive=positive[positive>0]
        floor=float(positive.min())*.5;floors.append(floor);overlaps.append(e)
        features.extend([np.log(np.maximum(e,floor)),(e==0).astype(float)])
    raw=np.stack(features,axis=1)
    mean=raw[split['train']].mean(0);scale=np.maximum(raw[split['train']].std(0),1e-8)
    raw=(raw-mean)/scale
    dev=args.device
    if dev=='cuda': torch.cuda.reset_peak_memory_stats()
    c=torch.tensor(code[types],dtype=torch.float32,device=dev)
    x=torch.tensor(raw,dtype=torch.float32,device=dev)
    y=torch.tensor(target,dtype=torch.float32,device=dev)
    pi=torch.tensor(pre,device=dev);qi=torch.tensor(post,device=dev)
    train=torch.tensor(split['train'],device=dev);val=split['val']
    prevalence=float(target[split['train']].mean())
    grid=[];saved={}

    def fit(family,lam):
        k=c.shape[1]
        bias=torch.tensor(np.log(prevalence/(1-prevalence)),dtype=torch.float32,device=dev,requires_grad=True)
        geometry=torch.zeros(4,device=dev,requires_grad=True)
        out=torch.zeros(k,device=dev,requires_grad=True);inc=torch.zeros(k,device=dev,requires_grad=True)
        affinity=torch.zeros(k,k,device=dev,requires_grad=True)
        parameters=[bias,geometry] if family=='geometry' else [bias,geometry,out,inc,affinity]
        opt=torch.optim.LBFGS(parameters,max_iter=160,history_size=12,line_search_fn='strong_wolfe',
                              tolerance_grad=1e-7,tolerance_change=1e-12)
        trace=[]
        def scores():
            result=bias+x@geometry
            if family=='identity_geometry':
                matrix=(c@affinity)@c.T
                result=result+(c@out)[pi]+(c@inc)[qi]+matrix[pi,qi]
            return result
        def objective():
            s=scores()
            penalty=sum(v.square().sum() for v in parameters[1:])
            loss=F.binary_cross_entropy_with_logits(s[train],y[train])+lam*penalty
            return loss
        def closure():
            opt.zero_grad(set_to_none=True);loss=objective();loss.backward()
            trace.append(float(loss.detach()))
            return loss
        t0=time.perf_counter();grad=np.inf
        for _ in range(4):
            opt.step(closure)
            closure()
            grad=max(float(v.grad.abs().max()) for v in parameters)
            if grad<1e-5: break
        with torch.no_grad(): prediction=scores().cpu().double().numpy()
        row={'family':family,'lambda':lam,'train':metrics(prediction[split['train']],target[split['train']]),
             'val':metrics(prediction[val],target[val]),'gradient_inf':grad,
             'settled':bool(grad<1e-5),'optimizer_iterations':int(opt.state[parameters[0]]['n_iter']),
             'function_evaluations':len(trace),'seconds':time.perf_counter()-t0}
        arrays={'bias':bias.detach().cpu().numpy(),'geometry':geometry.detach().cpu().numpy(),
                'out':out.detach().cpu().numpy(),'in':inc.detach().cpu().numpy(),
                'affinity':affinity.detach().cpu().numpy()}
        return row,arrays,prediction,trace

    for family in ('geometry','identity_geometry'):
        for lam in (1e-6,1e-5,1e-4,1e-3,.01,.1):
            row,arrays,prediction,trace=fit(family,lam)
            key=f'{family}_{lam:g}'
            np.savez(args.data/(key+'.npz'),**arrays)
            (args.data/(key+'_curve.json')).write_text(json.dumps(trace),encoding='utf-8')
            grid.append(row);saved[key]=(arrays,prediction)
            print(json.dumps({k:row[k] for k in ('family','lambda','gradient_inf','settled','optimizer_iterations','seconds')}|
                             {'val_log_loss':row['val']['log_loss']}),flush=True)
    # Extend an edge optimum before scoring the test set.
    for family in ('geometry','identity_geometry'):
        selected=min([r for r in grid if r['family']==family],key=lambda r:r['val']['log_loss'])
        if selected['lambda'] in (1e-6,.1):
            lam=1e-7 if selected['lambda']==1e-6 else 1.
            row,arrays,prediction,trace=fit(family,lam)
            key=f'{family}_{lam:g}';saved[key]=(arrays,prediction);grid.append(row)
            np.savez(args.data/(key+'.npz'),**arrays)
            (args.data/(key+'_curve.json')).write_text(json.dumps(trace),encoding='utf-8')
    best={family:min([r for r in grid if r['family']==family],key=lambda r:r['val']['log_loss'])
          for family in ('geometry','identity_geometry')}
    chosen=min(best.values(),key=lambda r:r['val']['log_loss'])
    lock={'selected_family':chosen['family'],'selected_lambda':chosen['lambda'],
          'split_sha256':{k:hashlib.sha256(np.stack([pre[idx],post[idx]],1).tobytes()).hexdigest() for k,idx in split.items()}}
    (args.data/'occurrence_selection_locked.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    results={}
    for family,row in best.items():
        prediction=saved[f"{family}_{row['lambda']:g}"][1]
        results[family]={**row,'test':metrics(prediction[split['test']],target[split['test']])}
    arrays,prediction=saved[f"{chosen['family']}_{chosen['lambda']:g}"]
    program=args.data/'occurrence_program.npz'
    np.savez(program,**arrays,code=strength.values['code'],types=strength.types,
             feature_mean=mean,feature_scale=scale,floors=floors)
    deployed=OccurrenceProgram.load(program)
    expected=expit(prediction)
    actual=deployed.probability(types[pre],types[post],*overlaps)
    error=float(np.max(np.abs(actual-expected)))
    report={'status':'completed','scope':'Observed cell-pair probability on one-animal panel; fixed historical type-pair holdout',
            'cells':n,'pairs':len(pre),'splits':{k:{'n':len(v),'positive':int(target[v].sum())} for k,v in split.items()},
            'selection':lock,'best':results,'grid':grid,
            'contact_coverage_of_positives':float(np.mean(overlaps[0][target>0]>0)),
            'deployment_max_probability_error':error,
            'settled':chosen['settled'],'program':str(program),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if dev=='cuda' else None,
            'seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.data/'occurrence_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('grid','best')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--generator',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    main(p.parse_args())
