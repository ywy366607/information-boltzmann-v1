"""Preregistered rank-normalization grid extension; original results preserved."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.matching import TypeMatching, PairBatch, optimize
from scripts.ib.generative.v3d_stable_matching import prepare,linear_fit,write_json,digest


def main(args):
    started=time.perf_counter()
    original=json.loads((args.run/'report.json').read_text())
    if original['status']!='completed':
        raise ValueError('Complete the original grid before this separate extension')
    data=prepare(args.work)
    if data['files']!=original['provenance']['data_sha256']:
        raise ValueError('Data changed after the primary fit')
    batch=PairBatch(torch.tensor(data['pre'],device='cuda'),torch.tensor(data['post'],device='cuda'),
        torch.tensor(data['features'],dtype=torch.float32,device='cuda'),
        torch.tensor(data['target'],dtype=torch.float32,device='cuda'))
    batches={name:batch.selected(torch.tensor(idx,device='cuda')) for name,idx in data['splits'].items()}
    train=data['splits']['train']
    coef,geometry,_=linear_fit(data['n'],data['pre'],data['post'],data['features'],data['target'],train)
    _,null,_=linear_fit(data['n'],data['pre'],data['post'],data['features'][:,:0],data['target'],train)
    fits=args.run/'rank_extension'; fits.mkdir(exist_ok=True)
    candidates=[r for r in original['all_fits'] if r['family']=='directed256']
    records=[]

    def fit(seed,l2,anchor=None,suffix='',steps=4000):
        path=fits/f'directed256_s{seed}_l{l2:g}{suffix}.pt'
        record_path=path.with_suffix('.json')
        if path.exists() and record_path.exists():
            record=json.loads(record_path.read_text())
            if (record['data_sha256']!=data['files'] or record['adam_updates']!=steps
                    or record['refinement_script_sha256']!=digest(Path(__file__))):
                raise ValueError('Refinement source/data changed; use a new run directory')
            return record
        model=TypeMatching(data['n'],256,8,seed=seed,device='cuda')
        model.initialize_linear(coef)
        if anchor:
            model.load_state_dict(torch.load(anchor['model_path'],map_location='cuda',weights_only=False)['state'])
        t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        result=optimize(model,batches['train'],batches['val'],l2,steps=steps,polish=200)
        record={**result,'family':'directed256','rank':256,'symmetric':False,'seed':seed,'lambda':l2,
            'warm_start':anchor is not None,'suffix':suffix,'model_path':str(path),
            'seconds':time.perf_counter()-t0,'parameters':sum(p.numel() for p in model.parameters()),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'data_sha256':data['files'],'refinement_script_sha256':digest(Path(__file__))}
        temporary=path.with_suffix('.tmp')
        torch.save({'specification':model.specification(),'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'types':data['names'].tolist(),'feature_mean':data['feature_mean'],
            'feature_scale':data['feature_scale'],'data_sha256':data['files']},temporary)
        temporary.replace(path)
        write_json(record_path,record)
        print(json.dumps({k:record[k] for k in ('seed','lambda','suffix','val_mse','gradient_l2','seconds')}),flush=True)
        return record

    seeds=[r['seed'] for r in original['selected']['directed256']['fits']]
    for seed in seeds:
        anchor=min((r for r in candidates if r['seed']==seed),key=lambda r:abs(np.log(r['lambda']/70)))
        low=fit(seed,70,anchor); records.append(low)
        high=fit(seed,140,low); records.append(high)
    candidates+=records
    means={l:float(np.mean([r['val_mse'] for r in candidates if r['lambda']==l]))
           for l in sorted({r['lambda'] for r in candidates})}
    best=min(means,key=means.get)
    if best==original['selected']['directed256']['lambda']:
        selected=copy.deepcopy(original['selected']['directed256'])
        selected['validation_means']=means
        selected['edge_of_searched_grid']=best in (min(means),max(means))
    else:
        chosen=[r for r in candidates if r['lambda']==best]
        extensions=[]; adopted=[]
        for record in chosen:
            extra=fit(record['seed'],best,record,suffix='_extended',steps=1000)
            improvement=record['objective']-extra['objective']
            extensions.append({'seed':record['seed'],'objective_improvement':improvement,
                'relative_objective_improvement':improvement/abs(record['objective']),
                'validation_change':extra['val_mse']-record['val_mse'],'refined':extra})
            adopted.append(extra if improvement>0 else record)
        selected={'lambda':best,'validation_means':means,'fits':adopted,
            'edge_of_searched_grid':best in (min(means),max(means)),
            'extension_check':extensions,'numerically_settled':all(abs(r['relative_objective_improvement'])<1e-4 for r in extensions),
            'cold_start':fit(seeds[0],best,suffix='_cold')}
    write_json(fits/'selection_locked.json',{'selected':selected,'test_evaluated':False,
        'criterion':'mean validation MSE; lambda=70,140 registered before 256-dimensional results'})
    test=data['splits']['test'];target=data['target'][test]
    nullsq=(null[test]-target)**2;geosq=(geometry[test]-target)**2
    scores=[];predictions=[]
    for record in selected['fits']+[selected['cold_start']]:
        saved=torch.load(record['model_path'],map_location='cuda',weights_only=False)
        model=TypeMatching(**saved['specification'],device='cuda');model.load_state_dict(saved['state'])
        with torch.no_grad():
            tb=batches['test'];prediction=model(tb.pre,tb.post,tb.features).double().cpu().numpy()
        explained=1-float(np.mean((prediction-target)**2)/nullsq.mean())
        score={'seed':record['seed'],'test_explained':explained,
            'test_gain_over_geometry':explained-original['baselines_test_explained']['geometry']}
        if record is selected['cold_start']:selected['cold_test']=score
        else:scores.append(score);predictions.append(prediction)
    selected['test_by_seed']=scores
    selected['test_mean']=float(np.mean([r['test_explained'] for r in scores]))
    selected['test_std']=float(np.std([r['test_explained'] for r in scores],ddof=1))
    sq=np.mean([(prediction-target)**2 for prediction in predictions],axis=0)
    rng=np.random.default_rng(20261021);gains=[]
    for _ in range(1000):
        draw=rng.integers(0,len(test),len(test));gains.append(float((geosq[draw]-sq[draw]).sum()/nullsq[draw].sum()))
    selected['gain_pair_bootstrap_95']=np.quantile(gains,[.025,.975]).tolist()
    result=copy.deepcopy(original)
    result['selected']['directed256']=selected
    result['all_fits'] += [{k:v for k,v in r.items() if k!='curve'} for r in records]
    result['predictions']['P4_256_gain_under_2pt']=selected['test_mean']-result['selected']['directed64']['test_mean']<.02
    result['rank_grid_extension']={'primary_report_sha256':digest(args.run/'report.json'),
        'preregistration':'CGB_V3D_PREREG_20261021.md final section','added_lambdas':[70,140],
        'seconds':time.perf_counter()-started}
    write_json(args.run/'refined_report.json',result)
    print(json.dumps({'selected_lambda':best,'test_mean':selected['test_mean'],'test_std':selected['test_std'],
        'settled':selected['numerically_settled'],'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--work',type=Path,default=Path('F:/fly_stream_overflow/cgb_v1'))
    main(parser.parse_args())
