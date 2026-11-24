"""Integrate the original successor law over the actual native address cells.

Consumes finite frozen geometry capsules, never reloads or trains a body.
Reports nested Sobol estimates, observed successor location, and original
four-point routing. Future labels are scored after all predictive masses.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import stdtr, stdtrit
from scipy.stats import qmc

from scripts.ib_local import successor_address_geometry as geometry
from scripts.ib_local.successor_distribution import quadrature


@lru_cache(maxsize=16)
def directions(df,width,power,seed):
    sampler=qmc.Sobol(d=width,scramble=True,seed=seed)
    uniform=sampler.random_base2(power)
    return torch.from_numpy(stdtrit(df,np.clip(uniform,1e-12,1-1e-12))).float()


def measure(case,actual_relation,power,seed):
    started=time.perf_counter()
    packet=case['geometry'];d=case['distribution']
    df=float(d['df']);width=d['mean'].shape[-1]
    direction=directions(df,width,power,seed).to(d['mean'])
    points=d['mean']+d['scale2'].sqrt()*direction
    result=geometry.addresses(packet,points)
    def masses(ids):
        identity,count=ids.unique(return_counts=True)
        return {str(int(i)):int(n)/len(ids) for i,n in zip(identity,count)}
    nested={str(p):masses(result['identity'][:2**p]) for p in (power-2,power)}
    four,_=quadrature(d,case['budget'])
    original=geometry.addresses(packet,four[0])
    actual=geometry.addresses(packet,case['actual_latent'])
    known_raw=case['actual_raw']
    # Separately record whether omitted encoder-kernel coordinates change scope.
    raw_proj=torch.einsum('kew,bw->bke',packet['axes'],known_raw)[0]
    tolerance=torch.finfo(known_raw.dtype).eps*(1+raw_proj.abs()+packet['bounds'].abs())*8
    raw_scope=((packet['signs']*(raw_proj-packet['bounds'])<=tolerance)|~packet['valid']).all(-1)
    raw_score=torch.nn.functional.normalize(case['actual_latent'],dim=-1)@torch.nn.functional.normalize(packet['centers'],dim=-1).T
    raw_score=raw_score[0].masked_fill(~((packet['identity']>0)&raw_scope),-2)
    score,index=raw_score.max(0)
    raw_identity=int(packet['identity'][index]) if float(score)>=packet['threshold'] else -1
    standard=(case['actual_latent']-d['mean'])/d['scale2'].sqrt()
    pit=stdtr(df,standard.numpy())
    z=case['actual_latent'];center=packet['centers']
    radius=points.norm(dim=-1)
    displacement=(points-d['mean']).norm(dim=-1)
    # Density ranking is an observed diagnostic over current legal centers;
    # no inferred cell mass is replaced by normalized center density.
    density=torch.distributions.StudentT(df,d['mean'],d['scale2'].sqrt()).log_prob(center).sum(-1)
    density=density.masked_fill(packet['identity']<=0,-torch.inf)
    ids=packet['identity'][density.argsort(descending=True)]
    target_rank=torch.nonzero(ids==actual_relation).flatten()
    return dict(name=case['name'],samples=len(points),mass_by_size=nested,
        actual_relation=actual_relation,actual_lifted_relation=int(actual['identity'][0]),
        actual_full_raw_relation=raw_identity,actual_similarity=float(actual['similarity'][0]),
        four_point_relations=original['identity'].tolist(),threshold=packet['threshold'],
        actual_relation_mass=nested[str(power)].get(str(actual_relation),0.),
        fresh_mass=nested[str(power)].get('-1',0.),
        actual_standardized_RMS=float(standard.square().mean().sqrt()),
        actual_marginal_95_fraction=float(((pit>=.025)&(pit<=.975)).mean()),
        mean_norm=float(d['mean'].norm()),actual_norm=float(z.norm()),
        median_draw_norm=float(radius.median()),actual_displacement_norm=float((z-d['mean']).norm()),
        median_draw_displacement=float(displacement.median()),
        existing_center_norm_median=float(center[packet['identity']>0].norm(dim=-1).median()),
        actual_center_density_rank=None if not len(target_rank) else int(target_rank[0])+1,
        legal_center_count=int((packet['identity']>0).sum()),
        seconds=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--power',type=int,default=14)
    p.add_argument('--seed',type=int,default=191119)
    a=p.parse_args();torch.set_num_threads(1)
    rows=[]
    for path in sorted(a.work.glob('geometry_*.pt')):
        capsule=torch.load(path,map_location='cpu',weights_only=False)
        cases=[measure(c,capsule['actual_relation'],a.power,a.seed) for c in capsule['cases']]
        rows.append(dict(cycle=capsule['cycle'],choice=capsule['choice'],checks=capsule['checks'],cases=cases))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(status='running',observations=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(cycle=capsule['cycle'],cases=len(cases),
            selected=[c for c in cases if c['name'].startswith(f'action:{capsule["choice"]}:') or c['name']=='actual_response'])),flush=True)
    assert len(rows)==3,len(rows)
    report=dict(status='completed',valid=all(all(r['checks'].values()) for r in rows),
        method='nested scrambled Sobol integration of product Student-t through original native address cells',
        training_updates=0,world_calls=0,power=a.power,seed=a.seed,observations=rows)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
