"""Frozen-weight OWT diagnosis of CBIM v3 boundary and dynamical regime."""
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir: sys.path.pop(0)
if repo_root not in sys.path: sys.path.insert(0, repo_root)

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.ib_local.cbim_malecns_v3 import CBIMMaleCNSV3


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('results/cbim_malecns_v3_3000/BBest.pt'))
    p.add_argument('--data',type=Path,default=Path('data/ib_owt_gpt2/validation.npy'))
    p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--score',type=int,default=512)
    p.add_argument('--output',type=Path,default=Path('results/published/cbim_v3_dynamics.json'))
    a=p.parse_args(); torch.manual_seed(101)
    saved=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
    cfg=saved['config']; model=CBIMMaleCNSV3(cfg['graph'],velocities=cfg['velocities'],content_dim=cfg['content_dim']).cuda().eval()
    model.load_state_dict(saved['model']); data=np.load(a.data,mmap_mode='r')
    ids=torch.as_tensor(np.array(data[:a.warmup+a.score+1]),device='cuda',dtype=torch.long)
    pos=model.readout.position_encoding()

    @torch.no_grad()
    def one(state, token, arm='keep'):
        if arm=='reset': state=model.initial_state(1,device='cuda')
        field,resource,fatigue=model.unpack(state)
        field,resource,envelope,_=model.source(field,resource,fatigue,token[None])
        if arm=='resource_one': resource=torch.ones_like(resource)
        if arm!='no_transport': field,_=model.transport(field)
        if arm!='no_collision': field,_=model.collision(field)
        if arm!='no_outflow': field,_=model.outflow(field,resource,fatigue,envelope)
        local=.5*field.square().sum(-1)
        fatigue=model.fatigue_decay*fatigue+(1-model.fatigue_decay)*local
        if arm=='fatigue_zero': fatigue=torch.zeros_like(fatigue)
        state=torch.cat((field,resource[...,None],fatigue[...,None]),-1)
        logits=model.decoder(model.readout(field,pos))
        return state,logits

    state=model.initial_state(1,device='cuda')
    with torch.no_grad():
        for t in range(a.warmup): state,_=one(state,ids[t])
    warm=state.clone(); arms=['keep','reset','resource_one','fatigue_zero','no_outflow','no_collision','no_transport']
    scores={}
    for arm in arms:
        s=warm.clone(); total=0.
        with torch.no_grad():
            for j in range(a.score):
                s,logits=one(s,ids[a.warmup+j],arm)
                total+=float(F.cross_entropy(logits,ids[a.warmup+j+1,None]))
        scores[arm]=total/a.score

    scales={}
    for factor in [.25,.5,1.,2.,4.]:
        f,r,z=model.unpack(warm.clone()); s=torch.cat((f*factor,r[...,None],z[...,None]),-1); total=0.
        with torch.no_grad():
            for j in range(a.score):
                s,logits=one(s,ids[a.warmup+j])
                total+=float(F.cross_entropy(logits,ids[a.warmup+j+1,None]))
        scales[str(factor)]=total/a.score

    base=warm.clone(); f,r,z=model.unpack(base); noise=torch.randn_like(f); noise*=1e-4*f.norm()/noise.norm().clamp_min(1e-12)
    pert=torch.cat((f+noise,r[...,None],z[...,None]),-1); initial=float(noise.norm()); trace=[]
    with torch.no_grad():
        for j in range(a.score):
            base,_=one(base,ids[a.warmup+j]); pert,_=one(pert,ids[a.warmup+j])
            delta=float((pert[...,:model.d]-base[...,:model.d]).norm())
            trace.append(np.log(max(delta,1e-30)/initial))
    x=np.arange(1,min(129,len(trace)+1)); lyap=float(np.polyfit(x,np.array(trace[:len(x)]),1)[0])
    report={'checkpoint_step':saved['step'],'warmup_tokens':a.warmup,'score_tokens':a.score,'nll':scores,
            'delta_vs_keep':{k:v-scores['keep'] for k,v in scores.items()},'state_scale_nll':scales,
            'conditional_lyapunov_per_token_first128':lyap,'log_perturbation_ratio':trace}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
