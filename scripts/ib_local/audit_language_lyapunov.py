"""Finite-time driven GDN tangent-QR diagnostic on real held-out bytes.

Frozen base GDN only: no optimizer, byte table, latent memory or action/world
feedback. Parallel symmetric finite differences estimate JVPs; QR is periodic.
This is not Clark's minimum-norm/cavity solver or a full-body spectrum.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--interval',type=int,default=8)
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--seed',type=int,default=61)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('preserve prior audit')
    if min(a.steps,a.interval,a.rank,a.epsilon)<=0 or a.steps%a.interval: raise ValueError('invalid budget')
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').to(device).eval()
    weights={k.removeprefix('body.base.'):v for k,v in saved['model'].items() if k.startswith('body.base.')}
    base.load_state_dict(weights,strict=True)
    for param in base.parameters(): param.requires_grad_(False)
    state=saved['recurrent'].to(device)
    root=Path('data/ib_owt_10000')
    off=json.loads((root/'tokenizer.json').read_text())['offset']
    raw=np.load(root/'validation.npy',mmap_mode='r')
    buf=np.asarray(raw[:2*(a.warmup+a.steps)])
    data=(buf[buf>=off]-off)[:a.warmup+a.steps].astype(np.uint8)
    if len(data)!=a.warmup+a.steps: raise ValueError('insufficient validation bytes')
    torch.manual_seed(a.seed)
    with torch.no_grad():
        for t in range(0,a.warmup,a.interval):
            ids=torch.tensor(data[t:min(t+a.interval,a.warmup)].astype(np.int64),device=device)[None]
            _,state,_=base.predict_logits(ids,state)
        shape=state.shape; dimension=state.numel()
        q,_=torch.linalg.qr(torch.randn(dimension,a.rank,device=device))
        sums=torch.zeros(a.rank,device=device,dtype=torch.float64)
        history=[]; t0=time.perf_counter()
        for t in range(0,a.steps,a.interval):
            direction=q.T.reshape(a.rank,shape[0],shape[2]).permute(1,0,2).contiguous()
            start=torch.cat((state,state+a.epsilon*direction,state-a.epsilon*direction),dim=1)
            ids=torch.tensor(data[a.warmup+t:a.warmup+t+a.interval].astype(np.int64),device=device)[None]
            _,out,_=base.predict_logits(ids.expand(1+2*a.rank,-1),start)
            state=out[:,:1].contiguous()
            tangent=((out[:,1:1+a.rank]-out[:,1+a.rank:])/(2*a.epsilon)).permute(1,0,2).reshape(a.rank,-1).T.contiguous()
            q,r=torch.linalg.qr(tangent)
            diag=torch.diagonal(r).abs()
            if not torch.isfinite(diag).all() or bool((diag==0).any()): raise FloatingPointError('invalid QR')
            sums+=diag.double().log()
            if (t+a.interval)%128==0 or t+a.interval==a.steps:
                history.append(dict(bytes=t+a.interval,estimates=(sums/(t+a.interval)).cpu().tolist()))
    result=dict(scope='frozen trained base GDN, fixed validation-byte drive, recurrent state only; no table/action/learning',
        estimator='central finite difference JVP + periodic QR, finite-time ordered tangent-growth rates',
        checkpoint=str(a.checkpoint), checkpoint_sha256=digest(a.checkpoint),source_sha256=digest(Path(__file__)),
        validation_sha256=hashlib.sha256(data.tobytes()).hexdigest(),device=str(device),state_dimension=dimension,
        warmup_bytes=a.warmup,bytes=a.steps,qr_interval=a.interval,rank=a.rank,epsilon_l2=a.epsilon,seed=a.seed,
        estimates_per_byte=(sums/a.steps).cpu().tolist(),history=history,elapsed_seconds=time.perf_counter()-t0,
        limits='Not the full spectrum or asymptotic exponents; real-input conditional rates depend on frozen weights, replayed input, state norm, finite horizon and epsilon.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
