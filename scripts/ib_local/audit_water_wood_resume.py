"""Stream-hash complete carried state after two restored real transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import struct

import numpy as np
import torch
from torch import nn

from scripts.ib_local import gdn_fused, run_navigation_residual_wood as legacy
from scripts.ib_local.navigation_checkpoint import CARRY, snapshot, restore
from scripts.ib_local.run_causal_water_reply import build


def digest(value):
    h=hashlib.sha256()
    def add(v):
        h.update(type(v).__qualname__.encode());h.update(b':')
        if torch.is_tensor(v):
            add(v.detach().cpu().contiguous().numpy())
        elif isinstance(v,np.ndarray):
            h.update(v.dtype.str.encode());h.update(str(v.shape).encode())
            h.update(memoryview(np.ascontiguousarray(v)).cast('B'))
        elif isinstance(v,np.random.Generator): add(v.bit_generator.state)
        elif isinstance(v,nn.Module): add((v.training,v.state_dict()))
        elif isinstance(v,dict):
            for k in sorted(v,key=repr): add(k);add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v:add(x)
        elif isinstance(v,(float,np.floating)):h.update(struct.pack('d',float(v)))
        elif isinstance(v,(str,int,bool,type(None),np.integer,torch.device)):
            h.update(str(v).encode())
        elif isinstance(v,bytes):
            h.update(str(len(v)).encode());h.update(v)
        elif hasattr(v,'__dict__'):add(vars(v))
        else:raise TypeError(f'undeclared state hash type: {type(v)}')
        h.update(b';')
    add(value)
    return h.hexdigest()


def live_parts(engine):
    skip=legacy._DERIVED.difference(CARRY)
    table=dict(arrays={n:getattr(engine.table,n) for n in
        ('rows','count','tag','err','epoch','last_write')},
        history=int(engine.table.history),events=engine.table.events,
        config={n:getattr(engine.table,n) for n in
        ('orders','heads','slots_log2','gain_floor','err_floor','prior_weight')})
    return dict(model=engine.model.state_dict(),optimizer=engine.optimizer.state_dict(),
        engine={k:v for k,v in vars(engine).items() if k not in skip},
        world=engine.world.state_dict(),table=table,
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()))


def audit(engine):
    initial=snapshot(engine)
    try:
        first=engine.step(torch.randn(64,32,device=engine.device))
        before={k:digest(v) for k,v in live_parts(engine).items()}
        restore(engine,initial)
        second=engine.step(torch.randn(64,32,device=engine.device))
        after={k:digest(v) for k,v in live_parts(engine).items()}
    finally:
        restore(engine,initial)
    report=dict(equal_parts={k:before[k]==after[k] for k in before},
        complete_row_equal=digest(first)==digest(second),restored_to_start=True,
        state_hashes=before,scope='all declared model/buffer, optimizer, engine, table, world and RNG state; two actual full learning transitions')
    assert all(report['equal_parts'].values()) and report['complete_row_equal'],report
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint)
    report=audit(engine)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
