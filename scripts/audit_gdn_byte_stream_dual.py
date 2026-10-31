from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from fine_grain.active_gdn2 import ActiveInferenceGDN2


def feat(v:int): return [1. if (v>>j)&1 else -1. for j in range(8)]

def run(length:int, seed:int=0):
    torch.manual_seed(seed); m=ActiveInferenceGDN2(8,8,8)
    s=m.initial_state(1,torch.device('cpu'),torch.float32)
    raw=(b"the language life lives in a continuous stream ")[:length]
    for t,v in enumerate(raw):
        tok=torch.tensor([[feat(v)]],dtype=torch.float32)
        c=m.atlas[(t%8):(t%8)+1].unsqueeze(0)
        s=m.assimilate(s,tok,c,torch.ones(1),update_motion=False)
    probe=raw[-8:]
    tokens=torch.tensor([[feat(v) for v in probe]],dtype=torch.float32)
    centers=m.atlas.unsqueeze(0)
    keys=m.address(tokens,centers)
    gram=keys[0].detach()@keys[0].detach().T
    rawerr=float(m.association_probe_error(s,keys,tokens).detach())
    dualerr=float((m.association_dual_readout(s,keys)-tokens).abs().amax().detach())
    return {"length":len(raw),"raw_error":rawerr,"dual_error":dualerr,"gram_condition":float(torch.linalg.cond(gram).detach()),"rank":int(torch.linalg.matrix_rank(keys[0].detach()))}

out={"experiment":"gdn_byte_stream_dual_readout_20261030","status":"mechanism_diagnostic_only","rows":[run(n) for n in (8,16,32,48)]}
p=ROOT/'results/published/gdn_byte_stream_dual_readout_20261030.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+"\n",encoding='utf-8'); print(json.dumps(out,indent=2))
