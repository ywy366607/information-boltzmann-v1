from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from fine_grain.active_gdn2 import ActiveInferenceGDN2

def feat(v): return [1. if (v>>j)&1 else -1. for j in range(8)]
def run(length):
    torch.manual_seed(0); a=ActiveInferenceGDN2(8,8,8); d=ActiveInferenceGDN2(8,8,8)
    sa=a.initial_state(1,torch.device('cpu'),torch.float32); sd=d.initial_state(1,torch.device('cpu'),torch.float32)
    sa.precision.fill_(1.); sd.precision.fill_(1.)
    raw=(b"the language life lives in a continuous stream ")[:length]
    last_keys=[]; last_vals=[]
    for t,v in enumerate(raw):
        tok=torch.tensor([[feat(v)]],dtype=torch.float32); c=a.atlas[(t%8):(t%8)+1].unsqueeze(0)
        sa=a.assimilate(sa,tok,c,torch.ones(1),update_motion=False)
        key=d.address(tok,c).squeeze(1); key=key/key.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        sd=d.association_delta_update(sd,key,tok[:,0])
        last_keys.append(key); last_vals.append(tok[:,0])
    keys=torch.cat(last_keys[-8:],dim=0).unsqueeze(0); vals=torch.cat(last_vals[-8:],dim=0).unsqueeze(0)
    dual_a=a.association_dual_keys(keys); dual_d=d.association_dual_keys(keys)
    return {"length":len(raw),"accum_dual_error":float((a.association_dual_readout(sa,keys)-vals).abs().amax().detach()),"delta_dual_error":float((d.association_dual_readout(sd,keys)-vals).abs().amax().detach()),"gram_condition":float(torch.linalg.cond(keys[0]@keys[0].T).detach())}
out={"experiment":"gdn_byte_stream_delta_reference_20261030","status":"mechanism_diagnostic_only","rows":[run(n) for n in (8,16,32,48)]}
p=ROOT/'results/published/gdn_byte_stream_delta_reference_20261030.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)+"\n",encoding='utf-8'); print(json.dumps(out,indent=2))
