from pathlib import Path
import torch, json
import sys
sys.path.insert(0,str(Path('.').resolve()))
from fine_grain.active_gdn2 import ActiveInferenceGDN2

def feat(v): return [1. if (v>>j)&1 else -1. for j in range(8)]
def run(beta,length=47):
 m=ActiveInferenceGDN2(8,8,8); s=m.initial_state(1,torch.device('cpu'),torch.float32); s.precision.fill_(1.); raw=(b"the language life lives in a continuous stream ")[:length]; ks=[]; vs=[]
 for t,v in enumerate(raw):
  tok=torch.tensor([[feat(v)]],dtype=torch.float32); c=m.atlas[(t%8):(t%8)+1].unsqueeze(0); k=m.address(tok,c).squeeze(1); k=k/k.norm(dim=-1,keepdim=True).clamp_min(1e-6); s=m.association_delta_update(s,k,tok[:,0],beta); ks.append(k); vs.append(tok[:,0])
 keys=torch.cat(ks[-8:],0).unsqueeze(0); vals=torch.cat(vs[-8:],0).unsqueeze(0); err=float((m.association_dual_readout(s,keys)-vals).abs().amax().detach()); return err
out={"length":47,"beta_results":{str(b):run(b) for b in (.25,.5,.75,1.,1.25,1.5)}}; p=Path('results/published/gdn_delta_beta_sweep_20261030.json'); p.write_text(json.dumps(out,indent=2)+"\n",encoding='utf-8'); print(json.dumps(out,indent=2))
