"""Production-eager CUDA component audit on a complete live checkpoint fork."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_online_learning import FlyOnlineLearner
from information_boltzmann.core import triton_online_credit as fused
from collections import defaultdict
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--edge-block',type=int,default=256)
parser.add_argument('--edge-warps',type=int,default=4)
parser.add_argument('--mask-dormant',action='store_true')
args=parser.parse_args()
run=ROOT/args.run
saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
cfg=saved['config']
m=FlyReservoirLM(ROOT/cfg['graph'],vocab_size=50257,d_model=cfg['d_model'],injection='topographic',read_surface='output',synapse_model='coba',use_alif=True,use_stp=True).cuda()
with torch.no_grad():
 weights=dict(saved['model'])
 for name in ('edge_weight_e','edge_weight_i'): getattr(m,name).copy_(weights.pop(name))
 m.load_state_dict(weights,strict=False)
learner=FlyOnlineLearner(m,lr=cfg['lr'],lr_synapse=cfg['lr_synapse'],lr_sensory=cfg['lr_sensory'])
learner.load_state_dict(saved['learner'])
cursor=saved['train_cursor']; previous=learner.previous_token
print('Profiling complete-state fork at',cursor,flush=True)
del saved,weights

timings=defaultdict(list)
measuring=False
edge_update=fused.update_local_coba_edges
def tuned_edge_update(*positional,**kwargs):
 kwargs.update(block=args.edge_block,mask_dormant=args.mask_dormant,num_warps=args.edge_warps)
 return edge_update(*positional,**kwargs)
fused.update_local_coba_edges=tuned_edge_update

def wrap(obj,attr,label):
 original=getattr(obj,attr)
 def timed(*args,**kwargs):
  if not measuring: return original(*args,**kwargs)
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  begin.record();result=original(*args,**kwargs);end.record()
  timings[label].append((begin,end))
  return result
 setattr(obj,attr,timed)

for obj,attr,label in [(m,'step','component.physics'),(m.topographic_writer,'forward','component.write'),
 (m.output_read,'forward','component.read'),(m.decoder,'forward','component.decoder'),
 (learner,'_physical_gradients','component.physical_credit'),(learner,'_gate_gradients','component.gate_credit'),
 (fused,'update_projection_traces','component.sensory_trace'),(fused,'update_local_coba_edges','component.edge_credit'),
 (learner.eprop,'compute_learning_signal_whole_brain','component.feedback'),(learner.eprop,'update_eligibility_coba','component.recurrent_trace'),
 (learner.optimizer,'step','component.optimizer')]: wrap(obj,attr,label)
train=np.load(ROOT/cfg['data']/'train.npy',mmap_mode='r')
tokens=torch.tensor(np.asarray(train[cursor+1:cursor+17],dtype=np.int64),device='cuda')
for target in tokens[:4]:
 learner.step(torch.tensor([previous],device='cuda'),target[None]);previous=int(target)
torch.cuda.synchronize()
start=time.perf_counter()
measuring=True
for target in tokens[4:12]:
 learner.step(torch.tensor([previous],device='cuda'),target[None]);previous=int(target)
torch.cuda.synchronize()
wall=(time.perf_counter()-start)/8
rows=[{'component':label,'cuda_event_ms_per_token':sum(a.elapsed_time(b) for a,b in pairs)/8} for label,pairs in timings.items()]
result={'wall_ms_per_token':wall*1000,'components':rows,
        'source_checkpoint':str(run/'last.pt'),'train_cursor':cursor,
        'credit_revision':'destination-coba-compensated-v1',
        'edge_block':args.edge_block,'edge_warps':args.edge_warps,'mask_dormant':args.mask_dormant,
        'measurement':'actual eager learner, nested inclusive CUDA events; no extrapolation to graphs',
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
print(json.dumps(result,indent=2),flush=True)
args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
