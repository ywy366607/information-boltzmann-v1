"""Measure fixed 256-token real-OWT BPE updates; optional full CUDA Graph replay."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ib_bpe_window import BPEWindow,clock_inputs

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=256);p.add_argument('--particles',type=int,default=512);p.add_argument('--hidden',type=int,default=128);p.add_argument('--updates',type=int,default=3);p.add_argument('--graph',action='store_true');p.add_argument('--no-recompute',action='store_true');p.add_argument('--recompute-stride',type=int,default=1);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 def record(value):(a.output/'status.json').write_text(json.dumps(value,indent=2,allow_nan=False))
 record({'status':'running','tokens_per_update':a.tokens,'particles':a.particles,'graph':a.graph})
 try:
  torch.set_num_threads(1);torch.manual_seed(11);torch.cuda.set_per_process_memory_fraction(.68)
  manifest=json.loads((a.data/'manifest.json').read_text());assert manifest['tokenizer']=='gpt2'
  train=np.load(a.data/'train.npy',mmap_mode='r')
  model=BPEWindow(hidden=a.hidden,particles=a.particles,recompute=not a.no_recompute,recompute_stride=a.recompute_stride).cuda()
  opt=torch.optim.AdamW(model.parameters(),lr=3e-4,foreach=True,capturable=a.graph)
  generator=torch.Generator(device='cuda').manual_seed(11)
  work_stream=torch.cuda.Stream();work_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(work_stream):state=model.core.initialize(torch.tensor([50256],device='cuda'),generator)
  torch.cuda.current_stream().wait_stream(work_stream)
  x,v=state.x,state.v
  ids=torch.empty(a.tokens,dtype=torch.long,device='cuda');targets=torch.empty_like(ids)
  clocks=clock_inputs(0,a.tokens,model.steps,'cuda');noise=torch.empty((a.tokens*model.steps*4,a.particles,4),device='cuda')
  def inputs(offset):
   prefix=50256 if offset==0 else int(train[offset-1])
   ids.copy_(torch.tensor(np.concatenate(([prefix],train[offset:offset+a.tokens-1])),device='cuda',dtype=torch.long))
   targets.copy_(torch.tensor(np.array(train[offset:offset+a.tokens]),device='cuda',dtype=torch.long))
   clocks.copy_(clock_inputs(offset,a.tokens,model.steps,'cuda'));noise.normal_(generator=generator)
  def update():
   opt.zero_grad(set_to_none=True)
   loss,new_x,new_v=model(x,v,ids,targets,clocks,noise)
   loss.backward()
   # Avoid scalar transfers; validate finite loss/gradients after the update.
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
   opt.step()
   with torch.no_grad():x.copy_(new_x);v.copy_(new_v)
   return loss,norm
  inputs(0);started=time.perf_counter()
  work_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(work_stream):loss,norm=update()
  torch.cuda.current_stream().wait_stream(work_stream);torch.cuda.synchronize();warm=time.perf_counter()-started
  print('warmup',warm,float(loss.detach()),flush=True)
  x,v=x.detach(),v.detach()
  capture_seconds=None;offset=a.tokens
  if a.graph:
   # Capture records kernels but does not execute an optimizer update.
   # Reuse static buffer shapes; the first replay consumes the next window.
   graph=torch.cuda.CUDAGraph();started=time.perf_counter()
   with torch.cuda.graph(graph,stream=work_stream):loss,norm=update()
   torch.cuda.synchronize();capture_seconds=time.perf_counter()-started
   print('capture',capture_seconds,flush=True)
  rows=[]
  for _ in range(a.updates):
   torch.cuda.synchronize();started=time.perf_counter();inputs(offset)
   if a.graph:graph.replay()
   else:loss,norm=update()
   torch.cuda.synchronize();elapsed=time.perf_counter()-started
   row={'offset':offset,'seconds':elapsed,'nll':float(loss.detach()),'grad_norm':float(norm.detach())}
   assert np.isfinite(row['nll']) and np.isfinite(row['grad_norm'])
   rows.append(row);offset+=a.tokens;print(row,flush=True)
  torch.save({'model':model.state_dict(),'x':x.detach(),'v':v.detach(),'optimizer':opt.state_dict(),'events':offset,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'last_loss':float(loss.detach())},a.output/'last.pt')
  record({'status':'complete','tokens_per_update':a.tokens,'particles':a.particles,'hidden':a.hidden,'vocab':50257,'parameters':sum(p.numel() for p in model.parameters()),'graph':a.graph,'warmup_seconds':warm,'capture_seconds':capture_seconds,'updates':rows,'median_seconds':float(np.median([r['seconds'] for r in rows])),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'scope':'runtime calibration with actual updates on real BPE tokens; one learned initialization then persistent state; no convergence/criticality claim'})
 except Exception as e:
  record({'status':'failed','error':repr(e),'tokens_per_update':a.tokens,'particles':a.particles,'graph':a.graph});raise
if __name__=='__main__':main()
