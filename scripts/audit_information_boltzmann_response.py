"""Frozen real-text checkpoint response audit, with explicit finite-scale limits."""
import argparse,json,math,time,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.state import PhaseState
from fine_grain.information_boltzmann.diagnostics import phase_difference,renormalize

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gamma-scales',type=float,nargs='+');p.add_argument('--events',type=int,default=256);p.add_argument('--fused-ou',action='store_true');a=p.parse_args()
 if a.events<=32:raise ValueError('events must exceed burn-in32')
 if a.gamma_scales and any(not math.isfinite(v) or v<=0 for v in a.gamma_scales):raise ValueError('gamma multipliers must be finite and positive')
 a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
 cfg=json.loads(a.config.read_text());model=InformationBoltzmann.from_config(cfg).cuda()
 tokens=np.load(a.data/'train.npy',mmap_mode='r')
 cases=[('age_000500.pt',e,'position') for e in [.003,.001,.0003]]+[('age_000500.pt',.001,'velocity'),('last.pt',.001,'position')]
 if a.fused_ou:
  from scripts.ib_fused_ou import install_fused_ou
  install_fused_ou(model)
 if a.gamma_scales:cases=[('age_000500.pt',.001,'position') for _ in a.gamma_scales]
 original_damping=model.force.damping
 results=[]
 for case_index,(file,eps,direction) in enumerate(cases):
  scale=a.gamma_scales[case_index] if a.gamma_scales else 1.0
  model.force.damping=lambda x, scale=scale: original_damping(x)*scale
  saved=torch.load(a.run/file,map_location='cpu',weights_only=True);model.load_state_dict(saved['model'])
  state=PhaseState(saved['state']['x'].cuda(),saved['state']['v'].cuda(),saved['state']['time'])
  shadow=PhaseState(state.x.clone(),state.v.clone(),state.time)
  if direction=='position':shadow.x[0,0]+=eps/math.sqrt(model.force.kappa)
  else:shadow.v[0,0]+=eps
  gen=torch.Generator(device='cuda');gen.set_state(saved['generator'].cpu())
  sequence=[saved['current_token'],*map(int,tokens[saved['events']:saved['events']+a.events-1])]
  rates=[];mismatches=0;relative_errors=[];energies=[];losses=[];started=time.time()
  for event_index,token in enumerate(sequence):
   realized=float(torch.linalg.vector_norm(phase_difference(shadow,state,model.force.kappa)))
   before=gen.get_state();base,_,stats=model._advance_steps(state,token,gen);after=gen.get_state();gen.set_state(before)
   pert,_,other=model._advance_steps(shadow,token,gen);gen.set_state(after)
   mismatches+=stats.get('pairs')!=other.get('pairs')
   shadow,distance=renormalize(base,pert,eps,model.force.kappa)
   energies.append(float(.5*(base.v.square()+model.force.kappa*base.x.square()).sum(-1).mean()))
   logits=model.decode(base);target=torch.tensor([int(tokens[saved['events']+event_index])],device='cuda');losses.append(float(torch.nn.functional.cross_entropy(logits[None],target)))
   rates.append(math.log(distance/realized)/model.event_interval);relative_errors.append(abs(realized/eps-1));state=base
  arr=np.array(rates)
  result={'checkpoint':file,'update':saved['updates'],'epsilon':eps,'direction':direction,'events':a.events,'gamma_multiplier':scale,'fused_ou':a.fused_ou,'burn_in':32,'mean_rate':float(arr[32:].mean()),'prefix_rates':{str(n):float(arr[32:n].mean()) for n in [64,128,256] if n<=a.events},'block_rates':[float(x.mean()) for x in np.array_split(arr,4)],'max_relative_initial_norm_error':max(relative_errors),'events_with_different_accepted_pairs':mismatches,'rates':rates,'mean_energy':float(np.mean(energies)),'max_energy':max(energies),'continuation_nll':float(np.mean(losses)),'seconds':time.time()-started,'interpretation':'frozen_weights_common_input_noise_finite_phase_response_not_full_learning_system_exponent'}
  results.append(result);(a.output/'response.json').write_text(json.dumps(results,indent=2));print({k:v for k,v in result.items() if k!='rates'},flush=True)
if __name__=='__main__':main()
