"""Particle-count runtime probe from a checkpoint empirical measure; no learning claim."""
import json,time,torch,numpy as np,sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.state import PhaseState
from scripts.ib_shared_projections import install_shared_projections
from scripts.ib_fused_ou import install_fused_ou

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.6)
 cfg=json.loads(a.config.read_text());m=InformationBoltzmann.from_config(cfg).cuda();s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m.load_state_dict(s['model']);install_shared_projections(m);install_fused_ou(m)
 rows=[]
 with torch.no_grad():
  for n in [32,128,512,1024]:
   # Repeated equal-weight support preserves the checkpoint empirical measure,
   # but changes finite-N collision sampling; these are not trained larger models.
   state=PhaseState(s['state']['x'].cuda().repeat(n//32,1),s['state']['v'].cuda().repeat(n//32,1),s['state']['time'])
   g=torch.Generator(device='cuda');g.set_state(s['generator'].cpu())
   m._advance_steps(state,s['current_token'],g)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   times=[]
   for repeat in range(3):
    g.set_state(s['generator'].cpu());torch.cuda.synchronize();start=time.perf_counter()
    _,_,stats=m._advance_steps(state,s['current_token'],g)
    torch.cuda.synchronize();times.append(time.perf_counter()-start)
   row={'particles':n,'frozen_event_seconds':times,'median_seconds':float(np.median(times)),'candidates':stats['candidates'],'accepted':stats['accepted'],'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'limitations':'one fixed real checkpoint event repeated3 after shape warmup; repeated empirical support; no backward/optimizer/accounting; not language capability or trained-size comparison'}
   rows.append(row);print(row,flush=True)
   a.output.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
