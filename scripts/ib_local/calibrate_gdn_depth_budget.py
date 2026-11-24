"""Real-OWT CUDA memory calibration, zero optimizer steps and no body mutation.

Includes FP32 Adam moments, two full delayed-policy gradient carriers, and
serial proposal gradients while an observation graph is live. It is a native
backbone/credit-buffer budget, not a trained or complete new language body.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers',type=int,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--heads',type=int,required=True)
    p.add_argument('--ffn',type=int,required=True)
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--proposal-bytes',type=int,default=128)
    p.add_argument('--content-budget',type=float,default=32.)
    p.add_argument('--data',type=Path,default=Path('data/ib_owt_10000'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(configuration=vars(args)|{'data':str(args.data),'output':str(args.output)},
        status='starting',optimizer_updates=0,world_calls=0,live_body_changed=False,
        dtype='float32',scope=__doc__,stages=[])
    def write():
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    write()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real budget calibration')
    # Bound the test itself; do not consume the whole user's graphics card.
    torch.cuda.set_per_process_memory_fraction(.46)
    torch.manual_seed(111432)
    gdn_fused.enable()
    start=time.perf_counter()
    def stage(name):
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        report['stages'].append(dict(stage=name,seconds=time.perf_counter()-start,
            allocated_mib=torch.cuda.memory_allocated()/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            reserved_mib=torch.cuda.memory_reserved()/2**20,
            device_total_used_mib=(total-free)/2**20))
        try:
            used=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True,timeout=5)
            report['stages'][-1]['nvidia_smi_used_mib']=float(used.stdout.splitlines()[0].strip())
        except (OSError,ValueError,subprocess.SubprocessError):
            pass
        report['status']=name;write();print(json.dumps(report['stages'][-1]),flush=True)
    try:
        with torch.device('cuda'):
            model=GatedDeltaNetLM(256,args.width,args.layers,args.heads,
                'decay_then_delta',args.ffn,4,args.content_budget,decay_init='log_timescale')
        report['native_config']=model.config_dict()
        parameters=list(model.parameters())
        report['parameters']=sum(p.numel() for p in parameters)
        report['parameter_grad_adam_mib']=report['parameters']*16/2**20
        report['S_and_P_mib']=2*args.layers*args.heads*(args.width//args.heads)**2*4/2**20
        report['saved_window_S_mib']=args.layers*args.heads*(args.tokens+1)*(args.width//args.heads)**2*4/2**20
        optimizer=torch.optim.AdamW(parameters,lr=3e-4)
        for parameter in parameters:
            optimizer.state[parameter].update(step=torch.tensor(0.),
                exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
        old_actor=[torch.zeros_like(parameter) for parameter in parameters]
        actor=[torch.zeros_like(parameter) for parameter in parameters]
        versions=[p._version for p in parameters]
        stage('parameters_Adam_and_two_policy_carriers')
        offset=json.loads((args.data/'tokenizer.json').read_text())['offset']
        raw=np.load(args.data/'train.npy',mmap_mode='r')[:8192]
        real=(raw[raw>=offset]-offset).astype(np.int64)[:args.tokens+1]
        if len(real)!=args.tokens+1 or real.max()>255:raise ValueError('real byte stream required')
        ids=torch.tensor(real[:-1],device='cuda')[None]
        targets=torch.tensor(real[1:],device='cuda')[None]
        # Largest possible 128-byte speech + 240-byte first mother half.
        prefix=min(args.tokens,args.proposal_bytes+240)
        loss,_,_=model(ids[:,:prefix],targets[:,:prefix])
        stage('observation_graph_live')
        for i in range(4):
            sampled,_,_=model(ids[:,i:i+args.proposal_bytes],targets[:,i:i+args.proposal_bytes])
            gradients=torch.autograd.grad(sampled,parameters)
            with torch.no_grad():
                for destination,gradient in zip(actor,gradients):destination.add_(gradient)
            del sampled,gradients
        stage('four_serial_proposal_gradients')
        loss.backward();del loss
        stage('observation_backward_with_policy_carriers')
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=model(ids,targets)
        loss.backward();del loss
        stage('full_window_backward_with_policy_carriers')
        report.update(status='complete',weights_unchanged=all(p._version==v for p,v in zip(parameters,versions)),
            whole_body_reserve_mib=192,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            conservative_budget_mib=torch.cuda.max_memory_reserved()/2**20+192,
            context_scope='device_total_used includes desktop/driver; reserve is for remaining whole-loop modules')
        write()
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='bounded_memory_rejection',error=str(error),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write();print(json.dumps(dict(status=report['status'],peak_mib=report['peak_allocated_mib'])),flush=True)
    print(json.dumps({k:report.get(k) for k in ('status','parameters','peak_allocated_mib','conservative_budget_mib','weights_unchanged')}),flush=True)


if __name__=='__main__':main()
