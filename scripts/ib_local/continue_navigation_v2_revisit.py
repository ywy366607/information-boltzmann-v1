"""Continue one completed navigation individual through real OWT A -> B -> A.

This is a controlled post-run language intervention; the individual, optimizer,
both memories and recurrent state remain continuous. It is not dialogue or a
separate evaluation-only learning algorithm. Each target byte is scored before
its real observation is committed; the second A uses exactly the first A bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest
from scripts.ib_local.latent_navigation_v2 import LatentNavigationV2,V2GraphTrainer
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local.train_gdn_fast_table import FLOOR,LEXICAL,TableGDN


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates-per-phase',type=int,default=200)
    parser.add_argument('--validation-bytes',type=int,default=32768)
    args=parser.parse_args()
    if args.updates_per_phase<=0 or args.validation_bytes<=0: raise ValueError('invalid budget')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); gdn_fused.enable()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False); cfg=saved['config']
    if saved['cycle']!=cfg['cycles']: raise ValueError('requires completed navigation checkpoint')
    for name,sha in cfg['sources'].items():
        if digest(name)!=sha: raise ValueError(f'checkpoint source changed: {name}')
    root=Path('data/ib_owt_10000'); offset=json.loads((root/'tokenizer.json').read_text())['offset']
    train=np.load(root/'train.npy',mmap_mode='r'); lo,hi=cfg['source_raw_bounds']
    raw=np.asarray(train[lo:hi]); source=(raw[raw>=offset]-offset).astype(np.uint8)
    if hashlib.sha256(source.tobytes()).hexdigest()!=cfg['source_byte_sha256']: raise ValueError('source mismatch')
    length=args.updates_per_phase*512
    a_start,b_start=12_000_000,14_000_000
    if b_start+length>len(source): raise EOFError('A/B source exceeds recorded world')
    a=source[a_start:a_start+length]; b=source[b_start:b_start+length]
    validraw=np.load(root/'validation.npy',mmap_mode='r'); buf=np.asarray(validraw[:2*(args.validation_bytes+1)])
    valid=(buf[buf>=offset]-offset)[:args.validation_bytes+1].astype(np.uint8)
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').cuda()
    body=TableGDN(base,True).cuda()
    model=LatentNavigationV2(body,threshold=cfg['threshold'],mode=cfg['memory']).cuda()
    model.load_state_dict(saved['model'])
    if cfg['disable_latent']: model.read_gain.requires_grad_(False)
    optimizer=torch.optim.AdamW(body.parameters(),lr=3e-4,foreach=True)
    extra=[p for name,p in model.named_parameters() if not name.startswith('body.')]
    optimizer.add_param_group(dict(params=extra,lr=3e-4)); optimizer.load_state_dict(saved['optimizer'])
    table=RetrievedCreditTable.from_state_dict(saved['table']); pending=int(saved['pending'])
    torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    config=dict(scope='controlled real-text A-B-A continuing actual individual',
        checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),
        code_sha256=digest(Path(__file__)),updates_per_phase=args.updates_per_phase,
        a_start=a_start,b_start=b_start,phase_bytes=length,a_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        b_sha256=hashlib.sha256(b.tobytes()).hexdigest(),validation_bytes=args.validation_bytes,
        a_positions_seen_before=int(saved['world']['seen'][a_start:a_start+length].sum()),
        b_positions_seen_before=int(saved['world']['seen'][b_start:b_start+length].sum()))
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    validation_before=validate(model,table,valid)
    runner=V2GraphTrainer(model,optimizer,tokens=512,projections=32,capture=True)
    runner.state.copy_(saved['recurrent'].cuda())
    named=list(body.named_parameters()); groups=(
        [p for n,p in named if n.removeprefix('base.') in LEXICAL],
        [p for n,p in named if n.removeprefix('base.') not in LEXICAL],extra)
    started=time.perf_counter(); rows=[]
    with (args.output/'metrics.jsonl').open('w') as log:
        for name,phase in (('A_first',a),('B',b),('A_revisit',a)):
            for step in range(args.updates_per_phase):
                incoming=phase[step*512:(step+1)*512]
                stream=np.concatenate((np.array([pending],np.uint8),incoming))
                _,probs,_=table.run_credit(stream,max_window=512)
                ids=torch.tensor(stream[:-1].astype(np.int64),device='cuda')[None]
                targets=torch.tensor(stream[1:].astype(np.int64),device='cuda')[None]
                logp=torch.tensor(np.log(np.maximum(probs,FLOOR)),device='cuda')[None]
                directions=torch.randn(64,32,device='cuda')
                loss,_,_,diag=runner.step(ids,targets,logp,directions)
                norms=torch.stack([torch.nn.utils.clip_grad_norm_(g,1.,foreach=True) for g in groups])
                if not torch.isfinite(norms).all(): raise FloatingPointError('nonfinite replay update')
                runner.finish(); optimizer.step(); pending=int(incoming[-1])
                row=dict(phase=name,step=step+1,nll=float(diag['ce']),short=float(diag['latent']),
                    long=float(diag['latent_long']),gain=float(model.read_gain.detach()),
                    clock=int(model.memory.clock),merge_count=int(model.memory.merged_total),
                    source_sha256=hashlib.sha256(incoming.tobytes()).hexdigest())
                rows.append(row); log.write(json.dumps(row)+'\n')
            log.flush()
    after=validate(model,table,valid)
    result=dict(config=config,counts={name:sum(x['phase']==name for x in rows) for name in ('A_first','B','A_revisit')},
        mean_nll={name:sum(x['nll'] for x in rows if x['phase']==name)/args.updates_per_phase
                  for name in ('A_first','B','A_revisit')},validation_before=validation_before,
        validation_after=after,elapsed_seconds=time.perf_counter()-started,
        claim='continuous real-text A-B-A; phase exposure and training differ; not isolated retention effect')
    result['revisit_saving']=result['mean_nll']['A_first']-result['mean_nll']['A_revisit']
    # Record externally observed positions without changing the world's cursor.
    world_state=saved['world']
    world_state['seen'][a_start:a_start+length]=True
    world_state['seen'][b_start:b_start+length]=True
    snapshot=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),recurrent=runner.state.cpu(),
        table=table.state_dict(),pending=pending,baseline=saved['baseline'],world_cursor=saved['world_cursor'],
        world=world_state,
        world_rng=saved['world_rng'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        config=cfg,cycle=saved['cycle'],revisit=config)
    tmp=args.output/'last.pending.pt'; torch.save(snapshot,tmp,pickle_protocol=5); tmp.replace(args.output/'last.pt')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__': main()
