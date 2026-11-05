"""Common-random-number perturbation of a complete online v4 language loop.

Three serial in-memory continuations: baseline, exact replay, perturbed body.
Each uses the real learning engine and actual world, table and action paths.
Never overwrites a checkpoint; this is finite-difference response, not a
Lyapunov spectrum of a smooth map through stochastic discrete events.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.language_action import digest
from scripts.ib_local.run_language_navigation_kanli_v4 import build


def run(cycles, epsilon, direction_seed):
    torch.manual_seed(7369)
    torch.cuda.manual_seed_all(7369)
    engine, table, _, cfg = build(mode='active', seed=11, equal=True, capture=False)
    state = engine.recurrent
    if epsilon:
        gen=torch.Generator(device=state.device).manual_seed(direction_seed)
        v=torch.randn(state.shape,device=state.device,dtype=state.dtype,generator=gen)
        state.add_(epsilon*v/v.norm())
    rows=[]
    for turn in range(cycles):
        result=engine.step(torch.randn(64,32,device='cuda'))
        # Copy immediately: the live optimiser and fixed-size memories continue
        # to evolve independently in each replay.
        params=torch.cat([p.detach().reshape(-1).float().cpu() for p in engine.model.parameters()])
        body=engine.recurrent.detach().reshape(-1).float().cpu()
        m=engine.model.memory
        rows.append(dict(turn=turn+1,choice=result['choice'],utterance=result['utterance_hex'],
            source=result['world']['source'],world_start=result['world']['start'],
            response_nll=result['response_nll'],progress=result['metal_g'],
            choice_probability=result['choice_probability'],episodes=int(m.episodes),
            memory_source=m.source.detach().reshape(-1).float().cpu().clone(),
            recurrent=body.clone(),parameters=params.clone(),
            byte_table_count_sum=float(table.count.sum()),
            table_rows_sha256=hashlib.sha256(table.rows.tobytes()).hexdigest()))
    return rows,cfg


def comparison(a,b):
    out=[]
    for x,y in zip(a,b):
        out.append(dict(turn=x['turn'],same_choice=x['choice']==y['choice'],
            same_utterance=x['utterance']==y['utterance'],same_world_start=x['world_start']==y['world_start'],
            same_table=x['table_rows_sha256']==y['table_rows_sha256'],
            recurrent_l2=float((x['recurrent']-y['recurrent']).norm()),
            parameter_l2=float((x['parameters']-y['parameters']).norm()),
            memory_source_l2=float((x['memory_source']-y['memory_source']).norm()),
            response_nll_difference=y['response_nll']-x['response_nll'],
            progress_difference=y['progress']-x['progress'],
            choice_probability_l1=float(np.abs(np.asarray(x['choice_probability'])-y['choice_probability']).sum())))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--seed',type=int,default=101)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('preserve existing report')
    if a.cycles<=0 or a.epsilon<=0: raise ValueError('invalid perturbation')
    torch.set_num_threads(1); gdn_fused.enable()
    baseline,cfg=run(a.cycles,0.,a.seed)
    replay,_=run(a.cycles,0.,a.seed)
    reproducibility=comparison(baseline,replay)
    verified=all(x['same_choice'] and x['same_utterance'] and x['same_world_start'] and
                 x['same_table'] and max(x['recurrent_l2'],x['parameter_l2'],x['memory_source_l2'])==0
                 for x in reproducibility)
    if not verified:
        result=dict(scope='complete closed-loop online replay',replay_verified=False,
                    reproducibility=reproducibility,perturbed_comparison=None,
                    warning='Perturbation interpretation withheld because identical replay differed.')
    else:
        perturbed,_=run(a.cycles,a.epsilon,a.seed)
        result=dict(scope='complete v4 equal-window loop: GDN, latent memory, byte table, generated action, world and online learning',
            experiment='common-random-seed, one initial recurrent-state perturbation; no checkpoint written',
            config=cfg,script_sha256=digest(Path(__file__)),cycles=a.cycles,epsilon_l2=a.epsilon,seed=a.seed,
            replay_verified=True,reproducibility=reproducibility,perturbed_comparison=comparison(baseline,perturbed),
            limitation='Discrete sampled actions, matching and table changes break smooth tangent assumptions; finite response is not a Lyapunov exponent. Initial in-memory continuation starts from the older checkpoint, not a fully restored v4 terminal individual.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
