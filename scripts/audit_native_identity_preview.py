"""Re-evaluate native posterior draws on the saved actual eight-fact design."""
import argparse
import gzip
import json
from pathlib import Path

import torch

from scripts.analyze_native_path_composition import load
from scripts.ib_local.causal_relation_program import RelationBank, posterior, mechanism_hypotheses
from scripts.ib_local.identity_preview import observation_noise
from scripts.run_native_relation_mode_use import sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('observations', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    designs = {}
    with gzip.open(args.observations/'operations.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            if row['kind']!='relation_signal':
                continue
            d = row['data']
            if d['operation']=='_bound_evidence_design':
                designs[d['call_id']] = d
            if (d['operation']=='mechanism' and d['state']['identity']==213
                    and d['kind']=='mother' and d['prediction']['bound_evidence']==8):
                selected = d
                break
        else:
            raise ValueError('actual eight-fact relation 213 record required')
    facts = selected['state']['constraints']
    # The mechanism's accepted design is the most recent nested actual call.
    design = designs[max(designs)]
    facts = design['facts']
    ids = tuple(f"mother:{int(e['observed'])}:{e['fact_id']}" for e in facts)
    assert len(ids)==8 and len(set(ids))==8
    with (args.observations/'tensors.bin').open('rb') as f:
        data = {k:torch.from_numpy(load(f,selected['prediction'][k])) for k in ('x','y','query','noise')}
    permutation = [0,7,6,1,2,3,4,5]
    bank = RelationBank(256,128,1,8,2,'cpu',torch.float64)
    reports = []
    for dtype in (torch.float64,torch.float32):
        x,y,q = [data[k].to(dtype).requires_grad_() for k in ('x','y','query')]
        a = posterior(x,y,q,selected['prediction']['ridge'])
        b = posterior(x[permutation],y[permutation],q,selected['prediction']['ridge'])
        a['noise']=b['noise']=data['noise'].to(dtype)
        a['fact_identities']=ids
        b['fact_identities']=tuple(ids[i] for i in permutation)
        w,e=bank.directions.to(dtype),bank.observation_directions.to(dtype)
        pa=mechanism_hypotheses(a,q,w,e)
        pb=mechanism_hypotheses(b,q,w,e)
        ga=torch.autograd.grad(pa.square().mean(),(x,y,q),retain_graph=True)
        gb=torch.autograd.grad(pb.square().mean(),(x,y,q))
        eps=observation_noise(ids,e)
        eps_p=observation_noise(b['fact_identities'],e)
        subset=observation_noise((ids[0],'mother:1:episode:999999'),e)
        reports.append(dict(dtype=str(dtype),mean_max_abs=float((a['mean']-b['mean']).abs().max().detach()),
            hypothesis_max_abs=float((pa-pb).abs().max().detach()),
            gradient_max_abs=[float((u-v).abs().max()) for u,v in zip(ga,gb)],
            noise_permutation_max_abs=float((eps[:,permutation]-eps_p).abs().max()),
            retained_fact_noise_max_abs=float((eps[:,0]-subset[:,0]).abs().max())))
    result=dict(status='completed',actual_call_id=selected['call_id'],fact_ids=ids,
        actual_shape={k:list(v.shape) for k,v in data.items()},results=reports,
        record_sha256=sha(args.observations/'operations.jsonl.gz'),
        implementation_sha256=sha(Path('scripts/ib_local/identity_preview.py')))
    write(args.output,result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
