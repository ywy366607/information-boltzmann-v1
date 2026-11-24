"""Frozen real-checkpoint check of content-dependent question transmission."""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.run_causal_water_reply import build, sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    before=sha256_file(args.checkpoint)
    e,_,_,meta=build(args.checkpoint,args.device,table_storage=args.output/'table',allow_external_archive=True)
    if not hasattr(e.model,'question_to_hidden'):
        raise ValueError('checkpoint has no explicitly migrated question transition')
    q=e._question_turn_context.detach().clone()
    # Both byte conditions were actually self-heard in the carried individual.
    m=e.model.memory
    eligible=m.occupied & (m.word_count>0)
    bytes_seen=m.words[eligible,0].unique().tolist()
    if len(bytes_seen)<2:
        raise ValueError('two actual byte contexts required')
    shifts=[];rows=[]
    with torch.no_grad():
        for byte in bytes_seen[:2]:
            ids=torch.tensor([[byte]],device=e.device)
            lp=torch.full((1,1,e.model.base.vocab_size),-5.54517744448,device=e.device)
            e.model._question_context=None
            plain=e.model.predict_observed(ids,e.recurrent,lp)
            e.model._question_context=q
            changed=e.model.predict_observed(ids,e.recurrent,lp)
            shifts.append(changed[0]-plain[0])
            rows.append(dict(actual_byte=int(byte),state_change=float((changed[1]-plain[1]).norm()),
                hidden_change=float((changed[3]-plain[3]).norm()),
                logit_change=float(shifts[-1].norm())))
    e.model._question_context=None
    report=dict(cycle=meta['cycle'],source_sha256=before,optimizer_updates=0,world_calls=0,
        question_norm=float(q.norm()),contexts=rows,
        context_dependent_logit_shift=float((shifts[0]-shifts[1]).abs().max()),
        source_unchanged=sha256_file(args.checkpoint)==before,
        scope='actual trained transition numerical audit; not a dialogue capability evaluation')
    report['passed']=(all(r['state_change']>0 and r['hidden_change']>0 for r in rows)
        and report['context_dependent_logit_shift']>1e-5 and report['source_unchanged'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('trained question remains a content-independent output bias')


if __name__=='__main__':main()
