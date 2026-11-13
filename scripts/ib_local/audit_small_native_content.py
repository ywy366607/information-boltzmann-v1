"""Hold the current body/table fixed; change only old stored native content.

Compare an actual record with the same record observed after an equal-length
number intervention. This distinguishes use of THIS stored value from a generic
benefit of reading any digit-containing passage. No optimizer update is made.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local.audit_small_native_failure import provenance
from scripts.ib_local.language_action import peek
from scripts.ib_local.pretrained_gdn_experience import LocalExperience
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import (
    consume_prompt,encode_case,fingerprint,probability_tensor,request_branch,sha256,
)


def collect_alternative(engine,row,alternative):
    original=row['outputs'][0]
    view=dict(row,input=row['input'].replace(original,alternative,1))
    prefix,_=encode_case(view);needle=prefix.index(alternative.encode())
    bank=engine.model.local_replay
    with request_branch(engine) as table,provenance(bank) as metadata:
        starts=[int(m.memory.clock) for m in bank.memories]
        consume_prompt(engine.model,table,prefix,128)
        selected=[]
        for i,m in enumerate(bank.memories):
            records={}
            for slot,p in metadata[i].items():
                lo=p['start']-starts[i];hi=p['end']-starts[i]
                if lo<needle+len(original) and hi>needle:
                    records[(lo,hi)]=copy.deepcopy(m.capsules[slot])
            selected.append(records)
    return selected


@torch.no_grad()
def measure(engine,row,alternative):
    candidates=collect_alternative(engine,row,alternative)
    value=row['outputs'][0];prefix,_=encode_case(row)
    needle=prefix.index(value.encode());bank=engine.model.local_replay
    with request_branch(engine) as table,provenance(bank) as metadata:
        starts=[int(m.memory.clock) for m in bank.memories]
        _,state,pending=consume_prompt(engine.model,table,prefix,128)
        pairs=[]
        for i,m in enumerate(bank.memories):
            for slot,p in metadata[i].items():
                lo=p['start']-starts[i];hi=p['end']-starts[i]
                if (lo,hi) in candidates[i]:
                    pairs.append((i,slot,candidates[i][lo,hi]))
        assert len({i for i,slot,c in pairs})==4
        table.run_credit(np.array([pending,32],dtype=np.uint8),max_window=1)
        logp=probability_tensor(engine.model,peek(table,32)[None])
        ids=torch.tensor([[32]],device=engine.device)
        factual=engine.model.predict(ids,state,logp)[0][0,0].float()
        for i,slot,c in pairs:
            m=bank.memories[i]
            m.capsules[slot]=LocalExperience(c.inputs,c.boundary,
                                           int(m.memory.read_credit_version[slot]))
        changed=engine.model.predict(ids,state,logp)[0][0,0].float()
        a,b=ord(value[0]),ord(alternative[0])
        return dict(case=row['index'],value=value,alternative=alternative,
            replaced_records=[dict(layer=i+1,slot=slot,positions=c.inputs.shape[1]) for i,slot,c in pairs],
            factual_first_digit_nll=float(-factual.log_softmax(-1)[a]),
            changed_memory_first_digit_nll=float(-changed.log_softmax(-1)[a]),
            factual_content_nll_benefit=float(factual.log_softmax(-1)[a]-changed.log_softmax(-1)[a]),
            stored_value_contrast=float((factual[a]-factual[b])-(changed[a]-changed[b])),
            max_logit_change=float((factual-changed).abs().max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.device=='cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint,device=args.device)
    assert engine.model.local_replay.schema in (2,3,4,5)
    before=fingerprint(engine.model,engine.table);sha=sha256(args.checkpoint)
    rows=[json.loads(l) for l in args.data.read_text().splitlines()][:8]
    start=time.time();records=[]
    for row in rows:
        alternative=next(x['outputs'][0] for x in rows if x['outputs'][0][0]!=row['outputs'][0][0])
        records.append(measure(engine,row,alternative))
        print('CONTENT',row['index'],records[-1]['factual_content_nll_benefit'],flush=True)
    result=dict(status='complete',device=args.device,optimizer_steps=0,records=records,
        seconds=time.time()-start,body_unchanged=fingerprint(engine.model,engine.table)==before,
        checkpoint_unchanged=sha256(args.checkpoint)==sha,
        favorable_nll_cases=sum(x['factual_content_nll_benefit']>0 for x in records),
        favorable_contrast_cases=sum(x['stored_value_contrast']>0 for x in records),
        mean_content_nll_benefit=sum(x['factual_content_nll_benefit'] for x in records)/len(records),
        mean_stored_value_contrast=sum(x['stored_value_contrast'] for x in records)/len(records))
    assert result['body_unchanged'] and result['checkpoint_unchanged']
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
