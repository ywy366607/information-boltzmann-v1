"""Frozen conditional-content readers: same-time vs observed-successor values.

Use actual unseen RULER observations. The researcher's true-record/position
oracle isolates content usability; it is NEVER an organism retrieval rule.
No body, Adam, experience preference or world state is trained by these readers.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.audit_small_native_information import capture_reads
from scripts.ib_local.audit_small_native_failure import provenance
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import (
    consume_prompt,encode_case,fingerprint,probability_tensor,request_branch,sha256,
)


@contextmanager
def queries(bank):
    result={};original=[]
    for i,m in enumerate(bank.memories):
        previous=m.retrieve_context;original.append((m,previous))
        def read(query,block,i=i,previous=previous):
            result[i]=query.detach().cpu()
            return previous(query,block)
        m.retrieve_context=read
    try:yield result
    finally:
        for m,previous in original:m.retrieve_context=previous


@torch.no_grad()
def collect(engine,rows,split):
    records=[];model=engine.model;bank=model.local_replay
    for order,row in enumerate(rows):
        prefix,_=encode_case(row);digit=prefix.index(row['outputs'][0].encode())
        with request_branch(engine) as table,provenance(bank) as metadata:
            starts=[int(m.memory.clock) for m in bank.memories]
            _,state,pending=consume_prompt(model,table,prefix,128)
            table.run_credit(np.array([pending,32],dtype=np.uint8),max_window=1)
            logp=probability_tensor(model,peek(table,32)[None])
            with capture_reads(bank) as captures,queries(bank) as query:
                model.predict(torch.tensor([[32]],device=engine.device),state,logp)
                for i,m in enumerate(bank.memories):
                    slot,meta=next((s,p) for s,p in metadata[i].items()
                        if p['start']-starts[i]<=digit<p['end']-starts[i])
                    r=captures[i][slot];position=digit-(meta['start']-starts[i])
                    w=r['weights'];e=r['encoded'];count=len(e)
                    assert 0<position<count-1, 'declare boundary cases rather than fabricate a successor'
                    successor=(w[:-1,None]*e[1:]).sum(0)/w[:-1].sum()
                    width=len(row['outputs'][0]);end=min(position+width,count)
                    records.append(dict(case=row['index'],split=split,layer=i+1,
                        label=ord(row['outputs'][0][0])-48,
                        features=dict(query=m.encoder(query[i]).flatten().cpu(),
                            current_pool=r['pooled'],successor_pool=successor,
                            uniform_pool=e.mean(0),
                            digit_bag=e[position:end].mean(0),
                            first_digit=e[position]),
                        first_digit_mass=float(w[position]),
                        predecessor_mass=float(w[position-1]),
                        sequence_mass=float(w[position:end].sum()),
                        positions=count,first_digit_position=position))
        if order%32==31:print('COLLECT',split,order+1,flush=True)
    return records


def reader_report(records):
    reports=[]
    for layer in range(1,5):
        training=[r for r in records if r['split']=='train' and r['layer']==layer]
        testing=[r for r in records if r['split']=='test' and r['layer']==layer]
        labels=torch.tensor([r['label'] for r in training]);truth=torch.tensor([r['label'] for r in testing])
        counts=torch.bincount(labels,minlength=10)
        for name in ('query','current_pool','successor_pool','uniform_pool','digit_bag','first_digit',
                     'query_current_pool','query_successor_pool'):
            def feature(r):
                if name.startswith('query_'):
                    return torch.cat((r['features']['query'],r['features'][name[6:]]))
                return r['features'][name]
            x=torch.stack([feature(r) for r in training]).double()
            tx=torch.stack([feature(r) for r in testing]).double()
            # Training-only balancing/centering. Report the entire declared
            # penalty grid; no test-set model/penalty selection.
            weights=counts[labels].double().reciprocal()
            normalized=weights/weights.sum();center=(x*normalized[:,None]).sum(0)
            y=F.one_hot(labels,10).double();target=(y*normalized[:,None]).sum(0)
            centered=x-center;gram=centered.T@(weights[:,None]*centered)
            for penalty in (1e-6,1e-4,1e-2):
                ridge=penalty*gram.diag().mean().clamp_min(1e-12)
                fit=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0]),
                    centered.T@(weights[:,None]*(y-target)))
                scores=(tx-center)@fit+target
                prediction=scores.argmax(-1)
                reports.append(dict(layer=layer,features=name,penalty=penalty,
                    correct=int((prediction==truth).sum()),questions=len(testing),
                    accuracy=float((prediction==truth).float().mean()),
                    train_examples=len(training),majority_accuracy=float((truth==counts.argmax()).float().mean()),
                    test_predictions=prediction.tolist(),test_labels=truth.tolist()))
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','train','test','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--train-count',type=int,default=256)
    parser.add_argument('--test-count',type=int,default=56)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    from scripts.ib_local import gdn_fused
    gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint)
    train=[json.loads(l) for l in args.train.read_text().splitlines()][128:128+args.train_count]
    test=[json.loads(l) for l in args.test.read_text().splitlines()][8:8+args.test_count]
    assert len(train)==args.train_count and len(test)==args.test_count
    assert not {r['outputs'][0] for r in train}&{r['outputs'][0] for r in test}
    checksum=sha256(args.checkpoint);before=fingerprint(engine.model,engine.table)
    start=time.time()
    records=collect(engine,train,'train')+collect(engine,test,'test')
    readers=reader_report(records)
    path=args.output.with_suffix('.features.pt')
    if path.exists():raise FileExistsError(path)
    torch.save(records,path)
    result=dict(status='complete',body_optimizer_updates=0,checkpoint_sha256=checksum,
        oracle='True historical record and first position selected by researcher only',
        train_indices=[r['index'] for r in train],test_indices=[r['index'] for r in test],
        train_data_sha256=sha256(args.train),test_data_sha256=sha256(args.test),
        reader_reports=readers,features_path=str(path),features_sha256=sha256(path),
        seconds=time.time()-start,body_unchanged=fingerprint(engine.model,engine.table)==before,
        checkpoint_unchanged=sha256(args.checkpoint)==checksum,
        source_files={str(Path(__file__)):sha256(Path(__file__)),
            str(Path('scripts/ib_local/audit_small_native_information.py')):
                sha256(Path('scripts/ib_local/audit_small_native_information.py'))})
    for layer in range(1,5):
        subset=[r for r in records if r['layer']==layer]
        result.setdefault('attention',[]).append(dict(layer=layer,
            first_digit_mass=float(np.mean([r['first_digit_mass'] for r in subset])),
            predecessor_mass=float(np.mean([r['predecessor_mass'] for r in subset])),
            sequence_mass=float(np.mean([r['sequence_mass'] for r in subset]))))
    assert result['body_unchanged'] and result['checkpoint_unchanged']
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
