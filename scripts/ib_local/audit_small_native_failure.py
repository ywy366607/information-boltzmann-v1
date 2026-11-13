"""Read-only provenance, query and old-input credit audit of native replay.

Keep source checkpoints unchanged. Equal-length number substitutions diagnose
conditional delivery; cuts happen only at the answer after identical observed
prompts. This is not another training or capability experiment.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import (
    encode_case, consume_prompt, probability_tensor, request_branch, sha256, fingerprint,
)


@contextmanager
def provenance(bank):
    metadata=[{} for _ in bank.memories]
    originals=[]
    for i,m in enumerate(bank.memories):
        old=m.commit
        originals.append(old)
        def commit(i=i,m=m,old=old):
            start=int(m.memory.clock)
            result=old()
            if result is not None:
                metadata[i][result]=dict(start=start,end=start+m.capsules[result].inputs.shape[1],
                    version=int(m.memory.read_credit_version[result]))
            return result
        m.commit=commit
    try:yield metadata
    finally:
        for m,old in zip(bank.memories,originals):m.commit=old


def probe(engine,row,alternative,*,swap=False,cut='none',differentiate=False):
    value=row['outputs'][0]
    if len(value)!=len(alternative) or value[0]==alternative[0]:raise ValueError('paired values required')
    view=dict(row,input=row['input'].replace(value,alternative,1)) if swap else row
    prefix,_=encode_case(view)
    needle=prefix.index((alternative if swap else value).encode())
    bank=engine.model.local_replay
    result=dict(case=row['index'],value=value,alternative=alternative,swapped=swap,cut=cut,
                needle_start=needle,needle_end=needle+len(value),prompt_bytes=len(prefix))
    captures={};read_originals=[];context_originals=[];position_reads={}
    with request_branch(engine) as table, provenance(bank) as metadata:
        starts=[int(m.memory.clock) for m in bank.memories]
        with torch.no_grad():
            _,state,pending=consume_prompt(engine.model,table,prefix,128)
        for i,m in enumerate(bank.memories):
            old=m.read;read_originals.append(old)
            def read(query,i=i,m=m,old=old):
                r=old(query)
                if cut=='force_digit_record':
                    # Researcher-only oracle route: does the existing reader
                    # benefit if an overlapping record is actually delivered?
                    # Neither targets nor this selector enter training/body.
                    choices=[]
                    for slot,p in metadata[i].items():
                        lo=p['start']-starts[i];hi=p['end']-starts[i]
                        overlap=max(0,min(hi,needle+len(value))-max(lo,needle))
                        if overlap:choices.append((overlap,slot))
                    if choices:
                        from scripts.ib_local.latent_navigation_v2 import TransitionRead
                        _,slot=max(choices)
                        slots=torch.full_like(r.slots,slot)
                        weights=torch.zeros_like(r.weights);weights[:,0]=1
                        r=TransitionRead(r.value,r.long_value,slots,weights)
                captures[i]=(query.detach().cpu(),r)
                return r
            m.read=read
            if hasattr(m,'retrieve_context'):
                previous=m.retrieve_context;context_originals.append((m,previous))
                def context_read(query,block,i=i,m=m,previous=previous):
                    out=previous(query,block)
                    captures[i]=(query.detach().cpu(),m.last_read)
                    position_reads[i]=m.position_read
                    return out
                m.retrieve_context=context_read
        inputs=[]
        for i,m in enumerate(bank.memories):
            for slot,capsule in enumerate(m.capsules):
                if capsule is not None and differentiate:
                    capsule.inputs.requires_grad_()
                    inputs.append((i,slot,capsule.inputs))
        enabled=bank.enabled
        suppressed=[m.suppressed_record for m in bank.memories]
        try:
            if cut=='all_local':bank.enabled=False
            if cut=='needle_record':
                for i,m in enumerate(bank.memories):
                    for slot,p in metadata[i].items():
                        lo=p['start']-starts[i];hi=p['end']-starts[i]
                        if lo<=needle<hi:
                            m.suppressed_record=(slot,p['version'])
            with torch.no_grad():
                table.run_credit(np.array([pending,32],dtype=np.uint8),max_window=1)
                logp=probability_tensor(engine.model,peek(table,32)[None])
            with torch.set_grad_enabled(differentiate):
                logits,_,_=engine.model.predict(torch.tensor([[32]],device=engine.device),state,logp)
                current=logits[0,0].float()
                loss=-current.log_softmax(-1)[ord(value[0])]
                result.update(first_digit_nll=float(loss.detach()),
                    digit_margin=float((current[ord(value[0])]-current[ord(alternative[0])]).detach()),
                    prediction=chr(int(current.argmax())),layers=[])
                gradients=(torch.autograd.grad(loss,[x for i,s,x in inputs],allow_unused=True)
                           if differentiate else [None]*len(inputs))
            for i,m in enumerate(bank.memories):
                records=[]
                for slot,p in metadata[i].items():
                    lo=p['start']-starts[i];hi=p['end']-starts[i]
                    c=m.capsules[slot]
                    if c is None or c.version!=p['version']:continue
                    records.append(dict(slot=slot,start=lo,end=hi,
                        overlap_digits=max(0,min(hi,needle+len(value))-max(lo,needle)),
                        source_position=hi-1,
                        source_char=chr(prefix[hi-1]) if hi<=len(prefix) else None))
                item=dict(layer=i+1,records=records)
                if i in captures:
                    q,r=captures[i]
                    item.update(query=q.flatten().tolist(),slots=r.slots.detach().flatten().tolist(),
                        weights=r.weights.detach().flatten().tolist())
                    item['current_request_mass']=sum(w for s,w in zip(item['slots'],item['weights'])
                        if s in metadata[i])
                    item['direct_digit_record_mass']=sum(w for s,w in zip(item['slots'],item['weights'])
                        if any(p['slot']==s and p['overlap_digits']>0 for p in records))
                    if i in position_reads and position_reads[i] is not None:
                        pr=position_reads[i];mask=[]
                        offset=0
                        for slot,length in zip(pr['slots'].tolist(),pr['lengths']):
                            p=metadata[i].get(slot)
                            start=None if p is None else p['start']-starts[i]
                            positions=(pr['value_positions'][offset:offset+length].tolist()
                                       if 'value_positions' in pr else range(length))
                            mask.extend([start is not None and needle<=start+j<needle+len(value)
                                         for j in positions])
                            offset+=length
                        mask=torch.tensor(mask,dtype=torch.bool)
                        item['direct_digit_position_mass']=float(pr['weights'][0,mask].sum().detach())
                if differentiate:
                    norm=digits=0.;per=[]
                    for (layer,slot,x),g in zip(inputs,gradients):
                        if layer!=i or g is None:continue
                        norm+=float(g.square().sum())
                        p=metadata[i].get(slot)
                        d=0.
                        if p is not None:
                            lo=p['start']-starts[i]
                            left=max(0,needle-lo);right=min(x.shape[1],needle+len(value)-lo)
                            if right>left:d=float(g[:,left:right].square().sum());digits+=d
                        per.append(dict(slot=slot,norm=float(g.norm()),digit_norm=d**.5))
                    item.update(old_input_grad_norm=norm**.5,direct_digit_input_grad_norm=digits**.5,
                                input_gradients=per)
                result['layers'].append(item)
        finally:
            bank.enabled=enabled
            for m,old,s in zip(bank.memories,read_originals,suppressed):
                m.read=old;m.suppressed_record=s
            for m,previous in context_originals:m.retrieve_context=previous
    return result


def window_probe(engine,row):
    """Actual128-byte training alignment, including earlier queries in window.

    A final single-step query alone cannot exclude a route through earlier
    reads in this SAME differentiable window. Audit that route explicitly.
    """
    prefix,gold=encode_case(row)
    raw=np.frombuffer(bytes([engine.pending])+prefix+gold+b'\n',dtype=np.uint8)
    needle=1+prefix.index(row['outputs'][0].encode())
    first=len(prefix)+2  # first digit target, after the leading answer space
    start=((first-1)//128)*128
    bank=engine.model.local_replay
    answer=[]
    with request_branch(engine) as table, provenance(bank) as metadata:
        starts=[int(m.memory.clock) for m in bank.memories]
        state=engine.recurrent.detach().clone()
        for pos in range(0,start,128):
            part=raw[pos:pos+129]
            _,prob,_=table.run_credit(part,max_window=128)
            ids,targets,logp=engine._inputs(part,prob)
            with bank.observation(engine.optimizer.param_groups[-1],128),torch.no_grad():
                _,state,_,features,read=engine.model.predict_observed(ids,state,logp)
                engine.model.commit_observed(features,read)
                bank.settle_and_commit()
                state=state.detach()
        inputs=[]
        for i,m in enumerate(bank.memories):
            for slot,c in enumerate(m.capsules):
                if c is not None:
                    c.inputs.requires_grad_();inputs.append((i,slot,c.inputs))
        part=raw[start:min(start+129,len(raw))]
        _,prob,_=table.run_credit(part,max_window=128)
        ids,targets,logp=engine._inputs(part,prob)
        with bank.observation(engine.optimizer.param_groups[-1],len(part)-1):
            logits,_,_,_,_=engine.model.predict_observed(ids,state,logp)
            query=first-1-start
            loss=-logits[0,query].float().log_softmax(-1)[raw[first]]
            gs=torch.autograd.grad(loss,[x for i,s,x in inputs],allow_unused=True)
            for i,m in enumerate(bank.memories):
                digit_norm=total_norm=0.;details=[]
                for (layer,slot,x),g in zip(inputs,gs):
                    if layer!=i or g is None:continue
                    total_norm+=float(g.square().sum())
                    p=metadata[i].get(slot)
                    if p is None:continue
                    lo=p['start']-starts[i];hi=p['end']-starts[i]
                    left=max(0,needle-lo);right=min(x.shape[1],needle+7-lo)
                    overlap=max(0,right-left)
                    d=float(g[:,left:right].square().sum()) if overlap else 0.
                    digit_norm+=d
                    read=m.last_read
                    mass=float((read.weights[:query+1]*(read.slots[:query+1]==slot)).sum().detach())
                    details.append(dict(slot=slot,start=lo,end=hi,digit_overlap=overlap,
                                        cumulative_read_mass=mass,digit_gradient_norm=d**.5))
                answer.append(dict(layer=i+1,old_input_gradient_norm=total_norm**.5,
                    direct_digit_input_gradient_norm=digit_norm**.5,records=details))
    return dict(case=row['index'],first_digit_nll=float(loss.detach()),layers=answer,
                needle_start=needle,window_start=start,query_position=query)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--oracle-only',action='store_true')
    p.add_argument('--window-only',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint)
    # Replay derivatives must be live for the diagnostic, independent of arm.
    engine.model.local_replay.set_held(False)
    before=fingerprint(engine.model,engine.table);checkpoint_sha=sha256(args.checkpoint)
    rows=[json.loads(l) for l in args.data.read_text().splitlines()][:8]
    report=dict(status='running',optimizer_steps=0,records=[],checkpoint=str(args.checkpoint),
                checkpoint_sha256=checkpoint_sha)
    start=time.time()
    for row in rows:
        if args.window_only:
            report['records'].append(window_probe(engine,row))
            print('WINDOW',row['index'],flush=True)
            continue
        alt=next(r['outputs'][0] for r in rows if r['outputs'][0][0]!=row['outputs'][0][0])
        cuts=('none','force_digit_record') if args.oracle_only else ('none','all_local','needle_record')
        for cut in cuts:
            for swap in (False,True):
                report['records'].append(probe(engine,row,alt,swap=swap,cut=cut,
                                               differentiate=cut=='none' and not swap and not args.oracle_only))
        print('CASE',row['index'],flush=True)
    report.update(status='complete',seconds=time.time()-start,
                  body_unchanged=before==fingerprint(engine.model,engine.table),
                  checkpoint_unchanged=checkpoint_sha==sha256(args.checkpoint))
    assert report['body_unchanged'] and report['checkpoint_unchanged']
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
