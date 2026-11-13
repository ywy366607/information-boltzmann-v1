"""Frozen whole-body route interventions for the failed short RULER adaptation.

Cross neural parameters, lexical table and causal-memory state independently.
No optimizer steps. All mixtures are diagnostic counterfactuals, not promoted
individuals. Logit decomposition includes the lexical INPUT port in the decoder
component; only the final geometric-mixture term is isolated algebraically.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local import gdn_fused
import scripts.ib_local.small_gdn_ruler as adapter
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import build


def margin_components(decoder, log_table, weight, original, alternative):
    """Exact log-odds decomposition, independent of softmax normalizer."""
    neural=float(decoder[original]-decoder[alternative])
    lexical=float(weight*(log_table[original]-log_table[alternative]))
    return dict(decoder_margin=neural,output_table_margin=lexical,total_margin=neural+lexical)


@contextmanager
def intervention(model, mode):
    handles=[]
    original=adapter.probability_tensor
    try:
        if mode=='uniform_table':
            adapter.probability_tensor=lambda m,p:original(m,np.full_like(p,1/256))
        elif mode=='no_output_table':
            handles.append(model.body.mix.register_forward_hook(
                lambda m,a,o:torch.full_like(o,-80.)))
        elif mode=='no_causal_injection':
            handles.append(model.to_hidden.register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
        elif mode!='normal':
            raise ValueError(mode)
        yield
    finally:
        adapter.probability_tensor=original
        for h in handles:h.remove()


@torch.no_grad()
def profile(engine,row,replacement,swapped=False,mode='normal'):
    value=row['outputs'][0]
    if row['input'].count(value)!=1 or len(value)!=len(replacement):
        raise ValueError('one equal-length value intervention required')
    view=dict(row,input=row['input'].replace(value,replacement,1)) if swapped else row
    prefix,gold=adapter.encode_case(view)
    # Nine-byte teacher profile, including the actually supervised newline.
    labels=gold+b'\n'
    captured={}
    def capture_decoder(module,args,out):captured['decoder']=out[0,-1].float()
    def capture_weight(module,args,out):captured['weight']=F.softplus(out[0,-1,0].float())
    losses=[];table_losses=[];decoder_losses=[];weights=[];digit=None;max_error=0.
    with adapter.request_branch(engine) as bank, intervention(engine.model,mode):
        handles=[engine.model.base.decoder.register_forward_hook(capture_decoder),
                 engine.model.body.mix.register_forward_hook(capture_weight)]
        try:
            current,state,pending=adapter.consume_prompt(engine.model,bank,prefix)
            for position,target in enumerate(labels):
                # Same live lexical state used by this pending-byte prediction.
                prob=peek(bank,pending)
                if mode=='uniform_table':prob=np.full_like(prob,1/256)
                log_table=torch.from_numpy(np.log(np.maximum(prob,adapter.FLOOR))).to(current.device)
                decoder,weight=captured['decoder'],captured['weight']
                reconstruction=decoder+weight*log_table
                max_error=max(max_error,float((reconstruction-current.float()).abs().max()))
                losses.append(float(-current.float().log_softmax(-1)[target]))
                table_losses.append(float(-log_table[target]))
                decoder_losses.append(float(-decoder.log_softmax(-1)[target]))
                weights.append(float(weight))
                if position==1:
                    a,b=ord(value[0]),ord(replacement[0])
                    digit=dict(**margin_components(decoder,log_table,weight,a,b),
                        predicted=chr(int(current.argmax())),
                        value_first_probability=float(current.float().softmax(-1)[a]),
                        alternative_first_probability=float(current.float().softmax(-1)[b]),
                        table_first_probability=float(prob[a]),
                        table_alternative_probability=float(prob[b]),weight=float(weight))
                if position+1<len(labels):
                    current,state,pending=adapter.advance(engine.model,bank,state,pending,target)
        finally:
            for h in handles:h.remove()
    if max_error>1e-4:
        raise AssertionError(f'logit decomposition mismatch{max_error}')
    return dict(case=row['index'],value=value,replacement=replacement,swapped=swapped,mode=mode,
        first_digit=digit,per_byte_nll=losses,table_per_byte_nll=table_losses,
        decoder_per_byte_nll=decoder_losses,table_weights=weights,max_decomposition_error=max_error)


def copy_parameters(model,values):
    with torch.no_grad():
        for name,p in model.named_parameters():p.copy_(values[name])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','adapted','data','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);gdn_fused.enable()
    source,_,_,source_meta=build(args.source)
    adapted,_,_,adapted_meta=build(args.adapted)
    engines=[source,adapted]
    parameters=[{n:v.detach().cpu().clone() for n,v in e.model.named_parameters()} for e in engines]
    memories=[e.model.memory for e in engines]
    tables=[e.table for e in engines]
    original_fingerprints=[adapter.fingerprint(e.model,e.table) for e in engines]
    rows=[json.loads(x) for x in args.data.read_text().splitlines()][:args.limit]
    alternatives=[]
    for i,row in enumerate(rows):
        alternatives.append(next(rows[(i+j)%len(rows)]['outputs'][0] for j in range(1,len(rows))
            if rows[(i+j)%len(rows)]['outputs'][0][0]!=row['outputs'][0][0]))
    report=dict(status='running',protocol=__doc__,optimizer_steps=0,source=source_meta,
        adapted=adapted_meta,data_sha256=adapter.sha256(args.data),records=[],
        source_files={str(Path(__file__)):adapter.sha256(__file__)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def write():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    write();start=time.time()
    try:
        for theta,lexical,history in itertools.product(range(2),repeat=3):
            source.model.memory=memories[history]
            source.table=tables[lexical]
            copy_parameters(source.model,parameters[theta])
            for row,other in zip(rows,alternatives):
                for swapped in (False,True):
                    r=profile(source,row,other,swapped)
                    r.update(theta=theta,lexical=lexical,history=history)
                    report['records'].append(r)
            write();print('CELL',theta,lexical,history,flush=True)
        # Whole-route cuts at each real endpoint; no artificial state transplant.
        for endpoint in range(2):
            source.model.memory=memories[endpoint];source.table=tables[endpoint]
            copy_parameters(source.model,parameters[endpoint])
            for mode in ('no_output_table','uniform_table','no_causal_injection'):
                for row,other in zip(rows,alternatives):
                    for swapped in (False,True):
                        r=profile(source,row,other,swapped,mode)
                        r.update(theta=endpoint,lexical=endpoint,history=endpoint)
                        report['records'].append(r)
                write();print('CUT',endpoint,mode,flush=True)
        source.model.memory=memories[0];source.table=tables[0]
        copy_parameters(source.model,parameters[0])
        restored=[adapter.fingerprint(e.model,e.table)==f for e,f in zip(engines,original_fingerprints)]
        assert all(restored)
        assert adapter.sha256(args.source)==source_meta['sha256']
        assert adapter.sha256(args.adapted)==adapted_meta['sha256']
        report.update(status='complete',states_restored=restored,checkpoints_unchanged=True,
                      seconds=time.time()-start)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:write()


if __name__=='__main__':main()
