"""One actual decision, whole-body frozen check of native-width delivery.

Correct-record intervention is outside the individual. No reference is supplied
to the read program, no optimizer runs, and the actual lifetime is untouched.
"""
from contextlib import ExitStack
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.ib_local.delayed_prediction_credit import fork_view,anchor_participation,release_view
from scripts.ib_local.native_contextual_recall import native_successor_residual
from scripts.ib_local.pretrained_gdn import new_recurrent_cache,enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def install(shadow,block,actual,mode,slot):
    normal=shadow.retrieve
    anchor_participation(shadow,actual)
    held=shadow.retrieve
    first=True
    projection=shadow.inlet.projection
    if mode!='baseline':
        inlet=copy.copy(shadow.inlet)
        inlet._modules=dict(inlet._modules)
        inlet.projection=torch.nn.Identity()
        shadow.inlet=inlet
    frozen={}
    def retrieve(query,replay):
        nonlocal first
        legacy=(held if first else normal)(query,replay)
        first=False
        if mode=='baseline':
            return legacy
        old=projection(legacy.to(query.device)).to(query.dtype)
        native=native_successor_residual(shadow,block,query,replay,
                query_boundary=shadow._query_boundary,
                forced_slot=slot if mode in ('native_oracle_record','native_oracle_position') else None,
                forced_value_position=getattr(shadow,'_oracle_value_position',None),frozen_cache=frozen)
        # The actual earlier reads in this held prefix remain fixed. Only the
        # current last read is intervened upon, not the entire prior context.
        result=old.clone()
        result[-1]=native[-1]
        return result
    shadow.retrieve=retrieve


@torch.no_grad()
def run_branch(learner,record,slots,gold,tokenizer,mode,generate,*,fact_offsets=None,prefix_reference=None):
    device=next(learner.model.parameters()).device
    shadows={i:fork_view(w,record.views[i]) for i,w in learner.memories.items()}
    for layer,shadow in shadows.items():
        install(shadow,learner.model.model.layers[layer],record.reads[layer],mode,slots[layer])
    cache=new_recurrent_cache(record.boundary['seen_tokens'])
    cache.states=copy_tree(record.boundary['states'],device)
    tokens=record.inputs.to(device)
    offset=record.clock_offset
    emitted,losses,alignments=[],[],[]
    prefix_copy=None;prefix_error=None
    try:
        for index in range(16 if generate else len(gold)):
            with ExitStack() as stack:
                for layer,shadow in shadows.items():
                    shadow._query_boundary=cache.states[layer]
                    if mode=='native_oracle_position':
                        shadow._oracle_value_position=(fact_offsets[layer]+index-1 if index<len(gold) else None)
                    stack.enter_context(shadow.attach(learner.model,layer=layer,capture=False,clock_offset=offset))
                output=learner.model(tokens,past_key_values=cache,use_cache=True)
            logits=output.logits[0,-1].float()
            if index==0:
                first=float((logits.cpu()-record.logits).abs().max())
                if mode=='baseline' and not generate:
                    prefix_copy=output.logits[0,:-1].detach().float().cpu().clone()
                if prefix_reference is not None:
                    prefix_error=float((output.logits[0,:-1].float().cpu()-prefix_reference).abs().max())
            if not generate and mode!='baseline':
                per_layer={}
                for layer,shadow in shadows.items():
                    data=shadow._native_last_alignment
                    weight=data['head_weights']
                    mask=torch.tensor([s==slots[layer] and fact_offsets[layer]<=p<fact_offsets[layer]+7
                                       for s,p in zip(data['slots'],data['value_positions'])])
                    top=weight.argmax(-1).tolist()
                    per_layer[str(layer+1)]=dict(fact_activity_mass=weight[:,mask].sum(-1).tolist(),
                        top_value_position=[data['value_positions'][i] for i in top])
                alignments.append(per_layer)
            cache=output.past_key_values
            offset+=tokens.numel()
            if generate:
                token=int(logits.argmax());emitted.append(token)
                if token==tokenizer.eos_token_id:
                    break
            else:
                token=gold[index]
                losses.append(float(F.cross_entropy(logits[None],torch.tensor([token],device=device))))
            tokens=torch.tensor([[token]],device=device)
            del output,logits
        return (dict(text=tokenizer.decode(emitted,skip_special_tokens=True),emitted=emitted,
                     first_logit_difference=first,prefix_logit_difference=prefix_error) if generate else
                dict(answer_nll=sum(losses)/len(losses),token_losses=losses,first_logit_difference=first,
                     _prefix=prefix_copy,prefix_logit_difference=prefix_error,alignment=alignments))
    finally:
        for shadow in shadows.values():
            release_view(shadow)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','case-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started=time.time()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    case=json.loads(args.case_report.read_text())['pending_case']
    _,cases=read_cases(args.manifest,[2048],payload['next_case']+1)
    row=cases[payload['next_case']][1]
    first=payload['pending'].records[0]
    lo=case['incoming_seen']+row['token_position_answer'];hi=lo+len(row['outputs'][0])-1
    slots={};fact_offsets={}
    for layer,view in first.views.items():
        candidates=[slot for slot in view.memory.occupied.nonzero().flatten().tolist()
                    if int(view.memory.birth[slot])-64<=lo and hi<=int(view.memory.birth[slot])+63]
        if len(candidates)!=1:
            raise RuntimeError('one actually retained fact window required')
        slots[layer]=candidates[0]
        fact_offsets[layer]=lo-(int(view.memory.birth[candidates[0]])-64)
    report=dict(status='loading',source=str(args.checkpoint),source_sha256=sha256(args.checkpoint),
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/native_contextual_recall.py')},
        case=case['case'],oracle_slots={str(k+1):v for k,v in slots.items()},
        results={},retained_updates=0,new_parameters=0,new_lean_coverage=0)
    write_json(args.output,report)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    _,gold=encode_case(tokenizer,row)
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=payload['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    live=learner.cache
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    try:
        prefix_reference=None
        for mode in ('baseline','native_oracle_record','native_oracle_position','native_natural'):
            score=run_branch(learner,first,slots,gold,tokenizer,mode,False,
                             fact_offsets=fact_offsets,prefix_reference=prefix_reference)
            prefix=score.pop('_prefix')
            if mode=='baseline':
                prefix_reference=prefix
            generated=run_branch(learner,first,slots,gold,tokenizer,mode,True,
                                 fact_offsets=fact_offsets,prefix_reference=prefix_reference)
            result={**score,**generated,**numeric_scores(generated['text'],row['outputs'])}
            report['results'][mode]=result
            report['status']='running'
            write_json(args.output,report)
            print(mode,result['answer_nll'],repr(result['text']),flush=True)
        baseline=report['results']['baseline']
        if baseline['emitted']!=[r.emitted for r in payload['pending'].records] or baseline['first_logit_difference']>.125:
            raise AssertionError('baseline does not reconstruct the actual answer')
        if learner.cache is not live or learner.updates!=state['updates']:
            raise AssertionError('read-only branch changed actual history')
        if not torch.equal(torch.get_rng_state(),rng) or not torch.equal(torch.cuda.get_rng_state(),cuda_rng):
            raise AssertionError('read-only branch changed RNG')
        report.update(status='complete',source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            actual_cache_preserved=True,source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        report['seconds']=time.time()-started
        write_json(args.output,report)


if __name__=='__main__':
    main()
