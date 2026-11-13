"""Use a record actually archived by the body, with a current known query.

The researcher identifies the fact's slot for an oracle delivery diagnostic.
No old text is supplied from the dataset to the reactivation program: its old
vectors and joint native boundary come exclusively from that stored record.
Only the already observed current question supplies query inputs.
"""
import argparse
from pathlib import Path
import json
import time
from dataclasses import replace

import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn_experience import execute_joint_query
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','case-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-grad',action='store_true')
    p.add_argument('--perturb-record',action='store_true',
                   help='External causal audit: change only the recorded digits, hold weights/query fixed.')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started=time.time()
    report=dict(status='loading',protocol=__doc__,retained_updates=0,new_lean_coverage=0,
        sources={str(f):sha256(f) for f in (args.checkpoint,args.manifest,args.case_report)},
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/pretrained_gdn_experience.py')})
    write_json(args.output,report)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    history=json.loads(args.case_report.read_text())['pending_case']
    _,cases=read_cases(args.manifest,[2048],payload['next_case']+1)
    row=cases[payload['next_case']][1]
    first=payload['pending'].records[0]
    lo=history['incoming_seen']+row['token_position_answer']
    archives={};slots={}
    for layer,view in first.views.items():
        selected=[s for s in view.memory.occupied.nonzero().flatten().tolist()
                  if int(view.memory.birth[s])-64<=lo and
                     lo+len(row['outputs'][0])<=int(view.memory.birth[s])+64]
        if len(selected)!=1:
            raise RuntimeError('actual fact window was not retained')
        slots[layer]=selected[0]
        archives[layer]=getattr(view.capsules[selected[0]],'joint',None)
        if archives[layer] is None:
            raise RuntimeError('no joint archive: cannot fabricate an old boundary')
    values=list(archives.values())
    if any(archive is not values[0] for archive in values):
        raise AssertionError('the two entries must reference one shared real window')
    archive=values[0]
    report.update(case=payload['next_case'],oracle_slots={str(k+1):v for k,v in slots.items()},
        archive_seen=archive.boundary['seen_tokens'],old_vectors_from_body=True,
        archive_native_layers=len(archive.boundary['states']),
        natural_last_reads={str(i+1):dict(slots=first.reads[i]['slots'][-1].tolist(),
                                       weights=first.reads[i]['weights'][-1].tolist()) for i in slots})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefix,_=encode_case(tokenizer,row)
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=payload['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    live=learner.cache
    device=next(model.parameters()).device
    query_ids=torch.tensor([prefix[-128:]],device=device)
    query=model.get_input_embeddings()(query_ids).detach()
    try:
        with torch.no_grad():
            output=execute_joint_query(model,archive,query)
            cache=output.past_key_values
            logits=output.logits[0,-1]
            emitted=[]
            for _ in range(16):
                token=int(logits.argmax());emitted.append(token)
                output=model(input_ids=torch.tensor([[token]],device=device),past_key_values=cache,use_cache=True)
                cache=output.past_key_values;logits=output.logits[0,-1]
                if token==tokenizer.eos_token_id:
                    break
            prediction=tokenizer.decode(emitted,skip_special_tokens=True)
            report.update(status='running',prediction=prediction,emitted=emitted,
                          references=row['outputs'],**numeric_scores(prediction,row['outputs']))
            write_json(args.output,report)
            print('JOINT',repr(prediction),flush=True)
            if args.perturb_record:
                reference=row['outputs'][0]
                replacement=''.join(str(9-int(digit)) for digit in reference)
                fact_position=row['token_position_answer']
                candidates=[start for start in range(max(0,fact_position-3),fact_position+4)
                    if [tokenizer.decode([token]).strip() for token in prefix[start:start+len(reference)]]==list(reference)]
                if len(candidates)!=1:
                    raise ValueError('one observed digit span required; do not alter an ambiguous record')
                position=candidates[0]
                frame_start=archive.boundary['seen_tokens']-history['incoming_seen']
                offset=position-frame_start
                if not 0<=offset<=128-len(reference):
                    raise ValueError('digit intervention must stay in the actual archived window')
                true_vectors=model.get_input_embeddings()(torch.tensor([prefix[frame_start:frame_start+128]],device=device)).cpu()
                if not torch.equal(true_vectors,archive.incoming_features):
                    raise AssertionError('clock-derived raw window does not match the recorded vectors')
                ids=[]
                for digit in replacement:
                    encoded=tokenizer.encode(' '+digit,add_special_tokens=False)
                    choices=[t for t in encoded if tokenizer.decode([t]).strip()==digit]
                    if len(choices)!=1:
                        raise ValueError('this causal audit requires one token per digit')
                    ids.append(choices[0])
                changed=archive.incoming_features.clone()
                changed[:,offset:offset+len(ids)]=model.get_input_embeddings()(torch.tensor([ids],device=device)).cpu()
                counterfactual=replace(archive,incoming_features=changed)
                output=execute_joint_query(model,counterfactual,query)
                cache=output.past_key_values;logits=output.logits[0,-1]
                generated=[]
                for _ in range(16):
                    token=int(logits.argmax());generated.append(token)
                    output=model(input_ids=torch.tensor([[token]],device=device),past_key_values=cache,use_cache=True)
                    cache=output.past_key_values;logits=output.logits[0,-1]
                    if token==tokenizer.eos_token_id:
                        break
                text=tokenizer.decode(generated,skip_special_tokens=True)
                report['record_digit_intervention']=dict(expected=replacement,prediction=text,
                    changed_input_positions=list(range(offset,offset+len(ids))),
                    weights_query_and_joint_boundary_held=True,
                    **numeric_scores(text,[replacement]))
                write_json(args.output,report)
                print('RECORD_INTERVENTION',repr(text),flush=True)
        if args.audit_grad:
            gradients={};forward=None;losses={}
            for held in (False,True):
                learner.optimizer.zero_grad(set_to_none=True)
                for _,parameter in learner.native:
                    parameter.grad=None
                output=execute_joint_query(model,archive,query,detach_history=held)
                if forward is None:
                    forward=output.logits.detach().cpu()
                else:
                    difference=float((forward-output.logits.detach().cpu()).abs().max())
                    if difference!=0:
                        raise AssertionError('cutting old derivatives changed the forward')
                    report['full_held_forward_difference']=difference
                # Only actual, already observed WORLD question tokens. No
                # answer label, self reward, or optimizer step enters this audit.
                loss=F.cross_entropy(output.logits[0,:-1].float(),query_ids[0,1:])
                loss.backward()
                key='held' if held else 'full'
                losses[key]=float(loss.detach())
                gradients[key]={name:(torch.zeros_like(p,device='cpu',dtype=torch.float32) if p.grad is None
                                     else p.grad.detach().float().cpu()) for name,p in learner.native}
                del output,loss
            report['current_observed_world_ce']=losses
            report['old_native_credit_by_layer']={str(i+1):
                sum(float((gradients['full'][name]-gradients['held'][name]).square().sum())
                    for name in gradients['full'] if name.startswith(f'layer{i}.'))**.5
                for i in learner.layers}
        if learner.cache is not live or learner.updates!=state['updates']:
            raise AssertionError('reactivation overwrote the actually-lived state')
        report.update(status='complete',actual_cache_preserved=True,
            sources_unchanged=all(sha256(f)==h for f,h in report['sources'].items()),
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        report['seconds']=time.time()-started
        write_json(args.output,report)


if __name__=='__main__':
    main()
