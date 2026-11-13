"""Frozen competence floor with carried state, memory reads and world markers.

Research branches start from saved complete states. They never reset S at a
question boundary, never update slow weights and never replace the actual life.
An EOS marker is an input event, not a cache-clearing operation.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream,PretrainedStreamBackend
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


@contextmanager
def native_only(learner,enabled):
    saved=[]
    if enabled:
        for wrapper in learner.memories.values():
            prior=wrapper.__dict__.get('retrieve')
            def retrieve(query,replay,width=wrapper.encoder.out_features):
                return query.new_zeros(len(query),width).float().cpu()
            saved.append((wrapper,prior))
            wrapper.retrieve=retrieve
    try:
        yield
    finally:
        for wrapper,prior in saved:
            if prior is None:
                wrapper.__dict__.pop('retrieve',None)
            else:
                wrapper.retrieve=prior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','reference','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=2)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--length',type=int,default=512,choices=(512,2048,4096))
    p.add_argument('--freeze-pending-teaching',action='store_true',
                   help='External diagnostic only: suppress pending slow updates, keep inputs/state.')
    p.add_argument('--conditions',nargs='+',choices=(
        'current_reads','current_native_only','current_native_marker',
        'current_reads_marker','reference_native_only'))
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started=time.time()
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    reference=torch.load(args.reference,map_location='cpu',weights_only=False,mmap=True)
    refstate=reference.get('learner',reference)
    manifest,cases=read_cases(args.manifest,[args.length],args.start+args.count)
    cases=cases[args.start:args.start+args.count]
    pending_teaching=sum(payload['stream']['learn_mask'])
    if pending_teaching and not args.freeze_pending_teaching:
        raise ValueError('frozen diagnostic must not consume outstanding teaching updates')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=payload['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    report=dict(status='running',protocol=__doc__,retained_updates=0,results=[],
        sources={str(f):sha256(f) for f in (args.checkpoint,args.reference,args.manifest)},
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/irreversible_credit_stream.py')},
        dataset_seed=manifest['seed'],configured_length=args.length,
        frozen_pending_teaching=(pending_teaching if args.freeze_pending_teaching else 0))
    write_json(args.output,report)
    conditions=[('current_reads',state,payload.get('stream'),False,False),
                ('current_native_only',state,payload.get('stream'),True,False),
                ('current_native_marker',state,payload.get('stream'),True,True),
                ('current_reads_marker',state,payload.get('stream'),False,True),
                ('reference_native_only',refstate,None,True,False)]
    if args.conditions is not None:
        conditions=[condition for condition in conditions if condition[0] in args.conditions]
    backend=None
    try:
        for name,source,history,disable,marker in conditions:
            for _,row in cases:
                learner.load_state_dict(source)
                backend=PretrainedStreamBackend(learner)
                stream=IrreversibleCreditStream(backend)
                if history is not None:
                    stream.restore_latest(history)
                    if args.freeze_pending_teaching:
                        # Research branch only. The actual checkpoint and its
                        # observed inputs/targets/caches are never modified.
                        stream.learn_mask=[False]*len(stream.learn_mask)
                before=learner.cache._seen_tokens
                with native_only(learner,disable):
                    if marker:
                        stream.receive([tokenizer.eos_token_id],phase='world_marker',teach=False)
                    prefix,_=encode_case(tokenizer,row)
                    stream.receive(prefix,phase='world',teach=False)
                    generated=stream.generate(eos_token_id=tokenizer.eos_token_id,max_tokens=16)
                backend.close();backend=None
                if learner.updates!=source['updates']:
                    raise AssertionError('frozen branch updated parameters')
                if learner.cache._seen_tokens-before!=len(prefix)+len(generated)+int(marker):
                    raise AssertionError('state was reset or an event was omitted')
                prediction=tokenizer.decode(generated,skip_special_tokens=True)
                item=dict(condition=name,index=row['index'],references=row['outputs'],prediction=prediction,
                    **numeric_scores(prediction,row['outputs']),incoming_seen=before,
                    outgoing_seen=learner.cache._seen_tokens,optimizer_updates=0,question_resets=0)
                report['results'].append(item)
                report['seconds']=time.time()-started
                write_json(args.output,report)
                print(name,row['index'],item['first_number_exact'],repr(prediction),flush=True)
        report.update(status='complete',sources_unchanged=all(sha256(f)==h for f,h in report['sources'].items()),
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        report['seconds']=time.time()-started
        write_json(args.output,report)


if __name__=='__main__':
    main()
