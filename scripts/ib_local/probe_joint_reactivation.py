"""Check a joint historical boundary before designing another local decoder.

Research branches only, zero updates. Reconstruct the actual pre-fact boundary
from a saved earlier arrival state, then use CURRENT parameters. The fact window
and question suffix are real previously observed inputs, not reference labels.
All24 native states carry; neither branch replaces the actual committed life.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import torch

from scripts.ib_local.pretrained_gdn import (enable_streaming_convolution,clone_cache,
                                            cache_nbytes,new_recurrent_cache)
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream,PretrainedStreamBackend
from scripts.ib_local.probe_continuous_readiness import native_only
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json,assert_same


def pin_inputs(value):
    def before(module,args,kwargs):
        if args:
            return (value,*args[1:]),kwargs
        return args,dict(kwargs,hidden_states=value)
    return before


@torch.no_grad()
def consume(learner,tokens,cache):
    device=next(learner.model.parameters()).device
    for start in range(0,len(tokens),128):
        x=torch.tensor([tokens[start:start+128]],device=device)
        logits,cache=learner.predict(x,cache)
    return logits[0,-1],cache


@torch.no_grad()
def answer(learner,tokenizer,tokens,cache):
    logits,cache=consume(learner,tokens,cache)
    emitted=[]
    for _ in range(16):
        token=int(logits.argmax());emitted.append(token)
        logits,cache=consume(learner,[token],cache)
        if token==tokenizer.eos_token_id:
            break
    return tokenizer.decode(emitted,skip_special_tokens=True),emitted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','incoming','checkpoint','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-reencode-history',action='store_true',
                   help='Diagnostic: relearn real old world inputs without pinning their old memory injection.')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started=time.time()
    report=dict(status='loading',protocol=__doc__,results={},retained_updates=0,
        executed_updates=0,new_lean_coverage=0,
        native_reencode_history=args.native_reencode_history,
        sources={str(f):sha256(f) for f in (args.incoming,args.checkpoint,args.manifest)},
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/pretrained_gdn_experience.py')})
    write_json(args.output,report)
    incoming=torch.load(args.incoming,map_location='cpu',weights_only=False,mmap=True)
    actual=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    number=actual['next_case']
    if incoming['next_case']!=number:
        raise ValueError('the actual question must immediately follow the saved arrival')
    _,cases=read_cases(args.manifest,[2048],number+1)
    row=cases[number][1]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefix,_=encode_case(tokenizer,row)
    lo=incoming['learner']['cache']['seen_tokens']+row['token_position_answer']
    record=actual['pending'].records[0]
    slots={}
    for layer,view in record.views.items():
        choices=[slot for slot in view.memory.occupied.nonzero().flatten().tolist()
                 if int(view.memory.birth[slot])-64<=lo<int(view.memory.birth[slot])+64]
        if len(choices)!=1:
            raise RuntimeError('one actually recorded fact window required')
        slots[layer]=choices[0]
    view=next(iter(record.views.values()))
    slot=slots[next(iter(record.views))]
    window_start=int(view.memory.birth[slot])-64-incoming['learner']['cache']['seen_tokens']
    if window_start!=128-len(incoming['stream']['inputs']):
        raise ValueError('this diagnostic reconstructs the first complete arriving window only')
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=incoming['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    backend=PretrainedStreamBackend(learner)
    stream=IrreversibleCreditStream(backend)
    stream.restore_latest(incoming['stream'])
    try:
        # Stops BEFORE the next target arrives, hence the pending teaching
        # cannot update anything. The actually-lived cache was preserved across
        # that later real update, so this is its exact pre-window native boundary.
        stream.receive(prefix[:window_start],phase='world',teach=False)
        backend.close()
        if learner.updates!=state['updates'] or len(stream.inputs)!=128:
            raise AssertionError('boundary reconstruction must not learn or flush')
        before=clone_cache(learner.cache)
        differences={}
        for layer,slot in slots.items():
            expected=record.views[layer].capsules[slot].boundary
            differences[str(layer+1)]={
                'recurrent_max_abs':float((before.states[layer]['recurrent_state'].cpu()-expected['recurrent_state']).abs().max()),
                'conv_max_abs':max(float((a.cpu()-b).abs().max()) for a,b in
                                   zip(before.states[layer]['conv_state'],expected['conv_state']))}
        report['boundary_differences']=differences
        write_json(args.output,report)
        for layer,slot in slots.items():
            expected=record.views[layer].capsules[slot].boundary
            assert_same(copy_tree(before.states[layer]),expected,'actual_pre_fact_boundary')
        report.update(boundary_matches_both_actual_capsules=True,
            old_window_start=window_start,old_window_tokens=128,current_query_tokens=128,
            fact_local_position=lo-(incoming['learner']['cache']['seen_tokens']+window_start),
            joint_boundary_bytes=cache_nbytes(before),joint_layers=len(before.states))
        learner.load_state_dict(actual['learner'])
        actual_live=learner.cache
        device=next(model.parameters()).device
        # Replay the real historical window. These two archived inputs already
        # contain its actual earlier memory injections, so pin them rather than
        # introducing future table contents or recursively inventing old reads.
        with native_only(learner,True),ExitStack() as stack,torch.no_grad():
            for layer,slot in ([] if args.native_reencode_history else slots.items()):
                held=record.views[layer].capsules[slot].complete_inputs.to(device)
                hook=model.model.layers[layer].register_forward_pre_hook(pin_inputs(held),with_kwargs=True)
                stack.callback(hook.remove)
            _,after=consume(learner,prefix[window_start:window_start+128],before)
        end=window_start+128
        frozen_after=copy_tree(after.states)
        with native_only(learner,True):
            for mode,tokens in [('all_interference',prefix[end:]),
                                ('joint_two_slices',prefix[-128:])]:
                text,emitted=answer(learner,tokenizer,tokens,after)
                report['results'][mode]=dict(prediction=text,emitted=emitted,
                    references=row['outputs'],**numeric_scores(text,row['outputs']),
                    query_inputs=len(tokens),question_state_resets=0)
                assert_same(frozen_after,copy_tree(after.states),'held_joint_boundary')
                report['status']='running'
                write_json(args.output,report)
                print(mode,repr(text),flush=True)
        if learner.cache is not actual_live or learner.updates!=actual['learner']['updates']:
            raise AssertionError('research reactivation changed the actual latest state')
        report.update(status='complete',actual_live_preserved=True,
            sources_unchanged=all(sha256(f)==h for f,h in report['sources'].items()),
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        backend.close()
        report['seconds']=time.time()-started
        write_json(args.output,report)


if __name__=='__main__':
    main()
