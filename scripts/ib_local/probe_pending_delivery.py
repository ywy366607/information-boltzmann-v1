"""One saved actual decision: held natural read vs researcher-specified record.

No training or lifetime continuation. Branches inspect a real pre-feedback
decision, including a stored long successor which the 65-token replay omits.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.ib_local.delayed_prediction_credit import fork_view, anchor_participation, release_view
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution, new_recurrent_cache
from scripts.ib_local.pretrained_gdn_experience import copy_tree,LocalExperience
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def execute_local_query(block, capsule, query, *, layer, detach_history=False):
    """Ask a current bounded query of a reexecuted historical local program.

    Stored inputs and the incoming boundary are held observations. Full mode
    differentiates through the stored program's final S/conv into this query;
    held mode cuts only that historical state, preserving the query computation.
    Neither local cache replaces the individual's live cache. For native FLA
    queries<=64 under autograd, the caller must use its chunk derivative bridge.
    """
    if not 1 <= query.shape[1] <= 128 or query.shape[0] != 1:
        raise ValueError('one bounded current query of at most128 inputs required')
    inputs=capsule.complete_inputs
    if not 65 <= inputs.shape[1] <= 128:
        raise ValueError('only the actual stored65..128 participation is executable')
    cache=new_recurrent_cache()
    cache.states=[{} for _ in range(layer)]+[
        {} if capsule.boundary is None else copy_tree(capsule.boundary,query.device)]
    block(inputs.to(query.device),past_key_values=cache,use_cache=True)
    if detach_history:
        from scripts.ib_local.pretrained_gdn import clone_cache
        cache=clone_cache(cache)
    return block(query,past_key_values=cache,use_cache=True)[0]


def intervention(shadow, record, held, slot, mode):
    normal=shadow.retrieve
    anchor_participation(shadow,held)
    actual_first=shadow.retrieve
    first=True

    def retrieve(query,replay):
        nonlocal first
        value=(actual_first if first else normal)(query,replay)
        first=False
        if mode in ('natural','correct_native_boundary','native_query_state',
                    'query_complete_record','query_complete_natural'):
            return value
        result=value.clone()
        def complete(capsule):
            return replay(LocalExperience(capsule.complete_inputs,capsule.boundary,capsule.version))
        if mode in ('no_current_read','correct_native_boundary_no_extra'):
            result[-1]=0
        elif mode=='correct_record_65':
            response=replay(record.capsules[slot])[:, -1].float().cpu()
            result[-1]=shadow.encoder(response)[0]
        elif mode=='correct_record_observed_127':
            result[-1]=shadow.encoder(record.memory.long_successor[slot:slot+1])[0]
        elif mode=='correct_record_complete':
            result[-1]=shadow.encoder(complete(record.capsules[slot])[:, -1].float().cpu())[0]
        elif mode=='complete_natural':
            prior_offsets=shadow.query_clock_offsets
            if prior_offsets is not None:
                shadow.query_clock_offsets=prior_offsets[-1:]
            try:
                result[-1]=normal(query[-1:],complete)[0]
            finally:
                shadow.query_clock_offsets=prior_offsets
        else:
            raise ValueError(mode)
        return result

    shadow.retrieve=retrieve


@contextmanager
def native_state_read(attention, shadow, state):
    """Oracle record, actual current Q and the pretrained gate/norm/O decoder."""
    query={}
    def capture(module,inputs,output):
        if not shadow._replaying:
            query['value']=output[0]
    def read(module,inputs):
        if shadow._replaying:
            return
        current=inputs[0]
        q=query['value'].reshape(1,-1,attention.num_heads,attention.head_k_dim)[:, -1:]
        q=q.float()/(q.float().square().sum(-1,keepdim=True).sqrt()+1e-6)
        old=torch.einsum('bthk,bhkv->bthv',q,state)/attention.head_k_dim**.5
        replaced=torch.cat((current[:, :-1],old.to(current.dtype)),dim=1)
        return (replaced,*inputs[1:])
    a=attention.q_conv1d.register_forward_hook(capture)
    b=attention.o_norm.register_forward_pre_hook(read)
    try:
        yield
    finally:
        a.remove(); b.remove()


@contextmanager
def query_program_read(block,shadow,view,*,layer,slot,natural,query_history):
    """Current query asks the old native program; deliver in the same output space."""
    def deliver(module,inputs,output):
        if shadow._replaying:
            return
        query=shadow.current['source']
        query_history.append(query)
        joined=torch.cat(query_history,dim=1)[:,-128:]
        # Only this temporary answer computation carries the query, never a
        # growing lifetime log or a replacement for the live recurrent state.
        query_history[:]=[joined]
        if natural:
            read=shadow.read(query[0,-1:])
            candidates=zip(read.slots[0].tolist(),read.weights[0])
        else:
            candidates=[(slot,query.new_tensor(1.))]
        original=output[0][:,-1:]
        revised=original.float()
        for chosen,weight in candidates:
            if not bool(view.memory.occupied[chosen]):
                continue
            capsule=view.capsules[chosen]
            shadow._replaying=True
            try:
                response=execute_local_query(block,capsule,joined,layer=layer)[:,-1:]
            finally:
                shadow._replaying=False
            revised=revised+weight.to(original.device)*(response.float()-original.float())
        result=torch.cat((output[0][:,:-1],revised.to(original.dtype)),dim=1)
        return (result,*output[1:])
    hook=block.register_forward_hook(deliver)
    try:
        yield
    finally:
        hook.remove()


@torch.no_grad()
def branch(learner, record, slots, gold, tokenizer, mode, *, generate, native_slots=None, read_states=None,
           prefix_reference=None):
    device=next(learner.model.parameters()).device
    shadows={i:fork_view(w,record.views[i]) for i,w in learner.memories.items()}
    for i,shadow in shadows.items():
        intervention(shadow,record.views[i],record.reads[i],slots[i],mode)
    cache=new_recurrent_cache(record.boundary['seen_tokens'])
    cache.states=copy_tree(record.boundary['states'],device)
    if mode.startswith('correct_native_boundary'):
        for layer,slot in native_slots.items():
            # Researcher intervention only: current convolution, other layers,
            # lifetime counters and live learner cache remain untouched.
            old=record.views[layer].capsules[slot].boundary['recurrent_state']
            cache.states[layer]['recurrent_state']=copy_tree(old,device)
    tokens=record.inputs.to(device)
    offset=record.clock_offset
    emitted,losses=[],[]
    first_error=None
    prefix_copy=None
    prefix_error=None
    query_history={i:[] for i in shadows}
    try:
        for index in range(16 if generate else len(gold)):
            with ExitStack() as stack:
                for i,shadow in shadows.items():
                    stack.enter_context(shadow.attach(learner.model,layer=i,capture=False,clock_offset=offset))
                    if mode in ('query_complete_record','query_complete_natural'):
                        stack.enter_context(query_program_read(learner.model.model.layers[i],shadow,record.views[i],
                            layer=i,slot=slots[i],natural=mode=='query_complete_natural',
                            query_history=query_history[i]))
                    if mode=='native_query_state':
                        stack.enter_context(native_state_read(learner.model.model.layers[i].attn,shadow,read_states[i]))
                output=learner.model(tokens,past_key_values=cache,use_cache=True)
            logits=output.logits[0,-1].float()
            cache=output.past_key_values
            offset+=tokens.numel()
            if index==0:
                first_error=float((logits.cpu()-record.logits).abs().max())
                if mode=='natural' and not generate:
                    prefix_copy=output.logits[0,:-1].float().cpu().clone()
                if prefix_reference is not None:
                    prefix_error=float((output.logits[0,:-1].float().cpu()-prefix_reference).abs().max())
                    if prefix_error!=0:
                        raise AssertionError('last-query intervention changed the earlier prefix')
            if generate:
                token=int(logits.argmax())
                emitted.append(token)
                if token==tokenizer.eos_token_id:
                    break
            else:
                token=gold[index]
                losses.append(float(F.cross_entropy(logits[None],torch.tensor([token],device=device))))
            tokens=torch.tensor([[token]],device=device)
            del output,logits
        if generate:
            return dict(text=tokenizer.decode(emitted,skip_special_tokens=True),emitted=emitted,
                        first_logit_difference_from_actual=first_error,prefix_logit_difference=prefix_error)
        issued_ce=float(F.cross_entropy(record.logits[None],torch.tensor([gold[0]])))
        return dict(answer_nll=sum(losses)/len(losses),token_losses=losses,
                    first_target_ce_difference=abs(losses[0]-issued_ce),
                    first_logit_difference_from_actual=first_error,_prefix=prefix_copy,
                    prefix_logit_difference=prefix_error)
    finally:
        for shadow in shadows.values():
            release_view(shadow)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','case-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--modes',nargs='+',default=['natural','native_query_state'],
                        choices=('natural','native_query_state','correct_record_65',
                                 'correct_record_complete','complete_natural','no_current_read',
                                 'query_complete_record','query_complete_natural'))
    args=parser.parse_args()
    if args.modes[0]!='natural':
        parser.error('natural baseline must be first')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started=time.time()
    source_hash=sha256(args.checkpoint)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    number=payload['next_case']
    history=json.loads(args.case_report.read_text())
    case=(history['pending_case'] if 'pending_case' in history else
          next(c for c in history['arms']['full']['cases'] if c['case']==number))
    _,tasks=read_cases(args.manifest,[2048],number+1)
    row=tasks[number][1]
    fact_start=case['incoming_seen']+row['token_position_answer']
    fact_end=fact_start+len(row['outputs'][0])-1
    record=payload['pending'].records[0]
    report=dict(status='checking',source=str(args.checkpoint),source_sha256=source_hash,
        actual_case=number,actual_fact_tokens=[fact_start,fact_end],slots={},results={},
        retained_updates=0,protocol=__doc__,modes=args.modes,
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/pretrained_gdn_experience.py',
                      'scripts/ib_local/delayed_prediction_credit.py')})
    slots,native_slots={},{}
    for layer,view in record.views.items():
        choices=[]
        for slot in view.memory.occupied.nonzero().flatten().tolist():
            birth=int(view.memory.birth[slot])
            if birth-64<=fact_start and fact_end<=birth+63:
                choices.append(slot)
        if len(choices)!=1:
            raise RuntimeError('expected one retained real window containing the annotated fact')
        slots[layer]=choices[0]
        birth=int(view.memory.birth[choices[0]])
        report['slots'][str(layer+1)]=dict(slot=choices[0],source_time=birth,
            fact_in_reexecuted_input=fact_end<=birth,
            fact_in_observed_long_successor=True)
        later=[(int(view.memory.birth[slot])-64,slot)
               for slot in view.memory.occupied.nonzero().flatten().tolist()
               if int(view.memory.birth[slot])-64>fact_end and view.capsules[slot].boundary is not None]
        if not later:
            raise RuntimeError('no observed post-fact native boundary retained')
        time_after,slot_after=min(later)
        native_slots[layer]=slot_after
        report['slots'][str(layer+1)].update(native_boundary_slot=slot_after,
                                            native_boundary_time=time_after)
    write_json(args.output,report)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefix,gold=encode_case(tokenizer,row)
    if fact_start-case['incoming_seen']>=len(prefix):
        raise ValueError('fact must precede actual answer')
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=payload['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    read_states={}
    with torch.no_grad():
        device=next(model.parameters()).device
        for layer,slot in (slots.items() if 'native_query_state' in args.modes else []):
            capsule=record.views[layer].capsules[slot]
            born=int(record.views[layer].memory.birth[slot])
            stop=fact_end-(born-64)+1
            held=new_recurrent_cache()
            held.states=[{} for _ in range(layer)]+[copy_tree(capsule.boundary,device)]
            old=model.model.layers[layer](capsule.complete_inputs[:, :stop].to(device),
                                          past_key_values=held,use_cache=True)
            read_states[layer]=held.states[layer]['recurrent_state']
            report['slots'][str(layer+1)]['native_query_state_prefix_inputs']=stop
            del old,held
    live=learner.cache
    original_rng=torch.get_rng_state().clone()
    original_cuda_rng=torch.cuda.get_rng_state().clone()
    report['status']='running'
    prefix_reference=None
    for mode in args.modes:
        score=branch(learner,record,slots,gold,tokenizer,mode,generate=False,native_slots=native_slots,read_states=read_states,
                     prefix_reference=prefix_reference)
        prefix=score.pop('_prefix')
        if mode=='natural':
            prefix_reference=prefix
        generated=branch(learner,record,slots,gold,tokenizer,mode,generate=True,native_slots=native_slots,read_states=read_states,
                         prefix_reference=prefix_reference)
        result={**score,**generated,**numeric_scores(generated['text'],row['outputs'])}
        report['results'][mode]=result
        print(mode,result['answer_nll'],repr(result['text']),flush=True)
        write_json(args.output,report)
    if learner.cache is not live or learner.updates!=state['updates']:
        raise AssertionError('diagnostic changed actual learner history')
    if not torch.equal(torch.get_rng_state(),original_rng) or not torch.equal(torch.cuda.get_rng_state(),original_cuda_rng):
        raise AssertionError('diagnostic changed learner random state')
    natural=report['results']['natural']
    report['natural_generated_matches_actual']=natural['emitted']==[r.emitted for r in payload['pending'].records]
    if (natural['first_logit_difference_from_actual']>.125 or natural['first_target_ce_difference']>.02
            or not report['natural_generated_matches_actual']):
        raise AssertionError('natural decision does not reconstruct the actually issued prediction')
    report.update(status='complete',seconds=time.time()-started,source_unchanged=sha256(args.checkpoint)==source_hash,
        actual_cache_preserved=True,source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
    write_json(args.output,report)


if __name__=='__main__':
    main()
