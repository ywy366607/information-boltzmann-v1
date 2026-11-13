"""Frozen RULER diagnosis: token boundary, rollout, read commits and kernels.

No optimizer steps. Uses source/32/64 checkpoints and a fixed small diagnostic
subset, not new benchmark scores. Alternative prompt partition uses the known
answer length only to reconstruct the supervised execution boundary.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.evaluate_pretrained_gdn_ruler import frozen_decode_replays
from scripts.ib_local.evaluate_pretrained_gdn_tasks import chunk_slices
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, new_recurrent_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def encode_case(tokenizer,row):
    """Preserve the legacy adapter to diagnose the original faulty checkpoints."""
    text = row['input']+row.get('answer_prefix','')
    return (tokenizer.encode(text,add_special_tokens=False),
            tokenizer.encode(' '+', '.join(row['outputs']),add_special_tokens=False))


def token_stats(logits, labels, tokenizer):
    prob = logits.float().softmax(-1)
    result = []
    for vec,distribution,label in zip(logits,prob,labels):
        order = vec.topk(3).indices.tolist()
        result.append(dict(gold=label,gold_text=tokenizer.decode([label]),
            probability=float(distribution[label]),nll=float(-distribution[label].log()),
            rank=1+int((vec>vec[label]).sum()),
            top=[dict(id=i,text=tokenizer.decode([i]),p=float(distribution[i])) for i in order]))
    return result


@torch.no_grad()
def probe(learner,tokenizer,row,mode):
    prefix,gold = encode_case(tokenizer,row)
    device = next(learner.model.parameters()).device
    learner.cache = new_recurrent_cache()
    if mode == 'published':
        for section in chunk_slices(len(prefix)):
            _,output = learner.observe(torch.tensor([prefix[section]],device=device),return_logits=True)
        first,cache = output[0,-1],learner.cache
        training_logits = None
    else:
        # Reconstruct the prefix before the actual128-token supervised window.
        boundary = len(prefix)+len(gold)-128
        for section in chunk_slices(boundary):
            learner.observe(torch.tensor([prefix[section]],device=device))
        before = learner.cache
        suffix = prefix[boundary:]+gold
        all_logits,_ = learner.predict(torch.tensor([suffix],device=device),before)
        training_logits = all_logits[0,len(prefix)-boundary-1:]
        output,cache = learner.predict(torch.tensor([prefix[boundary:]],device=device),before)
        first = output[0,-1]
    # Both teacher forcing and greedy rollout begin at identical full prompt state.
    labels = gold+[tokenizer.eos_token_id]
    natural = tokenizer.encode(row['input']+row.get('answer_prefix','')+' '+', '.join(row['outputs']),
                               add_special_tokens=False)
    if natural[:len(prefix)] != prefix:
        raise AssertionError('natural answer changed prompt boundary')
    natural_gold = natural[len(prefix):]
    with frozen_decode_replays(learner):
        batch,_ = learner.predict(torch.tensor([gold],device=device),cache)
        batch = torch.cat((first[None],batch[0]),dim=0)
        sequential = [first]
        seq_cache = cache
        for token in gold:
            out,seq_cache = learner.predict(torch.tensor([[token]],device=device),seq_cache)
            sequential.append(out[0,-1])
        sequential = torch.stack(sequential)
        free = []
        free_logits = []
        free_cache = cache
        current = first
        for _ in range(12):
            token = int(current.argmax())
            free.append(token)
            free_logits.append(current)
            if token == tokenizer.eos_token_id:
                break
            output,free_cache = learner.predict(torch.tensor([[token]],device=device),free_cache)
            current = output[0,-1]
        forced = {}
        for name,first_token in [('trained_first',gold[0]),('natural_first',natural_gold[0])]:
            selected = [first_token]
            forced_cache = cache
            token = first_token
            for _ in range(11):
                output,forced_cache = learner.predict(torch.tensor([[token]],device=device),forced_cache)
                token = int(output[0,-1].argmax())
                selected.append(token)
                if token==tokenizer.eos_token_id:
                    break
            forced[name] = dict(ids=selected,text=tokenizer.decode(selected))
    natural_output,_ = learner.predict(torch.tensor([natural_gold],device=device),cache)
    natural_logits = torch.cat((first[None],natural_output[0]),dim=0)
    stats = token_stats(batch,labels,tokenizer)
    answer_tokens = stats[:-1]
    answer_offset=row['input'].find(row['outputs'][0])
    answer_distance=(None if answer_offset<0 else len(prefix)-len(tokenizer.encode(
        row['input'][:answer_offset],add_special_tokens=False)))
    result = dict(mode=mode,gold=gold,natural_gold=natural_gold,
        prompt_tokens=len(prefix),observed_answer_distance_tokens=answer_distance,
        appended_text=tokenizer.decode(prefix+gold)[-90:],natural_text=tokenizer.decode(natural)[-90:],
        free_ids=free,free_text=tokenizer.decode(free),
        forced_first_token=forced,
        free_first_mismatch=next((i for i,(a,b) in enumerate(zip(free,labels)) if a!=b),None),
        batch=stats,sequential=token_stats(sequential,labels,tokenizer),
        natural=token_stats(natural_logits,natural_gold+[tokenizer.eos_token_id],tokenizer),
        batch_sequential_max=float((batch.float()-sequential.float()).abs().max()),
        batch_sequential_top_flips=int((batch.argmax(-1)!=sequential.argmax(-1)).sum()),
        first_nll=answer_tokens[0]['nll'],digits_mean_nll=sum(s['nll'] for s in answer_tokens[1:])/7,
        eos_rank=stats[-1]['rank'],eos_probability=stats[-1]['probability'])
    if training_logits is not None:
        result['training_128'] = token_stats(training_logits,labels,tokenizer)
        result['training128_sequential_max'] = float((training_logits.float()-sequential.float()).abs().max())
        result['training128_sequential_top_flips'] = int((training_logits.argmax(-1)!=sequential.argmax(-1)).sum())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve audit evidence')
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model,_=load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    payload=torch.load(args.source,map_location='cpu',mmap=True,weights_only=False)
    source=payload.get('learner',payload)
    learner=PretrainedExperienceLearner(model,layers=source['layers'],capacity=source['capacity'])
    checkpoints=[('source',args.source),('held32',args.root/'learning_pair/held_previous.pt'),
                 ('held64',args.root/'learning_pair/held_last.pt')]
    cases=[]
    for length,index in [(512,1),(512,2),(2048,0)]:
        rows=[json.loads(x) for x in (args.root/f'heldout/{length}/niah_single_1/test.jsonl').read_text().splitlines()]
        cases.append((length,index,rows[index]))
    report=dict(status='running',kind='frozen_generation_diagnosis',protocol=__doc__,
        optimizer_updates=0,records=[],checkpoints={str(p):sha256(p) for _,p in checkpoints})
    start=time.time()
    write_json(args.output,report)
    try:
        for name,path in checkpoints:
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            state=payload.get('learner',payload)
            for length,index,row in cases:
                for mode in ('published','deferred_prompt_commit'):
                    learner.load_state_dict(state)
                    item=probe(learner,tokenizer,row,mode)
                    item.update(checkpoint=name,length=length,index=index)
                    report['records'].append(item)
                    write_json(args.output,report)
                    print(json.dumps(dict(checkpoint=name,length=length,index=index,mode=mode,
                        first_rank=item['batch'][0]['rank'],first_nll=item['first_nll'],
                        eos_rank=item['eos_rank'],free=item['free_text'],
                        kernel_flips=item['batch_sequential_top_flips'])),flush=True)
                if name!='source' and length==512 and index==2:
                    learner.load_state_dict(state)
                    with torch.no_grad():
                        for w in learner.memories.values():
                            w.inlet.projection.weight.zero_()
                    item=probe(learner,tokenizer,row,'published')
                    item.update(checkpoint=name,length=length,index=index,recall_disabled=True)
                    report['records'].append(item)
                    write_json(args.output,report)
                if name=='held64' and length==512 and index==2:
                    noise='The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n'*60
                    head,rest=row['input'].split('\n',1)
                    before_question,question=row['input'].rsplit('\nWhat',1)
                    for where,text in [('prefix_warmup',head+'\n'+noise+rest),
                                       ('delayed_needle',before_question+'\n'+noise+'What'+question)]:
                        learner.load_state_dict(state)
                        item=probe(learner,tokenizer,dict(row,input=text),'published')
                        item.update(checkpoint=name,length=length,index=index,input_intervention=where)
                        report['records'].append(item)
                        write_json(args.output,report)
            learner.load_state_dict(state)
            assert_same(state,learner.state_dict())
        report.update(status='complete',source_state_restored=True,
            checkpoints_unchanged=all(sha256(p)==h for p,h in report['checkpoints'].items()),
            seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if not report['checkpoints_unchanged']:
            raise AssertionError('checkpoint changed during audit')
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        write_json(args.output,report)


if __name__=='__main__':
    main()
