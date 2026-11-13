"""Does forgetting outside the training window affect a later real answer?

Frozen held-SFT native weights; external recall disabled. Perturb only historical
decay rates, restore them before the final128 prompt tokens, then score answers.
This is a forward causal diagnostic, not a finite-difference learning algorithm.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, new_recurrent_cache, clone_cache
from scripts.ib_local.pretrained_gdn_experience import copy_tree
from scripts.ib_local.probe_pretrained_gdn_instruction import prompt, encode_answer
from scripts.ib_local.probe_pretrained_gdn_state_route import select_equal_question_pairs


def historical_interval(prompt_tokens):
    start, end = 128, prompt_tokens-129
    if end <= start:
        raise ValueError('need a nonempty historical interval beyond the last128-token prompt suffix')
    return start, end


@contextmanager
def shift_decay(model, offset):
    parameters = [b.attn.A_log for b in model.model.layers]
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(offset)
        yield
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, originals):
                p.copy_(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = select_equal_question_pairs(args.data/'dev-v1.1.json', tokenizer, 256, 640, 4)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    model.requires_grad_(False)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    with torch.no_grad():
        for name, value in source['native'].items():
            layer, parameter = name.split('.',1)
            model.model.layers[int(layer[5:])].attn.get_parameter(parameter).copy_(value.to('cuda'))
    original_A = [b.attn.A_log.detach().clone() for b in model.model.layers]
    report = dict(kind='historical_forgetting_late_answer_effect',status='running',
        checkpoint=str(args.checkpoint),training_updates=0,external_recall=False,
        perturbation='all24 A_log offset only after first128 prompt tokens and strictly before the final129 prompt tokens',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        offsets=dict(less_forgetting=-.25,normal=0.,more_forgetting=.25),
        rates_relative_to_normal=dict(less_forgetting=float(torch.exp(torch.tensor(-.25))),
                                      normal=1.,more_forgetting=float(torch.exp(torch.tensor(.25)))),records=[])

    def save():
        temp=args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(args.output)

    @torch.no_grad()
    def consume(ids,cache):
        logits=None
        for i in range(0,len(ids),128):
            output=model(ids[i:i+128][None],past_key_values=cache,use_cache=True)
            cache=output.past_key_values
            logits=output.logits[0,-1].float()
        return logits,cache

    @torch.no_grad()
    def state_difference(a,b):
        return [float((x['recurrent_state']-y['recurrent_state']).float().norm())
                for x,y in zip(a.states,b.states)]

    @torch.no_grad()
    def run_question(paragraph,qa,offset):
        prefix=torch.tensor(tokenizer.encode(prompt(paragraph['context'],qa['question'],'early'),
                                             add_special_tokens=False),device='cuda')
        start,end=historical_interval(len(prefix))
        cache=new_recurrent_cache(source['cache']['seen_tokens'])
        cache.states=copy_tree(source['cache']['states'],'cuda')
        _,cache=consume(prefix[:start],cache)
        written=clone_cache(cache)
        with shift_decay(model,offset):
            _,cache=consume(prefix[start:end],cache)
        first_logits,cache=consume(prefix[end:],cache)
        scores=[]
        for candidate in paragraph['qa']:
            # SentencePiece can segment an isolated answer differently from
            # that same answer following the actual prompt. Reuse the task's
            # verified full-prefix extension, never standalone span tokens.
            full,boundary=encode_answer(tokenizer,prompt(paragraph['context'],qa['question'],'early'),candidate['answer'])
            if boundary!=len(prefix):
                raise AssertionError('answer prompt boundary differs')
            ids=torch.tensor(full[boundary:],device='cuda')
            total=F.cross_entropy(first_logits[None],ids[:1],reduction='sum')
            if len(ids)>1:
                out=model(ids[:-1][None],past_key_values=clone_cache(cache),use_cache=True)
                total+=F.cross_entropy(out.logits[0].float(),ids[1:],reduction='sum')
            scores.append(float(total)/len(ids))
        return scores,written,cache,dict(start=start,end=end,
            latest_perturbed_token_distance_from_first_answer=len(prefix)-end,
            prompt_tokens=len(prefix))

    save()
    try:
        for index,paragraph in enumerate(cases):
            row=dict(ids=[q['id'] for q in paragraph['qa']],conditions={})
            started=time.perf_counter()
            for name,offset in report['offsets'].items():
                first=run_question(paragraph,paragraph['qa'][0],offset)
                second=run_question(paragraph,paragraph['qa'][1],offset)
                matrix=[first[0],second[0]]
                row['conditions'][name]=dict(matrix=matrix,
                    answer_nll=(matrix[0][0]+matrix[1][1])/2,
                    question_swap_gap=(matrix[1][0]-matrix[0][0]+matrix[0][1]-matrix[1][1])/2,
                    written_question_state_difference_by_layer=state_difference(first[1],second[1]),
                    answer_question_state_difference_by_layer=state_difference(first[2],second[2]),
                    interval=first[3])
                del first,second
            row['late_nll_derivative_wrt_past_decay_offset']=(row['conditions']['more_forgetting']['answer_nll']-
                row['conditions']['less_forgetting']['answer_nll'])/.5
            row['seconds']=time.perf_counter()-started
            report['records'].append(row)
            save()
            print(index+1,json.dumps({k:dict(nll=v['answer_nll'],gap=v['question_swap_gap']) for k,v in row['conditions'].items()}),flush=True)
        for p,value in zip([b.attn.A_log for b in model.model.layers],original_A):
            if not torch.equal(p,value):
                raise AssertionError('a temporary gate intervention persisted into current weights')
        report['summary']={name:dict(
            mean_answer_nll=sum(r['conditions'][name]['answer_nll'] for r in report['records'])/len(report['records']),
            mean_question_swap_gap=sum(r['conditions'][name]['question_swap_gap'] for r in report['records'])/len(report['records']))
                for name in report['offsets']}
        report.update(status='complete',current_parameters_restored_exactly=True,
            more_forgetting_worsens_answer_pairs=sum(r['late_nll_derivative_wrt_past_decay_offset']>0 for r in report['records']))
        save()
        print(json.dumps(report['summary'],indent=2),flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':
    main()
