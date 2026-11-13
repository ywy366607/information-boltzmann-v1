"""Intervene on S, convolution state, and recall after short answer-only SFT.

Reconstruct the held arm of the earlier real SQuAD protocol, then freeze all
weights. Equal-token-length questions keep the post-cut passage suffix exactly
identical. This separates carried information from old-computation learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.probe_pretrained_gdn_instruction import prompt, encode_answer, HEADER
from scripts.ib_local.probe_pretrained_gdn_sft_credit import windows


def zero_tree(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, tuple):
        return tuple(zero_tree(x) for x in value)
    if isinstance(value, list):
        return [zero_tree(x) for x in value]
    return value


def cut_cache(cache, kind):
    result = clone_cache(cache)
    keys = {'S': ('recurrent_state',), 'conv': ('conv_state',),
            'all': ('recurrent_state', 'conv_state')}[kind]
    for state in result.states:
        for key in keys:
            if key in state:
                state[key] = zero_tree(state[key])
    return result


def select_equal_question_pairs(path, tokenizer, minimum, maximum, limit):
    payload = json.loads(path.read_text(encoding='utf-8'))
    paragraphs = [(a['title'], p) for a in payload['data'] for p in a['paragraphs']]
    random.Random(23).shuffle(paragraphs)
    selected = []
    for title, paragraph in paragraphs:
        context = paragraph['context']
        length = len(tokenizer.encode(context, add_special_tokens=False))
        if not minimum <= length <= maximum:
            continue
        groups = {}
        pair = None
        for qa in paragraph['qas']:
            if not qa['answers'] or len(tokenizer.encode(qa['question'], add_special_tokens=False)) > 40:
                continue
            answer = qa['answers'][0]
            if not 1 <= len(tokenizer.encode(' '+answer['text'], add_special_tokens=False)) <= 16:
                continue
            if context[answer['answer_start']:answer['answer_start']+len(answer['text'])] != answer['text']:
                continue
            prefix = tokenizer.encode(prompt(context, qa['question'], 'early'), add_special_tokens=False)
            item = dict(id=qa['id'], question=qa['question'], answer=answer['text'])
            for prior in groups.get(len(prefix), []):
                if prior['answer'].lower() in item['answer'].lower() or item['answer'].lower() in prior['answer'].lower():
                    continue
                pair = [prior, item]
                break
            if pair:
                break
            groups.setdefault(len(prefix), []).append(item)
        if not pair:
            continue
        for qa in pair:
            prefix = tokenizer.encode(prompt(context, qa['question'], 'early'), add_special_tokens=False)
            qa['question_to_answer_tokens'] = len(prefix)-len(tokenizer.encode(
                HEADER+'Question: '+qa['question'], add_special_tokens=False))
        selected.append(dict(title=title, context=context, context_tokens=length, qa=pair))
        if len(selected) == limit:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--previous-sft', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    from transformers import AutoTokenizer
    task = json.loads(args.task.read_text(encoding='utf-8'))
    previous = json.loads(args.previous_sft.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = dict(short=select_equal_question_pairs(args.data/'dev-v1.1.json', tokenizer, 256, 640, 8),
                  long=select_equal_question_pairs(args.data/'dev-v1.1.json', tokenizer, 1024, 2048, 4))
    if not groups['short']:
        raise ValueError('no equal-length real question pairs')
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=(1, 12), capacity=64, detach_replay=True)
    report = dict(kind='frozen_state_vs_recall_after_held_sft', status='running',
        seed=11, old_native_credit_enabled=False, protocol='4 warm questions then16 held train questions; freeze before interventions',
        task_sha256=hashlib.sha256(args.task.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups={name:dict(paragraphs=len(items), records=[]) for name,items in groups.items()},
        reconstruction_nll_differences=[],
        interventions={
            'intact':'S, convolution and external recall all retained',
            'no_recall':'zero all recalled-value contributions throughout the example',
            'zero_S':'clear all24 live S matrices at first answer-supervised window; retain convolution and recall',
            'zero_S_no_recall':'same live S cut, external recall off throughout',
            'zero_conv_no_recall':'clear convolution at answer window; S retained, recall off',
            'zero_all_no_recall':'clear S and convolution at answer window, recall off; aligned suffix must yield zero question gap'},
        score='answer NLL and correct-question minus swapped-question conditioning; joint candidate matching is not free generation accuracy')

    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output)

    def train_one(paragraph, qa):
        ids, mask, _ = windows(tokenizer, paragraph['context'], qa['question'], qa['answer'])
        total, count = 0., 0
        for start in range(0, len(mask), 128):
            x, y = ids[start:start+128][None].cuda(), ids[start+1:start+129][None].cuda()
            active = mask[start:start+128][None].cuda()
            if bool(active.any()):
                row = learner.step(x, y, loss_mask=active)
                total += row['supervised_nll_sum']
                count += row['supervised_tokens']
            else:
                learner.observe(x)
        return total/count

    save()
    try:
        examples = [(p,q) for p in task['train'] for q in p['qa']]
        for index,(p,q) in enumerate(examples[:20]):
            nll = train_one(p,q)
            if index >= 4:
                expected = previous['arms']['held']['records'][index-4]
                if expected['id'] != q['id']:
                    raise AssertionError('reconstruction example changed')
                report['reconstruction_nll_differences'].append(nll-expected['answer_nll'])
            print('RECONSTRUCT', index+1, nll, flush=True)
        report['reconstruction_max_abs_nll_difference'] = max(abs(x) for x in report['reconstruction_nll_differences'])
        if report['reconstruction_max_abs_nll_difference'] > .005:
            raise AssertionError('reconstruction differs materially from original held route')
        report['held_optimizer_updates'] = learner.updates
        source = learner.state_dict()
        checkpoint = args.output.with_suffix('.pt')
        torch.save(source, checkpoint)
        report['frozen_checkpoint'] = str(checkpoint)
        save()

        @torch.no_grad()
        def score(paragraph, question, answer, condition):
            learner.load_state_dict(source)
            for memory in learner.memories.values():
                memory.suppress_older_than = 0 if 'no_recall' in condition else None
            ids, mask, _ = windows(tokenizer, paragraph['context'], question, answer)
            cut_done = False
            total, count = 0.,0
            for start in range(0, len(mask), 128):
                active = mask[start:start+128][None].cuda()
                if bool(active.any()) and not cut_done:
                    if condition.startswith('zero_S'):
                        learner.cache = cut_cache(learner.cache,'S')
                    elif condition.startswith('zero_conv'):
                        learner.cache = cut_cache(learner.cache,'conv')
                    elif condition.startswith('zero_all'):
                        learner.cache = cut_cache(learner.cache,'all')
                    cut_done = True
                x = ids[start:start+128][None].cuda()
                _, logits = learner.observe(x, return_logits=True)
                if bool(active.any()):
                    y = ids[start+1:start+129][None].cuda()
                    _,ce = language_loss(logits,y,active)
                    total += float(ce)*int(active.sum())
                    count += int(active.sum())
            return total/count

        for group, paragraphs in groups.items():
            conditions = list(report['interventions']) if group == 'short' else ['intact','no_recall','zero_all_no_recall']
            for index, paragraph in enumerate(paragraphs):
                # Verify equality of the suffix seen after the intervention for
                # EACH fixed answer, including padding and the supervision mask.
                for answer in paragraph['qa']:
                    variants = [windows(tokenizer,paragraph['context'],q['question'],answer['answer'])
                                for q in paragraph['qa']]
                    cut = int(torch.nonzero(variants[0][1])[0])//128*128
                    if not torch.equal(variants[0][0][cut:],variants[1][0][cut:]) or not torch.equal(variants[0][1][cut:],variants[1][1][cut:]):
                        raise AssertionError('question replacement changed the post-cut suffix/alignment')
                row = dict(ids=[q['id'] for q in paragraph['qa']], context_tokens=paragraph['context_tokens'],
                           question_distance=[q['question_to_answer_tokens'] for q in paragraph['qa']], conditions={})
                started = time.perf_counter()
                for condition in conditions:
                    matrix = [[score(paragraph,q['question'],a['answer'],condition)
                               for a in paragraph['qa']] for q in paragraph['qa']]
                    gap = sum(matrix[1-i][i]-matrix[i][i] for i in range(2))/2
                    if condition == 'zero_all_no_recall' and abs(gap) > 1e-6:
                        raise AssertionError('question survives a complete aligned history/recall cut')
                    row['conditions'][condition] = dict(matrix=matrix,
                        answer_nll=sum(matrix[i][i] for i in range(2))/2,question_swap_gap=gap,
                        matching_correct=gap>0)
                row['seconds'] = time.perf_counter()-started
                report['groups'][group]['records'].append(row)
                save()
                print(group,index+1,json.dumps({k:v['question_swap_gap'] for k,v in row['conditions'].items()}),flush=True)
            records = report['groups'][group]['records']
            if records:
                report['groups'][group]['summary'] = {c:dict(
                    mean_answer_nll=sum(r['conditions'][c]['answer_nll'] for r in records)/len(records),
                    mean_question_swap_gap=sum(r['conditions'][c]['question_swap_gap'] for r in records)/len(records),
                    matching_correct=sum(r['conditions'][c]['matching_correct'] for r in records),
                    paragraphs=len(records)) for c in conditions}
                save()
        learner.load_state_dict(source)
        for m in learner.memories.values():
            m.suppress_older_than=None
        report.update(status='complete',training_updates_during_interventions=0)
        save()
        print('COMPLETE',flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
