"""Real SQuAD question-conditioned long-context task readiness, without training.

Hold the passage fixed and exchange two real questions/answers. Score only the
answer, not easy passage tokens. Early and near layouts separate question use
from the additional burden of maintaining it across128-token boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time


HEADER = 'Read the passage and answer the question using a short span from the passage.\n\n'


def prompt(context, question, layout):
    if layout == 'early':
        return HEADER+f'Question: {question}\nPassage: {context}\nAnswer:'
    if layout == 'near':
        return HEADER+f'Passage: {context}\nQuestion: {question}\nAnswer:'
    raise ValueError(layout)


def select_examples(path, tokenizer, count, *, minimum=256, maximum=640):
    payload = json.loads(path.read_text(encoding='utf-8'))
    paragraphs = [(article['title'], p) for article in payload['data'] for p in article['paragraphs']]
    random.Random(11).shuffle(paragraphs)
    selected = []
    for title, paragraph in paragraphs:
        context = paragraph['context']
        context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        if not minimum <= context_tokens <= maximum:
            continue
        choices = []
        for qa in paragraph['qas']:
            if not qa['answers'] or len(tokenizer.encode(qa['question'], add_special_tokens=False)) > 40:
                continue
            answer = qa['answers'][0]
            text = answer['text']
            if not 1 <= len(tokenizer.encode(' '+text, add_special_tokens=False)) <= 16:
                continue
            start = answer['answer_start']
            if context[start:start+len(text)] != text:
                continue
            choices.append(dict(id=qa['id'], question=qa['question'], answer=text,
                                answer_start=start))
        pair = None
        for i, a in enumerate(choices):
            for b in choices[i+1:]:
                if a['answer'].lower() in b['answer'].lower() or b['answer'].lower() in a['answer'].lower():
                    continue
                pair = [a, b]
                break
            if pair:
                break
        if pair:
            selected.append(dict(title=title, context=context, context_tokens=context_tokens, qa=pair))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'only{len(selected)} eligible real paragraphs, expected{count}')
    return selected


def encode_answer(tokenizer, text, answer):
    prefix = tokenizer.encode(text, add_special_tokens=False)
    sequence = tokenizer.encode(text+' '+answer, add_special_tokens=False)
    if sequence[:len(prefix)] != prefix or len(sequence) <= len(prefix):
        raise ValueError('tokenization changed the prompt/answer boundary')
    return sequence, len(prefix)


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads((args.data/'manifest.json').read_text())
    result = dict(kind='squad_real_question_context_answer', source=manifest,
        scope='task readiness and short route checks, not official SQuAD capability evaluation',
        train=select_examples(args.data/'train-v1.1.json', tokenizer, 32),
        validation=select_examples(args.data/'dev-v1.1.json', tokenizer, args.paragraphs),
        criterion='same passage, two annotated questions with distinct answer spans; no invented distractors')
    for split in ('train', 'validation'):
        for sample in result[split]:
            for qa in sample['qa']:
                text = prompt(sample['context'], qa['question'], 'early')
                seq, boundary = encode_answer(tokenizer, text, qa['answer'])
                question_end = len(tokenizer.encode(HEADER+'Question: '+qa['question'], add_special_tokens=False))
                qa['question_to_answer_tokens'] = boundary-question_end
                if qa['question_to_answer_tokens'] <= 128:
                    raise AssertionError('early question remains within one credit window')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(train_paragraphs=len(result['train']),
        validation_paragraphs=len(result['validation']),
        min_question_distance=min(q['question_to_answer_tokens'] for s in result['validation'] for q in s['qa'])), indent=2))


def baseline(args):
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, new_recurrent_cache

    torch.set_num_threads(1)
    torch.manual_seed(11)
    task = json.loads(args.task.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    report = dict(kind='real_squad_question_use_calibration', status='running',
        training_updates=0, task_sha256=hashlib.sha256(args.task.read_bytes()).hexdigest(),
        score_unit='answer nats/token', conditions=['early', 'near'], records=[],
        note='native pretrained body only; no new recall inlet or SFT. Fresh cache per variant is an explicit frozen diagnostic, not continuous learning.')

    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output)

    @torch.no_grad()
    def score(text, answer):
        sequence, boundary = encode_answer(tokenizer, text, answer)
        n = len(sequence)-1
        needed = ((n+127)//128)*128+1
        ids = torch.tensor(sequence+[tokenizer.eos_token_id]*(needed-len(sequence)), device='cuda')
        cache = new_recurrent_cache()
        total = 0.
        for start in range(0, needed-1, 128):
            output = model(ids[start:start+128][None], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            first, last = max(boundary-1-start, 0), min(n-start, 128)
            if last > first:
                total += float(F.cross_entropy(output.logits[0, first:last].float(),
                    ids[start+first+1:start+last+1], reduction='sum'))
        return dict(nll=total, tokens=len(sequence)-boundary, mean=total/(len(sequence)-boundary))

    save()
    try:
        for index, sample in enumerate(task['validation']):
            started = time.perf_counter()
            conditions = {}
            for layout in ('early', 'near'):
                matrix = [[score(prompt(sample['context'], q['question'], layout), a['answer'])
                           for a in sample['qa']] for q in sample['qa']]
                gap = sum(matrix[1-i][i]['mean']-matrix[i][i]['mean'] for i in range(2))/2
                # Question-dependent matching cancels question-independent answer
                # preference. Positive margin favors both real pairings jointly.
                margin = sum(matrix[i][1-i]['mean']-matrix[i][i]['mean'] for i in range(2))
                conditions[layout] = dict(scores=matrix, mean_swap_gap=gap,
                    paired_assignment_margin=margin,
                    correct_answer_mean=sum(matrix[i][i]['mean'] for i in range(2))/2)
            row = dict(index=index, ids=[q['id'] for q in sample['qa']],
                context_tokens=sample['context_tokens'],
                question_distance=[q['question_to_answer_tokens'] for q in sample['qa']],
                conditions=conditions, seconds=time.perf_counter()-started)
            report['records'].append(row)
            save()
            print(json.dumps(dict(paragraph=index+1, early_gap=conditions['early']['mean_swap_gap'],
                                  near_gap=conditions['near']['mean_swap_gap'])), flush=True)
        report['summary'] = {layout: dict(
            mean_correct_answer_nll=sum(r['conditions'][layout]['correct_answer_mean'] for r in report['records'])/len(report['records']),
            mean_question_swap_gap=sum(r['conditions'][layout]['mean_swap_gap'] for r in report['records'])/len(report['records']),
            correct_paired_assignments=sum(r['conditions'][layout]['paired_assignment_margin'] > 0 for r in report['records']),
            total_paragraphs=len(report['records'])) for layout in ('early', 'near')}
        report['status'] = 'complete'
        save()
        print(json.dumps(report['summary'], indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'baseline'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paragraphs', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve task and result provenance')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args)
    else:
        baseline(args)


if __name__ == '__main__':
    main()
