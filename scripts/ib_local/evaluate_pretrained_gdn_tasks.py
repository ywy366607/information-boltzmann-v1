"""Real QASPER answers, continuous learning, and finite-time latch diagnostics.

Evaluation branches restore the complete learner. Generation never receives
answer tokens or token-count stopping hints. Adaptation uses separate branches;
published source checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import sys

import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.probe_pretrained_gdn_instruction import HEADER, prompt, encode_answer
from scripts.ib_local.probe_pretrained_gdn_sft_credit import windows
from scripts.ib_local.run_pretrained_gdn_long_sft import select_papers, write_json
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def normalize_answer(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(prediction, references):
    predicted = normalize_answer(prediction).split()
    scores = []
    for reference in references:
        gold = normalize_answer(reference).split()
        common = sum((Counter(predicted) & Counter(gold)).values())
        f1 = (2*common/(len(predicted)+len(gold)) if predicted and gold else
              float(predicted == gold))
        scores.append((float(predicted == gold), f1))
    return dict(em=max(x[0] for x in scores), f1=max(x[1] for x in scores))


def chunk_slices(length):
    """All real input, no padding; retain a >=128-token final chunk."""
    if length < 128:
        raise ValueError('the existing replay port needs at least128 actual input tokens')
    start = 0
    while length-start >= 256:
        yield slice(start, start+128)
        start += 128
    yield slice(start, length)


def examples(papers):
    return [(p, q) for p in papers for q in p['qa']]


def references(raw, paper, qa):
    original = next(q for q in raw[paper['paper_id']]['qas'] if q['question_id'] == qa['id'])
    spans = [a['answer']['extractive_spans'][0].strip() for a in original['answers']
             if not a['answer']['unanswerable'] and len(a['answer']['extractive_spans']) == 1]
    return list(dict.fromkeys([qa['answer'], *spans]))


@torch.no_grad()
def answer(learner, tokenizer, paper, qa, *, layout='early', max_new_tokens=32,
           generate=True, refs=None):
    """Read prompt once; recompute only its bounded tail for autoregression.

    The tail's held incoming cache is unchanged between generated tokens.
    Recall queries occur at Answer:, injecting only at/after that position.
    The answer-independent split keeps the final real prompt tail128..255 long.
    """
    text = prompt(paper['context'], qa['question'], layout)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    sequence, boundary = encode_answer(tokenizer, text, qa['answer'])
    split = ((len(prefix)-128)//128)*128
    if split < 0:
        raise ValueError('short prompt is outside this evaluation port')
    for start in range(0, split, 128):
        learner.observe(torch.tensor(prefix[start:start+128], device='cuda')[None])
    tail = prefix[split:]
    incoming = clone_cache(learner.cache)
    query = len(tail)-1
    gold = sequence[boundary:]
    teacher_input = torch.tensor(tail+gold[:-1], device='cuda')[None]
    logits, _ = learner.predict(teacher_input, incoming, read_position=query)
    gold_logits = logits[0, query:query+len(gold)].float()
    nll = float(torch.nn.functional.cross_entropy(gold_logits,
                 torch.tensor(gold, device='cuda'), reduction='sum'))
    question_start = text.index('Question: ') if layout == 'early' else text.rindex('Question: ')
    result = dict(id=qa['id'], paper_id=paper['paper_id'], layout=layout,
        prompt_tokens=len(prefix), answer_tokens=len(gold), answer_nll=nll/len(gold),
        question_distance_tokens=(len(prefix)-len(tokenizer.encode(
            text[:question_start+len('Question: ')+len(qa['question'])],
            add_special_tokens=False))),
        references=refs or [qa['answer']])
    if not generate:
        return result
    generated = []
    first_token = int(gold_logits[0].argmax())
    for index in range(max_new_tokens):
        current = torch.tensor(tail+generated, device='cuda')[None]
        predicted, _ = learner.predict(current, incoming, read_position=query)
        token = int(predicted[0, -1].argmax())
        if index == 0 and token != first_token:
            raise AssertionError('teacher-forced suffix changed the greedy first token')
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if '\n' in tokenizer.decode(generated, skip_special_tokens=True):
            break
    decoded = tokenizer.decode(generated, skip_special_tokens=True).split('\n', 1)[0].strip()
    result.update(prediction=decoded, generated_tokens=len(generated),
                  hit_generation_cap=len(generated) == max_new_tokens,
                  **answer_metrics(decoded, result['references']))
    return result


def aggregate(rows):
    result = dict(questions=len(rows), answer_nll=sum(r['answer_nll']*r['answer_tokens']
        for r in rows)/sum(r['answer_tokens'] for r in rows))
    if rows and 'em' in rows[0]:
        result.update(em=sum(r['em'] for r in rows)/len(rows),
                      f1=sum(r['f1'] for r in rows)/len(rows))
    return result


def train_question(learner, tokenizer, paper, qa):
    ids, mask, accounting = windows(tokenizer, paper['context'], qa['question'], qa['answer'])
    rows = []
    for start in range(0, len(mask), 128):
        x = ids[start:start+128][None].cuda()
        active = mask[start:start+128][None].cuda()
        if bool(active.any()):
            rows.append(learner.step(x, ids[start+1:start+129][None].cuda(), loss_mask=active,
                                    read_position=int(active[0].nonzero()[0])))
        else:
            learner.observe(x)
    return dict(id=qa['id'], accounting=accounting, updates=len(rows),
        answer_nll=sum(r['supervised_nll_sum'] for r in rows)/accounting['answer_tokens'])


def frozen_set(learner, tokenizer, pairs, *, generate=False, max_new_tokens=32):
    state = learner.state_dict()
    rows = []
    try:
        for paper, qa in pairs:
            learner.load_state_dict(state)
            rows.append(answer(learner, tokenizer, paper, qa, generate=generate,
                               max_new_tokens=max_new_tokens))
    finally:
        learner.load_state_dict(state)
    return dict(summary=aggregate(rows), rows=rows)


@torch.no_grad()
def latch_trajectory(learner, tokenizer, paper, qa, continuation, *, perturb=0.):
    text = HEADER+f"Question: {qa['question']}\nPassage: {paper['context']}"
    ids = tokenizer.encode(text, add_special_tokens=False)
    last_logits = None
    for segment in chunk_slices(len(ids)):
        _, last_logits = learner.observe(torch.tensor(ids[segment], device='cuda')[None],
                                          return_logits=True)
    if perturb:
        generator = torch.Generator().manual_seed(11)
        for layer in learner.layers:
            state = learner.cache.states[layer]['recurrent_state']
            noise = torch.randn(state.shape, generator=generator).to(state)
            state.add_(noise*(perturb*state.float().norm()/noise.float().norm().clamp_min(1e-20)))

    def record(age, logits):
        return dict(age_tokens=age,
            states={str(i):learner.cache.states[i]['recurrent_state'].float().cpu().clone()
                    for i in learner.layers},
            # At age0 perturb occurs after the last prediction. Compare output
            # differences only after the first128 actual continuation tokens.
            probability=None if age == 0 else logits[0, -1].float().softmax(-1).cpu())

    result = [record(0, last_logits)]
    for start in range(0, 2048, 128):
        _, last_logits = learner.observe(torch.tensor(continuation[start:start+128],
                                            device='cuda')[None], return_logits=True)
        if start+128 in (128, 512, 2048):
            result.append(record(start+128, last_logits))
    return result


def trajectory_comparison(a, perturbed, b):
    def distance(left, right):
        return sum(float((left[k]-right[k]).square().sum()) for k in left)**.5
    def js(p, q):
        if p is None or q is None:
            return None
        mixture = (p+q)/2
        return float(((p*(p.clamp_min(1e-30).log()-mixture.clamp_min(1e-30).log())).sum()+
                     (q*(q.clamp_min(1e-30).log()-mixture.clamp_min(1e-30).log())).sum())/2)
    content0 = distance(a[0]['states'], b[0]['states'])
    perturb0 = distance(a[0]['states'], perturbed[0]['states'])
    return [dict(age_tokens=x['age_tokens'],
        content_distance=distance(x['states'], y['states']),
        perturbation_distance=distance(x['states'], z['states']),
        content_retained_fraction=distance(x['states'], y['states'])/max(content0, 1e-30),
        perturbation_remaining_fraction=distance(x['states'], z['states'])/max(perturb0, 1e-30),
        content_prediction_js=js(x['probability'], y['probability']),
        perturbation_prediction_js=js(x['probability'], z['probability']))
        for x, z, y in zip(a, perturbed, b)]


def paper_text(paper):
    paragraphs = [paper['title'], paper['abstract']]
    for section in paper['full_text']:
        paragraphs.extend([section['section_name'], *section['paragraphs']])
    return '\n\n'.join(p for p in paragraphs if p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--published', type=Path)
    parser.add_argument('--papers', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--latch-papers', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior measurements')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    task = json.loads(args.task.read_text(encoding='utf-8'))
    pair = json.loads(args.pair.read_text(encoding='utf-8'))
    if pair['status'] != 'complete':
        raise ValueError('complete matched full/held pair required')
    raw_dev = json.loads((args.data/'qasper-dev-v0.3.json').read_text(encoding='utf-8'))
    raw_train = json.loads((args.data/'qasper-train-v0.3.json').read_text(encoding='utf-8'))
    validation = select_papers(raw_dev, tokenizer, args.papers)
    if [p['paper_id'] for p in validation[:4]] != [p['paper_id'] for p in task['validation']]:
        raise AssertionError('original validation subset changed')
    train_papers = select_papers(raw_train, tokenizer, len(task['train'])+2)
    if [p['paper_id'] for p in train_papers[:len(task['train'])]] != [p['paper_id'] for p in task['train']]:
        raise AssertionError('training provenance mismatch')
    a_pairs = examples(task['train'][2:4])
    b_pairs = examples(train_papers[len(task['train']):])
    if {p['paper_id'] for p, _ in b_pairs} & {p['paper_id'] for p in validation}:
        raise AssertionError('adaptation B overlaps independent validation')
    paths = {a:Path(pair['arms'][a]['checkpoint']) for a in ('full', 'held')}
    hashes = {a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()}
    report = dict(status='running', kind='real_qasper_generation_continuation_latch',
        source_pair=str(args.pair), task_sha256=hashlib.sha256(args.task.read_bytes()).hexdigest(),
        source_checkpoints={a:str(p) for a,p in paths.items()}, source_checkpoint_sha256=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(validation_papers=args.papers, max_new_tokens=args.max_new_tokens,
            stopping='greedy until EOS/newline or fixed cap; no gold-length stopping',
            scoring='lowercase/punctuation/articles/whitespace normalized EM and token F1; real single-span references',
            layouts=['early','near'], continual='4 seen A questions ->4 unused train B questions ->same4 A; no phase reset',
            latch='two real questions on the same paper, same2048 original continuation tokens; compare content separation with local S perturbation recovery',
            latch_perturbation_relative_norm=.001,
            boundary='finite-time local evidence; slow forgetting or generic contraction alone is not an attractor latch'),
        arms={})
    write_json(args.output, report)
    previous_exception_hook = sys.excepthook
    def record_failure(error_type, error, traceback):
        report.update(status='failed', error=repr(error))
        write_json(args.output, report)
        previous_exception_hook(error_type, error, traceback)
    sys.excepthook = record_failure
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    source = torch.load(paths['full'], map_location='cpu', weights_only=False, mmap=True)
    learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
    learner.load_state_dict(source)
    smoke_paper,smoke_qa = examples(validation)[0]
    smoke = answer(learner,tokenizer,smoke_paper,smoke_qa,max_new_tokens=2)
    learner.load_state_dict(source)
    assert_same(source,learner.state_dict())
    report['interface_gate'] = dict(real_question_id=smoke_qa['id'],
        first_generated_token_matches_causal_teacher_prefix=True,
        full_source_state_restored_exact=True,
        smoke_only_generation_cap=2)
    write_json(args.output,report)
    print('GATE',json.dumps(report['interface_gate']),flush=True)
    for arm, path in paths.items():
        source = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        learner.load_state_dict(source)
        out = dict(generation=[], continual={}, latch=[])
        report['arms'][arm] = out
        for layout in ('early','near'):
            for paper, qa in examples(validation):
                learner.load_state_dict(source)
                row = answer(learner, tokenizer, paper, qa, layout=layout,
                    max_new_tokens=args.max_new_tokens, refs=references(raw_dev, paper, qa))
                row['original_validation'] = paper['paper_id'] in {p['paper_id'] for p in task['validation']}
                out['generation'].append(row)
                write_json(args.output, report)
                print('ANSWER',arm,layout,len(out['generation']),row['em'],round(row['f1'],4),flush=True)
        learner.load_state_dict(source)
        assert_same(source, learner.state_dict())
        out['frozen_source_restore_exact'] = True

        # Isolated live adaptation: probe writes/RNG restore, training writes stay.
        out['continual']['baseline_A'] = frozen_set(learner, tokenizer, a_pairs)
        out['continual']['baseline_B'] = frozen_set(learner, tokenizer, b_pairs)
        for phase, items in [('A',a_pairs),('B',b_pairs),('A_return',a_pairs)]:
            out['continual'][phase] = dict(training=[])
            for paper,qa in items:
                out['continual'][phase]['training'].append(train_question(learner, tokenizer, paper, qa))
                write_json(args.output, report)
            out['continual'][phase]['A_probe'] = frozen_set(learner, tokenizer, a_pairs,
                generate=True, max_new_tokens=args.max_new_tokens)
            out['continual'][phase]['B_probe'] = frozen_set(learner, tokenizer, b_pairs)
            write_json(args.output, report)
            print('CONTINUAL',arm,phase,out['continual'][phase]['A_probe']['summary'],flush=True)
        adapted = args.output.parent/f'{arm}_adapted_last.pt'
        torch.save(learner.state_dict(), adapted.with_suffix('.tmp'))
        adapted.with_suffix('.tmp').replace(adapted)
        out['adapted_checkpoint'] = str(adapted)

        for paper in validation:
            remaining = paper_text(raw_dev[paper['paper_id']])[len(paper['context']):]
            if not paper_text(raw_dev[paper['paper_id']]).startswith(paper['context']):
                raise AssertionError('continuation is not from this original paper prefix')
            continuation = tokenizer.encode(remaining, add_special_tokens=False)[:2048]
            if len(continuation) != 2048:
                continue
            trajectories = []
            for qa,perturb in [(paper['qa'][0],0.),(paper['qa'][0],.001),(paper['qa'][1],0.)]:
                learner.load_state_dict(source)
                trajectories.append(latch_trajectory(learner,tokenizer,paper,qa,continuation,
                                                    perturb=perturb))
            out['latch'].append(dict(paper_id=paper['paper_id'],
                curve=trajectory_comparison(*trajectories)))
            write_json(args.output, report)
            print('LATCH',arm,len(out['latch']),out['latch'][-1]['curve'][-1],flush=True)
            if len(out['latch']) == args.latch_papers:
                break
        if len(out['latch']) != args.latch_papers:
            raise ValueError('insufficient real continuation for registered latch probes')
        learner.load_state_dict(source)
        assert_same(source, learner.state_dict())
        out['generation_summary'] = {layout:aggregate([r for r in out['generation']
            if r['layout']==layout]) for layout in ('early','near')}
        out['source_checkpoint_unchanged'] = hashlib.sha256(path.read_bytes()).hexdigest() == hashes[arm]
        if not out['source_checkpoint_unchanged']:
            raise AssertionError('source checkpoint changed')
        write_json(args.output, report)
    report['status'] = 'complete'
    write_json(args.output, report)
    if args.published:
        args.published.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.published,report)
    print('COMPLETE',json.dumps({a:v['generation_summary'] for a,v in report['arms'].items()}),flush=True)


if __name__ == '__main__':
    main()
