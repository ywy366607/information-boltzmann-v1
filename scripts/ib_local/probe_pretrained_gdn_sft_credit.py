"""Short answer-only SFT pair on real, question-first SQuAD paragraphs.

Both arms retain the same architecture and retrieval algorithm. Only native
derivatives through re-executed old computations are cut in held. One shared
warm start is restored between arms; continuous state is not reset at examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.probe_pretrained_gdn_instruction import prompt, encode_answer


def windows(tokenizer, context, question, answer):
    seq, boundary = encode_answer(tokenizer, prompt(context, question, 'early'), answer)
    positions = ((len(seq)-2)//128+1)*128
    ids = torch.tensor(seq+[tokenizer.eos_token_id]*(positions+1-len(seq)))
    labels = torch.arange(1, positions+1)
    mask = (labels >= boundary) & (labels < len(seq))
    return ids, mask, dict(real_tokens=len(seq), padding_tokens=positions+1-len(seq),
                           answer_tokens=len(seq)-boundary, question_boundary=boundary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-examples', type=int, default=16)
    parser.add_argument('--eval-paragraphs', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    from transformers import AutoTokenizer
    task = json.loads(args.task.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=(1, 12), capacity=64, detach_replay=True)
    examples = [(p, q) for p in task['train'] for q in p['qa']]
    if 4+args.train_examples > len(examples):
        raise ValueError('insufficient distinct training question examples')
    report = dict(kind='real_squad_short_sft_native_credit', status='running',
        seed=11, task_sha256=hashlib.sha256(args.task.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        window=128, capacity=64, native_layers=[1, 12], warm_examples=4,
        train_examples_per_arm=args.train_examples, validation_paragraphs=args.eval_paragraphs,
        supervised='annotated answer tokens only; prompts consumed without optimizer updates',
        padding='EOS pads to the128-token compute window; ignored by answer loss, recorded separately, recurrence executes them',
        hypothesis='full improves answer NLL and question-dependent matching beyond held under the same budget',
        scope='budget-limited route study, not trained instruction capability or whole navigation',
        arms={})

    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output)

    def train_example(paragraph, qa, *, diagnose=False):
        ids, mask, accounting = windows(tokenizer, paragraph['context'], qa['question'], qa['answer'])
        updates = []
        question_slots = None
        for start in range(0, len(mask), 128):
            x, y = ids[start:start+128][None].cuda(), ids[start+1:start+129][None].cuda()
            active = mask[start:start+128][None].cuda()
            if bool(active.any()):
                row = learner.step(x, y, loss_mask=active, diagnose=diagnose and not updates)
                row['question_slot_retrieved'] = {str(layer): question_slots is not None and
                    question_slots[layer] in row['reads'][str(layer)]['slots'] for layer in learner.layers}
                updates.append(row)
            else:
                slots = learner.observe(x)
                if start == 0:
                    question_slots = slots
        return dict(id=qa['id'], accounting=accounting, updates=updates,
                    answer_nll=sum(r['supervised_nll_sum'] for r in updates)/accounting['answer_tokens'])

    @torch.no_grad()
    def evaluate():
        # Every question/answer variant starts from the same complete live
        # individual. Frozen local prompt writes are rolled back afterwards.
        source = learner.state_dict()
        rows = []
        try:
            for paragraph in task['validation'][:args.eval_paragraphs]:
                matrix = []
                for question in paragraph['qa']:
                    scores = []
                    for answer in paragraph['qa']:
                        learner.load_state_dict(source)
                        ids, mask, _ = windows(tokenizer, paragraph['context'], question['question'], answer['answer'])
                        total, count = 0., 0
                        for start in range(0, len(mask), 128):
                            x = ids[start:start+128][None].cuda()
                            _, logits = learner.observe(x, return_logits=True)
                            active = mask[start:start+128][None].cuda()
                            if bool(active.any()):
                                y = ids[start+1:start+129][None].cuda()
                                _, ce = language_loss(logits, y, active)
                                total += float(ce)*int(active.sum())
                                count += int(active.sum())
                        scores.append(dict(mean=total/count, nll=total, tokens=count))
                    matrix.append(scores)
                gap = sum(matrix[1-i][i]['mean']-matrix[i][i]['mean'] for i in range(2))/2
                rows.append(dict(ids=[q['id'] for q in paragraph['qa']], matrix=matrix, swap_gap=gap))
        finally:
            learner.load_state_dict(source)
        return dict(mean_answer_nll=sum(r['matrix'][i][i]['mean'] for r in rows for i in range(2))/(2*len(rows)),
            mean_question_swap_gap=sum(r['swap_gap'] for r in rows)/len(rows),
            paired_assignments_correct=sum(r['swap_gap'] > 0 for r in rows), paragraphs=len(rows), rows=rows)

    save()
    torch.cuda.reset_peak_memory_stats()
    try:
        for p, q in examples[:4]:
            train_example(p, q)
        shared = learner.state_dict()
        report['shared_warm_updates'] = learner.updates
        report['initial_validation'] = evaluate()
        save()
        print('INITIAL', json.dumps({k:v for k,v in report['initial_validation'].items() if k != 'rows'}), flush=True)
        for arm in ('full', 'held'):
            learner.load_state_dict(shared, detach_replay=arm == 'held')
            initial_updates = learner.updates
            report['arms'][arm] = dict(records=[])
            for index, (p, q) in enumerate(examples[4:4+args.train_examples]):
                started = time.perf_counter()
                row = train_example(p, q, diagnose=index == 0 and arm == 'full')
                row['seconds'] = time.perf_counter()-started
                report['arms'][arm]['records'].append(row)
                save()
                print(arm, index+1, round(row['answer_nll'], 5), flush=True)
            report['arms'][arm]['actual_optimizer_updates'] = learner.updates-initial_updates
            report['arms'][arm]['validation'] = evaluate()
            save()
            print('FINAL', arm, json.dumps({k:v for k,v in report['arms'][arm]['validation'].items() if k != 'rows'}), flush=True)
        full, held = [report['arms'][arm]['validation'] for arm in ('full', 'held')]
        report.update(status='complete',
            answer_nll_full_minus_held=full['mean_answer_nll']-held['mean_answer_nll'],
            question_swap_gap_full_minus_held=full['mean_question_swap_gap']-held['mean_question_swap_gap'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        save()
        print('COMPLETE', report['answer_nll_full_minus_held'], report['question_swap_gap_full_minus_held'], flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
