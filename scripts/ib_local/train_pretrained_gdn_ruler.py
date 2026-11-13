"""Matched answer-only RULER adaptation, separating recall from old credit.

Both arms preserve current-window training, live recurrence and reservoir
reads. Only derivatives into reexecuted old native computations differ.
This is a bounded task-learning comparison, not a convergence claim.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case, run_case
from scripts.ib_local.evaluate_pretrained_gdn_tasks import chunk_slices
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def supervised_sections(prefix, gold, eos):
    """Exactly128 supervised inputs; preceding prompt chunks are gradient-free.

    All real inputs execute once, with no padding. Short (<128) answers fit in
    the final128-token window; any variable-size remainder stays in the prompt.
    """
    sequence = prefix+gold+[eos]
    if len(gold)+1 > 128 or len(sequence)-1 < 256:
        raise ValueError('this admission supports short answers and >=256 real inputs')
    boundary = len(sequence)-1-128
    sections = [*chunk_slices(boundary),slice(boundary,len(sequence)-1)]
    for section in sections:
        positions = range(section.start+1,section.stop+1)
        yield (sequence[section.start:section.stop],sequence[section.start+1:section.stop+1],
               [p >= len(prefix) for p in positions])


def read_rows(data):
    if sha256(data['path']) != data['sha256']:
        raise ValueError('generated data changed')
    return [json.loads(x) for x in Path(data['path']).read_text().splitlines()]


@torch.no_grad()
def evaluate(learner,tokenizer,cells,score):
    incoming = learner.state_dict()
    results = []
    try:
        for data, rows in cells:
            samples = []
            for row in rows:
                learner.load_state_dict(incoming)
                samples.append(run_case(learner,tokenizer,row,data['max_new_tokens']))
            results.append(dict(length=data['length'],samples=samples,
                ruler_score=score([s['prediction'] for s in samples],[s['references'] for s in samples]),
                first_number_exact_count=sum(numeric_scores(s['prediction'],s['references'])['first_number_exact']
                    for s in samples),
                stopped_on_eos_count=sum(s['stopped_on_eos'] for s in samples),
                mean_eos_nll=sum(s['eos_nll'] for s in samples)/len(samples),
                answer_nll=sum(s['answer_nll']*s['gold_tokens'] for s in samples)/sum(s['gold_tokens'] for s in samples)))
            print(json.dumps(dict(kind='frozen_evaluation',length=data['length'],
                ruler_score=results[-1]['ruler_score'],answer_nll=results[-1]['answer_nll'])),flush=True)
    finally:
        learner.load_state_dict(incoming)
    assert_same(incoming,learner.state_dict())
    return results


def main():
    import importlib.util
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','train_manifest','test_manifest','official','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--examples',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--eval-limit',type=int,default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior experiment')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    train_manifest = json.loads(args.train_manifest.read_text())
    test_manifest = json.loads(args.test_manifest.read_text())
    training = [r for d in train_manifest['data'] for r in read_rows(d)][:args.examples]
    cells = [(d,read_rows(d)[:args.eval_limit]) for d in test_manifest['data']]
    if len(training) != args.examples or train_manifest['seed'] == test_manifest['seed']:
        raise ValueError('distinct sufficient train/test streams required')
    train_targets = {s for r in training for s in r['outputs']}
    test_targets = {s for _,rows in cells for r in rows for s in r['outputs']}
    if train_targets & test_targets:
        raise ValueError('random target overlap; refuse rather than select results')
    spec = importlib.util.spec_from_file_location('ruler_scores',args.official/'scripts/eval/synthetic/constants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state = payload.get('learner',payload)
    source_sha = sha256(args.checkpoint)
    model,_ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    report = dict(status='running',protocol=__doc__,source_updates=state['updates'],
        checkpoint=str(args.checkpoint),source_sha256=source_sha,
        training_manifest_sha256=sha256(args.train_manifest),test_manifest_sha256=sha256(args.test_manifest),
        train_seed=train_manifest['seed'],test_seed=test_manifest['seed'],examples_per_arm=len(training),
        bptt_window=128,train_native_layers=[i+1 for i in state['layers']],
        supervision='answer tokens and EOS only, no padding; native S continuous between training requests',
        answer_encoding='joint text/answer tokenization with independently verified prompt boundary',
        eval_protocol='independent frozen requests with empty native cache and restored full live reservoir',
        source_files={str(f):sha256(f) for f in [Path(__file__),
            Path('scripts/ib_local/pretrained_gdn_learning.py'),Path('scripts/ib_local/pretrained_gdn_experience.py'),
            Path('scripts/ib_local/evaluate_pretrained_gdn_ruler.py'),
            Path('scripts/ib_local/probe_pretrained_gdn_instruction.py'),
            Path('scripts/ib_local/analyze_ruler_generation.py')]},arms={})
    path = args.output/'report.json'
    write_json(path,report)
    start = time.time()
    try:
        for arm in ('held','full'):
            learner.load_state_dict(state,detach_replay=(arm=='held'))
            arm_report = dict(steps=[],evaluations=[])
            report['arms'][arm] = arm_report
            if arm == 'held':
                shared = evaluate(learner,tokenizer,cells,module.string_match_all)
                report['shared_initial_evaluation'] = shared
            for index,row in enumerate(training):
                if (args.output/'STOP').exists():
                    raise InterruptedError('user STOP requested')
                prefix,gold = encode_case(tokenizer,row)
                supervised = []
                prompt_observations = 0
                for x,y,mask in supervised_sections(prefix,gold,tokenizer.eos_token_id):
                    tensor = torch.tensor([x],device='cuda')
                    if any(mask):
                        result = learner.step(tensor,torch.tensor([y],device='cuda'),
                            loss_mask=torch.tensor([mask],device='cuda',dtype=torch.bool))
                        # Keep interpretable layer traffic, not every duplicated query receipt.
                        traffic = {layer:dict(queries=r['query_count'],
                            oldest_tokens=max(r['ages_tokens'],default=0),
                            old_read_participations=sum(a>128 for a in r['ages_tokens']))
                            for layer,r in result['reads'].items()}
                        supervised.append(dict(ce=result['ce'],sigreg=result['sigreg'],
                            supervised_tokens=result['supervised_tokens'],reads=traffic))
                    else:
                        learner.observe(tensor)
                        prompt_observations += 1
                item = dict(example=index+1,source_index=row['index'],prompt_tokens=len(prefix),
                    prompt_observations=prompt_observations,answer_tokens=len(gold)+1,
                    answer_distance_tokens=len(prefix)-row['token_position_answer'],updates=supervised,
                    total_updates=learner.updates)
                arm_report['steps'].append(item)
                print(json.dumps(dict(arm=arm,example=index+1,updates=learner.updates,
                    ce=[s['ce'] for s in supervised])),flush=True)
                write_json(path,report)
                if (index+1)%args.eval_every==0 or index+1==len(training):
                    result = evaluate(learner,tokenizer,cells,module.string_match_all)
                    arm_report['evaluations'].append(dict(example=index+1,results=result))
                    checkpoint = args.output/f'{arm}_last.pt'
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save(dict(learner=learner.state_dict(),examples=index+1),temporary)
                    if checkpoint.exists():
                        checkpoint.replace(args.output/f'{arm}_previous.pt')
                    temporary.replace(checkpoint)
                    arm_report['checkpoint'] = str(checkpoint)
                    write_json(path,report)
            # Fresh full arm is loaded from the same original state, never this terminal state.
        first = [report['arms'][a]['steps'][0]['updates'] for a in ('held','full')]
        assert_same(first[0],first[1])
        report.update(status='complete',first_forward_equal=True,
            source_checkpoint_unchanged=sha256(args.checkpoint)==source_sha,
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
            seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if not report['source_checkpoint_unchanged'] or not report['source_files_unchanged']:
            raise AssertionError('source changed during experiment')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),seconds=time.time()-start)
        raise
    finally:
        write_json(path,report)


if __name__ == '__main__':
    main()
