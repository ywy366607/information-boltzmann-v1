"""A diagnostic life with fixed slow parameters; native S and episodes still write.

Only to separate accumulated-state interference from feedback-driven parameter
drift if both learned arms fail. No new curriculum or learning algorithm.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.run_irreversible_credit import read_cases, save_latest
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream, PretrainedStreamBackend
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'pair_report', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = p.parse_args()
    pair = json.loads(args.pair_report.read_text())
    if pair['status'] != 'complete' or set(pair['arms']) != {'held', 'full'}:
        raise ValueError('finish the preregistered pair first')
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.time()
    report = dict(status='running', protocol=__doc__, cases=[], windows=[],
                  source_sha256=sha256(args.checkpoint), optimizer_updates=0,
                  question_resets=0, diagnostic_not_main_learner=True)
    write_json(args.output/'report.json', report)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    _, cases = read_cases(args.manifest, [512, 2048], 2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = payload.get('learner', payload)
    model, _ = load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
    learner.load_state_dict(state, detach_replay=False)
    learner.set_long_credit_cut(None)
    backend = PretrainedStreamBackend(learner)
    stream = IrreversibleCreditStream(backend, on_window=report['windows'].append)
    try:
        for index, (length, row) in enumerate(cases):
            prefix, _ = encode_case(tokenizer, row)
            stream.receive(prefix, phase='world')
            generated = stream.generate(eos_token_id=tokenizer.eos_token_id, max_tokens=16)
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            # The world still corrects; only slow parameter updates are off.
            text = '\nThe correct answer is '+', '.join(row['outputs'])+'.\n'
            stream.receive(tokenizer.encode(text, add_special_tokens=False), phase='feedback')
            report['cases'].append(dict(case=index, length=length, source_index=row['index'],
                prediction=prediction, references=row['outputs'], generated_ids=generated,
                **numeric_scores(prediction, row['outputs'])))
            report.update(events=stream.events, end_seen=learner.cache._seen_tokens,
                          optimizer_updates=learner.updates-state['updates'],
                          phase_counts=dict(stream.phase_counts), seconds=time.time()-started)
            write_json(args.output/'report.json', report)
            print('FIXED SLOW WEIGHTS', index+1, report['cases'][-1]['first_number_exact'],
                  repr(prediction), flush=True)
        save_latest(args.output/'latest.pt', learner, stream, len(cases), backend)
        report.update(status='complete', source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            first_number_exact_count=sum(c['first_number_exact'] for c in report['cases']),
            learned_arm_first_four={arm:sum(c['first_number_exact'] for c in d['cases'][:4])
                                   for arm,d in pair['arms'].items()},
            seconds=time.time()-started)
        if report['optimizer_updates'] != 0:
            raise AssertionError('diagnostic changed slow weights')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        backend.close()
        write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
