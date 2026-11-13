"""Continuous, self-hearing RULER route check on the existing pure GDN.

Each arm starts once from the same full source, then never restores between
questions. Real generated tokens stay in its state. External feedback teaches;
self-produced tokens do not. Held cuts only native replay derivatives age>128.
This is a bounded route validation, not a converged capability benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.irreversible_credit_stream import (
    IrreversibleCreditStream, PretrainedStreamBackend,
)
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def read_cases(manifest_path, lengths, count):
    manifest = json.loads(manifest_path.read_text())
    cells = {}
    for item in manifest['data']:
        if item['task'] == 'niah_single_1' and item['length'] in lengths:
            if sha256(item['path']) != item['sha256']:
                raise ValueError('source task data changed')
            rows = [json.loads(line) for line in Path(item['path']).read_text().splitlines()]
            if len(rows) < count:
                raise ValueError('insufficient preregistered cases')
            cells[item['length']] = rows[:count]
    if set(cells) != set(lengths):
        raise ValueError('missing configured task lengths')
    # Interleave lengths; the same natural life order is used in both arms.
    return manifest, [(length, cells[length][i]) for i in range(count) for length in lengths]


def save_latest(path, learner, stream, next_case, backend):
    backend.close()
    payload = dict(learner=learner.state_dict(), stream=stream.state_dict(), next_case=next_case)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--cases-per-length', type=int, default=8)
    parser.add_argument('--generation-tokens', type=int, default=16)
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.time()
    report = dict(status='loading', protocol=__doc__, arms={}, new_lean_coverage=0,
                  started_at=started, source_sha256=sha256(args.checkpoint),
                  manifest_sha256=sha256(args.manifest), window=128,
                  lengths=args.lengths, cases_per_length=args.cases_per_length,
                  admission_only=args.admission_only,
                  feedback_only_teaching=True, question_resets=0,
                  generation_tokens=args.generation_tokens,
                  source_files={str(Path(f)): sha256(f) for f in
                    [__file__, 'scripts/ib_local/irreversible_credit_stream.py',
                     'scripts/ib_local/pretrained_gdn_learning.py',
                     'scripts/ib_local/pretrained_gdn_experience.py']})
    report_path = args.output/'report.json'

    def write():
        report['seconds'] = time.time()-started
        write_json(report_path, report)

    write()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest, cases = read_cases(args.manifest, args.lengths, args.cases_per_length)
    report['task_seed'] = manifest['seed']
    report['official_revision'] = manifest['official_revision']
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    source = payload.get('learner', payload)
    model, loading = load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
    report.update(status='running', weight_loading=loading,
                  native_layers=[i+1 for i in source['layers']],
                  source_updates=source['updates'], source_seen_tokens=source['cache']['seen_tokens'])
    backend = None
    try:
        for arm in (('full',) if args.admission_only else ('held', 'full')):
            learner.load_state_dict(source, detach_replay=False)
            learner.set_long_credit_cut(128 if arm == 'held' else None)
            backend = PretrainedStreamBackend(learner)
            arm_report = dict(cases=[], windows=[], start_seen=learner.cache._seen_tokens,
                              start_updates=learner.updates)
            report['arms'][arm] = arm_report

            def window_done(item):
                # Keep per-layer statistics, omit repeated per-query receipts.
                reads = {}
                for layer, r in item.get('reads', {}).items():
                    reads[layer] = dict(queries=r['query_count'], unique_replays=r['unique_replays'],
                        old_reads=sum(a > 128 for a in r['ages_tokens']),
                        all_reads=len(r['ages_tokens']),
                        oldest_tokens=max(r['ages_tokens'], default=0))
                compact = {k:v for k,v in item.items() if k not in ('reads', 'credit')}
                compact['reads'] = reads
                arm_report['windows'].append(compact)

            stream = IrreversibleCreditStream(backend, on_window=window_done)
            count = min(2, len(cases)) if args.admission_only else len(cases)
            for index, (length, row) in enumerate(cases[:count]):
                if (args.output/'STOP').exists():
                    raise InterruptedError('STOP requested; last committed life is preserved')
                incoming_events, incoming_seen = stream.events, learner.cache._seen_tokens
                prefix, _ = encode_case(tokenizer, row)
                pre_updates = learner.updates
                # Same source prompt and candidate limit; no answer enters here.
                prompt_scores = stream.receive(prefix, phase='world', teach=False)
                before_speech = learner.cache._seen_tokens
                generated = stream.generate(eos_token_id=tokenizer.eos_token_id,
                                            max_tokens=args.generation_tokens)
                after_speech = learner.cache._seen_tokens
                if after_speech-before_speech != len(generated):
                    raise AssertionError('emitted tokens were not committed exactly once')
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                # Correction is external language, after the actual answer.
                feedback = '\nThe correct answer is '+', '.join(row['outputs'])+'.\n'
                feedback_ids = tokenizer.encode(feedback, add_special_tokens=False)
                feedback_scores = stream.receive(feedback_ids, phase='feedback', teach=True)
                seen = learner.cache._seen_tokens
                if seen != source['cache']['seen_tokens'] + stream.events:
                    raise AssertionError('native lifetime clock diverged from actual events')
                item = dict(case=index, source_index=row['index'], length=length,
                    prediction=prediction, generated_ids=generated,
                    references=row['outputs'], **numeric_scores(prediction, row['outputs']),
                    incoming_events=incoming_events, outgoing_events=stream.events,
                    incoming_seen=incoming_seen, outgoing_seen=seen,
                    self_heard_tokens=len(generated), feedback_tokens=len(feedback_ids),
                    updates=learner.updates-pre_updates,
                    prompt_nll=(sum(prompt_scores)/len(prompt_scores) if prompt_scores else None),
                    feedback_nll=sum(feedback_scores)/len(feedback_scores),
                    pending_inputs=len(stream.inputs), pending_targets=len(stream.targets),
                    history_chain=stream.chain)
                arm_report['cases'].append(item)
                arm_report.update(end_seen=seen, end_updates=learner.updates,
                                  events=stream.events, phase_counts=dict(stream.phase_counts))
                print('LIFE', arm, index+1, length, item['first_number_exact'],
                      repr(prediction), 'updates', item['updates'], flush=True)
                write()
            checkpoint = args.output/f'{arm}_latest.pt'
            saved = save_latest(checkpoint, learner, stream, count, backend)
            arm_report['checkpoint'] = str(checkpoint)
            arm_report['first_number_exact_count'] = sum(c['first_number_exact'] for c in arm_report['cases'])
            arm_report['case_count'] = len(arm_report['cases'])
            if args.admission_only:
                # Only fault-recovery validation uses a separate researcher
                # continuation; the accepted admission life ends at saved.
                from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same
                tail = tokenizer.encode('\nContinue reading the following text.\n', add_special_tokens=False)
                tail = (tail * 20)[:160]
                a_before = learner.updates
                a_metrics = []
                stream.on_window = a_metrics.append
                stream.receive(tail, phase='feedback', teach=True)
                state_a = learner.state_dict()
                stream_a = stream.state_dict()
                backend.close()
                disk = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                learner.load_state_dict(disk['learner'])
                backend = PretrainedStreamBackend(learner)
                resumed = IrreversibleCreditStream(backend)
                resumed.restore_latest(disk['stream'])
                resumed.receive(tail, phase='feedback', teach=True)
                assert_same(state_a, learner.state_dict())
                assert_same(stream_a, resumed.state_dict())
                report['admission'] = dict(complete_disk_next_update_equal=True,
                    discarded_research_continuations=2, researcher_updates=learner.updates-a_before,
                    generated_history_preserved=True,
                    maximum_reexecution_ce_difference=max((abs(x['reexecution_ce_difference'])
                        for x in arm_report['windows'] if x['reexecution_ce_difference'] is not None), default=0))
            backend.close()
            backend = None
            del saved
            write()
        report.update(status='complete', source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if not report['source_unchanged'] or not report['source_files_unchanged']:
            raise AssertionError('execution source changed')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        write()


if __name__ == '__main__':
    main()
