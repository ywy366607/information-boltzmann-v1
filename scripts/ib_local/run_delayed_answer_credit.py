"""World feedback settles actual emitted predictions in one irreversible life."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from scripts.ib_local.delayed_prediction_credit import PendingAnswer, settle_answer, align_numeric_feedback
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream, PretrainedStreamBackend
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


def save(path, learner, stream, next_case, pending=None):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(learner=learner.state_dict(), stream=stream.state_dict(),
                    next_case=next_case, pending=pending), temporary)
    temporary.replace(path)


def gradient_difference(full, held):
    a, b = full.pop('_native_gradients'), held.pop('_native_gradients')
    result = {}
    for layer in ('layer1.', 'layer12.'):
        names = [n for n in a if n.startswith(layer)]
        difference = sum(float((a[n]-b[n]).square().sum()) for n in names)**.5
        norm = sum(float(a[n].square().sum()) for n in names)**.5
        result[str(int(layer[5:-1])+1)] = dict(old_native_gradient_norm=difference,
                                                full_native_gradient_norm=norm)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--examples', type=int, default=16)
    p.add_argument('--admission-only', action='store_true')
    p.add_argument('--capture-pending-only', action='store_true',
                   help='Continue one actual answer and save before feedback; no slow update.')
    p.add_argument('--capture-full-window', action='store_true')
    p.add_argument('--capture-joint',action='store_true')
    p.add_argument('--feedback-contract', choices=('token_positions','answer_content'),default='token_positions')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.time()
    report = dict(status='loading', protocol=__doc__, arms={},
        source_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
        source=str(args.checkpoint), start=args.start, examples=args.examples,
        admission_only=args.admission_only,capture_pending_only=args.capture_pending_only,
        capture_full_window=args.capture_full_window,answer_contract=args.feedback_contract,
        source_files={str(Path(f)):sha256(f) for f in [__file__,
            'scripts/ib_local/delayed_prediction_credit.py',
            'scripts/ib_local/irreversible_credit_stream.py',
            'scripts/ib_local/pretrained_gdn_learning.py',
            'scripts/ib_local/pretrained_gdn_experience.py',
            'scripts/ib_local/prepare_pretrained_gdn_ruler.py']}, new_lean_coverage=0)
    for filename in report['source_files']:
        source_file = Path(filename).resolve()
        archived = args.output/'executed_sources'/source_file.relative_to(Path.cwd())
        archived.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_file,archived)

    def write():
        report['seconds'] = time.time()-started
        write_json(args.output/'report.json', report)

    write()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    count = 1 if args.capture_pending_only else (2 if args.admission_only else args.examples)
    manifest, all_cases = read_cases(args.manifest, [2048], args.start+count)
    cases = all_cases[args.start:args.start+count]
    report['task_seed'] = manifest['seed']
    report['official_revision'] = manifest['official_revision']
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('next_case', args.start) != args.start:
        raise ValueError('continue from the committed case cursor; do not skip or repeat history')
    source = payload['learner']
    model, _ = load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
    backend = None
    try:
        for arm in (('full',) if args.admission_only or args.capture_pending_only else ('held', 'full')):
            learner.load_state_dict(source, detach_replay=False)
            if args.capture_full_window:
                learner.enable_full_window_participation()
            if args.capture_joint:
                learner.enable_joint_participation()
            learner.set_long_credit_cut(128 if arm == 'held' else None)
            backend = PretrainedStreamBackend(learner)
            arm_report = dict(cases=[], windows=[], source_seen=learner.cache._seen_tokens,
                              source_updates=learner.updates)
            report['arms'][arm] = arm_report
            stream = IrreversibleCreditStream(backend, on_window=arm_report['windows'].append)
            if 'stream' in payload:
                stream.restore_latest(payload['stream'])
            source_events = stream.events
            report['status'] = 'running'
            write()
            for index, (_, row) in enumerate(cases):
                if (args.output/'STOP').exists():
                    raise InterruptedError('STOP requested')
                before_seen, before_updates = learner.cache._seen_tokens, learner.updates
                prefix, gold = encode_case(tokenizer, row)
                backend.set_prediction_capture(True)
                stream.receive(prefix, phase='world', teach=False)
                pending = PendingAnswer(capacity=16)
                emitted = stream.generate(eos_token_id=tokenizer.eos_token_id,
                                          max_tokens=16, pending=pending)
                backend.set_prediction_capture(False)
                prediction = tokenizer.decode(emitted, skip_special_tokens=True)
                if args.capture_pending_only:
                    backend.close()
                    save(args.output/'pending_recovery.pt',learner,stream,args.start+index,pending)
                    report.update(status='awaiting_external_feedback',retained_new_updates=learner.updates-before_updates,
                        pending_case=dict(case=args.start+index,prediction=prediction,references=row['outputs'],
                            incoming_seen=before_seen,outgoing_seen=learner.cache._seen_tokens,
                            events=stream.events,pending_cpu_bytes=pending.nbytes),
                        source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
                        source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()))
                    print('PENDING',args.start+index,repr(prediction),'seen',learner.cache._seen_tokens,flush=True)
                    return
                if args.feedback_contract=='answer_content':
                    teaching_positions,teaching_labels=align_numeric_feedback(pending,tokenizer,row['outputs'][0])
                else:
                    teaching_positions,teaching_labels=None,gold+[tokenizer.eos_token_id]
                feedback = tokenizer.encode('\nThe correct answer is '+', '.join(row['outputs'])+'.\n',
                                            add_special_tokens=False)
                if args.admission_only and index == 1:
                    backend.close()
                    save(args.output/'pending_recovery.pt', learner, stream, args.start+index, pending)
                stream.receive(feedback, phase='feedback', teach=False)
                backend.close()
                live_before = learner.state_dict()['cache'] if args.admission_only else None
                if args.admission_only and index == 0:
                    learner.set_long_credit_cut(None)
                    full = settle_answer(learner, pending, teaching_labels,positions=teaching_positions,
                        feedback_seen=learner.cache._seen_tokens, apply=False, return_native_gradients=True)
                    learner.set_long_credit_cut(128)
                    held = settle_answer(learner, pending, teaching_labels,positions=teaching_positions,
                        feedback_seen=learner.cache._seen_tokens, apply=False, return_native_gradients=True)
                    differences = gradient_difference(full, held)
                    if full['loss_ce'] != held['loss_ce']:
                        raise AssertionError('gradient-only control changed delayed forward')
                    report['gradient_admission'] = dict(layers=differences,
                        full_held_ce_equal=True, full_ce=full['loss_ce'], held_ce=held['loss_ce'])
                    learner.set_long_credit_cut(None)
                result = settle_answer(learner, pending, teaching_labels,positions=teaching_positions,
                                       feedback_seen=learner.cache._seen_tokens)
                max_ce_error = max(abs(r['ce']-r['lived_ce']) for r in result['rows'])
                original_error = max(r['inference_logit_max_difference'] for r in result['rows'])
                if result['replay_logit_max_difference'] > .125 or max_ce_error > .02 or original_error > .125:
                    raise AssertionError(f'held prediction replay mismatch: {max_ce_error}, '
                                         f'{result["replay_logit_max_difference"]}')
                if args.admission_only:
                    assert_same(live_before, learner.state_dict()['cache'])
                item = dict(case=args.start+index, source_index=row['index'], prediction=prediction,
                    references=row['outputs'], emitted=emitted, **numeric_scores(prediction, row['outputs']),
                    incoming_seen=before_seen, outgoing_seen=learner.cache._seen_tokens,
                    updates=learner.updates-before_updates, settlement=result,
                    max_target_ce_replay_error=max_ce_error, events=stream.events,
                    history_chain=stream.chain)
                arm_report['cases'].append(item)
                arm_report.update(end_seen=learner.cache._seen_tokens, end_updates=learner.updates,
                                  events=stream.events, phase_counts=dict(stream.phase_counts))
                if learner.cache._seen_tokens != arm_report['source_seen']+stream.events-source_events:
                    raise AssertionError('local delayed reexecution advanced real time')
                print('DELAYED', arm, args.start+index+1, item['first_number_exact'], repr(prediction),
                      'ce', result['loss_ce'], 'replay', max_ce_error, flush=True)
                backend.close()
                save(args.output/f'{arm}_latest.pt', learner, stream, args.start+index+1)
                write()
                if args.admission_only and index == 1:
                    accepted = learner.state_dict()
                    accepted_stream = stream.state_dict()
                    saved = torch.load(args.output/'pending_recovery.pt', map_location='cpu',
                                       weights_only=False, mmap=True)
                    learner.load_state_dict(saved['learner'])
                    backend = PretrainedStreamBackend(learner)
                    replay_stream = IrreversibleCreditStream(backend)
                    replay_stream.restore_latest(saved['stream'])
                    replay_stream.receive(feedback, phase='feedback', teach=False)
                    backend.close()
                    again = settle_answer(learner, saved['pending'], teaching_labels,positions=teaching_positions,
                                          feedback_seen=learner.cache._seen_tokens)
                    assert_same(result, again, 'delayed_recovery_metrics')
                    assert_same(accepted, learner.state_dict(), 'delayed_recovery_body')
                    assert_same(accepted_stream, replay_stream.state_dict(), 'delayed_recovery_stream')
                    report['disk_pending_recovery_equal'] = True
                    report['discarded_research_continuations'] = 1
            arm_report['correct'] = sum(c['first_number_exact'] for c in arm_report['cases'])
            arm_report['count'] = len(arm_report['cases'])
            backend.close()
            backend = None
            write()
        report.update(status='complete', source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if not report['source_unchanged'] or not report['source_files_unchanged']:
            raise AssertionError('source changed during execution')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        write()


if __name__ == '__main__':
    main()
