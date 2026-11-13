"""Paired local update test: preserve recall and cut only old native credit.

Every pair starts from one complete state. Both branches consume the same
current window, then predict the same untrained continuation without updates.
Only the held branch is retained to form the next common starting state.
This is a short update-route diagnostic, not a converged capability study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--future-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.pairs < 1 or args.future_tokens < 128 or args.future_tokens % 128:
        parser.error('positive pairs and future tokens a positive multiple of128 required')
    if args.output.exists():
        raise FileExistsError('preserve existing diagnostic evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    curriculum = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    seq = curriculum['stream']
    window, warm = 128, 4
    if len(seq) < (warm+args.pairs)*window+args.future_tokens+1:
        raise ValueError('insufficient real continuation')
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model, layers=(1, 12), capacity=32,
                                         detach_replay=True)
    report = dict(kind='paired_one_update_native_old_credit', status='running',
        seed=11, window=window, warm_updates=warm, pairs=args.pairs,
        future_tokens_per_pair=args.future_tokens, capacity=32, layers=[1, 12],
        unit='nats/token', curriculum_sha256=curriculum['stream_sha256'],
        dataset_revision=curriculum['dataset_revision'],
        intervention='same recall forward; detach old response before live encoder only in held',
        common_trajectory='held branch retained after each pair',
        primary='full minus held next-continuation NLL; negative favors old native credit',
        scope='overlapping short probes, not independent trials or convergence evidence',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=[])

    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output)

    save()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(warm):
            start = index*window
            learner.step(seq[start:start+window][None].cuda(),
                         seq[start+1:start+window+1][None].cuda())
        validation = curriculum['heldouts']['validation'][:args.future_tokens+1]
        for pair in range(args.pairs):
            started = time.perf_counter()
            start = (warm+pair)*window
            x = seq[start:start+window][None].cuda()
            target = seq[start+1:start+window+1][None].cuda()
            future = seq[start+window:start+window+args.future_tokens+1]
            before = learner.state_dict()
            learner.set_replay_gradient(False)
            full_record = learner.step(x, target, diagnose=pair == 0)
            full_future = learner.evaluate(future)
            full_validation = learner.evaluate(validation)
            full_native = {name: p.detach().cpu().clone() for name, p in learner.native}
            learner.load_state_dict(before, detach_replay=True)
            held_record = learner.step(x, target)
            # The optimizer is reached only after identical forward language
            # predictions and identical source/weight selections.
            for key in ('ce', 'nll_sum', 'sigreg', 'reads'):
                if full_record[key] != held_record[key]:
                    raise AssertionError(f'paired forward differs: {key}')
            held_future = learner.evaluate(future)
            held_validation = learner.evaluate(validation)
            differences = {}
            for name, p in learner.native:
                delta = full_native[name].float()-p.detach().float().cpu()
                differences[name] = dict(norm=float(delta.norm()), max_abs=float(delta.abs().max()),
                    changed_fraction=float((delta != 0).float().mean()))
            row = dict(pair=pair+1, common_updates_before=warm+pair,
                current_nll_full_minus_held=full_record['ce']-held_record['ce'],
                reads=full_record['reads'],
                future_full=full_future['nats_per_token'], future_held=held_future['nats_per_token'],
                future_full_minus_held=full_future['nats_per_token']-held_future['nats_per_token'],
                validation_full=full_validation['nats_per_token'],
                validation_held=held_validation['nats_per_token'],
                validation_full_minus_held=full_validation['nats_per_token']-held_validation['nats_per_token'],
                native_parameter_differences=differences,
                diagnostics=full_record['diagnostics'],
                experience_cpu_bytes=held_record['experience_cpu_bytes'],
                seconds=time.perf_counter()-started)
            report['records'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ('pair', 'future_full_minus_held',
                                                    'validation_full_minus_held', 'seconds')}), flush=True)
            del before, full_native
        rows = report['records']
        report.update(status='passed',
            mean_future_full_minus_held=sum(r['future_full_minus_held'] for r in rows)/len(rows),
            mean_validation_full_minus_held=sum(r['validation_full_minus_held'] for r in rows)/len(rows),
            future_improved_pairs=sum(r['future_full_minus_held'] < 0 for r in rows),
            validation_improved_pairs=sum(r['validation_full_minus_held'] < 0 for r in rows),
            max_source_age_tokens=max(age for r in rows for read in r['reads'].values()
                                      for age in read['ages_tokens']),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            executed_optimizer_updates=warm+2*args.pairs,
            retained_common_updates=warm+args.pairs,
            note='full branch optimizer steps discarded after paired observations; no live navigation state changed')
        save()
        print('PASSED', flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
