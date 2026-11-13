"""Frozen null-use versus whole learned-trajectory decomposition on real text.

No optimizer step, episode write, state transplant, or persistent flag change.
Each evaluation starts from its checkpoint's complete saved recurrent state.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


@contextmanager
def null_mode(learner, enabled):
    """Toggle only the read choice; restore flags even when a probe fails."""
    saved = {i: m.null_read_enabled for i, m in learner.memories.items()}
    try:
        for memory in learner.memories.values():
            memory.null_read_enabled = enabled
        yield
    finally:
        for i, memory in learner.memories.items():
            memory.null_read_enabled = saved[i]


def decompose(no_null_off, null_off, null_on):
    """Exact endpoint accounting, not a weights-only attribution."""
    direct = null_on - null_off
    trajectory = null_off - no_null_off
    total = null_on - no_null_off
    return dict(direct_null_use=direct, whole_learned_trajectory=trajectory,
                total=total, accounting_error=total-(direct+trajectory))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--pair-dir', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--published', type=Path)
    parser.add_argument('--tokens', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    if args.tokens < 128 or args.tokens % 128:
        parser.error('positive 128-aligned evaluation budget required')
    torch.set_num_threads(1)
    curriculum = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    if any(len(ids) < args.tokens+1 for ids in curriculum['heldouts'].values()):
        raise ValueError('heldout budget exceeds genuine text')
    report = dict(status='running', optimizer_updates=0, episode_writes=0,
        protocol=dict(tokens_per_source=args.tokens, window=128,
            cache='each frozen trajectory begins from its own saved full cache; no reset or bank transplant',
            intervention='null-trained body: same weights, moments, records and incoming cache; only enable flag toggled',
            trajectory='all differences accumulated by the paired training histories, including recurrent state and episode contents; not weights alone',
            interpretation='negative NLL differences are better; finite endpoint diagnostic, not new training'),
        dataset_revision=curriculum['dataset_revision'], stages={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = None
    try:
        for suffix in ('previous', 'last'):
            stage = dict(cells={}, sources={})
            report['stages'][suffix] = stage
            for arm in ('no_null', 'null'):
                path = args.pair_dir / f'{arm}_{suffix}.pt'
                digest = sha256(path)
                payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                state = payload['learner']
                if learner is None:
                    learner = PretrainedExperienceLearner(model, layers=state['layers'], capacity=state['capacity'])
                learner.load_state_dict(state)
                before = learner.state_dict()
                original_flags = [m.null_read_enabled for m in learner.memories.values()]
                if original_flags != [arm == 'null']*len(original_flags):
                    raise AssertionError('source arm and persisted null flags disagree')
                stage['sources'][arm] = dict(path=str(path), sha256=digest,
                    updates=state['updates'], step=payload['step'])
                for enabled in ((False,) if arm == 'no_null' else (False, True)):
                    label = f'{arm}_trained_' + ('on' if enabled else 'off')
                    with null_mode(learner, enabled):
                        values = {name: learner.evaluate(ids[:args.tokens+1])
                                  for name, ids in curriculum['heldouts'].items()}
                    assert_same(before, learner.state_dict(), 'frozen_complete_state')
                    stage['cells'][label] = values
                    print(suffix, label, {k: v['nats_per_token'] for k,v in values.items()}, flush=True)
                    write_json(args.output, report)
                if sha256(path) != digest:
                    raise AssertionError('source checkpoint changed')
                del before, state, payload
            if stage['sources']['null']['step'] != stage['sources']['no_null']['step']:
                raise AssertionError('paired endpoint steps differ')
            stage['decomposition'] = {name: decompose(
                stage['cells']['no_null_trained_off'][name]['nats_per_token'],
                stage['cells']['null_trained_off'][name]['nats_per_token'],
                stage['cells']['null_trained_on'][name]['nats_per_token'])
                for name in curriculum['heldouts']}
            write_json(args.output, report)
        report.update(status='complete', complete_persistent_states_unchanged=True,
                      all_source_checkpoints_unchanged=True)
        reference_path = args.pair_dir / 'pair.json'
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding='utf-8'))
            drift = []
            for stage in report['stages'].values():
                for arm in ('no_null', 'null'):
                    original = next(row for row in reference['arms'][arm]['evaluations']
                                    if row['step'] == stage['sources'][arm]['step'])
                    cell = stage['cells'][f'{arm}_trained_' + ('on' if arm == 'null' else 'off')]
                    for name, value in cell.items():
                        old = original['metrics'][name]
                        if old['tokens'] != value['tokens']:
                            raise ValueError('reference evaluation token budgets differ')
                        drift.append(dict(arm=arm, step=original['step'], source=name,
                            recomputed_minus_original=value['nats_per_token']-old['nats_per_token']))
            report['cross_process_baseline_recheck'] = dict(records=drift,
                maximum_absolute_difference=max(abs(r['recomputed_minus_original']) for r in drift),
                note='all intervention differences use newly recomputed same-process cells; historical scores are not subtraction baselines')
        write_json(args.output, report)
        if args.published:
            args.published.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.published, report)
        print('COMPLETE', report['stages']['last']['decomposition'], flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
