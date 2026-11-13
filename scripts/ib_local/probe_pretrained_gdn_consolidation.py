"""A real-text numerical audit of read-credit-driven consolidation.

Resume a trained complete checkpoint in a disposable learner. Perform one
128-token OWT update, then restore and repeat that update exactly. Measure
signed preference and retention changes, not a long-term capability gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.experience_consolidation import bounded_preference, replacement_slot
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same


def copy_tree_for_migration(state):
    """Copy metadata only; no tensor or observed experience is changed."""
    expected = dict(state, memories=dict(state['memories']))
    for layer, payload in state['memories'].items():
        updated = dict(payload)
        metadata = dict(payload['memory._extra_state'])
        metadata.update(consolidation_enabled=True, consolidation_rule='retention')
        updated['memory._extra_state'] = metadata
        expected['memories'][layer] = updated
    return expected


def copy_tree_for_null_migration(state,enabled_state):
    """Only the new null key, bias and enable flag may differ."""
    expected = dict(state,memories=dict(state['memories']))
    for layer,payload in state['memories'].items():
        updated = dict(payload)
        updated['null_key'] = enabled_state['memories'][layer]['null_key']
        updated['null_bias'] = enabled_state['memories'][layer]['null_bias']
        metadata = dict(payload['_extra_state'])
        metadata['null_read_enabled'] = True
        updated['_extra_state'] = metadata
        expected['memories'][layer] = updated
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--consolidation-rule',choices=('preference','evidence','retention'),default='retention')
    parser.add_argument('--stream-offset', type=int,
                        help='Use the next real stream window instead of the prior a_unseen probe')
    parser.add_argument('--save-migrated', type=Path,
                        help='Save the explicitly upgraded zero-new-update complete state')
    parser.add_argument('--enable-null',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve diagnostic evidence')
    if args.save_migrated and args.save_migrated.exists():
        raise FileExistsError('preserve existing migrated individual')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    # Keep mmap owner and original complete individual alive throughout audit.
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    curriculum = torch.load(args.curriculum, map_location='cpu', weights_only=False)
    ids = (curriculum['heldouts']['a_unseen'][:129] if args.stream_offset is None else
           curriculum['stream'][args.stream_offset:args.stream_offset+129])
    if len(ids) != 129:
        raise ValueError('one actual 128-token input/target window required')
    report = dict(kind='read_credit_consolidation_real_text_wiring', status='running',
        checkpoint=str(args.checkpoint), seed='restored checkpoint RNG',
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_updates=source['updates'], layers=list(source['layers']),
        consolidation_rule=args.consolidation_rule,
        capacity=source['capacity'], real_tokens=128, executed_updates=0,
        training_data=('OWT a_unseen continuation, not validation' if args.stream_offset is None
                       else f'OWT real stream offset {args.stream_offset}'),
        token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypothesis='current outcome changes both read preference and age-adjusted retention',
        scope='one real update plus exact repeat; not learned long-term latch or NLL advantage')

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        tmp.replace(args.output)

    save()
    try:
        model, _ = load_pure_gdn(args.model)
        enable_streaming_convolution(model)
        learner = PretrainedExperienceLearner(model, layers=source['layers'], capacity=source['capacity'])
        learner.load_state_dict(source, detach_replay=False)
        x, target = ids[:-1][None].cuda(), ids[1:][None].cuda()
        baseline = learner.state_dict()
        if args.consolidation_rule == 'retention':
            with torch.no_grad():
                old_logits, _ = learner.predict(x)
            learner.enable_credit_retention()
            with torch.no_grad():
                new_logits, _ = learner.predict(x)
            report['migration_forward_max_difference'] = float((old_logits-new_logits).abs().max())
            if report['migration_forward_max_difference'] != 0:
                raise AssertionError('retention migration changed current prediction')
            del old_logits, new_logits
        else:
            for wrapper in learner.memories.values():
                wrapper.memory.consolidation_enabled = True
                wrapper.memory.consolidation_rule = args.consolidation_rule
        before = learner.state_dict()
        if args.consolidation_rule == 'retention':
            expected = copy_tree_for_migration(baseline)
            assert_same(expected, before, 'migration_only_changes_retention_policy')
            report['migration_only_changes_retention_policy'] = True
            del expected
        del baseline
        if args.enable_null:
            with torch.no_grad():
                logits,_ = learner.predict(x)
                _,prior_ce = language_loss(logits,target)
            learner.enable_null_read()
            enabled = learner.state_dict()
            expected = copy_tree_for_null_migration(before,enabled)
            assert_same(expected,enabled,'null_migration_preserves_existing_individual')
            report['null_migration_preserves_existing_individual'] = True
            report['ce_before_null'] = float(prior_ce)
            report['new_null_parameters'] = sum(p.numel() for p in learner.null_parameters)
            before = enabled
            del expected,enabled,logits
        if args.save_migrated:
            args.save_migrated.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.save_migrated.with_suffix('.tmp')
            torch.save(before, tmp)
            tmp.replace(args.save_migrated)
            restored = torch.load(args.save_migrated, map_location='cpu', weights_only=False, mmap=True)
            assert_same(before, restored, 'migrated_disk_roundtrip')
            del restored
            report['migrated_checkpoint'] = str(args.save_migrated)
            report['migrated_new_updates'] = 0
            report['migration_disk_restore_exact'] = True
        prior = {}
        for layer, wrapper in learner.memories.items():
            m = wrapper.memory
            prior[layer] = dict(preference=bounded_preference(m).clone(),
                                version=m.read_credit_version.clone(),
                                occupied=m.occupied.clone(), bytes=wrapper.storage_nbytes)
        torch.cuda.reset_peak_memory_stats()
        row = learner.step(x, target, diagnose=True)
        report['executed_updates'] = 1
        after = learner.state_dict()
        layers = {}
        for layer, wrapper in learner.memories.items():
            m, old = wrapper.memory, prior[layer]
            same = old['occupied'] & m.occupied & (old['version'] == m.read_credit_version)
            current = bounded_preference(m)
            change = current - old['preference']
            multiplier_delta = current.exp() - old['preference'].exp()
            ages = m.clock - m.birth
            active = same & (change != 0)
            oldest = int(m.last_write.argmin())
            chosen = replacement_slot(m)
            layers[str(layer)] = dict(
                changed_surviving_records=int(active.sum()),
                changed_over_128_tokens=int((active & (ages > 128)).sum()),
                strengthened=int((same & (change > 0)).sum()),
                weakened=int((same & (change < 0)).sum()),
                max_preference_change=float(change[same].abs().max()) if bool(same.any()) else 0.,
                max_lifetime_multiplier_change=float(multiplier_delta[same].abs().max()) if bool(same.any()) else 0.,
                observed_multiplier_min=float(current.exp().min()),
                observed_multiplier_max=float(current.exp().max()),
                next_credit_victim=chosen, next_oldest_victim=oldest,
                victim_differs_from_oldest=chosen != oldest,
                preference=current.tolist(),
                cpu_storage_bytes=wrapper.storage_nbytes,
                storage_bytes_delta=wrapper.storage_nbytes-old['bytes'])
        report.update(layers=layers, update=row,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        if args.enable_null:
            report['null_update'] = {str(layer):dict(
                bias_change=float(wrapper.null_bias-before['memories'][layer]['null_bias']),
                key_change_norm=float((wrapper.null_key-before['memories'][layer]['null_key']).norm()),
                real_credit_sum=row['credit'][str(layer)]['credit_sum'],
                null_bias_gradient=row['credit'][str(layer)]['null_bias_gradient'])
                for layer,wrapper in learner.memories.items()}
            for layer,values in report['null_update'].items():
                if abs(values['real_credit_sum']-values['null_bias_gradient']) > 1e-5:
                    raise AssertionError('real credit lost its common competition with null')
            report['real_credit_balances_null_gradient'] = True
        # Restore all weights, optimizer, episodes, cache and RNG. Repeat only
        # for numerical acceptance; do not retain a second chronological step.
        learner.load_state_dict(before)
        repeated = learner.step(x, target, diagnose=True)
        report['executed_updates'] = 2
        assert_same(row, repeated, 'update_record')
        assert_same(after, learner.state_dict(), 'complete_next_update')
        report.update(status='passed', next_update_restore_exact=True,
                      snapshot_file_written=bool(args.save_migrated), old_individual_unchanged=True,
                      source_checkpoint_unchanged=(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                                                   == report['checkpoint_sha256']),
                      retained_new_updates=0, seconds=time.perf_counter()-started)
        save()
        print(json.dumps({key: report[key] for key in ('status', 'layers', 'next_update_restore_exact', 'seconds')}), flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error), seconds=time.perf_counter()-started)
        save()
        raise


if __name__ == '__main__':
    main()
