"""Read-only four-pillar report for a continuing timing-repaired OWT fly.

Consumes actual pre-update logs. Uses no model forward, GPU allocation,
optimizer step, state reset or retrospective rescoring by newer weights.
The fixed train-only unigram reference controls marginal token difficulty;
it is a reporting reference, not an evaluation learner or complete task control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_encounter(row, validation, reference_surprisal, *, window=32):
    b = np.asarray(row['B_curve'], dtype=np.float64)
    a1 = np.asarray(row['A1_curve'], dtype=np.float64)
    a2 = np.asarray(row['A2_replay_curve'], dtype=np.float64)
    if len(a1) != len(a2) or len(a1) < 2 or not len(b):
        raise ValueError('Recorded first-pass/revisit score lengths are invalid')
    if not all(np.isfinite(values).all() for values in (b, a1, a2)):
        raise ValueError('Nonfinite recorded score')
    end = int(row['fresh_validation_cursor'])
    start = end - len(b)
    if start < 0 or end > len(validation):
        raise ValueError('Validation reference slice lies outside the actual stream')
    reference = reference_surprisal[validation[start:end]]
    gain = reference - b
    ticks = int(row.get('physical_ticks_per_input', 1))
    events = row['event_interval'][1] - row['event_interval'][0]
    if events != len(b) + len(a2):
        raise ValueError('Actual exposure counter disagrees with B and A2 scores')
    if 'physical_tick_interval' in row:
        physical = row['physical_tick_interval'][1] - row['physical_tick_interval'][0]
        if physical != events * ticks:
            raise ValueError('Physical exposure counter disagrees with timing')
    updates = row['update_interval'][1] - row['update_interval'][0]
    if events % window or updates != events // window:
        raise ValueError('Recorded optimizer cadence changed within the encounter')
    # Returning to A[0] scores a B->A bridge. Pairs A[0]->A[1] onward are
    # identical to A1; the first return window has not yet updated its weights.
    first_window_end = min(window, len(a1))
    matched_opening = float((a1[1:first_window_end] - a2[1:first_window_end]).mean())
    block = row.get('recovery', {}).get('block_tokens', 16)
    residual = b - reference
    blocks = [float(residual[i:i+block].mean())
              for i in range(0, len(b)-block+1, block)]
    residual_drop = blocks[0] - float(np.mean(blocks[-2:])) if len(blocks) >= 3 else None
    return {
        'bptt_train_tokens': row['bptt_train_tokens'],
        'timing': 'legacy_one_tick' if ticks == 1 else f'pulse_plus_{ticks-1}_quiet',
        'physical_ticks_per_input': ticks,
        'writer_baseline_clock': row.get('writer_baseline_clock', 'physical'),
        'actual_exposure': {'input_events': events, 'physical_ticks': events*ticks,
                            'optimizer_updates': updates},
        'first_pass': {
            'tokens': len(b), 'validation_range': [start, end],
            'model_nll': float(b.mean()), 'fixed_unigram_nll': float(reference.mean()),
            'gain_over_fixed_unigram': float(gain.mean()),
            'gain_per_optimizer_window': [float(gain[i:i+window].mean())
                                          for i in range(0, len(b), window)]},
        'adaptation': {
            'recorded_recovery': row.get('recovery'),
            'reference_relative_block_curve': blocks,
            'reference_relative_opening_to_late_drop': residual_drop,
            'scope': 'descriptive trajectory; reference controls marginal difficulty only'},
        'revisit': {
            'matched_pairs': len(a1)-1,
            'matched_nll_gain': float((a1[1:] - a2[1:]).mean()),
            'first_return_window_matched_pairs': first_window_end-1,
            'first_return_window_preupdate_nll_gain': matched_opening,
            'return_bridge_nll': float(a2[0]),
            'B_exposure_events': len(b),
            'intervening_events_before_first_matched_pair': len(b)+1,
            'intervening_physical_ticks_before_first_matched_pair': (len(b)+1)*ticks,
            'scope': 'first window precedes return update; later replay scores include relearning'},
        'health': {key: row.get(key) for key in (
            'field_energy', 'firing_rate', 'centered_effective_rank', 'read_norm_mean',
            'vram_peak_mib', 'vram_reserved_mib')},
        'health_scope': row.get('health_scope', 'descriptive recorded physical health'),
    }


def build_summary(records, config, progress, validation, reference_surprisal):
    migration = config.get('origin', {}).get('timing_migration')
    boundary = migration['bptt_train_tokens'] if migration else 0
    rows = [summarize_encounter(row, validation, reference_surprisal,
                               window=config['window']) for row in records]
    new_rows = [row for row in rows if row['bptt_train_tokens'] > boundary
                and row['physical_ticks_per_input'] == config['physical_ticks_per_input']]
    totals = sum(row['first_pass']['tokens'] for row in new_rows)
    model_mean = (sum(row['first_pass']['model_nll'] * row['first_pass']['tokens']
                      for row in new_rows)/totals) if totals else None
    reference_mean = (sum(row['first_pass']['fixed_unigram_nll'] * row['first_pass']['tokens']
                          for row in new_rows)/totals) if totals else None
    completed = max(0, progress['bptt_train_tokens'] - boundary)
    writer_migration = config.get('origin', {}).get('writer_clock_migration', {})
    phase_boundary = max(boundary, writer_migration.get('bptt_train_tokens', 0))
    phase_clock = config.get('writer_baseline_clock', 'physical')
    phase_rows = [row for row in new_rows if row['bptt_train_tokens'] > phase_boundary
                  and row['writer_baseline_clock'] == phase_clock]
    phase_targets = sum(row['first_pass']['tokens'] for row in phase_rows)
    phase_model = (sum(row['first_pass']['model_nll']*row['first_pass']['tokens']
                       for row in phase_rows)/phase_targets) if phase_targets else None
    phase_reference = (sum(row['first_pass']['fixed_unigram_nll']*row['first_pass']['tokens']
                           for row in phase_rows)/phase_targets) if phase_targets else None
    phase_updates = max(0, progress['bptt_train_tokens']-phase_boundary)//config['window']
    result = {
        'protocol': 'recorded active pre-update four-pillar trajectory; state never reset',
        'timing_boundary_bptt_targets': boundary,
        'new_timing_fresh_training_targets': completed,
        'new_timing_fresh_training_updates': completed//config['window'],
        'capability_assessment': (
            'insufficient_joint_updates' if phase_updates < 3000
            else 'sufficient_minimum_updates; convergence and capability require validation evidence'),
        'current_progress': progress,
        'new_timing_validation': {
            'encounters': len(new_rows), 'fresh_targets': totals,
            'weighted_model_nll': model_mean, 'weighted_fixed_unigram_nll': reference_mean,
            'weighted_gain_over_fixed_unigram': (reference_mean-model_mean) if totals else None},
        'current_implementation_phase': {
            'writer_baseline_clock': phase_clock,
            'start_bptt_train_tokens': phase_boundary,
            'fresh_training_updates': phase_updates,
            'encounters': len(phase_rows), 'fresh_validation_targets': phase_targets,
            'weighted_model_nll': phase_model, 'weighted_fixed_unigram_nll': phase_reference,
            'weighted_gain_over_fixed_unigram': (phase_reference-phase_model) if phase_targets else None},
        'encounters': rows,
        'interpretation': (
            'Historical and repaired timing use different fresh texts and different continuous histories. '
            'Reference-relative scores and matched A pairs improve attribution; no paired architecture '
            'comparison or asymptotic superiority is asserted. Health is descriptive, not proof of NESS '
            'or criticality. First-return-window gains precede return updates; full replay includes relearning.'),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=Path('results/q8_fly_bptt32_adamw_continuous_100k'))
    parser.add_argument('--reference-train', type=Path,
                        default=Path('data/ib_owt_gpt2/train.npy'),
                        help='Original train-only array used for the decoder unigram initializer')
    parser.add_argument('--output', type=Path,
                        default=Path('results/published/fly_pulse_quiet_live_summary.json'))
    args = parser.parse_args()
    config = json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    progress = json.loads((args.run/'progress.json').read_text(encoding='utf-8'))
    # Ignore an incomplete last line while the producer writes it; reject
    # malformed complete records instead of silently dropping their evidence.
    raw = (args.run/'lifelong_evaluation.jsonl').read_text(encoding='utf-8')
    lines = raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
    records = [json.loads(line) for line in lines if line.strip()]
    original_train = np.load(args.reference_train, mmap_mode='r')
    counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(original_train), 1_000_000):
        counts += np.bincount(original_train[left:left+1_000_000].astype(np.int64), minlength=50257)
    reference = -np.log(counts/counts.sum())
    validation = np.load(Path(config['data'])/'validation.npy', mmap_mode='r')
    result = build_summary(records, config, progress, validation, reference)
    result['reference'] = {'train': str(args.reference_train),
                           'tokens': len(original_train), 'smoothing': 'add one',
                           'frozen': True, 'validation_labels_used_for_prior': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in (
        'new_timing_fresh_training_updates', 'capability_assessment', 'new_timing_validation')}, indent=2))


if __name__ == '__main__':
    main()
