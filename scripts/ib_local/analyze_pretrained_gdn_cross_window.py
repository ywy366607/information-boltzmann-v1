"""Read-only paired analysis of continuous pretrained-GDN experience credit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def journal(path):
    # A resumed branch can re-execute entries after its last complete snapshot.
    # The final record for a curriculum step is the retained trajectory.
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        records[row['step']] = row
    return [records[i] for i in sorted(records)]


def paired(full, held, *, traffic=None, phase=None):
    rows = [(f, h) for f, h in zip(full, held) if phase is None or f['phase'] == phase]
    if any(f['step'] != h['step'] or f['phase'] != h['phase'] for f, h in rows):
        raise ValueError('unmatched curriculum positions')
    groups = {}
    full_total = held_total = token_total = 0.
    for f, h in rows:
        if traffic is None:
            a, b, count = f['nll_sum'], h['nll_sum'], f['target_tokens']
            assert count == h['target_tokens']
        else:
            a, b = f['traffic'][traffic]['nll_sum'], h['traffic'][traffic]['nll_sum']
            count = f['traffic'][traffic]['tokens']
            assert count == h['traffic'][traffic]['tokens']
        groups.setdefault(f['phase'], []).append((a-b, count))
        full_total += a
        held_total += b
        token_total += count
    # Stratify by phase, preserving each phase's duration. Resample contiguous
    # nonoverlapping 16-window blocks within each phase, not independent tokens.
    rng = np.random.default_rng(11)
    numerator, denominator = np.zeros(2000), np.zeros(2000)
    for values in groups.values():
        arr = np.asarray(values)
        blocks = np.asarray([arr[i:i+16].sum(0) for i in range(0, len(arr), 16)])
        picks = rng.integers(0, len(blocks), size=(2000, len(blocks)))
        sampled = blocks[picks].sum(1)
        numerator += sampled[:, 0]
        denominator += sampled[:, 1]
    samples = numerator/np.maximum(denominator, 1)
    return dict(tokens=int(token_total), full=full_total/token_total,
        held=held_total/token_total, full_minus_held=(full_total-held_total)/token_total,
        block16_interval95=np.quantile(samples, [.025, .975]).tolist(),
        uncertainty='conditional on this seed/text trajectory, not corpus/seed uncertainty')


def age_summary(rows):
    result = {}
    for lo, hi, label in [(0, 128, 'through_128'), (128, 1024, '129_to_1024'),
                           (1024, 4096, '1025_to_4096'), (4096, float('inf'), 'over_4096')]:
        mass, count = 0., 0
        for row in rows:
            for read in row['reads'].values():
                for age, weight in zip(read['ages_tokens'], read['weights']):
                    if lo < age <= hi:
                        mass += weight
                        count += 1
        result[label] = dict(selected_events=count, weight_mass=mass)
    result['max_source_age_tokens'] = max(age for row in rows for read in row['reads'].values()
                                        for age in read['ages_tokens'])
    cuts = [r['diagnostics']['old_read_cut_nll_delta'] for r in rows
            if 'old_read_cut_nll_delta' in r['diagnostics']]
    result['old_read_cut'] = dict(probes=len(cuts), mean_delta_nll=float(np.mean(cuts)),
        median_delta_nll=float(np.median(cuts)), positive_fraction=float(np.mean(np.asarray(cuts) > 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    progress = {arm: json.loads((args.directory/arm/'progress.json').read_text()) for arm in ('full', 'held')}
    if any(p['status'] != 'complete' for p in progress.values()):
        raise RuntimeError('paired interpretation waits for both complete trajectories')
    rows = {arm: journal(args.directory/arm/'steps.jsonl') for arm in progress}
    if len(rows['full']) != len(rows['held']):
        raise ValueError('unequal actual update budgets')
    initial = {arm: json.loads((args.directory/arm/'initial_validation.json').read_text()) for arm in progress}
    if initial['full'] != initial['held']:
        raise AssertionError('common-state initial sidecar scores differ; investigate before attribution')
    full, held = rows['full'], rows['held']
    result = dict(kind='paired_pretrained_gdn_cross_window', status='complete',
        definition='full minus held: only native old-replay derivative differs, not recall forward architecture',
        old_navigation_individual_unchanged=True, seed=11, score_unit='nats/token',
        checks=json.loads((args.directory/'checks.json').read_text()),
        matched_initial_validation=initial['full'],
        overall=paired(full, held), fresh=paired(full, held, traffic='fresh'),
        revisit=paired(full, held, traffic='revisit'),
        phases={phase: paired(full, held, phase=phase) for phase in ['A_fresh', 'B_fresh', 'A_revisit']},
        recall={arm: age_summary(records) for arm, records in rows.items()},
        final_validation={arm: p['final_validation'] for arm, p in progress.items()},
        resources={arm: dict(peak_cuda_allocated_bytes=p['peak_cuda_allocated_bytes'],
            peak_cuda_reserved_bytes=p['peak_cuda_reserved_bytes'],
            max_experience_cpu_bytes=max(r['experience_cpu_bytes'] for r in rows[arm]),
            recurrence_bytes=sorted({r['recurrence_bytes'] for r in rows[arm]}),
            step_seconds_median=float(np.median([r['seconds'] for r in rows[arm]])),
            actual_updates=p['next_step'], target_tokens=p['target_tokens']) for arm, p in progress.items()},
        scope='Budget-limited online comparison, not convergence/whole-navigation/mother-world acceptance')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for arm, color in [('full', '#2166ac'), ('held', '#b35806')]:
            nll = np.array([r['nll_sum'] for r in rows[arm]])
            count = np.array([r['target_tokens'] for r in rows[arm]])
            axes[0].plot(np.arange(1, len(nll)+1), nll.cumsum()/count.cumsum(), label=arm, color=color)
            xs, ys = [], []
            for i in range(0, len(nll), 16):
                xs.append(i+len(nll[i:i+16])/2)
                ys.append(nll[i:i+16].sum()/count[i:i+16].sum())
            axes[1].plot(xs, ys, label=arm, color=color)
        for ax in axes:
            for boundary in (len(full)//3, 2*len(full)//3):
                ax.axvline(boundary, color='.6', linestyle='--', linewidth=1)
            ax.grid(alpha=.2)
            ax.set_ylabel('Nats / token')
            ax.legend()
        axes[0].set_title('Continuous A / B / A learning: cumulative prequential NLL')
        axes[1].set_title('Nonoverlapping 16-window means; state and learning never reset')
        axes[1].set_xlabel('Optimizer update after common warm start')
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    print(json.dumps({k: result[k] for k in ('overall', 'fresh', 'revisit')}, indent=2))


if __name__ == '__main__':
    main()
