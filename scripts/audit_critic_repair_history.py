"""Reconstruct post-unit-repair TD error statistics from committed real events.

The comparison is on the observed trajectory only; it does not modify the
individual or claim what its actions would have been with different precision.
"""
import argparse
import json
import math
from pathlib import Path

from scripts.audit_repaired_mother_loop import SEGMENTS, read_committed


def audit(segments, previous_segments=(), migration=None):
    rows = [row for segment in segments for row in read_committed(segment / 'metrics.jsonl')]
    cycles = [r['cycle'] for r in rows]
    if not cycles or cycles != list(range(3885, cycles[-1] + 1)):
        raise ValueError('expected the complete contiguous post-repair trajectory')
    beta, numerator, mass = .98, 0., 0.
    reconstructed = None
    max_error = 0.
    blocks = []
    for i, row in enumerate(rows):
        if migration and row['cycle'] == migration['cycle'] + 1:
            if reconstructed is None or not math.isclose(
                    reconstructed, migration['old_critic_error'], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError('migration does not match the preceding saved statistic')
            reconstructed = migration['new_critic_error']
        td = row['td_previous']
        error2 = (td['imagined'] - 100. * td['y'])**2
        observed = row['critic_rms']**2
        if reconstructed is None:
            reconstructed = (observed - (1-beta)*error2) / beta
        reconstructed = beta*reconstructed + (1-beta)*error2
        max_error = max(max_error, abs(reconstructed-observed))
        numerator = beta*numerator + (1-beta)*error2
        mass = beta*mass + (1-beta)
        if i % 32 == 0:
            blocks.append({'first': row['cycle'], 'last': row['cycle'], 'errors2': [], 'saved': []})
        block = blocks[-1]
        block['last'] = row['cycle']
        block['errors2'].append(error2)
        block['saved'].append(row['critic_rms'])
    for b in blocks:
        b['actual_td_rms'] = math.sqrt(sum(b.pop('errors2')) / (b['last']-b['first']+1))
        saved = b.pop('saved')
        b['mean_saved_rms'] = sum(saved)/len(saved)
    post = math.sqrt(numerator/mass)
    out = {'first': cycles[0], 'last': cycles[-1], 'events': len(cycles),
            'ema_beta': beta, 'replay_max_abs_squared_error': max_error,
            'saved_rms': rows[-1]['critic_rms'], 'post_repair_bias_corrected_rms': post,
            'relative_precision_if_post_repair_only': rows[-1]['critic_rms']/post,
            'blocks': blocks,
            'scope': 'same observed actions and TD targets; no policy counterfactual or model update'}
    if previous_segments:
        previous = [row for segment in previous_segments
                    for row in read_committed(segment / 'metrics.jsonl')]
        all_rows = previous + rows
        all_cycles = [r['cycle'] for r in all_rows]
        if all_cycles != list(range(all_cycles[0], cycles[-1]+1)):
            raise ValueError('previous and repaired trajectories must be contiguous')
        num, weight, max_target_error = 0., 0., 0.
        for r in all_rows:
            td = r['td_previous']
            boundary = bool(td.get('objective_boundary', False))
            # Old targets used gamma*V/10; the corrected target uses gamma*V/100.
            y = td['y'] if r['cycle'] >= 3885 or boundary else .9*td['g']+.1*td['y']
            expected = td['g'] if boundary else td['g'] + .8*r['value_now'] / (
                100. if r['cycle'] >= 3885 else 10.)
            max_target_error = max(max_target_error, abs(expected-td['y']))
            num = beta*num + (1-beta)*(td['imagined']-100*y)**2
            weight = beta*weight + (1-beta)
        out['recomputed_history'] = {'first': all_cycles[0], 'last': all_cycles[-1],
            'events': len(all_cycles), 'bias_corrected_rms': math.sqrt(num/weight),
            'target_replay_max_abs_error': max_target_error,
            'scope': 'recorded forecasts rescored against unit-corrected TD targets; observed trajectory fixed'}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional-segment', action='append', type=Path, default=[])
    parser.add_argument('--migration', type=Path,
                        help='record of an explicit critic-error boundary migration')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = SEGMENTS[0].parent
    migration = json.loads(args.migration.read_text(encoding='utf-8')) if args.migration else None
    if migration and not migration.get('passed'):
        raise ValueError('migration record did not pass preservation checks')
    result = audit((*SEGMENTS, *args.additional_segment),
                   previous_segments=(base/'admission', base/'learning'), migration=migration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temp.replace(args.output)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
