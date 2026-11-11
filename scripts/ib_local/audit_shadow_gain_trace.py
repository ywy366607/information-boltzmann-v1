"""Replay recorded action-mean arithmetic without modifying the individual.

The replacement is conditional on the actions and labels actually observed;
it is not a counterfactual rollout of a changed policy.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

BASE = Path('results/cgb_bfss_gdn_reference_20261110')
RUNS = ('joint_question_update_follow_16', 'responsive_mother_admission_8',
        'responsive_mother_v2_admission_8', 'responsive_mother_v3_admission_8',
        'mother_shadow_admission_1', 'mother_shadow_follow_8')


def audit(rows, decay=.95):
    rows = sorted(rows, key=lambda row: row['cycle'])
    selected = [row for row in rows if 2959 <= row['cycle'] <= 2993]
    if [row['cycle'] for row in selected] != list(range(2959, 2994)):
        raise ValueError('need one uninterrupted real record for cycles 2959 through 2993')
    first = selected[0]
    if first['choice'] != 'generate':
        raise ValueError('2959 action differs from recorded generate')
    actual = [float(v[3]) for v in first['wood_all_bases']]
    counterfactual = actual.copy()
    actual_label = float(first['wood_target'][3])
    replacement = actual[1]  # Keep the selected forecast unchanged in 2959.
    rate = 1-decay
    max_error = 0.
    gen_count = 0
    last = None
    for row in selected:
        recorded = [float(v[3]) for v in row['wood_all_bases']]
        max_error = max(max_error, *(abs(a-b) for a,b in zip(actual,recorded)))
        last = dict(cycle=row['cycle'], recorded_pre=recorded,
                    replay_pre=actual.copy(), replaced_pre=counterfactual.copy())
        option = {'silence': 0, 'generate': 1, 'reuse': 2}[row['choice']]
        gen_count += option == 1
        target = float(row['wood_target'][3])
        replacement_target = replacement if row['cycle'] == 2959 else target
        err, cf_err = target-actual[option], replacement_target-counterfactual[option]
        # SharedActionEvidence's joint intercept plus centered action update.
        actual = [v+rate*err*(1. if i == option else .4) for i,v in enumerate(actual)]
        counterfactual = [v+rate*cf_err*(1. if i == option else .4)
                          for i,v in enumerate(counterfactual)]
    if max_error > 2e-5:
        raise ValueError(f'action-mean replay mismatch: {max_error}')
    mother_rows = [row for row in rows if 2985 <= row['cycle'] <= 2993]
    if [row['cycle'] for row in mother_rows] != list(range(2985, 2994)):
        raise ValueError('need continuous mother shadow event records')
    shadow = []
    for prev, row in zip(mother_rows, mother_rows[1:]):
        settled = row['shadow_preview']['settled_two_turn']
        if settled is None or not settled['eligible']:
            continue
        option = settled['option']
        immediate = float(prev['direct_gain_mean'][option])
        continuation = .8*float(prev['internal_rollout']['baseline_future'][option])
        baseline = float(settled['baseline'])
        delta = .8*float(prev['shadow_preview']['future_delta'][option])
        residual = float(settled['target'])-baseline
        shadow.append(dict(cycle=row['cycle'], option=option, target=settled['target'],
            immediate=immediate, continuation=continuation,
            calibrated_internal_correction=baseline-immediate-continuation,
            baseline=baseline, delta=delta, residual=residual, product=residual*delta))
    phases = []
    for lo, hi in ((2945, 2958), (2960, 2968), (2969, 2976), (2977, 2984), (2985, 2993)):
        subset = [r for r in rows if lo <= r['cycle'] <= hi and r['choice'] == 'generate']
        phases.append(dict(cycles=[lo, hi], n=len(subset),
            mean_baseline=statistics.mean(r['wood_base'][3] for r in subset),
            mean_target=statistics.mean(r['wood_target'][3] for r in subset)))
    return dict(decay=decay, event_label=actual_label, replacement_label=replacement,
                later_generate_updates=gen_count-1, max_replay_error=max_error,
                final_pre=last, final_post_actual=actual, final_post_replaced=counterfactual,
                excess_post=[a-b for a,b in zip(actual,counterfactual)],
                shadow_settlements=shadow, generate_phases=phases,
                scope='fixed observed actions and labels; other network and policy effects not recomputed')


def main():
    rows = []
    for run in RUNS:
        rows.extend(json.loads(line) for line in (BASE/run/'metrics.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip())
    print(json.dumps(audit(rows), indent=2))


if __name__ == '__main__':
    main()
