"""Freeze a finite, actual-event activity and paired prediction audit; no Torch."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.analyze_complete_optimizer_loop import summary
from scripts.audit_logged_choice_effect import effects


def margin(rows, control):
    values = []
    for row in rows:
        internal = row.get('internal_rollout', {})
        a, b = internal.get('next_boundary_prequential_mse'), internal.get(control)
        if a is not None and b is not None and math.isfinite(a+b):
            values.append(b-a)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return dict(events=len(values), mean_control_minus_real=statistics.fmean(values),
        paired_standard_deviation=sd, standardized_paired_margin=statistics.fmean(values)/sd if sd else None,
        real_better=sum(x > 0 for x in values), real_worse=sum(x < 0 for x in values),
        scope='Mean paired advantage divided by per-event paired standard deviation; not an iid significance test.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--from-cycle', type=int, required=True)
    p.add_argument('--through-cycle', type=int, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    all_rows = []
    with args.metrics.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            if row['cycle'] >= args.from_cycle:
                all_rows.append(row)
            if row['cycle'] == args.through_cycle+1:
                break
    rows = [r for r in all_rows if r['cycle'] <= args.through_cycle]
    if [r['cycle'] for r in rows] != list(range(args.from_cycle, args.through_cycle+1)):
        raise ValueError('Requested actual-event interval is incomplete')
    if all_rows[-1]['cycle'] != args.through_cycle+1:
        raise ValueError('Two-event evaluation needs the following real settlement')
    layers = []
    for index in range(32):
        values = [r['bfss_dynamics']['momentum_norms'][index] for r in rows]
        layers.append(dict(layer=index+1, mean=statistics.fmean(values),
                           standard_deviation=statistics.stdev(values)))
    count = len(rows)
    vectors = lambda key: [r[key] for r in rows if isinstance(r.get(key), list)]
    positive = lambda key: [sum(v[i] > 0 for v in vectors(key))/len(vectors(key))
                           for i in range(len(vectors(key)[0]))]
    generated = [r for r in rows if r['choice'] == 'generate']
    result = dict(schema=1, first_cycle=args.from_cycle, last_cycle=args.through_cycle,
        events=count, source=str(args.metrics), main_written=False, model_updates=0, world_calls=0,
        loop=summary(rows), two_event_selection=effects(all_rows, horizon=2),
        action_discrimination=margin(rows, 'other_action_prequential_mse'),
        change_prediction=margin(rows, 'unchanged_state_prequential_mse'),
        momentum_norm_by_layer=layers,
        mean_quiet_state_change=statistics.fmean(r['bfss_dynamics']['state_change'] for r in rows),
        observed_content_correction_max=max(abs(v) for r in rows for v in r['bfss_dynamics']['event_correction_norms']),
        momentum_read_action_tv_mean=statistics.fmean(r['momentum_wood']['choice_probability_tv_without_p'] for r in rows),
        wood_confidence_positive_fraction=positive('wood_confidence_used'),
        direct_gain_alpha_positive_fraction=positive('direct_gain_alpha'),
        navigation_matched=sum(r['world'].get('navigation_source') == 'matched' for r in rows),
        generated_events=len(generated), generated_handover=sum(r['expression']['reason'] == 'handover' for r in generated),
        generated_budget_stops=sum(r['expression']['reason'] == 'budget' for r in generated),
        fixed_validation=next((r['fixed_validation'] for r in reversed(rows) if r.get('fixed_validation')), None),
        scope='Actual online prequential predictions and logged randomization contrasts; numerical activity is separate from causal usefulness.')
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(events=count, first=args.from_cycle, last=args.through_cycle,
        momentum_read_tv=result['momentum_read_action_tv_mean'],
        action_discrimination=result['action_discrimination']), indent=2))


if __name__ == '__main__':
    main()
