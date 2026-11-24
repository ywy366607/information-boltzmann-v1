"""Read completed real-event receipts; no Torch, checkpoint load, or training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


FIELDS = (
    'momentum_wood.choice_probability_tv_without_p',
    'momentum_wood.prediction_delta', 'momentum_wood.weight_step',
    'bfss_dynamics.state_change', 'outcome_error', 'progress',
    'wood_policy_tv', 'internal_transition_loss',
    'adaptive_dynamics.last_consumed_steps',
)


def value(row, name):
    for part in name.split('.'):
        row = row.get(part) if isinstance(row, dict) else None
    return row


def summarize(rows):
    result = dict(count=len(rows), cycles=[rows[0]['cycle'], rows[-1]['cycle']])
    for field in FIELDS:
        values = [r[field] for r in rows if isinstance(r[field], (int, float))
                  and math.isfinite(r[field])]
        if values:
            result[field] = dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values))
    result['nonzero_p_choice_cycles'] = sum(
        (r['momentum_wood.choice_probability_tv_without_p'] or 0) > 0 for r in rows)
    result['dynamics_write_cycles'] = sum(r['written_layers'] > 0 for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, default=1600)
    args = parser.parse_args()
    before, after, validations = [], [], []
    last = None
    for line in args.metrics.open(encoding='utf-8'):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        cycle = row.get('cycle', -1)
        if cycle > args.through:
            continue
        if row.get('fixed_validation'):
            validations.append(dict(cycle=cycle, **row['fixed_validation']))
        if not 1402 <= cycle <= args.through:
            continue
        adaptive = row.get('adaptive_dynamics', {})
        compact = dict(cycle=cycle, **{key: value(row, key) for key in FIELDS},
            written_layers=sum(x > 0 for x in adaptive.get('per_layer_parameter_step', [])))
        (before if cycle <= 1472 else after).append(compact)
        last = row
    adaptive = last['adaptive_dynamics']
    frequencies = [x for layer in adaptive['frequency'] for x in layer]
    norms = last['bfss_dynamics']['momentum_norms']
    report = dict(
        scope='Same continuing individual, actual completed natural-text events; descriptive before/after, not a paired architecture treatment.',
        source=str(args.metrics.resolve()), through_cycle=last['cycle'],
        before=summarize(before), after=summarize(after),
        adaptive=dict(frequency_range=[min(frequencies), max(frequencies)],
            coupling_range=[min(adaptive['coupling']), max(adaptive['coupling'])],
            context_gain_absmax=max(map(abs, adaptive['context_gain'])),
            content_coefficient_absmax=max(abs(x) for layer in adaptive['content_coefficients'] for x in layer),
            layers_with_gradient=sum(x > 0 for x in adaptive['per_layer_gradient_norm']),
            trace_bytes=adaptive['trace_bytes']),
        per_layer_momentum=[dict(layer=i+1, norm=x) for i, x in enumerate(norms)],
        strongest_momentum_layers=sorted(range(1, len(norms)+1), key=lambda i: norms[i-1], reverse=True)[:6],
        validations=validations[-4:],
        interpretation=dict(
            choice='P-port removal changes the current choice distribution; TV is immediate sensitivity, not a counterfactual lifetime learning benefit.',
            rhythm='Learned frequencies remain close to 0.8; do not magnify their tiny spread in the animation.',
            progress='Navigated input differs across events; progress means and validation shifts are not controlled attribution to dynamics.'),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(through=report['through_cycle'],
        p_choice_tv=report['after'][FIELDS[0]], writes=report['after']['dynamics_write_cycles'],
        omega=report['adaptive']['frequency_range']), ensure_ascii=False))


if __name__ == '__main__':
    main()
