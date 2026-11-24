"""CPU-only summary of completed preview-restore arms; reads JSON, never checkpoints.

--legacy and --fixed name arm directories (or their result.json files).
Writes JSON even when comparison checks fail, then exits 1. Missing observations
are labelled missing, not replaced by zero. Scores are nats/latent coordinate;
pre_outcome_gain uses the runner's separate 10*finite_U units.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


MISSING = 'missing'
BOUNDARIES = (0, 8, 16, 24, 32)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    values = list(values)
    valid = [v for v in values if numeric(v)]
    return dict(n=len(valid), missing=len(values)-len(valid),
                min=min(valid) if valid else MISSING,
                median=median(valid) if valid else MISSING,
                mean=mean(valid) if valid else MISSING,
                max=max(valid) if valid else MISSING)


def get(row, path):
    value = row
    for key in path.split('.'):
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    return value


def metric(rows, path):
    """Keep vector/matrix component indices; never pool unlike action channels."""
    values = [get(row, path) for row in rows]
    if not any(isinstance(v, list) for v in values):
        return stats(values)
    width = max(len(v) for v in values if isinstance(v, list))
    return {str(i): metric(rows, f'{path}.{i}') for i in range(width)}


def query_map(evaluation):
    return {(q['relation'], q['query']): q for q in evaluation.get('queries', [])}


def query_summary(evaluation):
    return dict(status='recorded' if evaluation else MISSING,
                **{field: stats(q.get(field) for q in evaluation.get('queries', []))
                   for field in ('score', 'mse')})


def read_summary(rows):
    facts = defaultdict(lambda: dict(target_reads=0, support_reads=0, relations=set(),
                                    clocks=set(), scores=[]))
    records, scores, missing_relations, missing_clocks, missing_supports = 0, [], 0, 0, 0
    for row in rows:
        for read in row.get('read_use', []):
            records += 1
            relation = read.get('relation', {})
            identity = (relation.get('identity'), relation.get('generation'))
            known_relation = None not in identity
            missing_relations += not known_relation
            clock = read.get('comparison_update_clock')
            missing_clocks += not numeric(clock)
            score = read.get('log_score')
            scores.append(score)
            missing_supports += 'support_ids' not in read
            uses = [(read.get('target_fact_id'), 'target_reads')]
            uses += [(fact, 'support_reads') for fact in set(read.get('support_ids', []))]
            for fact, role in uses:
                if not fact:
                    continue
                item = facts[fact]
                item[role] += 1
                if known_relation:
                    item['relations'].add(identity)
                if numeric(clock):
                    item['clocks'].add(clock)
                if role == 'target_reads':
                    item['scores'].append(score)
    by_fact = []
    for fact, item in sorted(facts.items()):
        by_fact.append(dict(fact_id=fact, target_reads=item['target_reads'],
            support_reads=item['support_reads'],
            relations=[list(r) for r in sorted(item['relations'])] or MISSING,
            cross_relation_reuse=len(item['relations']) > 1,
            distinct_update_clocks=len(item['clocks']) if item['clocks'] else MISSING,
            later_update_reuse=len(item['clocks']) > 1 if item['clocks'] else MISSING,
            target_log_score=stats(item['scores'])))
    present = sum('read_use' in r for r in rows)
    return dict(status=MISSING if not present else 'recorded' if present == len(rows) else 'partial',
        events=len(rows), missing_events=len(rows)-present,
        records=records, target_log_score=stats(scores), missing_support_records=missing_supports,
        missing_relation_records=missing_relations, missing_update_clock_records=missing_clocks,
        unique_facts=len(facts),
        cross_relation_facts=sum(len(v['relations']) > 1 for v in facts.values()),
        later_update_facts=sum(len(v['clocks']) > 1 for v in facts.values()),
        by_fact=by_fact,
        meaning='Observed target/support identity reuse across (relation identity, generation); '
                'versions do not create a new relation. Not a causal benefit or an origin-relation claim.')


def gain_summary(rows):
    result = dict(units='10*finite_U_nats_per_latent_coordinate',
                  missing_events=sum('pre_outcome_gain' not in r for r in rows),
                  target=metric(rows, 'pre_outcome_gain.target'))
    for name in ('base', 'raw', 'expected'):
        errors = []
        for row in rows:
            prediction, target = get(row, f'pre_outcome_gain.{name}'), get(row, 'pre_outcome_gain.target')
            errors.append(prediction-target if numeric(prediction) and numeric(target) else None)
        result[name] = dict(prediction=metric(rows, f'pre_outcome_gain.{name}'),
            signed_error=stats(errors), absolute_error=stats(abs(e) if e is not None else None for e in errors),
            squared_error=stats(e*e if e is not None else None for e in errors))
    return result


# These are recorded diagnostics, not newly reconstructed counterfactuals.
CONFIDENCE = ('wood_confidence_used', 'wood_confidence_next', 'wood_all_confidence',
              'wood_to_fire.confidence', 'direct_gain_alpha')
CONTINUATION = ('causal_program.pre_draft.expected_after_value',
    'causal_program.pre_draft.selection_value', 'causal_program.expected_after_value',
    'joint_language_search.predicted_after_value', 'joint_language_search.selection_value',
    'internal_rollout.predicted_future', 'internal_rollout.baseline_future',
    'internal_rollout.delivered_offset', 'internal_rollout.choice_probability_tv',
    'shadow_preview.future_delta', 'shadow_preview.delivered_offset',
    'imagination_offset', 'wood_to_fire.policy_logit_F_effect_max')
FINITE_U = ('causal_use.observed_use', 'causal_use.expected_use', 'metal_g',
    'use_transport.mean.append_gain', 'use_transport.mean.release_effect',
    'use_transport.mean.actual_U_gain', 'use_transport.mean.representation_shift',
    'causal_use.finite_U.committed_score_max_error', 'use_transport.decomposition_residual')


def arm_summary(path, label, checks):
    result_path = path/'result.json' if path.is_dir() else path
    result = json.loads(result_path.read_text(encoding='utf-8'))
    with result_path.with_name('events.jsonl').open(encoding='utf-8') as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    rows = [r for r in events if get(r, 'autonomous_event.kind') == 'external']
    evaluations = result.get('evaluations', [])
    by_boundary = {e['external']: e for e in evaluations}
    initial = query_map(by_boundary.get(0, {}))

    def check(name, actual, expected):
        checks[f'{label}.{name}'] = dict(ok=actual == expected, actual=actual, expected=expected)

    for key, expected in dict(status='completed', origin_cycle=2437, cycle=2469,
                              cycles=32, updates=128, external_bytes=30720).items():
        check(key, result.get(key, MISSING), expected)
    check('mode', result.get('mode', MISSING),
          'legacy_restore_simulation' if label == 'legacy' else 'normal_repaired_restore')
    check('runtime_revision', result.get('runtime_revision', MISSING), 0 if label == 'legacy' else 1)
    check('event_external_rounds', len(rows), 32)
    for key, expected in (('updates', 128), ('external_bytes', 30720)):
        values = [r.get(key) for r in events]
        check(f'event_{key}', sum(values) if all(numeric(v) for v in values) else MISSING, expected)
    check('external_cycle_sequence', [r.get('cycle', MISSING) for r in rows], list(range(2438, 2470)))
    check('evaluation_boundaries', sorted(by_boundary), list(BOUNDARIES))
    check('unique_evaluation_boundaries', len(by_boundary), len(evaluations))
    check('initial_unique_queries', len(initial), 64)
    for external, evaluation in by_boundary.items():
        queries = evaluation.get('queries', [])
        check(f'queries_{external}', len(queries), 64)
        check(f'query_identity_{external}', set(query_map(evaluation)) == set(initial), True)
        check(f'query_numbers_{external}',
              all(numeric(q.get(k)) for q in queries for k in ('score', 'mse')), True)
        check(f'evaluation_cycle_{external}', evaluation.get('cycle', MISSING), 2437+external)

    gains = gain_summary(rows)
    gains['by_choice'] = {choice: gain_summary([r for r in rows if r.get('choice') == choice])
                         for choice in sorted({r['choice'] for r in rows if 'choice' in r})}
    flags = {}
    for field in ('causal_use.finite_U.committed', 'causal_use.finite_U.verified'):
        values = [get(r, field) for r in rows]
        flags[field] = dict(true=sum(v is True for v in values), false=sum(v is False for v in values),
                            missing=sum(not isinstance(v, bool) for v in values))
    transport_queries = [q for row in rows for q in get(row, 'use_transport.queries') or []]
    comparisons = [q for row in rows for q in get(row, 'causal_use.comparisons') or []]
    credit = {phase: {key: metric(rows, f'read_credit.{phase}.{key}')
                     for key in ('records', 'participations', 'stale', 'credit_sum')}
              for phase in ('response', 'heldout')}
    for phase in credit:
        credit[phase]['totals'] = {
            key: (sum(get(r, f'read_credit.{phase}.{key}') for r in rows)
                  if all(numeric(get(r, f'read_credit.{phase}.{key}')) for r in rows) and rows else MISSING)
            for key in ('records', 'participations', 'stale', 'credit_sum')}
    summary = dict(result=str(result_path.resolve()),
        run={k: result.get(k, MISSING) for k in ('status', 'mode', 'origin_cycle', 'cycle',
            'cycles', 'updates', 'external_bytes', 'initial_source_sha256', 'runtime_revision')},
        event_kinds=dict(Counter(get(r, 'autonomous_event.kind') or MISSING for r in events)),
        fixed_queries={str(n): query_summary(by_boundary.get(n, {})) for n in BOUNDARIES},
        pre_outcome_gain=gains,
        finite_U=dict(units='nats_per_latent_coordinate', flags=flags,
            per_event={field: metric(rows, field) for field in FINITE_U},
            transport_missing_events=sum(get(r, 'use_transport.queries') is None for r in rows),
            comparisons_missing_events=sum(get(r, 'causal_use.comparisons') is None for r in rows),
            comparisons={key: metric(comparisons, key) for key in ('before', 'after', 'contribution', 'expected')},
            per_query={key: metric(transport_queries, key) for key in
                       ('append_gain', 'release_effect', 'actual_U_gain', 'representation_shift')}),
        confidence={field: metric(rows, field) for field in CONFIDENCE},
        continuation_recorded={field: metric(rows, field) for field in CONTINUATION},
        continuation_isolated_ranking_effect=MISSING,
        read_use=read_summary(rows), read_credit=credit)
    return summary, by_boundary


def compare(legacy, fixed):
    checks = {}
    left, le = arm_summary(Path(legacy), 'legacy', checks)
    right, fe = arm_summary(Path(fixed), 'fixed', checks)
    li, fi = query_map(le.get(0, {})), query_map(fe.get(0, {}))
    same_ids = set(li) == set(fi) and len(li) == 64
    checks['paired.initial_query_identities'] = dict(ok=same_ids)
    for field in ('score', 'mse'):
        equal = same_ids and all(numeric(li[k].get(field)) and
                                li[k].get(field) == fi[k].get(field) for k in li)
        checks[f'paired.initial_{field}_exact'] = dict(ok=equal)
    sha = left['run']['initial_source_sha256']
    checks['paired.source_sha256'] = dict(ok=sha != MISSING and sha == right['run']['initial_source_sha256'])
    paired = {}
    for n in BOUNDARIES:
        lm, fm = query_map(le.get(n, {})), query_map(fe.get(n, {}))
        aligned = same_ids and set(lm) == set(li) and set(fm) == set(li)
        changes = []
        if aligned:
            for key in sorted(li):
                record = dict(relation=key[0], query=key[1])
                for field in ('score', 'mse'):
                    values = [m[key].get(field) for m in (li, fi, lm, fm)]
                    if all(numeric(v) for v in values):
                        l0, f0, ln, fn = values
                        record[field] = dict(legacy_change=ln-l0, fixed_change=fn-f0,
                            fixed_minus_legacy_change=(fn-f0)-(ln-l0), fixed_minus_legacy=fn-ln)
                changes.append(record)
        paired[str(n)] = dict(
            status='aligned' if aligned else MISSING,
            **{field: {quantity: stats(get(r, f'{field}.{quantity}') for r in changes)
               for quantity in ('legacy_change', 'fixed_change', 'fixed_minus_legacy_change', 'fixed_minus_legacy')}
               for field in ('score', 'mse')})
        if n == 32:
            paired[str(n)]['queries'] = changes
    return dict(schema=1, valid=all(c['ok'] for c in checks.values()), checks=checks,
        conventions=dict(missing='Absent/non-finite values are missing, never zero.',
            score='Higher is better; MSE lower is better. Pair by (relation, query), not row order.',
            confidence='Vectors retain original component indices; gain channel is index 3. '
                       'direct_gain_alpha is a distinct recorded calibration, not substituted confidence.',
            continuation='Recorded future values/offsets/TV only; an isolated continuation ranking '
                         'counterfactual is missing (not supplied by this runner).',
            read_credit='Aggregate actual read credit, not attribution to the original writing actor.'),
        legacy=left, fixed=right, paired_fixed_minus_legacy=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = [(p/'result.json' if p.is_dir() else p).resolve() for p in (args.legacy, args.fixed)]
    if args.output.resolve() in inputs + [p.with_name('events.jsonl') for p in inputs]:
        parser.error('--output must not overwrite input result/events files')
    result = compare(args.legacy, args.fixed)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output.resolve()), valid=result['valid'],
                          failed_checks=[k for k, v in result['checks'].items() if not v['ok']])))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
