"""Audit saved, real mother-world transitions across continuation segments.

Reads only newline-committed metrics and existing fixed validation sidecars;
does not load a body, interact with the world, or alter a running process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.ib_local.audit_generative_big_loop import calibration, group_summary, mean


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cgb_generative_joint_20261113'
SEGMENTS = (
    BASE / 'unit_repair_learning_3884',
    BASE / 'guarded_from_3887/body',
    BASE / 'guarded_from_3895_resume_20261114/body',
)


def read_committed(path: Path):
    if not path.exists():
        return []
    # A writer may currently be appending the final line. Never consume it
    # until the terminating newline makes the event visible in the log.
    data = path.read_bytes()
    complete = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in complete.splitlines() if line]


def one_block(rows):
    out = group_summary(rows)
    out['actual_updates'] = sum(r['updates'] for r in rows)
    out['real_commits'] = sum(bool(r['causal_program']['evidence_committed']) for r in rows)
    out['program_score_commits'] = sum(bool(r['generative_relation']['program_evidence_updated']) for r in rows)
    out['question_roles'] = dict(Counter(r['generative_relation']['query_role'] for r in rows))
    out['internal_steps'] = sum(sum(c['selected'] == 'internal' for c in
                                    r['generative_relation']['internal_admission']) for r in rows)
    out['internal_calibration'] = calibration(rows, 'internal_rollout')
    out['shadow_calibration'] = calibration(rows, 'shadow_preview')
    out['wood_to_fire'] = {
        'forecast_logit_abs_max_mean': mean([r['wood_to_fire']['policy_logit_F_effect_max'] for r in rows]),
        'direct_relation_logit_abs_max_mean': mean([r['wood_to_fire']['direct_read_logit_max'] for r in rows]),
        'question_byte_probability_tv_mean': mean([r['question_context']['decision_byte_probability_tv'] for r in rows]),
        'draft_choice_probability_tv_uniform_mean': out['draft_probability_mean_tv_uniform'],
    }
    out['retained_question'] = dict(
        used=sum(bool(r['retained_question']['used']) for r in rows),
        parent_carried=sum(bool(r['retained_question']['parent_evidence_carried']) for r in rows),
        observed_boundary_used=sum(bool(r['retained_question']['actual_boundary_used']) for r in rows))
    sources = defaultdict(list)
    for row in rows:
        for reading in row['world_learning']['readings']:
            sources[reading['source']].append(reading)
    out['world_sources'] = {
        name: dict(events=len(values), scored_bytes=sum(v['bytes'] for v in values),
                   pre_nll=mean([v['pre'] for v in values]),
                   post_nll=mean([v['post'] for v in values]),
                   mean_progress=mean([v['pre']-v['post'] for v in values]),
                   negative_progress=sum(v['post'] > v['pre'] for v in values))
        for name, values in sources.items()
    }
    out['world_sources']['delivered_bytes'] = dict(
        total=sum(r['external_bytes'] for r in rows),
        mother=sum(r['world'].get('mother_bytes', 0) for r in rows))
    replies = [r['world'].get('mother_reply') for r in rows if r['world'].get('mother_reply')]
    out['world_sources']['mother_reply_diversity'] = dict(
        events=len(replies), exact_distinct=len(set(replies)))
    return out


def ambient_score_key(row):
    world = row['world']
    mother_bytes, total_bytes = world.get('mother_bytes'), row.get('external_bytes')
    midpoint = ((mother_bytes + total_bytes) // 2
                if mother_bytes is not None and total_bytes is not None else None)
    return (world.get('start'), world.get('anchor'), midpoint, total_bytes)


def analyze(segments=SEGMENTS):
    rows = []
    segment_data = []
    parts = []
    for segment in segments:
        part = read_committed(segment / 'metrics.jsonl')
        parts.append(part)
        initial = segment / 'initial_validation.json'
        segment_data.append(dict(path=str(segment.relative_to(ROOT)), events=len(part),
                                 first=part[0]['cycle'] if part else None,
                                 last=part[-1]['cycle'] if part else None,
                                 initial_validation=json.loads(initial.read_text(encoding='utf-8'))
                                 if initial.exists() else None))
        rows.extend(part)
    if not rows:
        raise ValueError('No committed repaired events')
    cycles = [r['cycle'] for r in rows]
    if cycles != list(range(cycles[0], cycles[-1] + 1)):
        raise ValueError('Segments overlap or have a gap; do not splice them')
    if cycles[0] != 3885:
        raise ValueError('The repaired trajectory must start at 3885')
    for data, part in zip(segment_data, parts):
        data['summary'] = one_block(part) if part else None
    relation_events = defaultdict(list)
    anchor_relations = defaultdict(set)
    unknown_relations = defaultdict(set)
    for row in rows:
        role = row['causal_program']['roles']
        relation = (role['identity'], role['generation'])
        relation_events[relation].append(row)
        if role.get('anchor_identity') not in (None, 'unobserved'):
            anchor_relations[(role['anchor_identity'], tuple(role['open']))].add(relation)
        unknown_relations[role['unknown_identity']].add(relation)
    repeated = [events for events in relation_events.values() if len(events) > 1]
    def score(row, key):
        return (row['mother_boundary']['prequential'] or {}).get(key)
    first_last = [(score(es[0], 'mother_mse'), score(es[-1], 'mother_mse')) for es in repeated]
    first_last = [(a, b) for a, b in first_last if a is not None and b is not None]
    validation = [dict(cycle=r['cycle'], **r['fixed_validation']) for r in rows if 'fixed_validation' in r]
    # Match actual ambient source and its scored midpoint. Adjacent mother
    # lengths can yield the same integer midpoint. The mother prefix and
    # recurrent context can still differ, so this is a natural revisit, not a
    # controlled isolation of memory or an A-B-A relearning experiment.
    historic = []
    for name in ('admission', 'learning'):
        historic.extend(read_committed(BASE / name / 'metrics.jsonl'))
    by_source = defaultdict(list)
    for row in historic + rows:
        key = ambient_score_key(row)
        if all(v is not None for v in key):
            by_source[key].append(row)
    revisits = []
    for observations in by_source.values():
        observations.sort(key=lambda r: r['cycle'])
        for previous, current in zip(observations, observations[1:]):
            if current['cycle'] < 3885:
                continue
            same_response = previous['world'].get('mother_bytes') == current['world'].get('mother_bytes')
            revisits.append(dict(first_cycle=previous['cycle'], revisit_cycle=current['cycle'],
                gap=current['cycle']-previous['cycle'],
                mother_bytes=(previous['world'].get('mother_bytes'), current['world'].get('mother_bytes')),
                first_response_nll=previous.get('ambient_response_nll') if same_response else None,
                revisit_response_nll=current.get('ambient_response_nll') if same_response else None,
                first_suffix_pre=previous.get('heldout_pre_nll'),
                revisit_suffix_pre=current.get('heldout_pre_nll'),
                revisit_suffix_post=current.get('heldout_post_nll')))
    blocks = [one_block(rows[i:i+32]) for i in range(0, len(rows), 32)]
    unit_error = [abs(r['td_previous']['y'] -
                      (r['td_previous']['g'] + .8*r['value_now']/100)) for r in rows
                  if r.get('td_previous') is not None]
    older_path = ROOT / 'results/published/cgb_direct_water_wood_20261106.json'
    older = json.loads(older_path.read_text(encoding='utf-8'))['summary'] if older_path.exists() else None
    prior_capacity = None
    if older:
        real, blocked = older['train_real_causal'], older['train_blocked_causal']
        prior_capacity = dict(
            source=str(older_path.relative_to(ROOT)),
            cycles_per_arm=real['cycles'], updates_per_arm=real['optimizer_updates'],
            real_validation_final=real['validation_final'], blocked_validation_final=blocked['validation_final'],
            real_max_policy_tv=real['policy_tv_max'],
            scope=('Earlier same-lineage small body in a different language world; '
                   'evidence of prior functional capacity, not a matched current-world control'))
    return dict(
        scope='Observed saved real transitions; policy-selected, not matched causal controls',
        segments=segment_data, summary=one_block(rows), blocks_32=blocks,
        td_unit_max_abs_error=max(unit_error, default=None), fixed_validation=validation,
        validation_provenance=dict(
            mother_manifest='results/cgb_mother_continuation_20261110/validation/manifest.json',
            mother_scope=('11-05 real provider replies from a research-side fork; '
                          'fixed 8192-byte prefix, not a representative external benchmark'),
            owt_scope='Fixed 8192-byte prefix from the held-out OWT validation array'),
        relation_revisits=dict(identities=len(relation_events), repeated=len(repeated),
            repeated_events=sum(len(es)-1 for es in repeated), pairs_with_mother_scores=len(first_last),
            first_mean=mean([a for a, _ in first_last]), last_mean=mean([b for _, b in first_last]),
            last_lower=sum(b < a for a, b in first_last),
            scope='Descriptive first/last observations have different contexts and outcomes'),
        unknown_references=dict(
            observed_anchor_role_pairs=len(anchor_relations),
            anchor_role_pairs_seen_in_multiple_relations=sum(len(v)>1 for v in anchor_relations.values()),
            identical_unknown_names_seen_in_multiple_relations=sum(len(v)>1 for v in unknown_relations.values()),
            scope='Logged identity strings only; other paths may share underlying evidence'),
        natural_ambient_revisits=dict(n=len(revisits), records=revisits,
            scope=('Same source start, window anchor, ambient scored midpoint and extent; '
                   'body state and mother text differ, so not a controlled A-B-A test')),
        prior_small_body_evidence=prior_capacity,
        missing_evidence=['unexecuted candidate outcomes', 'matched loop-level control',
                          'controlled retention and relearning curve',
                          'larger body at adequate training budget'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-segment', action='append', type=Path, default=[],
                        help='Append a saved continuation; inspect segment summaries separately if the world changed')
    args = parser.parse_args()
    result = analyze((*SEGMENTS, *(p.resolve() for p in args.additional_segment)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(args.output)
    print(json.dumps({k: result[k] for k in ('summary', 'fixed_validation', 'relation_revisits',
                                              'td_unit_max_abs_error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
