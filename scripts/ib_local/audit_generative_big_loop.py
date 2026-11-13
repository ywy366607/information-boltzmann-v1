"""Read committed whole-loop evidence without model updates or world calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def paired(rows, left, right):
    values = [(left(r), right(r)) for r in rows]
    values = [(a, b) for a, b in values if a is not None and b is not None]
    return dict(n=len(values), first_mean=mean([a for a, _ in values]),
                second_mean=mean([b for _, b in values]),
                first_lower=sum(a < b for a, b in values),
                mean_first_minus_second=mean([a-b for a, b in values]))


def corr(a, b):
    if len(a) < 3:
        return None
    ma, mb = mean(a), mean(b)
    numerator = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return numerator/denominator if denominator else None


def group_summary(rows):
    mother = lambda r: r['mother_boundary']['prequential'] or {}
    boundary = lambda r: r['internal_rollout']
    search = [r['joint_language_search'] for r in rows]
    role_probabilities = [r['generative_relation']['query_probabilities'] for r in rows]
    return dict(events=len(rows), first_cycle=rows[0]['cycle'], last_cycle=rows[-1]['cycle'],
        g_mean=mean([r['metal_g'] for r in rows]),
        precision_mean=mean([r['precision'] for r in rows]),
        critic_rms_mean=mean([r['critic_rms'] for r in rows]),
        mother_before_learning=paired(rows, lambda r: mother(r).get('mother_mse'),
                                     lambda r: mother(r).get('original_mse')),
        delivered_program_next_mse=mean([r['causal_program']['decision_prequential']['mse']
            for r in rows if r['causal_program']['decision_prequential'] is not None]),
        saved_next_boundary_mse=mean([boundary(r).get('next_boundary_prequential_mse') for r in rows]),
        composed_transition_vs_wrong_action=paired(rows,
            lambda r: boundary(r).get('next_boundary_prequential_mse'),
            lambda r: boundary(r).get('other_action_prequential_mse')),
        draft_future_value_span_mean=mean([max(s['predicted_after_value'])-min(s['predicted_after_value']) for s in search]),
        draft_probability_max=max(max(s['distribution']) for s in search),
        draft_probability_mean_tv_uniform=mean([sum(abs(p-1/len(s['distribution']))
            for p in s['distribution'])/2 for s in search]),
        open_role_max_probability=max(max(p) for p in role_probabilities),
        actual_actions=dict(Counter(r['choice'] for r in rows)))


def calibration(rows, key):
    forecasts = [r[key].get('settled_two_turn') for r in rows]
    forecasts = [f for f in forecasts if f is not None]
    result = dict(n=len(forecasts), eligible=sum(f['eligible'] for f in forecasts))
    for name in ('baseline', 'prediction', 'uncalibrated'):
        pairs = [(f[name], f['target']) for f in forecasts if name in f]
        if pairs:
            result[name] = dict(mse=mean([(a-b)**2 for a, b in pairs]),
                target_correlation=corr([a for a, _ in pairs], [b for _, b in pairs]))
    return result


def checksum(path):
    digest = hashlib.sha256()
    buffer = bytearray(65536)
    with Path(path).open('rb', buffering=0) as stream:
        while (n := stream.readinto(buffer)):
            digest.update(memoryview(buffer)[:n])
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for phase in ('admission', 'learning'):
        for line in (args.run/phase/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
            if line:
                rows.append(json.loads(line))
    by_identity = defaultdict(list)
    for r in rows:
        roles = r['causal_program']['roles']
        by_identity[(roles['identity'], roles['generation'])].append(r)
    blocks = []
    for start in range(0, len(rows), 64):
        block = rows[start:start+64]
        blocks.append(dict(group_summary(block), internal_calibration=calibration(block, 'internal_rollout'),
                           shadow_calibration=calibration(block, 'shadow_preview')))
    report = dict(scope='observational committed real trajectory; no causal intervention or model execution',
        pipeline=json.loads((args.run/'pipeline.json').read_text()),
        summary=group_summary(rows), blocks=blocks,
        updates=sum(r['updates'] for r in rows), external_bytes=sum(r['external_bytes'] for r in rows),
        open_roles=dict(Counter(r['generative_relation']['query_role'] for r in rows)),
        real_commits=sum(r['causal_program']['evidence_committed'] for r in rows),
        rejected_commit_cycles=[r['cycle'] for r in rows if not r['causal_program']['evidence_committed']],
        program_score_commits=sum(r['generative_relation']['program_evidence_updated'] for r in rows),
        internal_steps=sum(sum(c['selected']=='internal' for c in r['generative_relation']['internal_admission']) for r in rows),
        internal_budget_hit=sum(r['generative_relation']['internal_budget_hit'] for r in rows),
        active_hot_reorganizations=rows[-1]['generative_relation']['reorganizations'],
        reorganization_semantics='hot-buffer sum, not a lifetime monotonic counter',
        relation_identities=len(by_identity), repeated_identities=sum(len(v)>1 for v in by_identity.values()),
        reused_identity_events=sum(len(v)-1 for v in by_identity.values()),
        fixed_validation=[dict(cycle=r['cycle'],**r['fixed_validation']) for r in rows if 'fixed_validation' in r],
        initial_validation=json.loads((args.run/'learning/initial_validation.json').read_text()),
        score_semantics=dict(mother='same-event raw original prior versus updated mother predictor, frozen before learning',
            raw_boundary='pending next_latent is overwritten by the delivered composed program; both MSE fields measure the same forecast',
            calibration='same saved previous prediction versus actual two-event gain',
            logits='draft logit span equals current precision times predicted future-value span'),
        unresolved=['old/new whole-body observations are policy selected, not matched controls',
            'no sufficiently trained larger same-loop body comparison',
            'available logs do not give true consequences of unexecuted drafts'])
    checkpoint = args.run/'learning/last.pt.gz'
    uncompressed = 0
    buffer = bytearray(65536)
    with gzip.open(checkpoint, 'rb') as stream:
        while (n := stream.readinto(buffer)):
            uncompressed += n
    report['checkpoint'] = dict(path=str(checkpoint), sha256=checksum(checkpoint),
        gzip_crc_passed=True, uncompressed_bytes=uncompressed,
        scope='stream integrity; cycle and referenced manifest still require checkpoint metadata inspection')
    report['relations'] = [dict(identity=identity[0],generation=identity[1],events=len(events),
        cycles=[r['cycle'] for r in events],
        versions=[r['causal_program']['roles']['version'] for r in events],
        commits=[r['causal_program']['evidence_committed'] for r in events],
        unknowns=[r['causal_program']['roles']['unknown_identity'] for r in events])
        for identity,events in by_identity.items()]
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('summary','updates','external_bytes','open_roles','real_commits',
        'rejected_commit_cycles','program_score_commits','internal_steps','relation_identities',
        'repeated_identities','reused_identity_events','checkpoint')},ensure_ascii=False,indent=2))
    print('block_calibration',json.dumps([dict(first=b['first_cycle'],last=b['last_cycle'],
        internal=b['internal_calibration'],shadow=b['shadow_calibration']) for b in blocks]))


if __name__ == '__main__':
    main()
