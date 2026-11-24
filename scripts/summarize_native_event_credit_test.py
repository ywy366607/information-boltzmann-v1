"""Summarize the existing four-round event-credit comparison without a model run."""
import argparse
import json
from pathlib import Path
import sqlite3
from statistics import mean, median

from scripts.audit_native_value_transition import summarize


def distribution(values):
    return dict(min=min(values), median=median(values), mean=mean(values), max=max(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('E:/cgb_event_credit_2469_20261118'))
    p.add_argument('--output', type=Path,
                   default=Path('results/published/cgb_event_credit_loop_2469_20261118.json'))
    args = p.parse_args()
    arms, checks, rows_by_arm, results = {}, {}, {}, {}
    for name in ('compressed', 'identified'):
        folder = args.work/name
        result = json.loads((folder/'result.json').read_text(encoding='utf8'))
        metadata = json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
        events = [json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in events if r['autonomous_event']['kind'] == 'external']
        rows_by_arm[name], results[name] = rows, result
        config = metadata['config']
        checks[name+'.budget'] = (result['status'], result['origin_cycle'], result['cycle'],
            result['cycles'], result['updates'], result['external_bytes'], result['provider_calls']) == (
            'completed', 2469, 2473, 4, 16, 3840, 0)
        checks[name+'.mode'] = config['event_credit']['mode'] == name
        checks[name+'.epistemic'] = config['epistemic_value']['schema'] == 1
        checks[name+'.reward'] = all(r['causal_use']['reward_version'] ==
            'finite_event_targeted_use_v3' for r in rows)
        checks[name+'.payment'] = all(r['causal_use']['finite_U'].get('verified', False)
            for r in rows if r['causal_use']['finite_U']['committed'])
        checks[name+'.delayed_identity'] = all(now.get('causal_program_delayed_fact') ==
            old['causal_program']['committed_episode_identity'] for old, now in zip(rows, rows[1:])
            if not old['episode_merged'] or name == 'identified')
        checks[name+'.metadata'] = metadata['cycle'] == result['cycle'] and (
            metadata['checkpoint_sha256'] == result['checkpoint_sha256'])
        snapshot = Path(config['causal_archive']['snapshot'])
        with sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro', uri=True) as db:
            foreign_keys = db.execute('PRAGMA foreign_key_check').fetchall()
            cold_events = db.execute('SELECT COUNT(*) FROM episodes WHERE origin_slot=-1').fetchone()[0]
        checks[name+'.foreign_keys'] = not foreign_keys
        evaluations = {}
        for label, evaluation in (('before', result['evaluations'][0]), ('after', result['evaluations'][-1])):
            evaluations[label] = {metric: distribution([q[metric] for q in evaluation['queries']])
                                 for metric in ('score', 'mse')}
        merged = [r for r in rows if r['episode_merged']]
        arms[name] = dict(cycle=result['cycle'], updates=result['updates'], bytes=result['external_bytes'],
            seconds=result['invocation']['seconds'], merged_events=len(merged),
            cold_events=cold_events, merge_branch_exercised=bool(merged),
            internal_events=sum(e['autonomous_event']['kind'] == 'internal' for e in events),
            choices=[r['choice'] for r in rows], gains=[r['metal_g'] for r in rows],
            gain_distribution=distribution([r['metal_g'] for r in rows]),
            common_queries=evaluations, checkpoint=result['checkpoint'],
            checkpoint_sha256=result['checkpoint_sha256'],
            actual_chain=[dict(cycle=r['cycle'], merged=r['episode_merged'], gain=r['metal_g'],
                committed=r['causal_use']['finite_U']['committed'],
                committed_score_error=r['causal_use']['finite_U'].get('committed_score_max_error'),
                event_identity=r['causal_program']['committed_episode_identity'],
                unknown=r['shared_unknown']['revision'],
                delayed_fact=r.get('causal_program_delayed_fact'),
                transition_samples=r['internal_transition_samples']) for r in rows],
            next_real_error_decomposition=summarize(events))
    a, b = results.values()
    checks['same_source'] = a['initial_source_sha256'] == b['initial_source_sha256'] == (
        '886144783de4e8158f609d187c72eb60ff98dc528d33e312d0eae17513145bac')
    checks['same_initial_queries'] = a['evaluations'][0] == b['evaluations'][0]
    left, right = rows_by_arm.values()
    same = {k: all(x.get(k) == y.get(k) for x, y in zip(left, right))
            for k in ('choice', 'choice_probability', 'world', 'utterance_hex', 'metal_g')}
    merged = sum(a['merged_events'] for a in arms.values())
    report = dict(valid=all(checks.values()), checks=checks, arms=arms, same_recorded_outputs=same,
        merge_branch_exercised=bool(merged),
        finding=('No natural merge in either arm. This is a normal-path paired control, '
                 'not real-body evidence that repaired merged events improve use.' if not merged else
                 'Natural merges occurred; inspect their event and next-boundary records.'),
        interface_verification='Two production fixtures force merges and check payment, shared U, '
            'SSD identity, later DECISION labels and foreign keys; distinct from natural exposure.',
        remaining='Continuation still mixes recalled actions, not the actual next actor distribution. '
            'Event credit repair does not resolve that separate whole-transition mismatch.',
        new_training_or_world_calls_by_summary=0)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(valid=report['valid'], checks=len(checks),
        merge_branch_exercised=bool(merged), same_outputs=same, output=str(args.output))))
    if not report['valid']:
        raise RuntimeError('comparison contract check failed')


if __name__ == '__main__':
    main()
