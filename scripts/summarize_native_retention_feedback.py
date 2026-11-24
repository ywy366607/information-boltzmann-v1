"""Report actual finite-use settlement separately from retention competition."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def distribution(xs):
    return (dict(min=min(xs), median=median(xs), mean=mean(xs), max=max(xs))
            if xs else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    arms, checks, records, results = {}, {}, {}, {}
    for name in ('observe', 'apply'):
        folder = a.work/name
        result = json.loads((folder/'result.json').read_text(encoding='utf8'))
        metadata = json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
        all_rows = [json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in all_rows if r['autonomous_event']['kind'] == 'external']
        records[name], results[name] = rows, result
        receipts = [r['read_credit'][part]['retention_feedback']
                    for r in rows for part in ('response', 'heldout')]
        observations = [x for receipt in receipts for x in receipt['observations']]
        identities = {(x['slot'],x['version']) for x in observations}
        checks[name+'.complete'] = (result['status'],result['origin_cycle'],result['cycle'],
            result['cycles'],result['updates'],result['external_bytes']) == ('completed',2473,2481,8,32,7680)
        checks[name+'.mode'] = metadata['config']['native_retention_feedback']['mode'] == name
        checks[name+'.settled'] = all(r['settled'] and r['stale'] == 0 for r in receipts)
        checks[name+'.writer'] = (sum(r['records'] for r in receipts) ==
                                   (len(observations) if name == 'apply' else 0))
        checks[name+'.reward'] = all(r['settlement_kind'] == 'finite_event_targeted_use_v3' for r in rows)
        checks[name+'.paid_U'] = all(r['causal_use']['finite_U'].get('verified',False)
            for r in rows if r['causal_use']['finite_U']['committed'])
        checks[name+'.metadata'] = metadata['checkpoint_sha256'] == result['checkpoint_sha256']
        checks[name+'.delayed_identity'] = all(now.get('causal_program_delayed_fact') ==
            old['causal_program']['committed_episode_identity'] for old,now in zip(rows,rows[1:]))
        # Empty-slot allocation is distinguished from replacing a live version.
        # The initial complete audit establishes a contiguous 2219 occupied slots.
        checks[name+'.no_paid_calls'] = result['provider_calls'] == 0
        evaluation = {label:{metric:distribution([x[metric] for x in v['queries']])
                      for metric in ('score','mse')} for label,v in
                      [('before',result['evaluations'][0]),('after',result['evaluations'][-1])]}
        arms[name] = dict(cycle=result['cycle'], updates=result['updates'], external_bytes=result['external_bytes'],
            checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
            measured_contributions=len(observations), unique_versioned_records=len(identities),
            accepted=sum(r['records'] for r in receipts),
            benefits=distribution([x['benefit'] for x in observations]),
            positive=sum(x['benefit'] > 0 for x in observations),
            negative=sum(x['benefit'] < 0 for x in observations),
            positive_to_negative=sum(x['preference_before'] > 0 > x['preference_after'] for x in observations),
            negative_to_positive=sum(x['preference_before'] < 0 < x['preference_after'] for x in observations),
            preference_updates=distribution([x['preference_after']-x['preference_before'] for x in observations]),
            window_seconds=distribution([r['seconds'] for r in receipts]),
            replay_error_max=max((r['replay_logit_max_error'] for r in receipts
                                  if r['replay_logit_max_error'] is not None), default=None),
            episode_slots=[r['episode_slot'] for r in rows], merges=sum(r['episode_merged'] for r in rows),
            merged_chain=[dict(cycle=r['cycle'], gain=r['metal_g'],
                fact=r['causal_program']['committed_episode_identity'],
                paid_U_verified=r['causal_use']['finite_U'].get('verified',False),
                next_delayed_fact=rows[i+1].get('causal_program_delayed_fact') if i+1<len(rows) else None)
                for i,r in enumerate(rows) if r['episode_merged']],
            internal_events=sum(r['autonomous_event']['kind']=='internal' for r in all_rows),
            actual_U=distribution([r['metal_g'] for r in rows]), common_old_queries=evaluation)
    checks['same_source'] = all(r['initial_source_sha256'] ==
        'e9f817bcd3dd5f13b6edead2e5c317fab98f681bdf56544237d32e738e75e0a6' for r in results.values())
    checks['same_initial_queries'] = results['observe']['evaluations'][0] == results['apply']['evaluations'][0]
    same = {key:all(x[key] == y[key] for x,y in zip(records['observe'],records['apply']))
            for key in ('choice','choice_probability','utterance_hex','world','metal_g')}
    report = dict(valid=all(checks.values()), checks=checks, arms=arms, same_outputs=same,
        same_final_old_queries=results['observe']['evaluations'][-1] == results['apply']['evaluations'][-1],
        scope='Online arrived-language use, not held-out evaluation of the new retention objective.',
        remaining='Hot retention statistics do not change read dose or the eight-fact relation release. '
            'No full-capacity hot competition means no demonstrated retention-driven future benefit.',
        summary_world_calls=0, summary_optimizer_updates=0)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
    if not report['valid']:
        raise RuntimeError('native retention comparison contract failed')


if __name__ == '__main__':
    main()
