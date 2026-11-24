"""Whole-loop contract and common-material summary for the 2469 value test."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('E:/cgb_epistemic_value_test_2469_20261118'))
    p.add_argument('--output',type=Path,default=Path('results/published/cgb_epistemic_value_loop_2469_20261118.json'))
    args = p.parse_args()
    arms, checks, actual_rows = {}, {}, {}
    initial = []
    for name in ('control','context'):
        folder = args.work/name
        result = json.loads((folder/'result.json').read_text(encoding='utf8'))
        events = [json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in events if r['autonomous_event']['kind']=='external']
        actual_rows[name] = rows
        checks[name+'.completed'] = result['status']=='completed'
        checks[name+'.budget'] = result['cycles']==4 and result['updates']==16 and result['external_bytes']==3840
        checks[name+'.provider'] = result['provider_calls']==0
        checks[name+'.identity'] = result['runtime_revision']==1
        checks[name+'.source'] = result['initial_source_sha256']=='886144783de4e8158f609d187c72eb60ff98dc528d33e312d0eae17513145bac'
        checks[name+'.reward'] = all(r['causal_use']['reward_version']=='finite_targeted_use_v2' for r in rows)
        checks[name+'.finite_U'] = all(not r['causal_use']['finite_U']['committed'] or
                                      r['causal_use']['finite_U']['verified'] for r in rows)
        before, after = result['evaluations'][0], result['evaluations'][-1]
        initial.append([(q['relation'],q['query'],q['score']) for q in before['queries']])
        forecasts = [r['causal_use'].get('epistemic_value') for r in rows]
        if name=='context':
            checks['context.recorded_every_event'] = all(f and f['schema']==1 for f in forecasts)
            checks['context.finite_forecasts'] = all(all(abs(v)<float('inf') for v in
                f['contextual_total']+f['old_total']) for f in forecasts)
        arms[name] = dict(cycle=result['cycle'],updates=result['updates'],bytes=result['external_bytes'],
            internal_events=sum(r['autonomous_event']['kind']=='internal' for r in events),
            seconds=result['invocation']['seconds'],
            common_before=dict(score=before['mean_score'],mse=before['mean_mse']),
            common_after=dict(score=after['mean_score'],mse=after['mean_mse']),
            gains=[r['metal_g'] for r in rows],choices=[r['choice'] for r in rows],
            utterances=[r['utterance_hex'] for r in rows],
            finite_U=[r['causal_use']['finite_U'] for r in rows],
            forecasts=forecasts, checkpoint=result['checkpoint'],
            checkpoint_sha256=result['checkpoint_sha256'])
    checks['same_initial_material'] = initial[0]==initial[1]
    probability_tv = [sum(abs(x-y) for x,y in zip(a['choice_probability'],b['choice_probability']))/2
                      for a,b in zip(actual_rows['control'],actual_rows['context'])]
    next_errors = []
    contextual_rows = actual_rows['context']
    for index, row in enumerate(contextual_rows[:-1]):
        f = row['causal_use']['epistemic_value']
        choice = ('silence','generate','reuse').index(row['choice'])
        target = 10*contextual_rows[index+1]['metal_g']
        next_errors.append(dict(prediction_cycle=row['cycle'], target_cycle=contextual_rows[index+1]['cycle'],
            actual_next_target=target, next_individual_U=contextual_rows[index+1]['causal_use']['finite_U']['committed'],
            old=f['old_future'][choice], contextual=f['contextual_future'][choice],
            old_squared_error=(f['old_future'][choice]-target)**2,
            contextual_squared_error=(f['contextual_future'][choice]-target)**2))
    out=dict(valid=all(checks.values()),checks=checks,arms=arms,
        changed_choices=sum(a!=b for a,b in zip(arms['control']['choices'],arms['context']['choices'])),
        changed_utterances=sum(a!=b for a,b in zip(arms['control']['utterances'],arms['context']['utterances'])),
        same_world_records=[a['world']==b['world'] for a,b in zip(actual_rows['control'],actual_rows['context'])],
        choice_probability_tv=probability_tv,
        within_context_changed_value_argmax=sum(
            max(range(len(f['old_total'])),key=lambda j:f['old_total'][j]) !=
            max(range(len(f['contextual_total'])),key=lambda j:f['contextual_total'][j])
            for f in arms['context']['forecasts']),
        next_real_settlement_errors=next_errors,
        scope='four-event complete-loop integration; path-specific returns retained separately; common old material compared')
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'], [k for k,v in checks.items() if not v]
    print(json.dumps(dict(valid=True,checks=len(checks),changed_choices=out['changed_choices'],
        common_after={k:v['common_after'] for k,v in arms.items()})))


if __name__=='__main__':
    main()
