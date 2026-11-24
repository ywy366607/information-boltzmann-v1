"""Replay the existing full life trajectory and retain its native value ensemble.

No new estimator enters the policy. The observer copies values already computed
by prospective_decision.value, then checks its native branch averaging against
the selected production continuation. Learning, physical activity, navigation,
world feedback, U and saving remain in the original continuation runner.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys


@contextmanager
def checkpoint_without_observer(engine):
    """Temporary measurement functions stay outside the complete body state."""
    observer=vars(engine).get('_utter')
    remove=bool(getattr(observer,'_native_ensemble_observer',False))
    if remove:
        del engine._utter
    try:
        yield
    finally:
        if remove:
            engine._utter=observer


def summarize_existing(args):
    """Publish completed measurements independently of the observer checkpoint."""
    from scripts.audit_native_value_transition import summarize
    from scripts.run_native_preview_restore_comparison import assert_replayed_event
    events=[json.loads(line) for line in (args.work/'replay/events.jsonl').read_text(encoding='utf8').splitlines()]
    reference=[json.loads(line) for line in (args.reference/'events.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(events)==len(reference),(len(events),len(reference))
    for i,(actual,recorded) in enumerate(zip(events,reference)):
        assert_replayed_event(actual,recorded,i)
    observations=json.loads((args.work/'ensemble_progress.json').read_text(encoding='utf8'))
    pairs=summarize(events)
    for row,pair in zip(observations,pairs):
        target=pair['next_policy_value'];v=row['original_forecast']
        values=[branch['value'] for branch in row['branches']]
        model_squared_distance=sum((x-target)**2 for x in values)/len(values)
        squared_mean_distance=(v-target)**2
        residual=model_squared_distance-squared_mean_distance-row['response_branch_variance']
        assert abs(residual)<1e-6,residual
        stages=[v]+[case['value'] for case in row['controlled_substitutions']['cases']]+[target]
        differences=[x-y for x,y in zip(stages,stages[1:])]
        row.update(actual_next_policy_value=target,realized_next_policy_discrepancy_squared=squared_mean_distance,
            actual_next_unknown=pair['next_unknown'],target_in_branch_range=row['range'][0]<=target<=row['range'][1],
            squared_distance_to_all_model_branches=model_squared_distance,
            finite_variance_identity_residual=residual,ordered_value_chain=stages,
            signed_differences=differences,reconstruction_residual=v-target-sum(differences))
    reference_result=json.loads((args.reference/'result.json').read_text(encoding='utf8'))
    source=hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            source.update(block)
    external=[r for r in events if r.get('autonomous_event',{}).get('kind')=='external']
    diagnosis=[]
    for row,next_event in zip(observations,external[1:]):
        cases={c['case']:c for c in row['controlled_substitutions']['cases']}
        target=row['actual_next_policy_value']
        names=('actual_response_predicted_boundary','actual_response_actual_latent',
               'actual_response_actual_raw','actual_response_actual_raw_identified')
        selected={name:dict(value=cases[name]['value'],absolute_error=abs(cases[name]['value']-target),
                    relation=cases[name]['details'][0]['address']['relation']) for name in names}
        diagnosis.append(dict(cycle=row['cycle'],target=target,
            actual_relation=next_event['causal_use']['relation'][0],cases=selected,
            fact_identity_effect=selected[names[3]]['value']-selected[names[2]]['value']))
    mean_errors={name:sum(d['cases'][name]['absolute_error'] for d in diagnosis)/len(diagnosis)
                 for name in names}
    base_error=mean_errors[names[0]]
    diagnostic_summary=dict(transitions=diagnosis,mean_absolute_errors=mean_errors,
        latent_substitution_error_reduction=1-mean_errors[names[1]]/base_error,
        latent_substitution_correct_relations=sum(d['cases'][names[1]]['relation']==d['actual_relation'] for d in diagnosis),
        original_correct_relations=sum(d['cases'][names[0]]['relation']==d['actual_relation'] for d in diagnosis),
        max_fact_identity_effect=max(abs(d['fact_identity_effect']) for d in diagnosis))
    evaluations=json.loads((args.work/'replay/evaluations.json').read_text(encoding='utf8'))
    reference_evaluations=json.loads((args.reference/'evaluations.json').read_text(encoding='utf8'))
    checks=dict(four_external=len(external)==4,
        updates=sum(r['updates'] for r in external)==16,
        external_bytes=sum(r['external_bytes'] for r in external)==3840,
        all_events_replayed=len(events)==len(reference),
        four_decisions=len(observations)==4,
        original_ensemble_exact=all(abs(r['original_forecast']-r['ensemble_mean'])<2e-6 for r in observations),
        actual_ports_match=all(max(r['controlled_substitutions'][key] for key in
            ('source_max_error','action_max_error'))<3e-5 for r in observations[:-1]),
        substitutions_private=all(all(r['controlled_substitutions']['checks'].values()) for r in observations[:-1]),
        fixed_queries_exact=evaluations==reference_evaluations,
        source_unchanged=source.hexdigest()==reference_result['initial_source_sha256'])
    saved=(args.work/'replay/result.json').exists()
    out=dict(valid=all(checks.values()),checks=checks,reference=str(args.reference),
        source_sha256=source.hexdigest(),observations=observations,updates=16,bytes=3840,
        diagnosis=diagnostic_summary,
        provider_calls=sum(r['world'].get('provider_calls',0) for r in external),
        measurement_complete=True,observer_checkpoint_saved=saved,
        retained_complete_reference_checkpoint=reference_result['checkpoint'],
        meaning='Realized next-policy discrepancy includes transition/question variation; measured branch spread remains model spread.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],checks
    if not saved:
        status=dict(status='completed_measurements',cycle=2497,cycles=4,updates=16,external_bytes=3840,
            observer_checkpoint_saved=False,checkpoint_error='local observed_utter closure; observer now excluded at save',
            complete_reference_checkpoint=reference_result['checkpoint'],report=str(args.output))
        (args.work/'replay/status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(valid=out['valid'],measurements=3,events=len(events),observer_checkpoint_saved=saved)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summarize-existing',action='store_true')
    args = p.parse_args()
    if args.summarize_existing:
        summarize_existing(args)
        return
    import torch
    from scripts.ib_local import run_mother_continuation as loader
    from scripts.ib_local import prospective_decision as prospective
    from scripts.ib_local import mother_checkpoint
    from scripts.run_native_preview_restore_comparison import main as run
    from scripts.audit_native_value_transition import summarize
    from scripts.audit_native_future_boundary import archived_vectors,measure_cases
    args.work.mkdir(parents=True,exist_ok=False)
    reference_events=[json.loads(line) for line in (args.reference/'events.jsonl').read_text(encoding='utf8').splitlines()]
    reference_external=[r for r in reference_events if 'causal_use' in r]
    archives=list((args.reference/'body').glob('causal_archive_*.sqlite'))
    assert len(archives)==1
    labels={int(r['causal_program']['committed_episode_identity'].split(':')[1]):
            archived_vectors(archives[0],int(r['causal_program']['committed_episode_identity'].split(':')[1]))
            for r in reference_external[:-1]}
    observations, calls = [], []
    original_prepare, original_value = loader.prepare_continuation, prospective.value
    original_save=mother_checkpoint.save

    def save(engine,*a,**kw):
        with checkpoint_without_observer(engine):
            return original_save(engine,*a,**kw)

    def value(engine, state, source, action, mother, latent, momentum):
        result = original_value(engine, state, source, action, mother, latent, momentum)
        calls.append(dict(source=source.detach().clone(),action=action.detach().clone(),
            mother=mother.detach().cpu().tolist(),value=float(result[0]),
            trace=copy.deepcopy(engine.prospective_decision_last)))
        return result

    def prepare(*a, **kw):
        loaded = original_prepare(*a, **kw)
        engine = loaded.engine
        utter = engine._utter
        def observed_utter():
            calls.clear()
            words, record = utter()
            s = engine._internal_source.reshape(1,-1)
            action = engine._internal_action[record['choice']:record['choice']+1]
            selected = [c for c in calls if torch.equal(c['source'],s) and torch.equal(c['action'],action)]
            # Silence and unavailable reuse share an endpoint, hence identical
            # calls occur twice. Keep one complete quadrature, not both copies.
            count = engine._program_forecasts['hypotheses'].shape[1]
            assert selected and len(selected)%count == 0, (len(selected),count)
            values = torch.tensor([c['value'] for c in selected],dtype=torch.float64)
            forecast = float(engine._program_forecasts['epistemic_after_value'][record['choice']])
            mean = float(values.mean())
            assert abs(mean-forecast)<2e-6,(mean,forecast)
            groups = []
            for c in selected[:count]:
                detail = c['trace'][0]
                qv = detail['question_values'];pq = detail['question_probability']
                qmean = sum(x*y for x,y in zip(qv,pq))
                qvar = sum(w*(x-qmean)**2 for x,w in zip(qv,pq))
                groups.append(dict(value=c['value'],mother=c['mother'],
                    address=detail['address'],dynamics=detail['dynamics'],
                    question_variance=qvar,question_probability=pq,question_values=qv,
                    unknowns=[case['identity'] for case in detail['cases']],
                    scopes=[case['scope'] for case in detail['cases']]))
            observations.append(dict(cycle=loaded.cycle+len(observations)+1,
                unknown=engine._program_question_roles['unknown_identity'],
                selected=record['choice'],quadrature=count,matching_calls=len(selected),
                original_forecast=forecast,ensemble_mean=mean,
                response_branch_variance=float((values-mean).square().mean()),
                response_branch_std=float(values.std(unbiased=False)),
                range=[float(values.min()),float(values.max())],
                mean_question_variance=sum(g['question_variance'] for g in groups)/count,
                branches=groups))
            if len(observations)<len(reference_external):
                reference=reference_external[len(observations)-1]
                episode=int(reference['causal_program']['committed_episode_identity'].split(':')[1])
                engine._audit_actual_choice=record['choice']
                observations[-1]['controlled_substitutions']=measure_cases(engine,labels[episode],episode)
                del engine._audit_actual_choice
            (args.work/'ensemble_progress.json').write_text(json.dumps(observations,
                ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            calls.clear()
            return words,record
        observed_utter._native_ensemble_observer=True
        engine._utter=observed_utter
        return loaded

    loader.prepare_continuation,prospective.value,mother_checkpoint.save=prepare,value,save
    try:
        sys.argv=[sys.argv[0],'--checkpoint',str(args.checkpoint),
            '--work',str(args.work/'replay'),'--cycles','4',
            '--native-decision','--native-word-execution',
            '--replay-prefix',str(args.reference/'events.jsonl')]
        run()
        summarize_existing(args)
    finally:
        loader.prepare_continuation,prospective.value=original_prepare,original_value
        mother_checkpoint.save=original_save


if __name__=='__main__':
    main()
