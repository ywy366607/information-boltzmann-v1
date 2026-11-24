"""Trace identified-mode interventions through the actual learning loop.

Pairs the old pending prediction with its ACTUAL native log_score target,
then follows optimizer events and subsequent choices/world landings. No
comparison of endpoint states at unequal event budgets is made.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_native_path_composition import load


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def observed(directory):
    source=directory/'operations.jsonl.gz'
    with gzip.open(source,'rt',encoding='utf8') as f:
        rows=[json.loads(line) for line in f]
    calls={r['data']['call_id']:r for r in rows if r['kind']=='relation_signal'}
    predictions=[]
    with (directory/'tensors.bin').open('rb') as stream:
        for cid in (947,950):
            row=calls[cid]
            d=row['data']
            assert d['operation']=='mechanism' and d['state']['identity']==9
            pred=d['prediction']
            key=pred['mean']['tensor_ref']['sha256']
            scores=[r for r in calls.values() if r['data']['operation']=='log_score' and
                r['event']==row['event'] and r['update_clock']==row['update_clock'] and
                r['data']['prediction']['mean']['tensor_ref']['sha256']==key]
            assert len(scores)==1
            score=scores[0]['data']
            target=load(stream,score['target'])
            mean=load(stream,pred['mean'])
            predictions.append(dict(call_id=cid,event=row['event'],update_clock=row['update_clock'],
                relation=[9,1],transport=d['mother'] is not None,
                target_sha256=score['target']['tensor_ref']['sha256'],
                actual_next_body_boundary_mse=float(((mean-target)**2).mean()),
                native_log_score=float(load(stream,score['score']).mean()),
                mean_norm=float(np.linalg.norm(mean)),target_norm=float(np.linalg.norm(target))))
    updates={r['update_clock']:r['data']['parameters'] for r in rows
             if r['kind']=='all_parameter_update_participation'}
    return dict(predictions=predictions,updates=updates,
        sources=[dict(path=str(p),sha256=sha(p)) for p in (source,directory/'tensors.bin')])


def external(result,n=None):
    rows=[r for r in result['receipts'] if r['autonomous_event']['kind']=='external']
    return rows if n is None else rows[:n]


def compare_choices(baseline,result):
    right=external(result)
    left=external(baseline,len(right))
    assert len(left)==len(right)
    rows=[]
    for i,(a,b) in enumerate(zip(left,right)):
        pa,pb=np.array(a['choice_probability']),np.array(b['choice_probability'])
        rows.append(dict(external_index=i+1,baseline_choice=a['choice'],intervened_choice=b['choice'],
            choice_probability_max_abs_change=float(abs(pa-pb).max()),
            choice_total_variation=float(abs(pa-pb).sum()/2),
            baseline_world_anchor=a['world']['anchor'],intervened_world_anchor=b['world']['anchor'],
            exact_utterance_equal=a['utterance_hex']==b['utterance_hex'],
            baseline_program_loss=a['causal_program_loss'],intervened_program_loss=b['causal_program_loss'],
            baseline_settlement=a['settlement'],intervened_settlement=b['settlement'],
            selected_relation=b['causal_program']['roles']['identity'],
            committed_episode=b['causal_program']['committed_episode_identity'],
            unknown_revision=b['shared_unknown']['revision']))
    return dict(events=rows,changed_sampled_choices=sum(r['baseline_choice']!=r['intervened_choice'] for r in rows),
        changed_world_landings=sum(r['baseline_world_anchor']!=r['intervened_world_anchor'] for r in rows),
        changed_utterances=sum(not r['exact_utterance_equal'] for r in rows),
        max_choice_probability_change=max(r['choice_probability_max_abs_change'] for r in rows))


def compare_updates(left,right):
    answer=[]
    for clock in sorted(right):
        a={r['name']:r for r in left[clock]}
        b={r['name']:r for r in right[clock]}
        assert a.keys()==b.keys()
        diff=[]
        for name in a:
            grad=abs(a[name]['gradient_norm']-b[name]['gradient_norm'])
            step=abs(a[name]['sampled_applied_step_rms']-b[name]['sampled_applied_step_rms'])
            if grad or step:
                diff.append(dict(name=name,gradient_norm_difference=grad,sampled_applied_step_rms_difference=step,
                    baseline_gradient_norm=a[name]['gradient_norm'],intervened_gradient_norm=b[name]['gradient_norm']))
        answer.append(dict(update_clock=clock,gradient_participants=len(a),
            gradient_norm_changed_parameters=sum(d['gradient_norm_difference']!=0 for d in diff),
            sampled_applied_step_rms_changed_parameters=sum(d['sampled_applied_step_rms_difference']!=0 for d in diff),
            maximum_gradient_norm_difference=max((d['gradient_norm_difference'] for d in diff),default=0),
            maximum_sampled_step_rms_difference=max((d['sampled_applied_step_rms_difference'] for d in diff),default=0),
            changes=sorted(diff,key=lambda d:d['gradient_norm_difference'],reverse=True)))
    return answer


def compare_physical_prefix(work, name):
    """Same committed event boundary, all 32 layers, no unequal endpoints."""
    changed=[]
    for source in sorted((work/name/'observations').glob('physical_*.npz')):
        with np.load(source) as b, np.load(work/'baseline/observations'/source.name) as a:
            assert np.array_equal(a['kind'],b['kind']) and np.array_equal(a['event'],b['event'])
            assert np.array_equal(a['updates'],b['updates'])
            i=-1
            assert a['kind'][i]=='event_after'
            row=dict(event=int(a['event'][i]),update_clock=int(a['updates'][i]))
            for key in ('node_energy','theta','omega'):
                x,y=a[key][i].astype(float),b[key][i].astype(float)
                d=y-x
                row[key]=dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(x)),
                    maximum_absolute_change=float(abs(d).max()),
                    changed_layers=int(np.any(d!=0,axis=-1).sum()),
                    per_layer_change_l2=np.linalg.norm(d,axis=-1).tolist())
            changed.append(row)
    return changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    names=['baseline','bridge_cut','matched_cut','pending_mode_cut','pending_matched']
    runs={name:json.loads((args.work/name/'result.json').read_text(encoding='utf8')) for name in names}
    assert all(r['before']==runs['baseline']['before'] for r in runs.values())
    observations={name:observed(args.work/name/'observations') for name in names}
    target=runs['pending_mode_cut']['pending_mode_interventions']
    control=runs['pending_matched']['pending_mode_interventions']
    assert len(target)==len(control)==2
    magnitude=[]
    for a,b in zip(target,control):
        assert a['original_call_id']==b['original_call_id']
        relative=abs(a['delta_frobenius']-b['delta_frobenius'])/a['delta_frobenius']
        assert relative<1e-6 and a['relative_harmonic_after']<1e-6 and b['relative_harmonic_change']<1e-6
        magnitude.append(dict(call_id=a['original_call_id'],target_delta=a['delta_frobenius'],
            matched_delta=b['delta_frobenius'],relative_norm_mismatch=relative,
            target_harmonic_fraction_remaining=a['relative_harmonic_after'],
            matched_harmonic_relative_change=b['relative_harmonic_change']))
    for i in (0,1):
        assert len({observations[name]['predictions'][i]['target_sha256'] for name in names})==1
    paths={name:compare_choices(runs['baseline'],r) for name,r in runs.items() if name!='baseline'}
    native_updates={name:compare_updates(observations['baseline']['updates'],observations[name]['updates'])
                    for name in ('pending_mode_cut','pending_matched')}
    report=dict(status='whole_learning_loop_mode_interventions_measured',
        full_start_states_equal=True,main_checkpoint_and_archive_unchanged=
            json.loads((args.work/'pending_summary.json').read_text(encoding='utf8'))['main_checkpoint_and_archive_unchanged'],
        target='fixed measured structural component and separately the reachable H1-response input directions of saved pending relation9',
        structural=dict(budget_external_events=8,gradient_updates_per_arm=32,
            group_read_calls={name:runs[name]['identified_group_read_calls'] for name in names[:3]},
            affected_read_calls={name:runs[name]['affected_scope_read_calls'] for name in names[:3]},
            same_checked_neural_optimizer_SP_pending_world_rng_after=
                runs['baseline']['after']==runs['bridge_cut']['after'],
            final_R={name:runs[name]['final_R'] for name in names[:3]},
            baseline_reads=runs['baseline']['read_counts'],
            active_control_changed_fields=[k for k in runs['baseline']['after']
                if runs['baseline']['after'][k]!=runs['matched_cut']['after'][k]],
            active_control_read_pairs=len(runs['matched_cut']['read_calls']),
            active_control_loaded_fact_sets_changed=sum(a['facts']!=b['facts'] for a,b in
                zip(runs['baseline']['read_calls'],runs['matched_cut']['read_calls'])),
            active_control_loaded_unknown_sets_changed=sum(a['unknowns']!=b['unknowns'] for a,b in
                zip(runs['baseline']['read_calls'],runs['matched_cut']['read_calls']))),
        pending=dict(budget_external_events=3,gradient_updates_per_arm=12,
            baseline_is_first_three_events_of_the_recorded_eight_event_baseline=True,
            mean_mse={name:float(np.mean([d['actual_next_body_boundary_mse'] for d in observations[name]['predictions']]))
                      for name in ('baseline','pending_mode_cut','pending_matched')},
            actual_target_scores={name:observations[name]['predictions'] for name in names},
            matched_amplitudes=magnitude,actual_interventions=dict(target=target,matched=control),
            native_update_comparisons=native_updates),
        whole_loop_paths=paths,
        all_layer_physical_prefix={name:compare_physical_prefix(args.work,name)
                                   for name in ('pending_mode_cut','pending_matched')},
        mapping_proofs=json.loads((args.work/'pending_input_lift.json').read_text(encoding='utf8')),
        actual_optimizer_calls=sum(r['actual_optimizer_calls'] for r in runs.values()),
        actual_text_world_interactions=sum(len(external(r)) for r in runs.values()),
        paid_world_calls=sum(r['provider_calls'] for r in runs.values()),
        mechanism_scope='input directions selected by H1 response; no claim that other full-edge Hodge components are held fixed',
        sources=[dict(path=str(args.work/name/'result.json'),sha256=sha(args.work/name/'result.json')) for name in names]
              +[s for o in observations.values() for s in o['sources']],
        code=[dict(path=str(ROOT/name),sha256=sha(ROOT/name)) for name in
              ('scripts/run_native_relation_mode_use.py','scripts/native_pending_mode_intervention.py',
               'scripts/analyze_native_relation_mode_use.py')])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status=report['status'],pending_mean_mse=report['pending']['mean_mse'],
        downstream={name:{k:v for k,v in item.items() if k!='events'}for name,item in paths.items()},
        structural=report['structural'],optimizer_calls=report['actual_optimizer_calls']),ensure_ascii=False))


if __name__=='__main__':
    main()
