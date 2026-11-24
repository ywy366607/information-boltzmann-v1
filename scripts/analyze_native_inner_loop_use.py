"""Read the complete four-arm internal-preview/feedback experiment."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.measure_native_inner_loop import analyze, load, sha, quantiles


def ports(directory):
    obs=directory/'observations'
    previews,commits,participants=[],[],[]
    with (obs/'tensors.bin').open('rb') as stream:
        for line in gzip.open(obs/'operations.jsonl.gz','rt',encoding='utf8'):
            row=json.loads(line);d=row['data']
            if row['kind']=='inner_loop_preview_ports':
                f=load(stream,d['selector_features']);prior=load(stream,d['selector_prior'])
                assert f.shape==(2,388) and prior.shape==(2,)
                out=dict(event=row['event'],held=d['hold_preview'],
                    selected=d['admission']['selected'],probability=d['admission']['probability'],
                    content_feature_gap=float(np.linalg.norm(f[1,:384]-f[0,:384])),
                    continuation_prior_gap=float(prior[1]-prior[0]),
                    role_tags=f[:,-4:].tolist())
                if d['hold_preview']:
                    assert out['content_feature_gap']==0 and out['continuation_prior_gap']==0
                previews.append(out)
            elif row['kind']=='inner_loop_feedback_commit':
                proposal=load(stream,d['proposed_delta']);actual=load(stream,d['committed_delta'])
                if d['hold_feedback']:assert not np.any(actual)
                else:assert np.array_equal(proposal,actual)
                commits.append(dict(event=row['event'],update_clock=row['update_clock']+1,
                    held=d['hold_feedback'],proposed_norm=float(np.linalg.norm(proposal)),
                    committed_norm=float(np.linalg.norm(actual))))
            elif row['kind']=='all_parameter_update_participation':
                participants.append(dict(event=row['event'],update_clock=row['update_clock'],parameters=d['parameters']))
    return dict(previews=previews,feedback=commits,parameter_updates=participants)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('work',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    names=('baseline','bridge_cut','matched_cut','order_only')
    reports,signals,reads={},{},{}
    for name in names:
        reports[name],signals[name]=analyze(args.work/name)
        reads[name]=ports(args.work/name)
    results={name:json.loads((args.work/name/'result.json').read_text(encoding='utf8')) for name in names}
    base=reports['baseline']
    comparisons={}
    # Equal external budget; autonomous event counts and actual input paths
    # are outcomes, explicitly retained rather than event-number matched.
    for name in names[1:]:
        r=reports[name]
        pairs=[]
        for a,b in zip(base['actions'],r['actions']):
            pa,pb=np.array(a['probability']),np.array(b['probability'])
            pairs.append(dict(baseline_event=a['event'],branch_event=b['event'],
                probability_max_abs=float(abs(pa-pb).max()),
                choice_changed=a['choice']!=b['choice'],anchor_changed=a['world_anchor']!=b['world_anchor'],
                utterance_changed=a['utterance_hex']!=b['utterance_hex'],
                heldout_pre_nll_delta=b['heldout_pre_nll']-a['heldout_pre_nll'],
                program_loss_delta=b['causal_program_loss']-a['causal_program_loss']))
        last0=base['phase']['samples'][-1]['sample'];last1=r['phase']['samples'][-1]['sample']
        delta=signals[name][f'X_o_{last1}']-signals['baseline'][f'X_o_{last0}']
        pa,pb=reads['baseline']['parameter_updates'],reads[name]['parameter_updates']
        changed_parameters=set()
        update_differences=[]
        for a,b in zip(pa,pb):
            ma={x['name']:x for x in a['parameters']};mb={x['name']:x for x in b['parameters']}
            entries=[]
            for key in ma.keys()|mb.keys():
                if ma.get(key)!=mb.get(key):
                    changed_parameters.add(key)
                    if key in ma and key in mb:
                        entries.append(dict(name=key,
                            gradient_norm_abs_change=abs(ma[key]['gradient_norm']-mb[key]['gradient_norm']),
                            baseline_gradient_norm=ma[key]['gradient_norm'],branch_gradient_norm=mb[key]['gradient_norm'],
                            baseline_sampled_step_rms=ma[key]['sampled_applied_step_rms'],
                            branch_sampled_step_rms=mb[key]['sampled_applied_step_rms']))
            if entries:
                update_differences.append(dict(update_clock=a['update_clock'],
                    changed_count=len(entries),largest_gradient_norm_changes=sorted(entries,
                        key=lambda x:x['gradient_norm_abs_change'],reverse=True)[:8]))
        comparisons[name]=dict(external_pairs=pairs,changed_choices=sum(x['choice_changed'] for x in pairs),
            changed_world_landings=sum(x['anchor_changed'] for x in pairs),changed_utterances=sum(x['utterance_changed'] for x in pairs),
            max_probability_difference=max(x['probability_max_abs'] for x in pairs),
            final_X_o_difference_by_layer=np.linalg.norm(delta,axis=1).tolist(),
            final_X_o_difference_norm=float(np.linalg.norm(delta)),
            changed_parameter_participation=sorted(changed_parameters),
            update_differences=update_differences,
            carried_equal={k:results[name]['after'][k]==results['baseline']['after'][k] for k in results['baseline']['after']})
    first_previews=[reads[name]['previews'][0] for name in names]
    first_pr=np.array([r['probability'] for r in first_previews])
    assert first_previews[0]['event']==first_previews[1]['event']==first_previews[2]['event']==first_previews[3]['event']
    # Commit switch begins at the later optimizer: its first selector agrees.
    assert np.array_equal(first_pr[0],first_pr[2]) and np.array_equal(first_pr[1],first_pr[3])
    probability=np.array([[a['probability'] for a in reports[n]['actions']] for n in names])
    interaction=probability[0]-probability[1]-probability[2]+probability[3]
    ends=[signals[n][f'X_o_{reports[n]["phase"]["samples"][-1]["sample"]}'] for n in names]
    phase_interaction=ends[0]-ends[1]-ends[2]+ends[3]
    array_path=args.output.with_suffix('.npz')
    np.savez_compressed(array_path,**{f'{name}_{key}':value for name in names for key,value in signals[name].items()})
    result=dict(status='completed',work=str(args.work),
        arrays=dict(path=str(array_path),sha256=sha(array_path)),
        arms={n:dict(summary=reports[n]['summary'],ports=reads[n],
             phase=reports[n]['phase'],
             preview_pairs=reports[n]['preview_pairs'],learning_updates=reports[n]['learning_updates'],
             committed_event_changes=reports[n]['committed_event_changes'],actions=reports[n]['actions'],
             sources=reports[n]['sources']) for n in names},
        first_same_boundary_preview=first_previews,
        comparisons=comparisons,
        factorial_contrast=dict(definition='live/live - held/live - live/held + held/held',
            choice_probability_by_external_index=interaction.tolist(),
            max_abs_choice_probability=float(abs(interaction).max()),
            final_X_o_norm=float(np.linalg.norm(phase_interaction)),
            final_X_o_by_layer=np.linalg.norm(phase_interaction,axis=1).tolist(),
            interpretation='paired finite trajectory effect under common exogenous RNG, preserving endogenous navigation and learning'))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(first_preview=first_previews,
        comparisons={n:{k:v for k,v in r.items() if k in ('changed_choices','changed_world_landings','changed_utterances','max_probability_difference','final_X_o_difference_norm')} for n,r in comparisons.items()},
        factorial=result['factorial_contrast']),ensure_ascii=False))


if __name__=='__main__':main()
