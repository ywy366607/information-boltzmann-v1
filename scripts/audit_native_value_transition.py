"""Exact full-transition value-error decomposition on existing real events.

The admission term below is a retrospective condition, NOT a prospective gate.
All next-policy terms use the actual next decision's probabilities and action
values, after its intervening physical, learning and relation transitions.
"""
import argparse
import json
from pathlib import Path


def summarize(events):
    rows = [r for r in events if 'causal_use' in r]
    out = []
    for old, now in zip(rows, rows[1:]):
        f = old['causal_use'].get('epistemic_value')
        nf = now['causal_use'].get('epistemic_value')
        if not f or not nf:
            continue
        k = ('silence','generate','reuse').index(old['choice'])
        j = ('silence','generate','reuse').index(now['choice'])
        q = nf['local_expected']
        on_policy = sum(p*v for p,v in zip(now['choice_probability'],q))
        selected = q[j]
        variance=sum(p*(v-on_policy)**2 for p,v in zip(now['choice_probability'],q))
        committed = now['causal_use']['finite_U']['committed']
        admitted = selected if committed else 0.
        actual = 10*now['metal_g']
        terms = dict(transport=f['contextual_future'][k]-on_policy,
                     policy_sampling=on_policy-selected,
                     admission=selected-admitted,
                     observed_response=admitted-actual)
        error = f['contextual_future'][k]-actual
        residual = error-sum(terms.values())
        assert abs(residual)<1e-12
        retained=set(old['causal_use']['finite_U'].get('supports',[]))
        next_queries={v['query'] for v in now['causal_use']['comparisons']}
        out.append(dict(prediction_cycle=old['cycle'], next_cycle=now['cycle'],
            previous_unknown=old['causal_use']['unknown'],next_unknown=now['causal_use']['unknown'],
            revised_support=sorted(retained),next_query_scope=sorted(next_queries),
            shared_facts=len(retained&next_queries),
            support_query_jaccard=(len(retained&next_queries)/len(retained|next_queries)
                                  if retained|next_queries else 1.),
            predicted_future=f['contextual_future'][k], next_action_values=q,
            next_probabilities=now['choice_probability'], next_policy_value=on_policy,
            selected_expected=selected, next_merged=now['episode_merged'],
            individual_commit=committed, actual=actual,
            error=error, components=terms, reconstruction_residual=residual,
            model_action_sampling_variance=variance,
            conditional_model_risk=terms['transport']**2+variance,
            conditional_model_risk_components=dict(value_bias_squared=terms['transport']**2,
                action_sampling_variance=variance),
            delayed_samples=now['internal_transition_samples'],
            delayed_fact=now.get('causal_program_delayed_fact')))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('E:/cgb_epistemic_value_test_2469_20261118/context'))
    p.add_argument('--output',type=Path,default=Path('results/published/cgb_value_transition_2469_20261118.json'))
    a=p.parse_args()
    events=[json.loads(x) for x in (a.work/'events.jsonl').read_text(encoding='utf8').splitlines()]
    rows=summarize(events)
    out=dict(source=str(a.work/'events.jsonl'), pairs=rows,
        equation='preview-actual = transport + policy_sampling + admission + observed_response',
        measurement='actual coupled learning/world trajectory; no new training or world calls',
        admission_role='retrospective observed branch, never supplied as a pre-world feature',
        squared_error_attribution='components add as signed errors; their squares include cross terms')
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':
    main()
