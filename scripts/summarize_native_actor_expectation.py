"""Bind the original finite-action risk identity to complete learning events."""
import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.audit_native_value_transition import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--actor-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    actor=json.loads(a.actor_report.read_text(encoding='utf8'))
    assert actor['source_unchanged'] and actor['world_calls']==actor['optimizer_updates']==0
    assert actor['encoded_actor']['logits_max_error']<3e-5
    assert actor['candidate_policy_decomposition']['reconstruction_error']<3e-5
    arms={}
    for name in ('observe','apply'):
        events=[json.loads(x) for x in (a.work/name/'events.jsonl').read_text(encoding='utf8').splitlines()]
        pairs=summarize(events)
        assert len(pairs)==7
        arms[name]=dict(pairs=pairs,summary=dict(
            next_policy_value_bias_squared=mean(x['components']['transport']**2 for x in pairs),
            model_action_sampling_variance=mean(x['model_action_sampling_variance'] for x in pairs),
            observed_response_error_squared=mean(x['components']['observed_response']**2 for x in pairs),
            actual_future_MSE=mean(x['error']**2 for x in pairs),
            changed_unknown=sum(x['previous_unknown']!=x['next_unknown'] for x in pairs),
            mean_support_query_jaccard=mean(x['support_query_jaccard'] for x in pairs),
            max_reconstruction_residual=max(abs(x['reconstruction_residual']) for x in pairs)))
    out=dict(valid=True,arms=arms,actual_actor=actor['encoded_actor'],
        actual_boundary_candidate_policy=actor['candidate_policy_decomposition'],
        learning_scope_cross=actor['learning_scope_cross'],
        candidate_condition_decomposition=actor['candidate_condition_decomposition'],
        source=str(a.work),actor_report=str(a.actor_report),
        risk_identity='E_action[(v-q(A))^2]=(v-E_actual_actor[q])^2+Var_actual_actor[q(A)]',
        payoff='q is the original pre-response calibrated finite-U use, in 10*g units; actual is the later original settlement',
        interpretation='sampling variance concerns the model q(A); observed response error is separately retained',
        new_training=0,new_world_calls=0)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(valid=True,arms={k:v['summary'] for k,v in arms.items()},
        actor=out['actual_actor'],candidate_policy=out['actual_boundary_candidate_policy']),ensure_ascii=False))


if __name__=='__main__':main()
