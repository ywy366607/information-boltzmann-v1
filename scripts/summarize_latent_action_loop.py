"""Full-life use/cost and learned proposal verification for latent action planning."""
import argparse
import json
from pathlib import Path

from scripts.summarize_joint_response import read_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original, _, reference = read_arm(Path('scratch/cgb_native_batch_scores_loop_2493_20261119/joint'))
    _, _, amortized = read_arm(Path('scratch/cgb_native_amortized_loop_2493_20261120/joint'))
    run, rows, candidate = read_arm(args.work/'joint')
    restored = json.loads((args.work/'restore.json').read_text(encoding='utf8'))
    config = json.loads((args.work/'joint/body/config.json').read_text(encoding='utf8'))
    routes = [route for row in rows for action in row['causal_use']['epistemic_value']['successor_routes'] for route in action]
    calibration = [row['causal_use']['native_amortization'] for row in rows]
    errors = []
    for route in routes:
        prediction = route['values'][0]
        estimate = sum(w*v for w,v in zip(route['weights'],prediction))
        estimate += sum(w*(v-prediction[i]) for i,w,v in zip(route['native_indices'],
            route['correction_weights'],route['native_values']))
        errors.append(abs(estimate-route['expected'][0]))
    planners = [r['action_planner'] for r in calibration]
    checks = dict(completed=(run['status'],run['origin_cycle'],run['cycle'],run['updates'],run['external_bytes'])==
        ('completed',2493,2497,16,3840), source=run['initial_source_sha256']==original['initial_source_sha256'],
        material=run['evaluations'][0]==original['evaluations'][0], original_law=config['epistemic_value']['schema']==5,
        configured=config['native_amortization']['planner']=='action_batch', restored=restored['valid'],
        restored_learning=restored['latent_action_records']==planners[-1]['records'],
        bounded=all(r['records']<=r['capacity'] for r in planners),
        real_labels=all(r['new_world_label'] for r in planners),
        original_U=all(row['causal_use']['finite_U']['verified'] for row in rows
                       if row['causal_use']['finite_U']['committed']),
        original_payment=all(row['causal_use']['reward_version']=='finite_relation_use_v5' for row in rows),
        successor_learning=all(row['causal_program']['successor_learning']['target_excluded'] for row in rows[1:]),
        mass=all(abs(sum(r['weights'])-1)<1e-6 for r in routes),
        correction=max(errors)<2e-6,
        full_support=all(all(q+1e-8>=p/2 for p,q in zip(r['weights'],r['proposal_weights'])) for r in routes),
        action_plans=all(r['planned_candidates']>0 for r in routes), paid_calls=run['provider_calls']==0)
    if config['native_amortization'].get('correction') == 'decision_budget':
        checks['decision_teacher_budget'] = all(0 < r['teacher_count'] <= 2 for r in planners)
    for arm in (reference,amortized,candidate):
        arm['use_sum_per_minute']=sum(arm['actual_use'])*60/arm['seconds']
        arm['fixed_old_gain_per_minute']=(arm['old_query_after']-arm['old_query_before'])*60/arm['seconds']
    report = dict(valid=all(checks.values()), checks=checks, control=reference, amortized=amortized,
        action_planner=candidate, calibration=calibration, maximum_readout_error=max(errors),
        speedup=reference['seconds']/candidate['seconds'],
        speedup_over_amortized=amortized['seconds']/candidate['seconds'],
        source_sha256=run['initial_source_sha256'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert report['valid'], checks
    lines=['\n\n## 同源完整四轮结果\n', '|指标|原合批|条件价值读出|批量行动规划|', '|---|---:|---:|---:|']
    for title,key in [('总秒数','seconds'),('实际用途均值','mean_actual_use'),
                      ('用途/分钟','use_sum_per_minute'),('旧事实改善/分钟','fixed_old_gain_per_minute'),
                      ('后继策略估值差MSE','actual_policy_discrepancy_MSE'),('真实后续收益MSE','realized_future_MSE')]:
        lines.append(f'|{title}|{reference[key]:.9g}|{amortized[key]:.9g}|{candidate[key]:.9g}|')
    lines.append(f'\n完整报告：`{args.output.as_posix()}`。检查通过；用途与时间按本次同源材料逐项比较。\n')
    with Path('docs/information_boltzmann/CGB_BATCHED_LATENT_ACTION_PLANNING_20261120.md').open('a',encoding='utf8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(valid=True,speedup=report['speedup'],use=candidate['mean_actual_use'])))


if __name__ == '__main__':
    main()
