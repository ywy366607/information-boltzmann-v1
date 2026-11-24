"""Same-source full-world validation of conditional native decision learning."""
import argparse
import json
from pathlib import Path

from scripts.summarize_joint_response import read_arm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    old, _, control = read_arm(Path('scratch/cgb_native_batch_scores_loop_2493_20261119/joint'))
    new, rows, candidate = read_arm(args.work/'joint')
    restore = json.loads((args.work/'restore.json').read_text(encoding='utf-8'))
    config = json.loads((args.work/'joint/body/config.json').read_text(encoding='utf-8'))
    routes = [r for row in rows for action in row['causal_use']['epistemic_value']['successor_routes'] for r in action]
    reports = [r['causal_use']['native_amortization'] for r in rows]
    reconstruction = []
    for r in routes:
        if r['allocated_draws'] == r['budget']:
            values = r['native_values']
            rebuilt = values[0]+sum(w*(v-values[0]) for w, v in zip(r['weights'], values))
        else:
            prediction = r['values'][0]
            rebuilt = sum(w*v for w, v in zip(r['weights'], prediction))+sum(
                w*(v-prediction[i]) for i, w, v in zip(r['native_indices'], r['correction_weights'], r['native_values']))
        reconstruction.append(abs(rebuilt-r['expected'][0]))
    checks = dict(complete=(new['status'], new['origin_cycle'], new['cycle'], new['updates'],
        new['external_bytes']) == ('completed', 2493, 2497, 16, 3840),
        same_source=old['initial_source_sha256']==new['initial_source_sha256'],
        same_initial_material=old['evaluations'][0]==new['evaluations'][0],
        same_law=config['epistemic_value']['schema']==5,
        configured=config['native_amortization']['schema']==1,
        restored=restore['valid'] and restore['native_amortization']==config['native_amortization'],
        bounded=all(r['records']<=r['capacity'] for r in reports),
        real_learning=all(r['new_world_label'] for r in reports),
        finite_U=all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed']),
        same_payment=all(r['causal_use']['reward_version']=='finite_relation_use_v5' for r in rows),
        successor_learning=all(r['causal_program']['successor_learning']['target_excluded'] for r in rows[1:]),
        original_mass=all(abs(sum(r['weights'])-1)<1e-6 for r in routes),
        readout=max(reconstruction)<2e-6, paid_calls=new['provider_calls']==0)
    for arm in (control, candidate):
        arm['use_sum_per_minute']=sum(arm['actual_use'])*60/arm['seconds']
        arm['fixed_old_gain_per_minute']=(arm['old_query_after']-arm['old_query_before'])*60/arm['seconds']
    report = dict(valid=all(checks.values()), checks=checks, control=control, amortized=candidate,
        source_sha256=new['initial_source_sha256'], calibration=reports,
        maximum_readout_error=max(reconstruction), speedup=control['seconds']/candidate['seconds'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert report['valid'], checks
    text='\n\n## 同源四轮结果\n\n原U、实际反馈学习及完整恢复检查通过。\n\n'
    text+='|指标|原合批schema5|条件端口学习|\n|---|---:|---:|\n'
    for title, key in [('实际用途均值','mean_actual_use'),('用途和/分钟','use_sum_per_minute'),
                       ('共同旧事实改善/分钟','fixed_old_gain_per_minute'),('总秒数','seconds'),
                       ('后继策略估值差MSE','actual_policy_discrepancy_MSE')]:
        text+=f'|{title}|{control[key]:.9g}|{candidate[key]:.9g}|\n'
    text+=f'\n实际学习和精算计数：`{args.output.as_posix()}`。\n'
    with Path('docs/information_boltzmann/CGB_CONDITIONAL_NATIVE_AMORTIZATION_20261120.md').open('a',encoding='utf-8') as f:
        f.write(text)
    print(json.dumps(dict(valid=True, speedup=report['speedup'], use=candidate['mean_actual_use'])))


if __name__=='__main__':
    main()
