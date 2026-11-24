"""Whole-life use and wall time of native residual quadrature at same source."""
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
    config = json.loads((args.work/'joint/body/config.json').read_text(encoding='utf-8'))
    restore = json.loads((args.work/'restore.json').read_text(encoding='utf-8'))
    routes = [route for row in rows for action in row['causal_use']['epistemic_value']['successor_routes']
              for route in action]
    reconstruction = max(abs(r['baseline_mean']+sum(w*v for w, v in
        zip(r['weights'], r['residuals']))-r['expected'][0]) for r in routes)
    checks = dict(complete=(new['status'], new['origin_cycle'], new['cycle'], new['updates'],
        new['external_bytes']) == ('completed', 2493, 2497, 16, 3840),
        same_source=new['initial_source_sha256'] == old['initial_source_sha256'],
        same_initial_material=new['evaluations'][0] == old['evaluations'][0],
        same_law=config['epistemic_value']['schema'] == 5,
        residual=config['successor_quadrature']['mode'] == 'residual',
        restored=restore['valid'] and restore['successor_quadrature']['mode'] == 'residual',
        same_payment=all(row['causal_use']['reward_version'] == 'finite_relation_use_v5' for row in rows),
        finite_U=all(row['causal_use']['finite_U']['verified'] for row in rows
                     if row['causal_use']['finite_U']['committed']),
        successor_learning=all(row['causal_program']['successor_learning']['target_excluded'] for row in rows[1:]),
        mass=all(abs(sum(r['weights'])-1) < 1e-6 and
                 abs(sum(r['joint']['address_mass'].values())-1) < 1e-6 for r in routes),
        original_budget=all(r['schema'] == 2 and r['branches'] <= r['budget'] for r in routes),
        readout=reconstruction < 2e-6, paid_calls=new['provider_calls'] == 0)
    for arm in (control, candidate):
        arm['use_sum_per_minute'] = sum(arm['actual_use'])/(arm['seconds']/60)
        arm['fixed_old_gain_per_minute'] = (arm['old_query_after']-arm['old_query_before'])/(arm['seconds']/60)
    report = dict(valid=all(checks.values()), checks=checks, control=control, residual=candidate,
                  maximum_readout_error=reconstruction,
                  source_sha256=new['initial_source_sha256'],
                  speedup=control['seconds']/candidate['seconds'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert report['valid'], checks
    text = '\n\n## 同源整圈结果\n\n四轮/16更新/3840字节，完整恢复和原支付通过。\n\n'
    text += '|指标|原合批schema5|分层残差积分|\n|---|---:|---:|\n'
    for label, key in [('实际用途均值','mean_actual_use'),('用途和/分钟','use_sum_per_minute'),
                       ('共同旧事实改善/分钟','fixed_old_gain_per_minute'),
                       ('总秒数','seconds'),('后继策略估值差MSE','actual_policy_discrepancy_MSE')]:
        text += f'|{label}|{control[key]:.9g}|{candidate[key]:.9g}|\n'
    text += f'\n原公式重构最大差{reconstruction:.9g}。报告：`{args.output.as_posix()}`。\n'
    with Path('docs/information_boltzmann/CGB_NATIVE_RESIDUAL_QUADRATURE_20261119.md').open('a', encoding='utf-8') as f:
        f.write(text)
    print(json.dumps(dict(valid=True, speedup=report['speedup'],
                         mean_use=candidate['mean_actual_use'], report=str(args.output))))


if __name__ == '__main__':
    main()
