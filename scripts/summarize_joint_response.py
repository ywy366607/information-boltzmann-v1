"""Same-source complete-loop use and cost comparison, schema5 versus schema6."""
import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.audit_native_value_transition import summarize


def read_arm(path):
    result = json.loads((path/'result.json').read_text(encoding='utf8'))
    rows = [json.loads(line) for line in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
    rows = [r for r in rows if r['autonomous_event']['kind'] == 'external']
    pairs = summarize(rows)
    return result, rows, dict(
        actual_policy_discrepancy_MSE=mean(p['components']['transport']**2 for p in pairs),
        realized_future_MSE=mean(p['error']**2 for p in pairs),
        response_component_MSE=mean(p['components']['observed_response']**2 for p in pairs),
        actual_use=[r['metal_g'] for r in rows], mean_actual_use=mean(r['metal_g'] for r in rows),
        actions=[r['choice'] for r in rows],
        unknowns=[r['shared_unknown']['identity'] for r in rows],
        utterance_hex=[r['utterance_hex'] for r in rows], pairs=pairs,
        old_query_before=result['evaluations'][0]['mean_score'],
        old_query_after=result['evaluations'][-1]['mean_score'],
        seconds=result['invocation']['seconds'],
        response_joint_scores=[r['causal_program'].get('mother_joint_prequential') for r in rows],
        runtime_profile=result.get('runtime_profile'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old, old_rows, control = read_arm(Path('scratch/cgb_joint_successor_2493_20261119/joint'))
    new, rows, candidate = read_arm(a.work/'joint')
    restore = json.loads((a.work/'restore.json').read_text(encoding='utf8'))
    config = json.loads((a.work/'joint/body/config.json').read_text(encoding='utf8'))
    checks = dict(
        complete=(new['status'], new['origin_cycle'], new['cycle'], new['updates'], new['external_bytes']) ==
                 ('completed', 2493, 2497, 16, 3840),
        same_source=old['initial_source_sha256'] == new['initial_source_sha256'],
        same_initial_queries=old['evaluations'][0] == new['evaluations'][0],
        same_payment=all(r['causal_use']['reward_version'] == 'finite_relation_use_v5' for r in rows),
        finite_U=all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed']),
        aligned_successor_learning=all(r['causal_program']['successor_learning']['target_excluded'] for r in rows[1:]),
        joint_response_observed=all(r['causal_program']['mother_joint_prequential'] is not None for r in rows),
        version=config['epistemic_value']['schema'] == 6,
        restored=restore['valid'] and restore['epistemic_schema'] == 6,
        natural_world=new['provider_calls'] == 0,
        cost_recorded=bool(candidate['runtime_profile']))
    out = dict(valid=all(checks.values()), checks=checks,
               control_schema5=control, response_schema6=candidate,
               source_sha256=new['initial_source_sha256'])
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    assert out['valid'], checks
    text = '\n\n## 同源完整运行结果\n\n各4轮、16更新、3840自然文本字节；实际U/v5、原生行动及完整恢复核对通过。\n\n'
    text += '|指标|schema5后继联合|schema6回应与后继联合|\n|---|---:|---:|\n'
    for label, key in [('下一策略期望差MSE','actual_policy_discrepancy_MSE'),
        ('实际后续收益MSE','realized_future_MSE'),('实际回应误差分量MSE','response_component_MSE'),
        ('实际用途均值','mean_actual_use'),('固定旧事实终点分数','old_query_after'),('总秒数','seconds')]:
        text += f'|{label}|{control[key]:.9f}|{candidate[key]:.9f}|\n'
    text += f'\n完整有符号误差分解及实际调用耗时：`{a.output.as_posix()}`。\n'
    with Path('docs/information_boltzmann/CGB_JOINT_RESPONSE_LOOP_20261119.md').open('a', encoding='utf8') as f:
        f.write(text)
    with Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md').open('a', encoding='utf8') as f:
        f.write('\n\n## 2026-11-19 联合回应—后继整圈完成\n\n同2493源4轮/16更新/3840字节，原有限U/v5及schema6恢复通过；结果见[CGB_JOINT_RESPONSE_LOOP_20261119.md](CGB_JOINT_RESPONSE_LOOP_20261119.md)末节。主2401、共同2493源保持。\n')
    print(json.dumps(dict(valid=True, checks=len(checks), control_seconds=control['seconds'],
                          joint_response_seconds=candidate['seconds'])), flush=True)


if __name__ == '__main__':
    main()
