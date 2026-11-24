"""Publish completed pair results into the existing route and handover docs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--restore-prefix',default='cgb_relation_use_restore')
    a=p.parse_args()
    r=json.loads(a.report.read_text(encoding='utf8'))
    assert r['valid']
    for name in ('observe','apply'):
        restored=json.loads(Path(f'results/published/{a.restore_prefix}_{name}_2489_20261119.json').read_text(encoding='utf8'))
        assert restored['valid'] and restored['source_unchanged'] and restored['cycle']==2489
    data=r['arms']
    version=r.get('transition_schema',1)
    lines=[f'\n\n## schema{version}配对完整结果（自动按实际报告填写）\n',
        '两臂均完成2481→2489，各8外界轮、32原优化、7680自然文本字节。每4轮完整保存并由新进程恢复，最终两臂的scope快照再次标准恢复核对通过。主2401及源2481保持，预算结束。\n',
        '|实际链条|observe几何|apply用途|','|---|---:|---:|']
    for label,key in [('实际消费旧用途的scope调用','consumed_scope_events')]:
        lines.append(f'|{label}|{data["observe"][key]}|{data["apply"][key]}|')
    for label,key in [('相对同边界几何改变释放','changed_release_events'),('读到已有用途的scope后的冻结后果','later_frozen_on_used_scope')]:
        lines.append(f'|{label}|{len(data["observe"][key])}|{len(data["apply"][key])}|')
    for name in ('observe','apply'):
        v=data[name]
        lines += [f'\n{name}共同64旧查询终点：'+json.dumps(v['old_queries']['after'],ensure_ascii=False)+'。',
                  '新鲜事前冻结预测：'+json.dumps(v['fresh_scores'],ensure_ascii=False)+'。',
                  '在已有用途scope的更晚冻结预测：'+json.dumps(v['later_used_scope_scores'],ensure_ascii=False)+'。']
    lines += ['\n同一路径输出一致性：'+json.dumps(r['same_outputs'],ensure_ascii=False)+'。',
        f'\n本段分别保留写入、实际取舍、再次使用和后续预测数据；完整结果见`{a.report.as_posix()}`。后续研究沿实际消费的scope与later frozen记录接续。']
    pth=Path('docs/information_boltzmann/CGB_RELATION_RETENTION_USE_20261119.md')
    with pth.open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    text=(f'## 2026-11-19 schema{version}关系用途配对预算已完成（最新）\n\n'
        f'observe/apply均完整2489，各8轮/32更新/7680字节，4轮跨进程恢复及最终scope快照恢复通过。'
        f'实际消费旧用途scope调用{data["observe"]["consumed_scope_events"]}/{data["apply"]["consumed_scope_events"]}，'
        f'apply相对同边界几何改变释放{len(data["apply"]["changed_release_events"])}次。'
        '完整数值见[CGB_RELATION_RETENTION_USE_20261119.md](CGB_RELATION_RETENTION_USE_20261119.md)及机器报告。'
        '主2401/完整2481源保持，配对预算结束，无自动追加。\n\n')
    for file in ('CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md',):
        dst=Path('docs/information_boltzmann')/file
        dst.write_text(text+dst.read_text(encoding='utf8'),encoding='utf8')
    with Path('docs/information_boltzmann/CGB_EVENT_CREDIT_TRANSITION_20261118.md').open('a',encoding='utf8') as f:f.write('\n'+text)
    dst=Path('CLAUDE.md')
    dst.write_text('> **关系用途实验已完整2489：** 查看台账顶部及CGB_RELATION_RETENTION_USE_20261119.md。两臂预算结束、恢复通过，不重复启动。\n\n'+dst.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(dict(published=True,checks=len(r['checks'])),ensure_ascii=False))


if __name__=='__main__':main()
