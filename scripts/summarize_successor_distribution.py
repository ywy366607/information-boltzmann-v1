"""Publish the bounded complete-loop acceptance, preserving per-event results."""
import argparse
import json
from pathlib import Path
import re
from statistics import mean

from scripts.audit_native_value_transition import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('work','control','output','measurement'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();arms={};checks={};sources=[];starts=[]
    for name,path,schema in [('mean',a.control,3),('distribution',a.work/'distribution',4)]:
        result=json.loads((path/'result.json').read_text(encoding='utf8'))
        metadata=json.loads((path/'body/last.metadata.json').read_text(encoding='utf8'))
        events=[json.loads(x) for x in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[r for r in events if r['autonomous_event']['kind']=='external']
        pairs=summarize(rows)
        checks[name+'.budget']=(result['origin_cycle'],result['cycle'],result['cycles'],result['updates'],result['external_bytes'])==(2493,2497,4,16,3840)
        checks[name+'.complete']=result['status']=='completed'
        config=metadata['config']['epistemic_value']
        checks[name+'.schema']=config['schema']==schema and config['word_execution']=='native_anchor'
        checks[name+'.finite_U']=all(r['settlement_kind']=='finite_relation_use_v5' and
            (not r['causal_use']['finite_U']['committed'] or r['causal_use']['finite_U']['verified']) for r in rows)
        checks[name+'.world']=result['provider_calls']==0
        learning=[r['causal_program'].get('successor_learning') for r in rows[1:]]
        if schema==4:
            checks['distribution.aligned_learning']=all(r and r['schema']==1 and r['target_excluded'] and
                r['event'] in r['evidence'] for r in learning)
        arms[name]=dict(actual_policy_discrepancy_MSE=mean(x['components']['transport']**2 for x in pairs),
            realized_future_MSE=mean(x['error']**2 for x in pairs),
            old_query_score_before=result['evaluations'][0]['mean_score'],
            old_query_score_after=result['evaluations'][-1]['mean_score'],
            seconds=result['invocation']['seconds'],actions=[r['choice'] for r in rows],
            actual_use=[r['metal_g'] for r in rows],utterance_hex=[r['utterance_hex'] for r in rows],
            aligned_learning=learning,pairs=pairs)
        sources.append(result['initial_source_sha256']);starts.append(result['evaluations'][0])
    restore=json.loads((a.work/'distribution_restore.json').read_text(encoding='utf8'))
    measurement=json.loads(a.measurement.read_text(encoding='utf8'))
    checks.update(same_source=sources[0]==sources[1],same_initial_queries=starts[0]==starts[1],
        restore=restore['valid'] and restore['epistemic_schema']==4 and restore['source_unchanged'],
        measured_three_transitions=measurement['valid'])
    out=dict(valid=all(checks.values()),checks=checks,arms=arms,source_sha256=sources[0],
        measurement=str(a.measurement),control_reused=str(a.control),
        purpose='bounded full-loop mechanism acceptance with original finite-U and actual delayed labels')
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],checks
    doc=Path('docs/information_boltzmann/CGB_SUCCESSOR_DISTRIBUTION_REPAIR_20261119.md')
    text='\n\n## 四步实码验收完成\n\n同2493源至2497，各4轮、16次真实优化、3840自然文本字节；完整恢复通过。\n\n'
    text+='|指标|原均值路径|协调后继分布|\n|---|---:|---:|\n'
    for label,key in [('下一实际策略期望差MSE','actual_policy_discrepancy_MSE'),
                      ('实际后续收益MSE','realized_future_MSE'),('共同旧查询终点得分','old_query_score_after'),
                      ('完整运行秒数','seconds')]:
        text+=f'|{label}|{arms["mean"][key]:.9f}|{arms["distribution"][key]:.9f}|\n'
    text+=f'\n机器结果：`{a.output.as_posix()}`；三条同边界比较：`{a.measurement.as_posix()}`。\n'
    original=doc.read_text(encoding='utf8')
    original=re.sub(r'^当前游标：.*$', '当前游标：四步实码验收完成，结果及用途评价见末节。',original,flags=re.MULTILINE)
    doc.write_text(original+text,encoding='utf8')
    ledger=Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md')
    ledger.write_text('## 2026-11-19 后继分布四步实码验收完成\n\n'
        '真实转移比较、完整学习与恢复结果见[CGB_SUCCESSOR_DISTRIBUTION_REPAIR_20261119.md](CGB_SUCCESSOR_DISTRIBUTION_REPAIR_20261119.md)末节。\n\n'
        +ledger.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(dict(valid=True,checks=len(checks),arms=arms),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
