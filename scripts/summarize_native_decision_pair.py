"""Publish the bounded full-loop prospective-decision comparison."""
import argparse
import json
from pathlib import Path
from statistics import mean
from scripts.audit_native_value_transition import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();checks={};arms={};initials={};sources={}
    for name,schema in (('control',2),('native',3)):
        folder=a.work/name
        result=json.loads((folder/'result.json').read_text(encoding='utf8'))
        metadata=json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
        restore=json.loads((a.work/f'{name}_restore.json').read_text(encoding='utf8'))
        rows=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[x for x in rows if x['autonomous_event']['kind']=='external']
        checks[name+'.budget']=(result['origin_cycle'],result['cycle'],result['cycles'],
            result['updates'],result['external_bytes'])==(2493,2497,4,16,3840)
        checks[name+'.schema']=metadata['config']['epistemic_value']['schema']==schema
        checks[name+'.reward']=all(x['settlement_kind']=='finite_relation_use_v5' for x in rows)
        checks[name+'.paid_U']=all(x['causal_use']['finite_U'].get('verified',False)
            for x in rows if x['causal_use']['finite_U']['committed'])
        checks[name+'.restore']=restore['valid'] and restore['source_unchanged'] and restore['epistemic_schema']==schema
        checks[name+'.no_paid_world']=result['provider_calls']==0
        pairs=summarize(rows)
        arms[name]=dict(cycle=result['cycle'],updates=result['updates'],external_bytes=result['external_bytes'],
            pairs=pairs,future_MSE=mean(x['error']**2 for x in pairs),
            model_value_bias_squared=mean(x['components']['transport']**2 for x in pairs),
            action_sampling_variance=mean(x['model_action_sampling_variance'] for x in pairs),
            old_query_score_before=mean(q['score'] for q in result['evaluations'][0]['queries']),
            old_query_score_after=mean(q['score'] for q in result['evaluations'][-1]['queries']),
            actions=[x['choice'] for x in rows],real_use=[x['metal_g'] for x in rows])
        initials[name]=result['evaluations'][0];sources[name]=result['initial_source_sha256']
    checks['same_initial_material']=initials['control']==initials['native']
    checks['same_initial_body']=sources['control']==sources['native']
    out=dict(valid=all(checks.values()),checks=checks,arms=arms,source_sha256=sources['control'],
        comparison='schema2 R/Q/P continuation vs schema3 predicted native decision programs',
        payoff='same finite_relation_use_v5, complete learning/navigation/internal activity/world feedback retained')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],[k for k,v in checks.items() if not v]
    doc=Path('docs/information_boltzmann/CGB_NATIVE_DECISION_CONTINUATION_20261119.md')
    with doc.open('a',encoding='utf8') as f:
        f.write('\n\n## 四轮整圈配对已完成\n\n两臂同2493至2497，各4轮/16更新/3840自然文本字节，原v5支付与完整恢复通过。\n\n'
            '|实际轨迹指标|schema2|schema3|\n|---|---:|---:|\n'+
            ''.join(f'|{label}|{arms["control"][key]:.9f}|{arms["native"][key]:.9f}|\n' for label,key in
                [('后继真实收益MSE','future_MSE'),('后继策略期望偏差平方','model_value_bias_squared'),
                 ('同模型行动抽样方差','action_sampling_variance'),('共同旧查询终点得分','old_query_score_after')])+
            f'\n逐项事件、误差分解及检查：`{a.output.as_posix()}`。本段预算完成。\n')
    ledger=Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md')
    ledger.write_text('## 2026-11-19 schema3原生决策延续配对已完成\n\n'
        '同2493源两臂均保存2497，各4轮/16更新/3840字节，原有限U支付及完整恢复通过。'
        '结果见[CGB_NATIVE_DECISION_CONTINUATION_20261119.md](CGB_NATIVE_DECISION_CONTINUATION_20261119.md)末节；主2401及研究源保持。\n\n'+
        ledger.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(dict(valid=True,checks=len(checks))))


if __name__=='__main__':main()
