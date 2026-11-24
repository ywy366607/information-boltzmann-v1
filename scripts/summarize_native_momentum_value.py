"""Same full body: carry the known P condition through epistemic continuation."""
import argparse
import json
from pathlib import Path
from statistics import mean
from scripts.audit_native_value_transition import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    checks={};arms={};initial={};source={}
    for name,schema in (('control',1),('carry',2)):
        folder=a.work/name
        result=json.loads((folder/'result.json').read_text(encoding='utf8'))
        meta=json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
        xs=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[x for x in xs if x['autonomous_event']['kind']=='external']
        checks[name+'.budget']=(result['status'],result['origin_cycle'],result['cycle'],result['cycles'],
            result['updates'],result['external_bytes'])==('completed',2489,2493,4,16,3840)
        checks[name+'.schema']=meta['config']['epistemic_value']['schema']==schema
        checks[name+'.same_reward']=all(x['settlement_kind']=='finite_relation_use_v5' for x in rows)
        checks[name+'.paid_U']=all(x['causal_use']['finite_U'].get('verified',False)
            for x in rows if x['causal_use']['finite_U']['committed'])
        checks[name+'.reported_schema']=all(x['causal_use']['epistemic_value']['schema']==schema for x in rows)
        checks[name+'.no_paid_calls']=result['provider_calls']==0
        restored=json.loads((a.work/f'{name}_restore.json').read_text(encoding='utf8'))
        checks[name+'.restore']=restored['valid'] and restored['source_unchanged'] and restored['epistemic_schema']==schema
        initial[name]=result['evaluations'][0];source[name]=result['initial_source_sha256']
        pairs=summarize(rows)
        arms[name]=dict(cycle=result['cycle'],checkpoint_sha256=result['checkpoint_sha256'],
            updates=result['updates'],external_bytes=result['external_bytes'],pairs=pairs,
            future_MSE=mean(x['error']**2 for x in pairs),
            model_value_bias_squared=mean(x['components']['transport']**2 for x in pairs),
            action_sampling_variance=mean(x['model_action_sampling_variance'] for x in pairs),
            old_query_score_before=mean(q['score'] for q in result['evaluations'][0]['queries']),
            old_query_score_after=mean(q['score'] for q in result['evaluations'][-1]['queries']),
            actual_payoffs=[x['metal_g'] for x in rows],actions=[x['choice'] for x in rows])
    checks['same_initial_material']=initial['control']==initial['carry']
    checks['same_initial_body']=source['control']==source['carry']
    out=dict(valid=all(checks.values()),checks=checks,arms=arms,
        source_sha256=source['control'],scope='four real coupled learning events per arm; persistent R/Q, original U, actor, world, optimizer and S/P',
        difference='epistemic schema1 vs schema2: prospective response includes known projected P; one-step carried-state forecast',
        payoff='original finite_relation_use_v5, not language NLL')
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],[k for k,v in checks.items() if not v]
    if a.publish:
        text=('\n\n## P条件延续：真实四轮配对完成\n\n'
            'control/carry均从完整2489至2493，各4外界轮、16原优化、3840自然文本字节；原v5结算、持存事实、导航、S/P和全参数更新保留，恢复核对通过。\n\n'
            '|实际轨迹指标|control|carry P|\n|---|---:|---:|\n'+
            ''.join(f'|{label}|{arms["control"][key]:.9f}|{arms["carry"][key]:.9f}|\n' for label,key in
              [('后继真实收益MSE','future_MSE'),('后继策略模型期望偏差平方','model_value_bias_squared'),
               ('同模型动作抽样方差','action_sampling_variance'),('共同旧查询终点log-score','old_query_score_after')])+
            f'\n逐项结果及检查：`{a.output.as_posix()}`。该预算完成，原主2401及两条研究源均保持。\n')
        doc=Path('docs/information_boltzmann/CGB_ACTOR_EXPECTATION_CONTRACT_20261119.md')
        with doc.open('a',encoding='utf8') as f:f.write(text)
        ledger=Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md')
        ledger.write_text('## 2026-11-19 P条件延续配对已完成\n\n'
            '两臂均完整2493，4轮/16更新/3840字节，原v5支付及完整恢复通过。'
            '结果见[CGB_ACTOR_EXPECTATION_CONTRACT_20261119.md](CGB_ACTOR_EXPECTATION_CONTRACT_20261119.md)末节及'
            f'`{a.output.as_posix()}`。主2401和研究源保持，预算结束。\n\n'+ledger.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(dict(valid=True,checks=len(checks),arms=arms),ensure_ascii=False))


if __name__=='__main__':main()
