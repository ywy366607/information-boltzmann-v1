"""Complete-loop comparison of coherent native-cell successor execution."""
import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.audit_native_value_transition import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();arms={};checks={};sources=[];starts=[]
    controls=[('mean',Path('scratch/cgb_self_execution_loop_2493_20261119/native'),3),
        ('independent',Path('scratch/cgb_successor_distribution_2493_20261119/distribution'),4),
        ('joint',a.work/'joint',5)]
    for name,path,schema in controls:
        result=json.loads((path/'result.json').read_text(encoding='utf8'))
        config=json.loads((path/'body/last.metadata.json').read_text(encoding='utf8'))['config']
        rows=[json.loads(x) for x in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[r for r in rows if r['autonomous_event']['kind']=='external'];pairs=summarize(rows)
        checks[name+'.complete']=(result['status'],result['origin_cycle'],result['cycle'],result['updates'],result['external_bytes'])==('completed',2493,2497,16,3840)
        checks[name+'.version']=config['epistemic_value']['schema']==schema
        checks[name+'.world']=result['provider_calls']==0
        checks[name+'.finite_U']=all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed'])
        learned=[r['causal_program'].get('successor_learning') for r in rows[1:]]
        if schema==5:
            checks['joint.learning']=all(d and d['target_excluded'] and d['event'] in d['evidence'] for d in learned)
            routes=[]
            for r in rows:
                receipt=r['causal_use'];choice=receipt['choice']
                actual=receipt['epistemic_value']['successor_routes'][choice]
                routes.append(dict(cycle=r['cycle'],choice=choice,
                    response_branches=[dict(weights=d['weights'],joint=d['joint']) for d in actual]))
        else:routes=[]
        arms[name]=dict(actual_policy_discrepancy_MSE=mean(d['components']['transport']**2 for d in pairs),
            realized_future_MSE=mean(d['error']**2 for d in pairs),
            actual_use=[r['metal_g'] for r in rows],actions=[r['choice'] for r in rows],
            utterance_hex=[r['utterance_hex'] for r in rows],unknowns=[r['shared_unknown']['identity'] for r in rows],
            old_query_before=result['evaluations'][0]['mean_score'],old_query_after=result['evaluations'][-1]['mean_score'],
            seconds=result['invocation']['seconds'],pairs=pairs,routes=routes,aligned_learning=learned)
        sources.append(result['initial_source_sha256']);starts.append(result['evaluations'][0])
    restore=json.loads((a.work/'restore.json').read_text(encoding='utf8'))
    checks.update(same_source=len(set(sources))==1,same_initial_material=starts[0]==starts[1]==starts[2],
        complete_restore=restore['valid'] and restore['epistemic_schema']==5 and bool(restore['joint_successor']))
    report=dict(valid=all(checks.values()),checks=checks,arms=arms,source_sha256=sources[0])
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert report['valid'],checks
    doc=Path('docs/information_boltzmann/CGB_JOINT_SUCCESSOR_GEOMETRY_20261119.md')
    text='\n\n## 完整生活对照完成\n\n同2493源，各4轮/16更新/3840自然文本字节；联合分布及有限学习状态完整恢复。\n\n'
    text+='|指标|均值|独立分布|联合方向与原地址积分|\n|---|---:|---:|---:|\n'
    for label,key in [('下一策略期望差MSE','actual_policy_discrepancy_MSE'),
        ('后续真实收益MSE','realized_future_MSE'),('旧查询终点得分','old_query_after'),('运行秒数','seconds')]:
        text+='|'+label+'|'+'|'.join(f'{arms[n][key]:.9f}' for n in ('mean','independent','joint'))+'|\n'
    with doc.open('a',encoding='utf8') as f:f.write(text+f'\n完整结果：`{a.output.as_posix()}`。\n')
    print(json.dumps(dict(valid=True,checks=len(checks))),flush=True)


if __name__=='__main__':main()
