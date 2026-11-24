"""Publish the actual U/next-policy comparison for the shared self executor."""
import argparse
import json
from pathlib import Path
from statistics import mean
from scripts.audit_native_value_transition import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out={};checks={};sources=[];initial=[]
    for name,path in [('historical',a.control),('native_anchor',a.work/'native')]:
        r=json.loads((path/'result.json').read_text(encoding='utf8'))
        m=json.loads((path/'body/last.metadata.json').read_text(encoding='utf8'))
        rows=[json.loads(x) for x in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[x for x in rows if x['autonomous_event']['kind']=='external']
        pairs=summarize(rows)
        checks[name+'.budget']=(r['status'],r['origin_cycle'],r['cycle'],r['cycles'],r['updates'],r['external_bytes'])==('completed',2493,2497,4,16,3840)
        checks[name+'.mode']=m['config']['epistemic_value']['schema']==3 and m['config']['epistemic_value'].get('word_execution','historical')==name
        checks[name+'.U']=all(x['settlement_kind']=='finite_relation_use_v5' and
            (not x['causal_use']['finite_U']['committed'] or x['causal_use']['finite_U']['verified']) for x in rows)
        checks[name+'.world']=r['provider_calls']==0
        out[name]=dict(future_MSE=mean(x['error']**2 for x in pairs),
            model_value_bias_squared=mean(x['components']['transport']**2 for x in pairs),
            action_sampling_variance=mean(x['model_action_sampling_variance'] for x in pairs),
            old_query_score_before=mean(x['score'] for x in r['evaluations'][0]['queries']),
            old_query_score_after=mean(x['score'] for x in r['evaluations'][-1]['queries']),
            actions=[x['choice'] for x in rows],real_use=[x['metal_g'] for x in rows],pairs=pairs)
        sources.append(r['initial_source_sha256']);initial.append(r['evaluations'][0])
    restore=json.loads((a.work/'native_restore.json').read_text(encoding='utf8'))
    checks['native_full_restore']=restore['valid'] and restore['source_unchanged'] and restore['word_execution']=='native_anchor'
    checks['same_source']=sources[0]==sources[1];checks['same_initial_material']=initial[0]==initial[1]
    result=dict(valid=all(checks.values()),checks=checks,arms=out,
        control_reused=str(a.control),source_sha256=sources[0],payoff='original finite_relation_use_v5')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert result['valid'],[k for k,v in checks.items() if not v]
    doc=Path('docs/information_boltzmann/CGB_NATIVE_SELF_EXECUTION_20261119.md')
    text='\n\n## 四轮原生自我执行对照完成\n\n同2493源至2497，各4轮/16更新/3840自然文本字节；历史位移控制复用原schema3完整运行。\n\n|指标|历史位移|当前原生执行锚点|\n|---|---:|---:|\n'
    for label,key in [('后继收益MSE','future_MSE'),('下一实际策略期望偏差平方','model_value_bias_squared'),('共同旧查询终点得分','old_query_score_after')]:
        text+=f'|{label}|{out["historical"][key]:.9f}|{out["native_anchor"][key]:.9f}|\n'
    with doc.open('a',encoding='utf8') as f:f.write(text+f'\n逐项事件与恢复检查：`{a.output.as_posix()}`。\n')
    ledger=Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md')
    ledger.write_text('## 2026-11-19 原生自我执行四轮核对完成\n\n同2493至2497，原U/v5和完整恢复通过；结果见[CGB_NATIVE_SELF_EXECUTION_20261119.md](CGB_NATIVE_SELF_EXECUTION_20261119.md)末节。\n\n'+ledger.read_text(encoding='utf8'),encoding='utf8')
    print(json.dumps(dict(valid=True,checks=len(checks))))

if __name__=='__main__':main()
