"""Same schema5 life, comparing original and batched native word execution."""
import argparse
import json
from pathlib import Path

from scripts.summarize_joint_response import read_arm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    old,orows,control=read_arm(Path('scratch/cgb_joint_successor_2493_20261119/joint'))
    new,rows,candidate=read_arm(a.work/'joint')
    config=json.loads((a.work/'joint/body/config.json').read_text(encoding='utf8'))
    restored=json.loads((a.work/'restore.json').read_text(encoding='utf8'))
    use_error=max(abs(x['metal_g']-y['metal_g']) for x,y in zip(rows,orows))
    checks=dict(complete=(new['status'],new['origin_cycle'],new['cycle'],new['updates'],new['external_bytes'])==
        ('completed',2493,2497,16,3840),same_source=new['initial_source_sha256']==old['initial_source_sha256'],
        initial_material=new['evaluations'][0]==old['evaluations'][0],schema=config['epistemic_value']['schema']==5,
        runtime=config['native_word_runtime']['mode']=='batch',
        restored=restored['valid'] and restored['native_word_runtime']=='batch',
        same_actions=[r['choice'] for r in rows]==[r['choice'] for r in orows],
        same_words=[r['utterance_hex'] for r in rows]==[r['utterance_hex'] for r in orows],
        same_unknowns=candidate['unknowns']==control['unknowns'],use_numerics=use_error<1e-4,
        finite_U=all(r['causal_use']['finite_U']['verified'] for r in rows if r['causal_use']['finite_U']['committed']),
        paid_calls=new['provider_calls']==0)
    report=dict(valid=all(checks.values()),checks=checks,control=control,batch=candidate,
                maximum_actual_use_difference=use_error,speedup=control['seconds']/candidate['seconds'])
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    text='\n\n## 完整四轮加速对照\n\n'
    text+=f'执行及逐轮对照：{checks}。最大用途差{use_error:.9g}。\n\n'
    text+=f"原总时间{control['seconds']:.3f}秒；合批{candidate['seconds']:.3f}秒，实测{report['speedup']:.3f}倍。\n"
    text+=f'完整结果：`{a.output.as_posix()}`。\n'
    with Path('docs/information_boltzmann/CGB_USE_COST_FRONTIER_20261119.md').open('a',encoding='utf8') as f:f.write(text)
    assert report['valid'],checks
    with Path('docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md').open('a',encoding='utf8') as f:
        f.write('\n\n## 2026-11-19 schema5话语轨迹合批整圈完成\n\n四轮/16更新/3840字节、原U及完整恢复通过；逐轮动作/话语/未知项与原schema5一致，用途差在数值门内。实测速度与完整调用树见[CGB_USE_COST_FRONTIER_20261119.md](CGB_USE_COST_FRONTIER_20261119.md)。\n')
    print(json.dumps(dict(valid=True,speedup=report['speedup'],maximum_use_difference=use_error)))


if __name__=='__main__':main()
