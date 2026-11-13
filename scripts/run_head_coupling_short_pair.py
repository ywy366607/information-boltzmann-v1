"""Sequential, durable whole-body comparison after the measured speed repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def manifest():
    harness=Path('scripts/ib_local/mengli_harness')
    files=list(Path('scripts/ib_local').glob('*.py'))+list(harness.glob('*.py'))+list(harness.glob('*.mjs'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def summarize(path):
    rows=[json.loads(x) for x in (path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    initial=json.loads((path/'initial_validation.json').read_text())
    chunks=[]
    for i in range(0,len(rows),32):
        part=rows[i:i+32]
        average=lambda key:sum(float(r[key]) for r in part)/len(part)
        chunks.append(dict(first=part[0]['cycle'],last=part[-1]['cycle'],
            response_nll=average('response_nll'),mother_nll=average('mother_response_nll'),
            ambient_nll=average('ambient_response_nll'),g=average('metal_g'),
            choices={a:sum(r['choice']==a for r in part) for a in ('silence','generate','reuse')}))
    result=dict(cycles=len(rows),initial=initial,chunks=chunks,
        external_bytes=sum(r['external_bytes'] for r in rows),
        optimizer_updates=sum(r['updates'] for r in rows),
        max_policy_replay_error=max(r['policy_replay_max_error'] for r in rows),
        seconds=rows[-1]['elapsed_seconds'],
        fixed_validation=[dict(cycle=r['cycle'],**r['fixed_validation'])
                          for r in rows if 'fixed_validation' in r])
    if 'learned_head_coupling' in rows[0]:
        diag=[r['learned_head_coupling'] for r in rows]
        result['coupling']=dict(updated_cycles=sum(d['parameter_change_norm']>0 for d in diag),
            max_replay_error=max(d['replay_max_error'] for d in diag),
            layer_gradient_max=[max(d['per_layer_gradient_norm'][i] for d in diag) for i in range(4)],
            parameter_change_norm_sum=sum(d['parameter_change_norm'] for d in diag))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared-cache',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=128)
    parser.add_argument('--mother-context',choices=('segmented','dsh'))
    parser.add_argument('--mother-harness-url')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    frozen=manifest();write(args.output/'runtime_manifest.json',frozen)
    report=dict(status='running',source=str(args.checkpoint),
                source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                cycles_per_arm=args.cycles,arms={},controller_pid=os.getpid(),started=time.time())
    report['mother_context']=args.mother_context or 'inherited'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                     CGB_MOMENTUM_BACKEND='fused',PYTHONUTF8='1')
    write(args.output/'status.json',report)
    for arm in ('fixed','weighted','free'):
        if (args.output/'STOP').exists():
            report['status']='stopped';write(args.output/'status.json',report);return
        if manifest()!=frozen:raise RuntimeError('runtime changed during comparison')
        report['active_arm']=arm;write(args.output/'status.json',report)
        cmd=[sys.executable,'-m','scripts.ib_local.run_mother_continuation',
             '--checkpoint',str(args.checkpoint),'--output',str(args.output/arm),
             '--cycles',str(args.cycles),'--shared-cache',str(args.shared_cache),
             '--stop-file',str(args.output/'STOP'),'--validate-every','64',
             '--validation-bytes','8192']
        if arm!='fixed':cmd+=['--head-coupling',arm]
        from scripts.ib_local.run_mother_pair_queue import mother_context_args
        cmd+=mother_context_args(args.mother_context,args.mother_harness_url,args.output,arm)
        with (args.output/(arm+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            report['learner_pid']=child.pid;write(args.output/'status.json',report)
            returncode=child.wait()
        if returncode:
            report.update(status='failed',failed_arm=arm,returncode=returncode)
            write(args.output/'status.json',report);return
        complete=json.loads((args.output/arm/'completed.json').read_text())
        report['arms'][arm]=summarize(args.output/arm)
        write(args.output/'status.json',report)
        if complete['status']!='complete':
            report['status']='stopped';write(args.output/'status.json',report);return
    report.update(status='complete',finished=time.time())
    report.pop('active_arm',None);report.pop('learner_pid',None)
    write(args.output/'status.json',report);write(args.output/'comparison.json',report)
    lines=['# 可学习头间动力学：同源128轮筛选','',
           '| 势能 | OWT 初始/终点 | 妈妈侧评 初始/终点 | 实际更新数 |',
           '|---|---|---|---|']
    for arm,data in report['arms'].items():
        first,last=data['initial'],data['fixed_validation'][-1]
        lines.append(f"| {arm} | {first['owt_nll']:.6f} / {last['owt_nll']:.6f} | "
                     f"{first['mother_nll']:.6f} / {last['mother_nll']:.6f} | {data['optimizer_updates']} |")
    lines+=['','固定侧评无梯度、无在线导航；每个真实个体保持持续状态。',
            '这是单种子短预算筛选，在线后果评分和分叉后的不同实际输入见 comparison.json。']
    (args.output/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
