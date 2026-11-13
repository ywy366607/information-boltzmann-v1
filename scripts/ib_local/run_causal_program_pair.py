"""Run the documented whole-body mother-world comparison on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def runtime_manifest():
    # Cover the actual inherited runtime, not just the newly added files.
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path('scripts/ib_local').glob('*.py'))}


def summary(directory):
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    result=dict(cycles=len(rows),chunks=[])
    for begin in range(0,len(rows),64):
        chunk=rows[begin:begin+64]
        mean=lambda name:sum(float(row[name]) for row in chunk)/len(chunk)
        result['chunks'].append(dict(first=chunk[0]['cycle'],last=chunk[-1]['cycle'],
            response_nll=mean('response_nll'),heldout_post_nll=mean('heldout_post_nll'),
            actual_g=mean('metal_g'),self_bytes=mean('self_bytes'),
            choices={name:sum(row['choice']==name for row in chunk) for name in ('silence','generate','reuse')},
            policy_replay_error=max(row['policy_replay_max_error'] for row in chunk)))
    if rows and 'causal_program' in rows[0]:
        programs=[r['causal_program'] for r in rows]
        result['program']=dict(final_active_relations=programs[-1]['active_relations'],
            identities_used=len({p['roles']['identity'] for p in programs}),
            committed=sum(p['evidence_committed'] for p in programs),
            repeated_identity_cycles=len(programs)-len({p['roles']['identity'] for p in programs}),
            question_weight_updated=sum(p['question_weight_step']>0 for p in programs),
            shadow_delivery_nonzero=sum(any(abs(x)>0 for x in r['shadow_preview']['delivered_offset']) for r in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=512)
    args=p.parse_args()
    if args.output.exists() or args.cycles<1: raise ValueError('new output directory and positive budget required')
    args.output.mkdir(parents=True)
    manifest=runtime_manifest()
    write(args.output/'runtime_manifest.json',manifest)
    report=dict(status='running',source=str(args.checkpoint),cycles_per_arm=args.cycles,
                arms={},start_time=time.time(),meaning='same complete origin; real responsive mother; sequential GPU')
    write(args.output/'status.json',report)
    env=os.environ.copy(); env['OMP_NUM_THREADS']='1'; env['MKL_NUM_THREADS']='1'
    for name,extra in (('joint_program',['--causal-program']),('inherited',[])):
        if (args.output/'STOP').exists():
            report['status']='stopped'; write(args.output/'status.json',report); return
        if runtime_manifest()!=manifest: raise RuntimeError('runtime changed during the controlled comparison')
        command=[sys.executable,'-m','scripts.ib_local.run_mother_continuation',
            '--checkpoint',str(args.checkpoint),'--output',str(args.output/name),
            '--cycles',str(args.cycles),'--shared-cache',str(args.output/'mother_cache.json'),
            '--stop-file',str(args.output/'STOP'),*extra]
        report['active_arm']=name; write(args.output/'status.json',report)
        with (args.output/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode:
            report.update(status='failed',failed_arm=name,returncode=result.returncode)
            write(args.output/'status.json',report); raise SystemExit(result.returncode)
        completed=json.loads((args.output/name/'completed.json').read_text(encoding='utf-8'))
        report['arms'][name]=dict(completed=completed,summary=summary(args.output/name))
        write(args.output/'status.json',report)
        if completed['status']!='complete':
            report['status']='stopped'; write(args.output/'status.json',report); return
    if runtime_manifest()!=manifest: raise RuntimeError('runtime changed during the controlled comparison')
    report.update(status='complete',elapsed_seconds=time.time()-report['start_time'])
    report.pop('active_arm',None)
    write(args.output/'status.json',report)
    write(args.output/'comparison.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
