"""One GPU queue: two real events, exact restore, then 510 continuous events.

The light owner never imports Torch or resets the individual. Resource gates
and event-boundary stopping are delegated to the existing supervisor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='active',stage='admission',pid=os.getpid(),
        source=str(args.checkpoint.resolve()),source_cycle=4208,target_cycle=4720,
        planned_world_events=512,phase_outputs={},started_at=time.time())
    statefile=args.output/'pipeline.json'; stop=args.output/'STOP'
    def write():
        tmp=statefile.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        tmp.replace(statefile)
    def run(stage,command,output):
        if stop.exists():
            raise InterruptedError('pipeline stop requested at a saved boundary')
        status.update(stage=stage)
        status['phase_outputs'][stage]=str(output.resolve());write()
        with (args.output/(stage+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;write()
            while child.poll() is None:
                if stop.exists() and stage in ('admission','learning'):
                    output.mkdir(parents=True,exist_ok=True)
                    (output/'STOP').write_text('pipeline requests complete event and save\n',encoding='utf8')
                time.sleep(2)
            status['child_exit_code']=child.returncode;write()
        if child.returncode:
            raise RuntimeError(f'{stage} process failed; see its log')
    def guarded(stage,source,n):
        output=args.output/stage
        run(stage,[sys.executable,'-u','scripts/run_guarded_mother_continuation.py',
            '--checkpoint',str(source),'--output',str(output),'--cycles',str(n),
            '--minimum-system-disk-mib','1536'],output)
        outcome=json.loads((output/'status.json').read_text(encoding='utf8'))
        status['latest_cycle']=outcome.get('final_cycle',status.get('latest_cycle',4208))
        checkpoint=output/'body/last.pt.gz'
        if checkpoint.exists():status['latest_checkpoint']=str(checkpoint.resolve())
        write()
        if outcome['status']!='complete':
            if outcome['status'] in ('stopped','stopped_before_start','source_changed_requires_review'):
                raise InterruptedError(f'{stage} ended as {outcome["status"]}; no automatic retry')
            raise RuntimeError(f'{stage} ended as {outcome["status"]}; no automatic retry')
        return checkpoint
    write()
    try:
        admission=guarded('admission',args.checkpoint,2)
        rows=[json.loads(line) for line in (admission.parent/'metrics.jsonl').read_text(
            encoding='utf8').splitlines() if line]
        checks=[]
        for i,row in enumerate(rows):
            q=row['shared_unknown']
            checks.append(dict(cycle=row['cycle'],committed=q['revision']['committed'],
                real_updates=row['updates']>0,
                identity_in_program=q['identity']==row['causal_program']['roles']['unknown_identity'],
                policy_replay=row['generative_relation']['binding_replay_error']<1e-5,
                delayed_program=i==0 or row['generative_relation']['program_evidence_updated']))
        passed=len(rows)==2 and all(all(v for k,v in c.items() if k!='cycle') for c in checks)
        status['admission_checks']=checks;status['admission_passed']=passed;write()
        if not passed:raise RuntimeError('real whole-event admission failed; latest body preserved')
        restored=args.output/'restore'
        run('restore',[sys.executable,'-u','-m','scripts.ib_local.audit_joint_restore',
            '--checkpoint',str(admission),'--output',str(restored),
            '--table-storage',str(args.output/'restore_table')],restored)
        audit=json.loads((restored/'restore_audit.json').read_text(encoding='utf8'))
        if not audit['passed']:raise RuntimeError('full real-body restore differs')
        status['restore_passed']=True;write()
        guarded('learning',admission,510)
        status.update(status='complete',stage='finished',ended_at=time.time());write()
    except BaseException as exc:
        status.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',
            error=f'{type(exc).__name__}: {exc}',ended_at=time.time());write()
        if not isinstance(exc,InterruptedError):raise


if __name__=='__main__':main()
