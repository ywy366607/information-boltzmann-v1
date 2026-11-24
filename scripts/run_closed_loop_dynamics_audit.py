"""Serial GPU audit; always return the unchanged original to its prior queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--reference-pid',type=int,required=True)
    p.add_argument('--cycles',type=int,default=8)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    status=dict(status='waiting_reference',cycles=a.cycles,results=[])
    status_path=a.work/'batch_status.json'
    write(status_path,status)
    try:
        # Windows process query uses only the stdlib helper and no Torch.
        import runpy
        helper=runpy.run_path(str(root/'scripts/run_guarded_mother_continuation.py'))
        while helper['process_memory'](a.reference_pid).get('error') is None:
            time.sleep(2)
        if not (a.work/'closed_0/report.json').exists():
            raise RuntimeError('reference process ended without a complete report')
        # Exact yoke reproduction is a gate before perturbed arms.
        arms=[('closed','fire2'),('yoked','fire2'),
              ('yoked',0),('closed',1),('closed',2),('closed',3),
              ('yoked',1),('yoked',2),('yoked',3),
              ('gdn',0),('gdn',1),('gdn',2),('gdn',3)]
        for condition,mask in arms:
            name=f'{condition}_{mask}'
            if (a.work/name/'report.json').exists():
                status['results'].append(dict(arm=name,exit_code=0,already_completed=True))
                continue
            status.update(status='running',arm=name,started_at=time.time())
            command=[sys.executable,'-u','-m','scripts.audit_closed_loop_dynamics',
                '--body',str(a.body),'--work',str(a.work),'--condition',condition,
                '--mask',str(0 if isinstance(mask,str) else mask),'--cycles',str(a.cycles)]
            if isinstance(mask,str):command.extend(['--fire-intervention',mask.removeprefix('fire')])
            with (a.work/f'{name}.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                status['child_pid']=child.pid; write(status_path,status)
                code=child.wait()
            status['results'].append(dict(arm=name,exit_code=code))
            if code: raise RuntimeError(f'{name} failed: inspect private log')
        status.update(status='completed',ended_at=time.time())
    except Exception as error:
        status.update(status='failed',error=repr(error),ended_at=time.time())
    finally:
        write(status_path,status)
        metadata=json.loads((a.body/'last.metadata.json').read_text(encoding='utf-8'))
        remaining=10417-int(metadata['cycle'])
        if remaining>0:
            guard=a.body.parent/f'guard_after_closed_loop_{metadata["cycle"]}'
            command=[sys.executable,'-u',str(root/'scripts/run_guarded_mother_continuation.py'),
                '--checkpoint',str(a.body/'last.pt.gz'),'--output',str(guard),
                '--body-directory',str(a.body),'--resume-body','--cycles',str(remaining),
                '--validate-every','128','--checkpoint-every','8','--world-source','text',
                '--text-navigation','stable_context_v1','--checkpoint-compression-level','1',
                '--optimizer-policy','budgeted','--autonomous-events',
                '--table-storage',str(a.body.parent/'table'),'--causal-storage',str(a.body.parent/'causal')]
            if guard.exists():
                status['resume_error']='Guard already exists; refusing a duplicate owner'
            else:
                log=(a.work/'resume_guard.log').open('w',encoding='utf-8')
                child=subprocess.Popen(command,cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                log.close()
                status.update(resume_pid=child.pid,resume_guard=str(guard/'status.json'))
                observer=root/'data/harness/workbench/training-observer.json'
                value=json.loads(observer.read_text(encoding='utf-8'))
                value['guard_status']=str(guard/'status.json')
                write(observer,value)
            write(status_path,status)


if __name__=='__main__':main()
