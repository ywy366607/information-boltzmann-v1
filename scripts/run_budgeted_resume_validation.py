"""Wait for the live single-GPU comparison, then audit complete budget restore.

No new training begins while another complete-body arm owns the GPU. This
queue never replaces or mutates the main checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_guarded_mother_continuation import resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    state = dict(status='waiting_comparison',pid=os.getpid(),comparison=str(a.comparison),
                 source=str(a.body),main_written=False)
    def publish():
        path = a.output/'status.tmp.json'
        path.write_text(json.dumps(state,indent=2)+'\n')
        path.replace(a.output/'status.json')
    publish()
    while True:
        prior = json.loads((a.comparison/'status.json').read_text())
        if (a.output/'STOP').exists():
            state.update(status='stopped_before_start'); publish(); return
        if prior['status'] == 'complete': break
        if prior['status'] not in ('running','waiting_resources'):
            state.update(status='upstream_not_completed',upstream_status=prior['status']); publish(); return
        time.sleep(10)
    commands = [[sys.executable,'-u','-m','scripts.audit_budgeted_body_resume',
                '--body',str(a.body),'--output',str(a.output),'--arm',arm]
                for arm in ('reference','first','resume')]
    commands += [[sys.executable,'-m',module,
        '--before',str(a.output/'reference_checkpoint/last.pt.gz'),
        '--after',str(a.output/'resume_checkpoint/last.pt.gz'),
        '--report',str(a.output/name),*flags] for module,name,flags in (
            ('scripts.ib_local.audit_runtime_checkpoint_pair','complete_state.json',['--exact-state']),
            ('scripts.audit_checkpoint_carried_fields','carried_state.json',[]))]
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_MODULE_LOADING='LAZY')
    for index,command in enumerate(commands):
        while True:
            reading = resources(a.output)
            state.update(status='waiting_resources',stage=index,resources=reading); publish()
            if (a.output/'STOP').exists():
                state.update(status='stopped_between_stages'); publish(); return
            if (reading['available_commit_mib']>=3328 and reading['system_disk_free_mib']>=512
                    and reading['output_disk_free_mib']>=2048): break
            time.sleep(10)
        state.update(status='running',command=command); publish()
        with (a.output/f'stage_{index}.log').open('w') as log:
            child = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid; publish()
            code=child.wait()
        if code:
            state.update(status='failed',exit_code=code); publish(); return
    state.update(status='complete',completed_at=time.time()); publish()


if __name__=='__main__': main()
