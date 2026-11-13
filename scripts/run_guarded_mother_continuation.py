"""Wait for host headroom, then continue one complete body on a single GPU.

This supervisor imports no Torch, changes no model state, and never kills
another application. Low headroom requests an event-boundary save and stop.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
        (name,ctypes.c_ulonglong) for name in ('total_physical','available_physical',
        'total_commit','available_commit','total_virtual','available_virtual','available_extended')]


def resources():
    value=MemoryStatus();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    system_disk=Path(os.environ.get('SystemDrive','C:')+'\\')
    return dict(available_commit_mib=value.available_commit/1048576,
                available_physical_mib=value.available_physical/1048576,
                system_disk_free_mib=shutil.disk_usage(system_disk).free/1048576)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (block:=stream.read(65536)):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,required=True)
    p.add_argument('--minimum-start-commit-mib',type=int,default=3328)
    p.add_argument('--minimum-system-disk-mib',type=int,default=512)
    args=p.parse_args()
    if os.name!='nt' or args.cycles<1:raise ValueError('Windows supervisor and positive budget required')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    code={str(f.relative_to(root)):digest(f) for f in (root/'scripts/ib_local').glob('*.py')}
    code['scripts/run_guarded_mother_continuation.py']=digest(Path(__file__))
    status=dict(status='waiting_resources',pid=os.getpid(),planned_cycles=args.cycles,
        source_checkpoint=str(args.checkpoint.resolve()),source_sha256=digest(args.checkpoint),
        code_sha256=code,minimum_start_commit_mib=args.minimum_start_commit_mib,
        minimum_system_disk_mib=args.minimum_system_disk_mib,started_at=time.time())
    def write():
        tmp=args.output/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        tmp.replace(args.output/'status.json')
    stop=args.output/'STOP'
    while True:
        status['resources']=resources();write()
        if stop.exists():status['status']='stopped_before_start';write();return
        if (status['resources']['available_commit_mib']>=args.minimum_start_commit_mib and
                status['resources']['system_disk_free_mib']>=args.minimum_system_disk_mib):break
        time.sleep(10)
    if any(digest(root/f)!=h for f,h in code.items()):
        status.update(status='source_changed_requires_review');write();return
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        CUDA_MODULE_LOADING='LAZY',TRITON_CACHE_DIR=str(root/'scratch/triton_cache'))
    command=[sys.executable,'-u','-m','scripts.ib_local.run_mother_continuation',
        '--checkpoint',str(args.checkpoint),'--output',str(args.output/'body'),
        '--cycles',str(args.cycles),'--validate-every','64','--validation-bytes','8192',
        '--mother-harness-url','http://127.0.0.1:8902','--stop-file',str(stop)]
    with (args.output/'body.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='running',child_pid=child.pid);write()
        while child.poll() is None:
            status['resources']=resources()
            if status['resources']['available_commit_mib']<96 and not stop.exists():
                stop.write_text('Host commit below96MiB: complete and save current event, then stop.\n')
                status['guard_requested_stop']=True
            write();time.sleep(2)
        status['exit_code']=child.returncode
    completed=args.output/'body/completed.json'
    status['status']='failed'
    if completed.exists():
        state=json.loads(completed.read_text());status.update(status=state['status'],final_cycle=state['final_cycle'],
            committed_cycles=state['cycles'])
    status['ended_at']=time.time();write()


if __name__=='__main__':main()
