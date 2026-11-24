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
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import time


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [
        (name,ctypes.c_ulonglong) for name in ('total_physical','available_physical',
        'total_commit','available_commit','total_virtual','available_virtual','available_extended')]


def process_memory(pid):
    """Read the supervised process's commit and resident bytes without Torch."""
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
                'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    handle = kernel.OpenProcess(0x410, False, int(pid))
    if not handle:
        return dict(error=ctypes.get_last_error())
    try:
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb):
            return dict(error=ctypes.get_last_error())
        return dict(working_set_mib=value.working_set/1048576,
                    private_commit_mib=value.private/1048576,
                    peak_working_set_mib=value.peak_working_set/1048576)
    finally:
        kernel.CloseHandle(handle)


def resources(output_directory=None):
    value=MemoryStatus();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    system_disk=Path(os.environ.get('SystemDrive','C:')+'\\')
    result = dict(available_commit_mib=value.available_commit/1048576,
                available_physical_mib=value.available_physical/1048576,
                total_commit_mib=value.total_commit/1048576,
                total_physical_mib=value.total_physical/1048576,
                system_disk_free_mib=shutil.disk_usage(system_disk).free/1048576)
    try:
        result['system_pagefile_mib'] = (system_disk/'pagefile.sys').stat().st_size/1048576
    except OSError:
        pass
    if output_directory is not None:
        result['output_disk_free_mib'] = shutil.disk_usage(output_directory).free/1048576
    return result


def ready_to_start(readings, args):
    return (readings['available_commit_mib'] >= args.minimum_start_commit_mib and
            readings['system_disk_free_mib'] >= args.minimum_system_disk_mib and
            readings['output_disk_free_mib'] >= args.minimum_output_disk_mib)


def stop_reason(readings):
    if readings['available_commit_mib'] < 96:
        return 'Host commit below 96 MiB'
    if readings['output_disk_free_mib'] < 128:
        return 'Output disk below 128 MiB'
    if readings['system_disk_free_mib'] < 384:
        return 'System disk below 384 MiB'
    return None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (block:=stream.read(65536)):h.update(block)
    return h.hexdigest()


def checkpoint_publisher(root):
    # Importing scripts.ib_local.checkpoint_commit first executes ib_local's
    # package initializer, which imports Torch. The supervisor must remain
    # stdlib-only: execute this pure helper directly, without that initializer.
    return runpy.run_path(str(Path(root)/'scripts/ib_local/checkpoint_commit.py'))['publish_checkpoint']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,required=True)
    p.add_argument('--body-directory',type=Path,
                   help='explicit existing complete-body output for continuous resume')
    p.add_argument('--resume-body',action='store_true')
    p.add_argument('--autonomous-events',action='store_true')
    p.add_argument('--adaptive-dynamics',action='store_true')
    p.add_argument('--world-source',choices=('mother','text'))
    p.add_argument('--text-navigation',choices=('legacy','stable_context_v1'))
    p.add_argument('--natural-text-manifest',type=Path)
    p.add_argument('--checkpoint-compression-level',type=int,choices=range(10))
    p.add_argument('--learning-audit-cycles',type=int,default=0)
    p.add_argument('--learning-audit-output',type=Path)
    p.add_argument('--table-storage',type=Path)
    # Keep the admission supervisor Torch-free while forwarding runtime knobs.
    p.add_argument('--activation-mode', choices=('tiered', 'disk', 'device'), default='tiered')
    p.add_argument('--activation-gpu-mib', type=int, default=512)
    p.add_argument('--activation-host-mib', type=int, default=128)
    p.add_argument('--causal-storage',type=Path)
    p.add_argument('--validate-every',type=int,default=64)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--optimizer-policy', choices=('adamw', 'budgeted'))
    p.add_argument('--minimum-start-commit-mib',type=int,default=3328)
    p.add_argument('--minimum-system-disk-mib',type=int,default=512)
    p.add_argument('--minimum-output-disk-mib',type=int,default=512)
    args=p.parse_args()
    if os.name!='nt' or args.cycles<1:raise ValueError('Windows supervisor and positive budget required')
    args.output.mkdir(parents=True,exist_ok=False)
    body_directory=args.body_directory or args.output/'body'
    if args.resume_body and args.checkpoint.resolve()!=(body_directory/'last.pt.gz').resolve():
        raise ValueError('resume must use the same body output and its complete checkpoint')
    root=Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    publish_checkpoint=checkpoint_publisher(root)
    code={str(f.relative_to(root)):digest(f) for f in (root/'scripts/ib_local').glob('*.py')}
    code['scripts/run_guarded_mother_continuation.py']=digest(Path(__file__))
    status=dict(status='waiting_resources',pid=os.getpid(),planned_cycles=args.cycles,
        source_checkpoint=str(args.checkpoint.resolve()),source_sha256=digest(args.checkpoint),
        code_sha256=code,minimum_start_commit_mib=args.minimum_start_commit_mib,
        minimum_system_disk_mib=args.minimum_system_disk_mib,
        minimum_output_disk_mib=args.minimum_output_disk_mib,started_at=time.time())
    def write():
        # Keep low-cost resource evidence across startup, compilation, events
        # and an orderly stop. The overwritten status alone loses the cause
        # of a transient disk/commit spike. This records no tensor or secret.
        row = dict(timestamp=time.time(), status=status['status'],
                   child_pid=status.get('child_pid'), resources=status.get('resources'),
                   child_memory=status.get('child_memory'))
        try:
            with (args.output/'resources.jsonl').open('a', encoding='utf-8') as history:
                history.write(json.dumps(row)+'\n')
        except OSError as error:
            status['resource_history_error'] = str(error)
        tmp=args.output/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        try:
            publish_checkpoint(tmp,args.output/'status.json')
        except PermissionError as error:
            # Telemetry publication must not orphan an otherwise live learner.
            # Resource checks continue; the next heartbeat retries publication.
            status['status_publication_error']=str(error)
            print(f'Status publication delayed; supervisor remains active: {error}',file=sys.stderr,flush=True)
    stop=args.output/'STOP'
    while True:
        status['resources']=resources(args.output);write()
        if stop.exists():status['status']='stopped_before_start';write();return
        if ready_to_start(status['resources'], args):break
        time.sleep(10)
    if any(digest(root/f)!=h for f,h in code.items()):
        status.update(status='source_changed_requires_review');write();return
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        CUDA_MODULE_LOADING='LAZY',
        TRITON_CACHE_DIR=os.environ.get('TRITON_CACHE_DIR',str(root/'scratch/triton_cache')))
    command=[sys.executable,'-u','-m','scripts.ib_local.run_mother_continuation',
        '--checkpoint',str(args.checkpoint),'--output',str(body_directory),
        '--cycles',str(args.cycles),'--validate-every',str(args.validate_every),'--validation-bytes','8192',
        '--checkpoint-every',str(args.checkpoint_every),
        '--mother-harness-url','http://127.0.0.1:8902','--stop-file',str(stop)]
    if args.resume_body:command.append('--resume-output')
    if args.autonomous_events:command.append('--autonomous-events')
    if args.adaptive_dynamics:command.append('--adaptive-dynamics')
    if args.optimizer_policy:
        command.extend(['--optimizer-policy', args.optimizer_policy])
    if args.world_source:
        command.extend(['--world-source',args.world_source])
    if args.text_navigation:
        command.extend(['--text-navigation',args.text_navigation])
    if args.natural_text_manifest:
        command.extend(['--natural-text-manifest',str(args.natural_text_manifest.resolve())])
    if args.checkpoint_compression_level is not None:
        command.extend(['--checkpoint-compression-level',str(args.checkpoint_compression_level)])
    if args.learning_audit_cycles:
        if args.learning_audit_cycles < 1 or args.learning_audit_output is None:
            raise ValueError('bounded learning audit requires an explicit output')
        command.extend(['--learning-audit-cycles',str(args.learning_audit_cycles),
                        '--learning-audit-output',str(args.learning_audit_output.resolve())])
    for flag,value in (('--table-storage',args.table_storage),('--causal-storage',args.causal_storage)):
        if value is not None:
            command.extend([flag,str(value.resolve())])
    command.extend(['--activation-mode', args.activation_mode,
                    '--activation-gpu-mib', str(args.activation_gpu_mib),
                    '--activation-host-mib', str(args.activation_host_mib)])
    status.update(body_directory=str(body_directory.resolve()),command=command)
    with (args.output/'body.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='running',child_pid=child.pid);write()
        while child.poll() is None:
            status['resources']=resources(args.output)
            status['child_memory']=process_memory(child.pid)
            reason = stop_reason(status['resources'])
            if reason is not None and not stop.exists():
                stop.write_text(reason+': complete and save current event, then stop.\n')
                status['guard_requested_stop']=True
                status['guard_stop_reason']=reason
            write();time.sleep(2)
        status['exit_code']=child.returncode
    completed=body_directory/'completed.json'
    status['status']='failed'
    if completed.exists():
        state=json.loads(completed.read_text())
        if state.get('requested_cycles')==args.cycles and state.get('source_sha256')==status['source_sha256']:
            status.update(status=state['status'],final_cycle=state['final_cycle'],committed_cycles=state['cycles'])
        else:
            status['previous_completion_not_current_run']=True
    status['ended_at']=time.time();write()


if __name__=='__main__':main()
