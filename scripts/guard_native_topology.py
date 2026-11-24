"""Stdlib-only serial supervisor for the approved native five-part protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def atomic_save_reserve_mib(body):
    """One new snapshot + one database copy; old snapshot already uses disk."""
    snapshot = (body/'last.pt.gz').stat().st_size
    database = max((p.stat().st_size for p in body.glob('causal_archive_*.sqlite')), default=0)
    return (snapshot+database+1048575)//1048576+256


class ResourcePressure:
    """Distinguish sustained exhaustion from Windows pagefile growth.

    A present commit limit is not a fixed RAM capacity. Windows can enlarge
    its managed pagefile during CUDA loading. Keep a hard emergency margin,
    but do not end a healthy event on a single low-headroom sample.
    """
    def __init__(self, disk_reserve_mib=1536):
        self.low_since = None
        self.disk_reserve_mib = disk_reserve_mib

    def check(self, reading, now):
        commit = reading['available_commit_mib']
        if reading['system_disk_free_mib'] < self.disk_reserve_mib:
            return 'system disk below atomic-save reserve'
        if commit < 96:
            return 'system commit below emergency reserve'
        if commit >= 512:
            self.low_since = None
            return None
        if self.low_since is None:
            self.low_since = now
        if now-self.low_since >= 30:
            return 'system commit persistently below 512 MiB for 30 seconds'
        return None


def write(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--cycles',type=int,default=8057)
    p.add_argument('--continue-collection',action='store_true')
    p.add_argument('--min-start-commit-mib',type=int,default=2048,
        help='initial allocation headroom, not total model RAM; runtime pressure is monitored separately')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    work=a.root/'work';status_path=a.root/'pipeline_status.json'
    helpers=runpy.run_path(str(repo/'scripts/run_guarded_mother_continuation.py'))
    status=dict(status='waiting_plain_equivalence',pid=os.getpid(),body=str(a.root/'body'),
        contract=str(repo/'docs/information_boltzmann/CGB_NATIVE_TOPOLOGY_EXECUTION_20261116.md'))
    write(status_path,status)
    while helpers['process_memory'](a.wait_pid).get('error') is None:
        time.sleep(2)
    def child(phase,cycles):
        if phase=='collect':
            while True:
                r=helpers['resources'](a.root)
                if r['available_commit_mib']>=a.min_start_commit_mib and r['system_disk_free_mib']>=2600:break
                status.update(status='waiting_resources',phase='collect',resources=r,
                    required_start_commit_mib=a.min_start_commit_mib,
                    required_start_disk_mib=2600,
                    reason='initial allocation headroom; Windows managed pagefile may grow during actual execution')
                write(status_path,status)
                if (work/'STOP').exists():raise RuntimeError('stopped while waiting for collection headroom')
                time.sleep(15)
        env=os.environ.copy()
        # The measured live tensors occupy ~0.6 GiB after an event while
        # the default pool retains ~3 GiB. Reclaim unused blocks proactively
        # during allocation, before WDDM pressure grows the C: pagefile.
        # This is an allocator policy, not a cap on live tensors or gradients.
        env['PYTORCH_CUDA_ALLOC_CONF']='backend:native,garbage_collection_threshold:0.4'
        status['cuda_allocator_config']=env['PYTORCH_CUDA_ALLOC_CONF']
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            env[name]='1'
        command=[sys.executable,'-u','-m','scripts.run_native_topology','--body',str(a.root/'body'),
            '--work',str(work),'--phase',phase,'--cycles',str(cycles)]
        with (a.root/f'{phase}.log').open('w',encoding='utf-8') as out, \
             (a.root/f'{phase}.err.log').open('w',encoding='utf-8') as err:
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            status.update(status='running',phase=phase,worker_pid=proc.pid)
            write(status_path,status)
            # Windows can expand its pagefile by several GiB while a CUDA
            # event is in flight. Preserve room for that growth as well as the
            # atomic snapshot; this does not alter the learner or observation.
            pressure=ResourcePressure(max(4096, atomic_save_reserve_mib(a.root/'body')))
            status['atomic_save_reserve_mib']=pressure.disk_reserve_mib
            while proc.poll() is None:
                r=helpers['resources'](a.root)
                memory=helpers['process_memory'](proc.pid)
                status.update(resources=r,worker_memory=memory)
                with (a.root/'resources.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(time=time.time(),phase=phase,pid=proc.pid,
                        resources=r,worker_memory=memory))+'\n')
                reason=pressure.check(r,time.monotonic())
                if reason:
                    # Admission is one bounded private event, never the live
                    # owner. Loading cannot honor an event-boundary STOP; the
                    # worker waits for headroom before starting collection.
                    worker_status=work/(phase+'_status.json')
                    state=json.loads(worker_status.read_text(encoding='utf-8')) if worker_status.exists() else {}
                    if phase=='collect' and state.get('status')=='running':
                        (work/'STOP').write_text('resource guard: event-boundary stop\n',encoding='utf-8')
                        status.update(resource_stop=r,resource_stop_reason=reason)
                    else:
                        status.update(admission_or_loading_resource_observation=r)
                write(status_path,status)
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f'{phase} failed with exit {proc.returncode}; inspect retained error log')
    try:
        plain=json.loads((work/'pair_plain_status.json').read_text(encoding='utf-8'))
        if plain['status']!='completed':raise RuntimeError('plain actual event not completed')
        if not a.continue_collection:child('pair_observed',1)
        observed=json.loads((work/'pair_observed_status.json').read_text(encoding='utf-8'))
        checks={k:plain['after'][k]==observed['after'][k] for k in plain['after']}
        checks.update(initial_same=plain['before']==observed['before'],
            receipts_same=plain['receipts']==observed['receipts'],
            observer_rng=all(observed['observer_rng_checks']),
            observer_source_unchanged=observed['observer_source_sha256']==hashlib.sha256(
                (repo/'scripts/native_topology_observer.py').read_bytes()).hexdigest())
        gate=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
            differing_engine_fields=[k for k,v in plain['after'].get('engine_fields',{}).items()
                if observed['after'].get('engine_fields',{}).get(k)!=v],
            differing_initial_fields=[k for k,v in plain['before'].get('engine_fields',{}).items()
                if observed['before'].get('engine_fields',{}).get(k)!=v],
            plain_seconds=plain['seconds'],observed_seconds=observed['seconds'],
            purpose='same actual complete-learning event with/without detached observer; not capability evidence')
        write(work/'observer_equivalence.json',gate)
        write(repo/'results/published/cgb_native_topology_observer_equivalence_20261116.json',gate)
        if not all(checks.values()):raise RuntimeError('passive observation changed a carried state; no collection allowed')
        if (work/'STOP').exists():raise RuntimeError('resource stop remains; collection not started')
        metadata=json.loads((a.root/'body/last.metadata.json').read_text(encoding='utf-8'))
        child('collect',min(a.cycles,10417-int(metadata['cycle'])))
        collection=json.loads((work/'collect_status.json').read_text(encoding='utf-8'))
        command=[sys.executable,'-m','scripts.analyze_native_topology','--observations',
            collection['observations_output'],'--output',str(work/'native_acquisition_analysis.json')]
        subprocess.run(command,cwd=repo,check=True,env=os.environ.copy())
        status.update(status='acquisition_finished',
            collection=json.loads((work/'collect_status.json').read_text(encoding='utf-8')),
            functional_intervention='not executed: requires identified modes and information identity')
        write(status_path,status)
    except Exception as error:
        status.update(status='failed',error=repr(error))
        write(status_path,status)
        raise


if __name__=='__main__':main()
