"""Recover, verify passive observation, then observe the SAME normal learner.

All production modules and original recovery receipts remain unchanged.
This runner never restores mother calls, resets live state or installs a
topology controller. Phase 5 requires actual discovered modes, not a preset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import shutil
import time

import numpy as np
import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation, ROOT
from scripts.ib_local.event_journal import EventJournal, complete_jsonl, execution_digest, rng_digest, receipt
from scripts.audit_spectrum_runtime import digest, carried
from scripts.native_topology_observer import NativeTopologyObserver, SOURCE_SHA256


def write(path, value):
    path=Path(path); temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def canonical_carried_value(value):
    """Compare module state, not allocation-specific tensor-storage pickle IDs."""
    if isinstance(value,torch.nn.Module):
        return dict(module_type=type(value).__module__+'.'+type(value).__qualname__,
            tensors=value.state_dict(),attributes={k:canonical_carried_value(v) for k,v in vars(value).items()
                if not k.startswith('_')})
    if isinstance(value,dict):return {k:canonical_carried_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(canonical_carried_value(v) for v in value)
    if isinstance(value,set):return sorted((canonical_carried_value(v) for v in value),key=repr)
    return value


def full_digest(engine):
    from scripts.ib_local.run_navigation_residual_wood import _DERIVED
    from scripts.ib_local.navigation_checkpoint import CARRY
    names=_DERIVED.difference(CARRY)
    state={k:canonical_carried_value(v) for k,v in vars(engine).items() if k not in names}
    result=carried(engine)
    result['all_carried_engine']=digest(state)
    result['engine_fields']={k:digest(v) for k,v in state.items()}
    result['engine_field_types']={k:type(v).__qualname__ for k,v in state.items()}
    result['fast_table']=digest({k:np.asarray(getattr(engine.table,k)) for k in
        ('rows','count','tag','err','epoch','last_write')})
    result['table_clock']=digest((engine.table.history,engine.table.events))
    h=hashlib.sha256();db=engine.model._causal_archive.db
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        h.update(name.encode())
        for row in db.execute('SELECT * FROM "'+name.replace('"','""')+'" ORDER BY 1'):
            h.update(pickle.dumps(row,protocol=4))
    result['sql_logical']=h.hexdigest()
    return result


def options(args):
    result=continuation_parser().parse_args(['--checkpoint',str(args.body/'last.pt.gz'),
        '--output',str(args.body),'--resume-output','--cycles',str(args.cycles),
        '--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(args.work/(args.phase+'_table')),
        '--causal-storage',str(args.work/(args.phase+'_causal')),
        '--checkpoint-every','8','--checkpoint-compression-level','1'])
    return result


def evaluation(loaded):
    """Preserve the saved independent validation cadence and data sources."""
    config=loaded.config;v=config.get('validation_schedule')
    if not v:return None
    from scripts.ib_local.run_language_navigation_v2 import validate
    from scripts.audit_language_contribution import language_contribution
    n=int(v['bytes_each'])
    raw=np.load(ROOT/'data/ib_owt_10000/validation.npy',mmap_mode='r')
    offset=json.loads((ROOT/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
    raw=np.asarray(raw[:2*(262144+n+1)])
    owt=(raw[raw>=offset]-offset)[262144:262144+n+1].astype(np.uint8)
    manifest=config['natural_text_corpus']
    data=Path(manifest['validation_path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest['validation_sha256']:
        raise ValueError('original natural validation changed')
    fine=np.frombuffer(data,dtype=np.uint8)[:n+1]
    defaults=continuation_parser().parse_args(['--output',str(loaded.output),'--cycles','1'])
    mother=np.frombuffer(defaults.mother_validation.read_bytes(),dtype=np.uint8)[:n+1]
    def evaluate():
        e=loaded.engine
        first=language_contribution(e.model,e.table,owt)
        second=language_contribution(e.model,e.table,fine)
        return dict(owt_nll=first['full_nll'],fineweb_nll=second['full_nll'],
            mother_nll=validate(e.model,e.table,mother),
            bytes_each=n,language_contribution=dict(owt=first,fineweb=second),
            scope='original independent sidecars; no live-state reset; excluded from native dynamics observation')
    return evaluate


def run(args):
    args.work.mkdir(parents=True,exist_ok=True)
    status_path=args.work/(args.phase+'_status.json')
    status=dict(status='loading',phase=args.phase,body=str(args.body),pid=os.getpid())
    status['observer_source_sha256']=SOURCE_SHA256
    write(status_path,status)
    opts=options(args)
    loaded=prepare_continuation(opts,resume_output=True)
    e=loaded.engine
    # Original identity is retained, but private admission does not publish a
    # competing workbench card or claim ownership of the original body.
    if args.phase!='collect':loaded.life_card=None
    original_code=execution_digest(ROOT)
    journal=EventJournal(args.body,cycle=e.autonomous_event_clock,checkpoint_sha=loaded.checkpoint_sha,
        code_sha=original_code,config=loaded.config,filename='events.autonomous.pending.jsonl',
        config_key='autonomous_event_journal')
    if args.phase=='recover':
        expected=len(journal.pending)
        if expected!=21:raise ValueError(f'expected 21 original pending events, found {expected}')
        # Production runtime verifies the original receipts before allowing a
        # fresh event. No observer is installed during original recovery.
        from scripts.ib_local.autonomous_runtime import run as production
        production(opts,loaded,None,ROOT)
        if e.autonomous_event_clock!=2821:
            raise ValueError('recovery did not end at last durable event 2821')
        status.update(status='completed',cycle=2360,event=e.autonomous_event_clock,
            verified_original_events=expected,source_sha=loaded.source_sha,
            production_execution_sha=original_code,carried=full_digest(e))
        write(status_path,status)
        return
    if journal.pending:raise ValueError('recover original receipts before observation')
    if args.phase=='collect':
        helpers=runpy.run_path(str(ROOT/'scripts/run_guarded_mother_continuation.py'))
        while True:
            resources=helpers['resources'](args.body)
            if resources['available_commit_mib']>=1024 and resources['system_disk_free_mib']>=1536:break
            status.update(status='waiting_resources_before_first_event',resources=resources)
            write(status_path,status)
            if (args.work/'STOP').exists():raise RuntimeError('stopped before collection; restored body unchanged')
            time.sleep(5)
    from scripts.ib_local.autonomous_cycle import dispatch
    observer=None
    if args.phase in ('pair_observed','collect'):
        directory=args.work/(args.phase+'_observations')
        if directory.exists():directory=args.work/f'{args.phase}_observations_from_{loaded.cycle}'
        observer=NativeTopologyObserver(e,directory)
        status['observations_output']=str(directory)
    start_cycle=loaded.cycle;cycle=start_cycle
    before=full_digest(e) if args.phase.startswith('pair_') else carried(e)
    updates=0;events=[];started=time.perf_counter()
    verify_rng=[]
    production_dispatch=dispatch
    def observed_dispatch(engine):
        nonlocal cycle,updates
        if observer:
            saved=rng_digest();observer.begin_event(cycle);verify_rng.append(saved==rng_digest())
        row=production_dispatch(engine)
        if observer:
            saved=rng_digest();observer.end_event(row);verify_rng.append(saved==rng_digest())
        if row['autonomous_event']['kind']=='external':
            cycle+=1
        return row
    try:
        if args.phase.startswith('pair_'):
            while cycle==start_cycle:
                row=observed_dispatch(e)
                events.append(receipt(row))
            if observer:observer.close()
            status.update(status='completed',start_cycle=start_cycle,final_cycle=cycle,
                before=before,after=full_digest(e),receipts=events,
                observer_rng_checks=verify_rng,seconds=time.perf_counter()-started,
                source_sha=loaded.source_sha)
            write(status_path,status)
            return
        # Keep the original production journal/save/validation/world loop. Only
        # its dispatch call is surrounded by detached observation.
        from scripts.ib_local import autonomous_runtime
        from scripts.native_topology_io import install_runtime_scan
        old_scan=install_runtime_scan(autonomous_runtime)
        original=autonomous_runtime.dispatch
        autonomous_runtime.dispatch=observed_dispatch
        opts.stop_file=args.work/'STOP'
        previous_counted=e.optimizer.step
        def resource_step(*a,**kw):
            result=previous_counted(*a,**kw)
            # Disk is checked at event boundaries below, not per parameter.
            return result
        def guarded_dispatch(engine):
            value=observed_dispatch(engine)
            # CUDA's reusable allocation pool also consumes Windows driver
            # commit. Under host pressure return only UNUSED blocks after a
            # complete event; never offload/reset live states or empty the
            # pool inside a forward/backward calculation.
            resource_reading=helpers['resources'](args.body)
            allocator=dict(allocated_mib=torch.cuda.memory_allocated()/1048576,
                reserved_mib=torch.cuda.memory_reserved()/1048576)
            reclaimed=False
            if (resource_reading['available_commit_mib']<1536
                    or resource_reading['system_disk_free_mib']<4096):
                torch.cuda.empty_cache()
                reclaimed=True
            allocator.update(reserved_after_mib=torch.cuda.memory_reserved()/1048576,
                released_unused_cache=reclaimed)
            from scripts.guard_native_topology import atomic_save_reserve_mib
            if shutil.disk_usage(args.body).free<atomic_save_reserve_mib(args.body)*1048576:
                opts.stop_file.write_text('native observer: disk headroom stop\n',encoding='utf-8')
            if not all(verify_rng[-2:]):
                raise RuntimeError('observer consumed RNG')
            write(status_path,dict(status='running',phase=args.phase,pid=os.getpid(),
                cycle=cycle,event=e.autonomous_event_clock,quiet_time=observer.quiet_time,
                observer_updates=observer.update_clock,output=str(observer.directory),
                cuda_allocator=allocator,
                completion=dict(structure='collecting',modes='pending actual analysis',
                    synchronization='pending sufficient native time',regulation='collecting',
                    functional_intervention='pending identified modes')))
            return value
        autonomous_runtime.dispatch=guarded_dispatch
        try:
            autonomous_runtime.run(opts,loaded,evaluation(loaded),ROOT)
        finally:
            autonomous_runtime.dispatch=original
            autonomous_runtime.complete_jsonl=old_scan
        status.update(status='completed' if cycle==start_cycle+args.cycles else 'stopped',
            start_cycle=start_cycle,cycle=cycle,event=e.autonomous_event_clock,
            seconds=time.perf_counter()-started,observer_rng_all=all(verify_rng),
            production_execution_sha=execution_digest(ROOT),
            stages='normal acquisition only; mode analysis and selective intervention remain explicit')
        write(status_path,status)
    finally:
        if observer:observer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--phase',choices=('recover','pair_plain','pair_observed','collect'),required=True)
    p.add_argument('--cycles',type=int,default=1)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        write(args.work/(args.phase+'_status.json'),dict(status='failed',phase=args.phase,
            pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':main()
