"""Whole-generator physical execution on a continuous real OWT individual."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.energy_observation import material_energy_components
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.drive_maintenance_audit import observations
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.source_protocol import SourceProtocol


def legacy_check(run, out, base_commit):
    folder=out/'legacy_source'/'cgb_before_full_step'
    folder.mkdir(parents=True)
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',base_commit,
                                   'information_boltzmann/cgb'],cwd=ROOT,text=True).splitlines()
    hashes={}
    for name in names:
        p=folder/Path(name).relative_to('information_boltzmann/cgb')
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(subprocess.check_output(['git','show',base_commit+':'+name],cwd=ROOT))
        hashes[name]=digest(p)
    sys.path.insert(0,str(folder.parent))
    old_load=importlib.import_module('cgb_before_full_step.exchange_checkpoint').load_exchange
    current,meta=load_exchange(run/'text_drive_mid.npz')
    old,old_meta=old_load(run/'text_drive_mid.npz')
    assert meta==old_meta
    session=copy.deepcopy(meta['session']);old_session=copy.deepcopy(session)
    protocol=SourceProtocol(**meta['source_protocol']);old_protocol=copy.deepcopy(protocol)
    observer=CarrierObservation(current,meta['growth_observer'],meta['carrier_observer'])
    current.event_observer,current.domain_observer=observer.native,observer.domain
    block=np.load(run/'fresh_train_1920_2432.npy',allow_pickle=False)
    records=[json.loads(row) for row in (run/'text_drive_intervals.jsonl').read_text().splitlines()]
    for index in range(256,512):
        row=protocol.advance(current,session,int(block[index]))
        expected=dict(records[index]);expected.pop('corpus_train_byte')
        assert row==expected
        if index<320:
            assert row==old_protocol.advance(old,old_session,int(block[index]))
        if index==319:
            same(current.state_dict(),old.state_dict())
            assert session==old_session
    final,final_meta=load_exchange(run/'text_drive_complete.npz')
    same(current.state_dict(),final.state_dict())
    assert session==final_meta['session']
    assert observer.state==final_meta['growth_observer'] and observer.carriers==final_meta['carrier_observer']
    return dict(prior_actual_future_bytes=256,independent_old_source_future_bytes=64,
                all_physics_rngs_scores_session_and_observers_exact=True,
                old_source_commit=base_commit,old_source_sha256=hashes)


def snapshot(path,model,session,observer,protocol,sources,reference,dt):
    observer.validate(model);protocol.validate(model)
    assert model.t==reference['start_ms']+4*protocol.intervals
    save_exchange(path,model,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(observer.state),
        carrier_observer=copy.deepcopy(observer.carriers),source_protocol=protocol.state_dict(),
        sources=sources,stage_reference=reference,dt_ms=dt,external_interval_ms=4.))


def run_arm(label,parent,meta,block,out,sources):
    start=time.perf_counter()
    model=fork_material(parent)
    before=model.state_dict()
    dt=1. if label=='symmetric_dt1' else 2.
    if label!='legacy_dt2':
        model.material_scheme='symmetric'
        migrated=model.state_dict()
        migrated.pop('material_scheme');migrated['schema']=before['schema']
        same(migrated,before)  # execution only: no physical energy/state/RNG change
    session=copy.deepcopy(meta['session']);session.update(phase='fresh_E',next_phase_byte=0)
    session['interventions'].append(dict(time_ms=model.t,numerical_scheme=model.material_scheme,
        dt_ms=dt,external_interval_ms=4.,external_work=0.,state_erased=False))
    protocol=SourceProtocol('text_drive')
    observer=CarrierObservation(model,meta['growth_observer'],meta['carrier_observer'])
    reference=dict(start_ms=model.t,work=model.work_in,heat=model.heat_out,source_charge=model.source_charge,
        energy=material_energy_components(model),domain_births=model.domain_births,domain_deaths=model.domain_deaths,
        carrier_count=len(observer.carriers['alive']),carrier_next_id=observer.carriers['next_id'],
        carrier_keys=sorted(observer.carriers['alive']),carrier_deaths=observer.carriers['deaths'],
        original_reactions=dict(model.reaction_counts),text_scored_bytes=session['scored_bytes'])
    snapshot(out/(label+'_start.npz'),model,session,observer,protocol,sources,reference,dt)
    fork=None;rows=[];windows=[];maximum_error=0.;last_event=-float('inf')
    with gzip.open(out/(label+'_events.jsonl.gz'),'wt',encoding='utf-8') as events, \
         gzip.open(out/(label+'_domains.jsonl.gz'),'wt',encoding='utf-8') as domains, \
         (out/(label+'_intervals.jsonl')).open('w',encoding='utf-8') as intervals:
        def check_time(row):
            nonlocal last_event
            assert last_event<=row['time_ms']<=model.t+dt
            last_event=row['time_ms']
        def native(row):
            check_time(row);observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):
            check_time(row);observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        model.event_observer,model.domain_observer=native,domain
        for index,target in enumerate(block):
            row=protocol.advance(model,session,int(target),dt_ms=dt)
            if fork is not None:
                assert row==fork_protocol.advance(fork,fork_session,int(target),dt_ms=dt)
            row['corpus_train_byte']=2432+index
            rows.append(row);intervals.write(json.dumps(row,separators=(',',':'))+'\n')
            observer.validate(model)
            maximum_error=max(maximum_error,max(abs(float(v)) for v in model.balances().values()))
            if (index+1)%32==0:
                info=observations(model,observer,protocol,session,reference)
                info['nll_nats_per_byte']={k:float(np.mean([x['scores'][k] for x in rows[-32:]])) for k in row['scores']}
                windows.append(info);atomic_json(out/(label+'_windows.json'),windows)
                status=dict(arm=label,bytes=index+1,time_ms=model.t,nodes=model.n,
                            work=info['source_work'],maximum_error=maximum_error)
                atomic_json(out/'progress.json',dict(status='running',**status));print(json.dumps(status),flush=True)
                events.flush();domains.flush();intervals.flush()
            if index+1==128:
                path=out/(label+'_mid.npz')
                snapshot(path,model,session,observer,protocol,sources,reference,dt)
                fork,fork_meta=load_exchange(path)
                fork_session=copy.deepcopy(fork_meta['session'])
                fork_protocol=SourceProtocol(**fork_meta['source_protocol'])
                fork_observer=CarrierObservation(fork,fork_meta['growth_observer'],fork_meta['carrier_observer'])
                fork.event_observer,fork.domain_observer=fork_observer.native,fork_observer.domain
        same(model.state_dict(),fork.state_dict())
        assert session==fork_session and protocol.state_dict()==fork_protocol.state_dict()
        assert observer.state==fork_observer.state and observer.carriers==fork_observer.carriers
        assert maximum_error<1e-8
        archive=out/(label+'_complete.npz')
        snapshot(archive,model,session,observer,protocol,sources,reference,dt)
    info=observations(model,observer,protocol,session,reference)
    info.update(archive=str(archive),sha256=digest(archive),primary=label=='symmetric_dt2',
        material_scheme=model.material_scheme,dt_ms=dt,external_interval_ms=4.,next_text_cursor=2688,
        original_reactions={k:model.reaction_counts.get(k,0)-reference['original_reactions'].get(k,0)
                            for k in set(model.reaction_counts)|set(reference['original_reactions'])},
        nll_nats_per_byte={k:float(np.mean([x['scores'][k] for x in rows])) for k in rows[0]['scores']},
        maximum_error=maximum_error,restored_future_intervals=128,
        all_records_physics_rngs_session_protocol_and_observers_exact=True,
        event_clocks_monotone_and_in_actual_interval=True,wall_seconds=time.perf_counter()-start)
    atomic_json(out/(label+'_summary.json'),info)
    return info


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert base=='d76830daf1bcc77bccd3afceb6e7764f3027ef47'
    parent,meta=load_exchange(args.archive);parent_hash=digest(args.archive)
    assert parent_hash=='6ec4fa467f768c630015a39c88a6d4cbdc572651821851d30fd9289ad48dc285'
    assert parent.t==12032. and parent.n==46
    corpus=meta['sources']['corpus'];data=args.data/'train.npy'
    assert digest(data)==corpus['files_sha256']['train.npy']
    raw=np.asarray(np.load(data,mmap_mode='r')[2432:2688])
    if len(raw)!=256 or np.any((raw<4)|(raw>=260)):
        raise ValueError('Declare actual document boundaries before proceeding')
    block=np.asarray(raw-4,np.uint8);np.save(args.out/'fresh_train_2432_2688.npy',block,allow_pickle=False)
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        text_selection=dict(split='train',start=2432,end=2688,token_offset=4,
                            sha256=__import__('hashlib').sha256(block.tobytes()).hexdigest()),
        protocol='256 actual OWT bytes, 4ms each; explicit numerical scheme/dt per full-state archive')
    paths=[Path(__file__),ROOT/'scripts/ib/generative/source_protocol.py',
        ROOT/'scripts/ib/generative/native_learning_stream.py',ROOT/'scripts/ib/generative/drive_maintenance_audit.py',
        ROOT/'scripts/ib/generative/domain_rate_audit.py',ROOT/'docs/information_boltzmann/CGB_FULL_STEP_PREREG_20261022.md',
        ROOT/'docs/information_boltzmann/CGB_FULL_STEP_DERIVATION_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        dest=args.out/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(args.out/'sources.json',sources)
    legacy=legacy_check(args.archive.parent,args.out,base)
    atomic_json(args.out/'legacy_equivalence.json',legacy)
    arms={label:run_arm(label,parent,meta,block,args.out,sources)
          for label in ('legacy_dt2','symmetric_dt2','symmetric_dt1')}
    assert digest(args.archive)==parent_hash
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=base,sources=sources,
        source_unchanged=True,code_sha256=hashes,arms=arms,legacy_equivalence=legacy,
        optimizer_updates=0,reference_temperature=1.,
        scope='Whole-kernel interface and same actual-clock numerical comparison, not a capability or ensemble convergence conclusion'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_drive_maintenance_20261022/text_drive_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
