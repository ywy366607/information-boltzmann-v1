"""Field-coupled physical storage on a complete, continuously learning body."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.charge_storage import select_storage_constitution
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.energy_observation import material_energy_components
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.trace_fate_audit import cohort,frame


def legacy_replay(previous):
    m,meta=load_exchange(previous/'natural_mid.npz')
    final,fmeta=load_exchange(previous/'natural_complete.npz')
    session=copy.deepcopy(meta['session'])
    obs=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    m.event_observer,m.domain_observer=obs.native,obs.domain
    data=np.load(previous/'actual_train_4096_4136.npy',allow_pickle=False)[16:32]
    expected=json.loads((previous/'f0_native_AB_summary.json').read_text())['scored_input_records'][16:]
    for target,row in zip(data,expected):
        actual=advance_byte(m,session,int(target),'temporal_order')
        assert all(row[k]==v for k,v in actual.items())
    same(m.state_dict(),final.state_dict())
    assert session==fmeta['session'] and obs.state==fmeta['growth_observer'] and obs.carriers==fmeta['carrier_observer']
    return dict(passed=True,actual_diagnostic_replay_bytes=16,old_constitution_full_future_exact=True,
                legacy_serialized_params_unchanged='polarized_storage' not in m.state_dict()['params'])


def save(path,m,obs,session,reference,sources,count,label,choice,inherited,last):
    obs.validate(m);assert m.t==reference['start_ms']+4*count
    partition=m.storage_charge_partition()
    partition={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in partition.items()}
    save_exchange(path,m,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(obs.state),
        carrier_observer=copy.deepcopy(obs.carriers),stage_reference=reference,sources=sources,
        inherited_external_observation=copy.deepcopy(inherited),dt_ms=2.,external_interval_ms=4.,
        storage_constitution_choice=choice,storage_charge_partition=partition,last_actual_byte=int(last),
        experience_protocol=dict(intervals=count,actual_text_exposures=count,
            fresh_train_exposures=min(count,128),fresh_validation_exposures=max(count-128,0),
            next_train_byte=4128+min(count,128),next_validation_byte=1280+max(count-128,0)),
        is_main_candidate=label=='reference',is_constitution_candidate=label=='polarized'))


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,meta=load_exchange(args.archive);parent_hash=digest(args.archive)
    assert parent_hash=='57b50f0b6dc155ff0af639938a938c85781c1ea5a66bb7033438a6c8db945134'
    assert parent.n==55 and parent.t==23424.
    replay=legacy_replay(args.archive.parent)
    corpus=meta['sources']['corpus'];blocks=[]
    for split,lo,hi in [('train',4128,4256),('validation',1280,1408)]:
        file=args.data/(split+'.npy');assert digest(file)==corpus['files_sha256'][file.name]
        raw=np.asarray(np.load(file,mmap_mode='r')[lo:hi]);assert len(raw)==128 and np.all((raw>=4)&(raw<260))
        block=np.asarray(raw-4,np.uint8);blocks.append(block);np.save(args.out/(split+'_bytes.npy'),block,allow_pickle=False)
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        selections=dict(train=[4128,4256],validation=[1280,1408]),
        scope='Same net-state continuous reference and an explicit physical-constitution branch; no winner selection')
    if args.resume_failed:
        diagnosis=json.loads((args.resume_failed/'accounting_diagnosis.json').read_text())
        assert diagnosis['replay_scores_exact'] and diagnosis['actual_failed_intervals']==108
        sources['failed_source_run']=str(args.resume_failed)
        sources['accounting_diagnosis_sha256']=digest(args.resume_failed/'accounting_diagnosis.json')
        sources['reconstructed_intervals']=108
        sources['numerical_repair']='TwoSum cumulative arithmetic only; no rebase, threshold or physical law change'
    paths=[Path(__file__),ROOT/'scripts/ib/generative/native_learning_stream.py',
        ROOT/'scripts/ib/generative/domain_rate_audit.py',ROOT/'scripts/ib/generative/trace_fate_audit.py',
        ROOT/'docs/information_boltzmann/CGB_POLARIZED_STORAGE_DERIVATION_20261022.md',
        ROOT/'docs/information_boltzmann/CGB_POLARIZED_STORAGE_PREREG_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    if args.resume_failed:
        paths.append(ROOT/'docs/information_boltzmann/CGB_STORAGE_ACCOUNTING_REPAIR_20261022.md')
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        dst=args.out/'source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    atomic_json(args.out/'sources.json',sources);atomic_json(args.out/'legacy_replay.json',replay)
    summaries={}
    for label,polarized in [('reference',False),('polarized',True)]:
        if args.resume_failed and label=='reference':
            # This complete natural trajectory already exists. Reuse exact
            # files, never advance its experience twice to repair a candidate.
            for source in args.resume_failed.glob('reference_*'):
                if source.is_file():
                    dst=args.out/source.name;shutil.copy2(source,dst)
                    assert digest(source)==digest(dst)
            summary=json.loads((args.out/'reference_summary.json').read_text())
            summary['archive']=str(args.out/'reference_complete.npz')
            summary['reused_exact_complete_reference']=True
            summaries[label]=summary
            atomic_json(args.out/'reference_summary.json',summary)
            continue
        m=fork_material(parent)
        choice=select_storage_constitution(m,polarized)
        if args.resume_failed:
            m.enable_compensated_ledgers()
        session=copy.deepcopy(meta['session']);session.update(phase='storage_train',next_phase_byte=0)
        obs=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
        reference=dict(start_ms=m.t,work=m.work_in,heat=m.heat_out,energy=material_energy_components(m),
            cohort=cohort(m,obs),carrier_next_id=obs.carriers['next_id'],carrier_deaths=obs.carriers['deaths'],
            births=m.domain_births,deaths=m.domain_deaths)
        inherited=meta['inherited_external_observation']
        last=meta['last_actual_byte'];rows=[];windows=[];maximum=0.;restored=None
        physical_array_bytes=sum(getattr(m,name).nbytes for name in m._arrays if not name.startswith('gene_'))
        physical_shapes={name:list(getattr(m,name).shape) for name in m._arrays if not name.startswith('gene_')}
        save(args.out/(label+'_start.npz'),m,obs,session,reference,sources,0,label,choice,inherited,last)
        sham=fork_material(m);ssession=copy.deepcopy(session)
        sobs=CarrierObservation(sham,obs.state,obs.carriers);sham.event_observer,sham.domain_observer=sobs.native,sobs.domain
        counts={};thermal_electrical=[]
        expected=[];event_expected=None;replayed_events=0
        if args.resume_failed:
            expected=[json.loads(s) for s in (args.resume_failed/'polarized_intervals.jsonl').read_text().splitlines()]
            with gzip.open(args.resume_failed/'polarized_events.jsonl.gz','rt',encoding='utf-8') as stream:
                event_expected=iter([json.loads(s) for s in stream])
        with gzip.open(args.out/(label+'_events.jsonl.gz'),'wt',encoding='utf-8') as events, \
                gzip.open(args.out/(label+'_domains.jsonl.gz'),'wt',encoding='utf-8') as domains, \
                (args.out/(label+'_intervals.jsonl')).open('w',encoding='utf-8') as interval:
            def native(row):
                nonlocal replayed_events
                obs.native(row)
                if expected and m.t<expected[-1]['after_ms']:
                    assert row==next(event_expected)
                    replayed_events+=1
                if row['kind']=='thermal':
                    key='boundary' if row.get('interface',-1)>=0 else 'internal'
                    counts[key]=counts.get(key,0)+1
                    assert bool(row['charge_changes'])==polarized
                    thermal_electrical.append(row['electrical_energy_delta'])
                events.write(json.dumps(row,separators=(',',':'))+'\n')
            def domain(row):obs.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
            m.event_observer,m.domain_observer=native,domain
            for index,target in enumerate(np.concatenate(blocks)):
                phase='storage_train' if index<128 else 'storage_validation'
                if index==128:
                    session.update(phase=phase,next_phase_byte=0)
                    rsession.update(phase=phase,next_phase_byte=0)
                constitution=m.port_constitution().state_dict()
                prediction=np.asarray(constitution['matching_world_voltage']);assert prediction.dtype.kind=='f'
                persistence=(m.world_charge/m.world_capacitance).copy()
                row=advance_byte(m,session,int(target),phase)
                if index<32:assert row==advance_byte(sham,ssession,int(target),phase)
                if index==31:
                    same(m.state_dict(),sham.state_dict());assert session==ssession
                    assert obs.state==sobs.state and obs.carriers==sobs.carriers
                if restored is not None:
                    assert constitution==restored.port_constitution().state_dict()
                    assert row==advance_byte(restored,rsession,int(target),phase)
                actual=m.world_charge/m.world_capacitance
                risks={key:.5*float(np.sum(m.world_capacitance*(value-actual)**2)) for key,value in
                    [('native_matching',prediction),('world_persistence',persistence)]}
                row.update(phase=phase,source_train_byte=4128+index if index<128 else None,
                    source_validation_byte=1280+index-128 if index>=128 else None,
                    native_constitution=constitution,quadratic_sensory_risk=risks)
                if index<len(expected):
                    assert row==expected[index]
                    if index+1==len(expected):
                        assert next(event_expected,None) is None
                rows.append(row);interval.write(json.dumps(row,separators=(',',':'))+'\n');last=int(target)
                maximum=max(maximum,max(abs(float(v)) for v in m.balances().values()))
                assert maximum<1e-8
                if (index+1)%32==0:
                    info=frame(m,obs,reference,index+1,phase);windows.append(info)
                    atomic_json(args.out/(label+'_windows.json'),windows)
                    status=dict(arm=label,interval=index+1,time_ms=m.t,nodes=m.n,maximum_error=maximum)
                    atomic_json(args.out/'progress.json',dict(status='running',**status));print(json.dumps(status),flush=True)
                if index+1==128:
                    path=args.out/(label+'_mid.npz')
                    save(path,m,obs,session,reference,sources,128,label,choice,inherited,last)
                    restored,rmeta=load_exchange(path);rsession=copy.deepcopy(rmeta['session'])
                    robs=CarrierObservation(restored,rmeta['growth_observer'],rmeta['carrier_observer'])
                    restored.event_observer,restored.domain_observer=robs.native,robs.domain
            same(m.state_dict(),restored.state_dict());assert session==rsession
            assert obs.state==robs.state and obs.carriers==robs.carriers
            path=args.out/(label+'_complete.npz');save(path,m,obs,session,reference,sources,256,label,choice,inherited,last)
        summary=frame(m,obs,reference,256,'storage_validation')
        summary.update(primary=label=='reference',constitution_candidate=polarized,
            archive=str(path),sha256=digest(path),next_train_byte=4256,next_validation_byte=1408,
            physical_array_bytes_at_start=physical_array_bytes,physical_shapes_at_start=physical_shapes,
            thermal_events=counts,thermal_electrical_energy_sum=float(np.sum(thermal_electrical)),
            thermal_events_with_nonzero_electrical_energy=sum(v!=0. for v in thermal_electrical),
            maximum_error=maximum,observer_sham_intervals=32,restored_future_intervals=128,
            reconstructed_failed_intervals=len(expected),reconstructed_native_events_exact=replayed_events,
            nll_by_phase={phase:{k:float(np.mean([r['scores'][k] for r in rows if r['phase']==phase]))
                for k in rows[0]['scores']} for phase in ('storage_train','storage_validation')},
            phase_sample_counts=dict(storage_train=128,storage_validation=128),
            quadratic_sensory_risk={phase:{k:float(np.mean([r['quadratic_sensory_risk'][k] for r in rows if r['phase']==phase]))
                for k in rows[0]['quadratic_sensory_risk']} for phase in ('storage_train','storage_validation')})
        summaries[label]=summary;atomic_json(args.out/(label+'_summary.json'),summary)
    assert summaries['reference']['physical_array_bytes_at_start']==summaries['polarized']['physical_array_bytes_at_start']
    assert summaries['reference']['physical_shapes_at_start']==summaries['polarized']['physical_shapes_at_start']
    assert parent_hash==digest(args.archive) and hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'report.json',dict(passed=True,protocol_kind='polarized_storage',sources=sources,
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=hashes,source_unchanged=True,legacy_replay=replay,arms=summaries,
        scope='New reciprocal charge/configuration material branch, not automatic migration or a demonstrated learning winner'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_native_timing_20261022/natural_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-failed',type=Path,default=None,
        help='Reuse complete reference and exactly reconstruct failed candidate with compensated arithmetic')
    main(p.parse_args())
