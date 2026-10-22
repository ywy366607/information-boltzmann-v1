"""Temporal ordering through the complete existing material generator."""
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
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.learning_observation import parameter_quench
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.trace_fate_audit import cohort,frame
from information_boltzmann.cgb.energy_observation import material_energy_components


def neutralize_charge(m):
    """An external, work/charge-accounted diagnostic preparation only."""
    before=m.energy(),m.free_energy()
    old_brain=m.charge.copy();old_world=m.world_charge.copy()
    m.charge[:]=0.;m.world_charge[:]=0.
    m.electrode_charge+=float(old_brain.sum())
    m.source_charge-=float(old_world.sum())
    du,dg=m.energy()-before[0],m.free_energy()-before[1]
    m.work_in+=du;m.entropy_balance+=du-dg
    error=max(abs(float(v)) for v in m.balances().values())
    assert error<1e-8
    return dict(external_work=du,free_energy_change=dg,
        electrode_charge_delta=float(old_brain.sum()),source_charge_delta=-float(old_world.sum()),
        maximum_error=error,scope='External diagnostic preparation, controller cost unmodeled; main history unchanged')


def panel(m,obs):
    ids=obs.state['ids']
    return dict(time_ms=m.t,ids=ids,coordinates=m.coordinates.tolist(),
        charge=m.charge.tolist(),world_voltage=(m.world_charge/m.world_capacitance).tolist(),
        charged=m.charged.tolist(),depleted=m.depleted.tolist(),
        free=m.free.tolist(),carried=m.carried.tolist(),cohort=cohort(m,obs),
        bridge_conductance=m.bridge_conductance().tolist(),
        boundary_conductance=m.effective_port_conductance().tolist(),
        nodes=m.n,bridges=len(m.pre),trace=int(m.trace.sum()+m.port_trace.sum()),
        bound_support=int(m.bound.sum()+m.port_bound.sum()),
        source_work=m.work_in,heat=m.heat_out,energy=m.energy(),
        maximum_error=max(abs(float(v)) for v in m.balances().values()))


def save(path,m,obs,session,*,sources,index,primary,reference=None,last_actual_byte=None,extra=None):
    obs.validate(m)
    metadata=dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(obs.state),
        carrier_observer=copy.deepcopy(obs.carriers),sources=sources,is_main_candidate=primary,
        dt_ms=2.,external_interval_ms=4.,last_actual_byte=last_actual_byte,
        experience_protocol=dict(intervals=index,actual_text_exposures=index,
            fresh_primary_exposures=index if primary else 0,
            next_train_byte=4096+index if primary else None,next_validation_byte=1280))
    if reference is not None:metadata['stage_reference']=reference
    if extra:metadata.update(extra)
    save_exchange(path,m,metadata)


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,meta=load_exchange(args.archive)
    parent_hash=digest(args.archive)
    assert parent_hash=='9cd3bdb7bd3d1349871b4968378d82950fac8234e64b6151575a700020b4867f'
    assert parent.n==55 and parent.t==23296.
    corpus=meta['sources']['corpus']
    assert digest(args.data/'train.npy')==corpus['files_sha256']['train.npy']
    raw=np.asarray(np.load(args.data/'train.npy',mmap_mode='r')[4096:4136])
    assert len(raw)==40 and np.all((raw>=4)&(raw<260))
    block=np.asarray(raw-4,np.uint8)
    np.save(args.out/'actual_train_4096_4136.npy',block,allow_pickle=False)
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        primary_selection=[4096,4128],diagnostic_future_selection=[4128,4136],
        future_noise_offsets=[0,1000000,2000000,3000000])
    paths=[Path(__file__),ROOT/'scripts/ib/generative/native_learning_stream.py',
        ROOT/'scripts/ib/generative/domain_rate_audit.py',ROOT/'scripts/ib/generative/trace_fate_audit.py',
        ROOT/'docs/information_boltzmann/CGB_NATIVE_TIMING_PREREG_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        dst=args.out/'source_snapshot'/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    atomic_json(args.out/'sources.json',sources)
    summaries={};primary_summary=None;windows=[]
    for family in range(4):
        for closed in (False,True):
            for order in ('AB','BA'):
                name=f'f{family}_'+('closed_' if closed else 'native_')+order
                primary=name=='f0_native_AB'
                m=fork_material(parent)
                if family:
                    for rng in (m.rng_reaction,m.rng_motion,m.rng_field):rng.bit_generator.advance(1000000*family)
                barrier=parameter_quench(m,write_rate=0.) if closed else None
                session=copy.deepcopy(meta['session']);session.update(phase='temporal_order',next_phase_byte=0)
                obs=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
                reference=dict(start_ms=m.t,work=m.work_in,heat=m.heat_out,energy=material_energy_components(m),
                    cohort=cohort(m,obs),carrier_next_id=obs.carriers['next_id'],carrier_deaths=obs.carriers['deaths'],
                    births=m.domain_births,deaths=m.domain_deaths)
                indices=list(range(32)) if order=='AB' else list(range(8,16))+list(range(8))+list(range(16,32))
                rows=[];max_error=0.;restored=None
                if primary:
                    save(args.out/'natural_start.npz',m,obs,session,sources=sources,index=0,primary=True,
                         reference=reference,last_actual_byte=meta.get('last_actual_byte',104))
                    sham=fork_material(m);ssession=copy.deepcopy(session)
                    sobs=CarrierObservation(sham,obs.state,obs.carriers)
                    sham.event_observer,sham.domain_observer=sobs.native,sobs.domain
                with gzip.open(args.out/(name+'_events.jsonl.gz'),'wt',encoding='utf-8') as log:
                    def native(row):obs.native(row);log.write(json.dumps(row,separators=(',',':'))+'\n')
                    def domain(row):obs.domain(row);row['domain_record']=True;log.write(json.dumps(row,separators=(',',':'))+'\n')
                    m.event_observer,m.domain_observer=native,domain
                    for index,source_index in enumerate(indices):
                        target=int(block[source_index])
                        row=advance_byte(m,session,target,'temporal_order')
                        if primary:assert row==advance_byte(sham,ssession,target,'temporal_order')
                        if restored is not None:assert row==advance_byte(restored,rsession,target,'temporal_order')
                        row.update(source_train_byte=4096+source_index)
                        rows.append(row)
                        max_error=max(max_error,max(abs(float(v)) for v in m.balances().values()))
                        if primary and index+1 in (16,32):
                            info=frame(m,obs,reference,index+1,'temporal_order')
                            windows.append(info)
                            if index+1==16:
                                save(args.out/'natural_mid.npz',m,obs,session,sources=sources,index=16,
                                     primary=True,reference=reference,last_actual_byte=target)
                                restored,rmeta=load_exchange(args.out/'natural_mid.npz');rsession=copy.deepcopy(rmeta['session'])
                                robs=CarrierObservation(restored,rmeta['growth_observer'],rmeta['carrier_observer'])
                                restored.event_observer,restored.domain_observer=robs.native,robs.domain
                assert max_error<1e-8
                if primary:
                    same(m.state_dict(),sham.state_dict());same(m.state_dict(),restored.state_dict())
                    assert session==ssession==rsession
                    assert obs.state==sobs.state==robs.state and obs.carriers==sobs.carriers==robs.carriers
                    save(args.out/'natural_complete.npz',m,obs,session,sources=sources,index=32,primary=True,
                         reference=reference,last_actual_byte=int(block[31]),
                         extra=dict(inherited_external_observation=meta['inherited_external_observation']))
                    primary_summary=frame(m,obs,reference,32,'temporal_order')
                    primary_summary.update(primary=True,archive=str(args.out/'natural_complete.npz'),
                        sha256=digest(args.out/'natural_complete.npz'),next_train_byte=4128,next_validation_byte=1280,
                        nll_by_phase=dict(temporal_order={k:float(np.mean([r['scores'][k] for r in rows])) for k in rows[0]['scores']}),
                        phase_sample_counts=dict(temporal_order=32),restored_future_intervals=16,observer_only_sham_bytes=32)
                    atomic_json(args.out/'natural_summary.json',primary_summary)
                    atomic_json(args.out/'natural_windows.json',windows)
                save(args.out/(name+'_ordered.npz'),m,obs,session,sources=sources,index=32,primary=False,
                     last_actual_byte=int(block[31]),extra=dict(diagnostic_barrier=barrier))
                before=panel(m,obs)
                preparation=neutralize_charge(m)
                save(args.out/(name+'_neutral.npz'),m,obs,session,sources=sources,index=32,primary=False,
                     last_actual_byte=int(block[31]),extra=dict(external_preparation=preparation))
                m.event_observer,m.domain_observer=obs.native,obs.domain
                after_preparation=panel(m,obs)
                future=[]
                for target in block[32:]:future.append(advance_byte(m,session,int(target),'diagnostic_common_future'))
                assert max(abs(float(v)) for v in m.balances().values())<1e-8
                save(args.out/(name+'_future.npz'),m,obs,session,sources=sources,index=40,primary=False,
                     last_actual_byte=int(block[-1]),extra=dict(external_preparation=preparation))
                result=dict(noise_family=family,closed_write_barrier=closed,order=order,
                    ordered=before,preparation=preparation,neutral=after_preparation,future=panel(m,obs),
                    scored_input_records=rows,diagnostic_future_records=future,maximum_error=max_error)
                summaries[name]=result
                atomic_json(args.out/(name+'_summary.json'),result)
                atomic_json(args.out/'progress.json',dict(status='running',completed_arms=len(summaries),total_arms=16,
                    last_arm=name,maximum_error=max_error))
                print(json.dumps(dict(completed_arms=len(summaries),arm=name,nodes=m.n,maximum_error=max_error)),flush=True)
    assert digest(args.archive)==parent_hash and hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'report.json',dict(passed=True,protocol_kind='native_timing',
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=sources,code_sha256=hashes,source_unchanged=True,arms=dict(natural=primary_summary),
        diagnostic_summaries={k:str(args.out/(k+'_summary.json')) for k in summaries},
        scope='Same complete generator, actual source ordering and work-accounted fast-charge preparation; not full V2 capability'))
    atomic_json(args.out/'progress.json',dict(status='completed',completed_arms=16))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_native_port_20261022/natural_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
