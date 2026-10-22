"""Native constitutive ports on the current complete physical individual."""
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
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.readout_observation import bridge_pool_port_current
from information_boltzmann.cgb.energy_observation import material_energy_components
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.trace_fate_audit import cohort,frame
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json


def independent_neutral_current(m):
    assert np.count_nonzero(m.charge[m.port_nodes])==0
    current=bridge_pool_port_current(m)
    if m.domain_reactions_enabled:
        lookup={int(node):p for p,node in enumerate(m.port_nodes)}
        for row in m.domain_generator_rates()['inverse_channels']:
            if row['parent'] in lookup:
                current[lookup[row['parent']]]+=row['accepted_rate']*m.charge[row['child']]
    return current


def external_prepare(m,world_voltage):
    before=m.energy(),m.free_energy();old=m.charge[m.port_nodes].copy();world=m.world_charge.copy()
    m.charge[m.port_nodes]=0.;m.electrode_charge+=float(old.sum())
    m.world_charge[:]=m.world_capacitance*np.asarray(world_voltage)
    m.source_charge+=float(np.sum(m.world_charge-world))
    du,dg=m.energy()-before[0],m.free_energy()-before[1]
    m.work_in+=du;m.entropy_balance+=du-dg
    m._validate()
    assert max(abs(float(v)) for v in m.balances().values())<1e-8
    return dict(time_ms=m.t,external_work=du,free_energy_change=dg,
        source_charge_delta=float(np.sum(m.world_charge-world)),electrode_charge_delta=float(old.sum()),
        body_port_charge_before=old.tolist(),world_voltage=np.asarray(world_voltage).tolist(),
        role='Declared ideal external preparation, not an autonomous action; controller cost beyond this work is not modeled')


def transfer_rates(m):
    result={}
    for event in m.events():
        if event.interface>=0 and m._charge_changes(event):
            result[(event.interface,event.kind,event.sign)]=(event,m.event_delta(event)[1],m.event_rate(event))
    return result


def constitutive_assay(parent,meta,out):
    before=parent.state_dict();p=parent.port_constitution();match=p.match_sensation()
    assert all(x=='unique_finite' for x in match['sensor_status'])
    v=np.asarray(match['matching_world_voltage']);dv=parent.params.charge_per_event/parent.world_capacitance
    prepared={};details={}
    for label,voltage in [('matching',v),('quantum_deviation',v+dv)]:
        m=fork_material(parent);preparation=external_prepare(m,voltage)
        observer=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
        m.event_observer,m.domain_observer=observer.native,observer.domain
        direct=independent_neutral_current(m)
        expected=p.body_current+p.sensory_current(voltage)
        np.testing.assert_allclose(direct,expected,atol=2e-13,rtol=1e-12)
        if label=='matching':np.testing.assert_allclose(direct,0.,atol=2e-13)
        path=out/(label+'_prepared.npz')
        save_exchange(path,m,dict(session=copy.deepcopy(meta['session']),growth_observer=copy.deepcopy(observer.state),
            carrier_observer=copy.deepcopy(observer.carriers),external_preparation=preparation,
            actual_source='finite world, no source for the next 4ms',is_main_candidate=False))
        restored,rmeta=load_exchange(path)
        robs=CarrierObservation(restored,rmeta['growth_observer'],rmeta['carrier_observer'])
        restored.event_observer,restored.domain_observer=robs.native,robs.domain
        sham=fork_material(m) if label=='matching' else None
        if sham is not None:
            sobs=CarrierObservation(sham,observer.state,observer.carriers)
            sham.event_observer,sham.domain_observer=sobs.native,sobs.domain
        initial_rates=transfer_rates(m);events=dict(m.reaction_counts)
        traces=m.trace.copy(),m.port_trace.copy()
        for _ in range(2):
            m.port_constitution().match_sensation()
            m.step(2.,None);restored.step(2.,None)
            if sham is not None:sham.step(2.,None)
        same(m.state_dict(),restored.state_dict())
        assert observer.state==robs.state and observer.carriers==robs.carriers
        if sham is not None:
            same(m.state_dict(),sham.state_dict())
            assert observer.state==sobs.state and observer.carriers==sobs.carriers
        save_exchange(out/(label+'_after_4ms.npz'),m,dict(session=copy.deepcopy(meta['session']),
            growth_observer=copy.deepcopy(observer.state),carrier_observer=copy.deepcopy(observer.carriers),
            external_preparation=preparation,actual_no_source_elapsed_ms=4.,new_text_exposures=0,is_main_candidate=False))
        prepared[label]=(m,initial_rates)
        details[label]=dict(preparation=preparation,independent_initial_current=direct.tolist(),
            native_expected_initial_current=expected.tolist(),time_ms=m.t,nodes=m.n,
            final_port_charge=m.charge[m.port_nodes].tolist(),final_world_charge=m.world_charge.tolist(),
            final_trace_count=int(m.trace.sum()+m.port_trace.sum()),
            original_reactions={k:m.reaction_counts.get(k,0)-events.get(k,0) for k in set(m.reaction_counts)|set(events)},
            restored_full_future_exact=True,read_only_sham_exact=True if sham is not None else None,
            maximum_balance_error=max(abs(float(x)) for x in m.balances().values()))
    a,ra=prepared['matching'];b,rb=prepared['quantum_deviation'];affinities=[]
    assert set(ra)==set(rb)
    for key,(event,dga,ratea) in ra.items():
        dgb,rateb=rb[key][1:]
        predicted=sum(dq*dv[event.interface] for node,dq in parent._charge_changes(event) if node==parent.n+event.interface)
        assert abs(dgb-dga-predicted)<2e-12
        affinities.append(dict(port=key[0],kind=key[1],sign=key[2],actual_delta_g_change=dgb-dga,
                               native_charge_work_prediction=predicted,matching_rate=ratea,deviated_rate=rateb))
    same(before,parent.state_dict())
    result=dict(passed=True,constitution=p.state_dict(),quantum_voltage_deviation=dv.tolist(),arms=details,
        native_affinity_changes=affinities,
        final_world_difference_l2=float(np.linalg.norm(a.world_charge-b.world_charge)),
        final_port_charge_difference_l2=float(np.linalg.norm(a.charge[a.port_nodes]-b.charge[b.port_nodes])),
        original_main_state_unchanged=True,
        scope='Current complete individual, instantaneous generator and 4 actual ms; current match is not a forecast proof')
    atomic_json(out/'constitutive_assay.json',result)
    return result


def save_natural(path,m,session,observer,reference,sources,index,parent_meta):
    observer.validate(m);assert m.t==reference['start_ms']+4*index
    inherited={k:copy.deepcopy(parent_meta[k]) for k in ('frozen_external_readouts','matched_training_frequencies','observed_aggregates')}
    save_exchange(path,m,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(observer.state),
        carrier_observer=copy.deepcopy(observer.carriers),stage_reference=reference,sources=sources,
        dt_ms=2.,external_interval_ms=4.,native_port_constitution=m.port_constitution().state_dict(),
        inherited_external_observation=inherited,
        experience_protocol=dict(intervals=index,new_text_exposures=index,next_train_byte=3840+index,
                                 next_validation_byte=1280,source='fresh train')))


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,meta=load_exchange(args.archive);parent_hash=digest(args.archive)
    assert parent_hash=='1f92108dcd7db7cd8aca7de0e9c69e55af2d7ebb7e05f6254823310910ddf19a'
    assert parent.n==55 and parent.t==22272.
    corpus=meta['sources']['corpus'];data=args.data/'train.npy'
    assert digest(data)==corpus['files_sha256']['train.npy']
    raw=np.asarray(np.load(data,mmap_mode='r')[3840:4096])
    if len(raw)!=256 or np.any((raw<4)|(raw>=260)):raise ValueError('Declare document boundaries before continuing')
    block=np.asarray(raw-4,np.uint8);np.save(args.out/'fresh_train_3840_4096.npy',block,allow_pickle=False)
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        selection=dict(split='train',start=3840,end=4096,token_offset=4),
        protocol='Native constitution before each of 256 new actual 4ms sensory exposures; no fitted head')
    paths=[Path(__file__),ROOT/'scripts/ib/generative/native_learning_stream.py',
        ROOT/'scripts/ib/generative/trace_fate_audit.py',ROOT/'scripts/ib/generative/domain_rate_audit.py',
        ROOT/'docs/information_boltzmann/CGB_NATIVE_PREDICTIVE_PORT_DERIVATION_20261022.md',
        ROOT/'docs/information_boltzmann/CGB_NATIVE_PORT_PREREG_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for path in paths:
        dst=args.out/'source_snapshot'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    atomic_json(args.out/'sources.json',sources)
    assay=constitutive_assay(parent,meta,args.out)
    m=fork_material(parent);session=copy.deepcopy(meta['session']);session.update(phase='fresh_H',next_phase_byte=0)
    observer=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    reference=dict(start_ms=m.t,work=m.work_in,heat=m.heat_out,energy=material_energy_components(m),
        cohort=cohort(m,observer),carrier_next_id=observer.carriers['next_id'],carrier_deaths=observer.carriers['deaths'],
        births=m.domain_births,deaths=m.domain_deaths,original_reactions=dict(m.reaction_counts))
    save_natural(args.out/'natural_start.npz',m,session,observer,reference,sources,0,meta)
    sham=fork_material(m);sham_session=copy.deepcopy(session)
    sobs=CarrierObservation(sham,observer.state,observer.carriers)
    sham.event_observer,sham.domain_observer=sobs.native,sobs.domain
    last=meta['last_actual_byte'];rows=[];windows=[];maximum_error=0.;restored=None
    with gzip.open(args.out/'natural_events.jsonl.gz','wt',encoding='utf-8') as events, \
         gzip.open(args.out/'natural_domains.jsonl.gz','wt',encoding='utf-8') as domains, \
         (args.out/'natural_intervals.jsonl').open('w',encoding='utf-8') as records:
        def native(row):observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        m.event_observer,m.domain_observer=native,domain
        for index,target in enumerate(block):
            constitution=m.port_constitution().state_dict()
            assert all(s=='unique_finite' for s in constitution['sensor_status'])
            prediction=np.asarray(constitution['matching_world_voltage'])
            persistence=(m.world_charge/m.world_capacitance).copy()
            previous=np.array([3. if (last>>bit)&1 else -3. for bit in range(8)])
            old_source=np.asarray(session['source_only_voltage'])
            held_mean=previous+(old_source-previous)*np.exp(-4*m.source_conductance/m.world_capacitance)
            row=advance_byte(m,session,int(target),'fresh_H')
            assert row==advance_byte(sham,sham_session,int(target),'fresh_H')
            if restored is not None:
                assert constitution==restored.port_constitution().state_dict()
                assert row==advance_byte(restored,restored_session,int(target),'fresh_H')
            actual=m.world_charge/m.world_capacitance
            risk={key:.5*float(np.sum(m.world_capacitance*(value-actual)**2)) for key,value in
                  [('native_matching',prediction),('world_persistence',persistence),('held_previous_source_mean',held_mean)]}
            row.update(phase='fresh_H',corpus_train_byte=3840+index,native_constitution=constitution,
                future_world_voltage=actual.tolist(),world_persistence=persistence.tolist(),
                held_previous_source_mean=held_mean.tolist(),quadratic_sensory_risk=risk)
            rows.append(row);records.write(json.dumps(row,separators=(',',':'))+'\n');last=int(target)
            maximum_error=max(maximum_error,max(abs(float(v)) for v in m.balances().values()))
            if (index+1)%32==0:
                info=frame(m,observer,reference,index+1,'fresh_H')
                info['nll_nats_per_byte']={key:float(np.mean([r['scores'][key] for r in rows[-32:]])) for key in row['scores']}
                info['quadratic_sensory_risk']={key:float(np.mean([r['quadratic_sensory_risk'][key] for r in rows[-32:]])) for key in risk}
                windows.append(info);atomic_json(args.out/'natural_windows.json',windows)
                status=dict(interval=index+1,time_ms=m.t,nodes=m.n,maximum_error=maximum_error,
                            quadratic_sensory_risk=info['quadratic_sensory_risk'])
                atomic_json(args.out/'progress.json',dict(status='running',**status));print(json.dumps(status),flush=True)
                events.flush();domains.flush();records.flush()
            if index+1==128:
                path=args.out/'natural_mid.npz';save_natural(path,m,session,observer,reference,sources,128,meta)
                restored,rmeta=load_exchange(path);restored_session=copy.deepcopy(rmeta['session'])
                robs=CarrierObservation(restored,rmeta['growth_observer'],rmeta['carrier_observer'])
                restored.event_observer,restored.domain_observer=robs.native,robs.domain
        same(m.state_dict(),restored.state_dict());same(m.state_dict(),sham.state_dict())
        assert session==restored_session==sham_session
        assert observer.state==robs.state==sobs.state and observer.carriers==robs.carriers==sobs.carriers
        assert maximum_error<1e-8
        path=args.out/'natural_complete.npz';save_natural(path,m,session,observer,reference,sources,256,meta)
    summary=frame(m,observer,reference,256,'fresh_H')
    summary.update(archive=str(path),sha256=digest(path),primary=True,
        nll_by_phase=dict(fresh_H={key:float(np.mean([r['scores'][key] for r in rows])) for key in rows[0]['scores']}),
        phase_sample_counts=dict(fresh_H=256),
        quadratic_sensory_risk={key:float(np.mean([r['quadratic_sensory_risk'][key] for r in rows])) for key in rows[0]['quadratic_sensory_risk']},
        out_of_source_amplitude_count=int(sum(np.count_nonzero(np.abs(r['native_constitution']['matching_world_voltage'])>3.) for r in rows)),
        native_voltage_count=256*8,maximum_error=maximum_error,next_train_byte=4096,next_validation_byte=1280,
        observer_only_sham_bytes=256,restored_future_intervals=128,
        all_native_values_physics_rngs_scores_session_observers_exact=True,
        scope='Constitutive sensory matching is evaluated as a prospective signal, not assumed a learned forecast')
    assert digest(args.archive)==parent_hash and hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'natural_summary.json',summary)
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_kind='native_port',sources=sources,source_unchanged=True,code_sha256=hashes,
        constitutive_assay=assay,arms=dict(natural=summary),body_optimizer_updates=0))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_prediction_observation_20261022/natural_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000');p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
