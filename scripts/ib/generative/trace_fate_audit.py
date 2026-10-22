"""Current full-generator native-memory conditions and real A-to-B-to-A."""
from __future__ import annotations

import argparse
import copy
import gzip
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
from information_boltzmann.cgb.learning_observation import native_learning_drift, parameter_quench
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.memory_intervention import matched_trace_condition, assert_matched_trace_states, contact_key
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def cohort(model,observer):
    result={}
    ids=observer.state['ids']
    for e,(i,j) in enumerate(zip(model.pre,model.post)):
        key='i:'+str(ids[i])+':'+str(ids[j])
        result[key]=dict(carrier_id=observer.carriers['alive'][key],trace=int(model.trace[e]),bound=int(model.bound[e]))
    for p,n in enumerate(model.port_installed):
        if n:
            key='p:'+str(p)
            result[key]=dict(carrier_id=observer.carriers['alive'][key],trace=int(model.port_trace[p]),bound=int(model.port_bound[p]))
    return result


def rates(model,observer):
    result={}
    for e in model.events():
        if e.kind in ('binding','transport','write','thermal','active'):
            key=contact_key(model,e,observer.state['ids'])
            row=result.setdefault(key,dict())
            row[e.kind+('+' if e.sign>0 else '-')]=float(model.event_rate(e))
    return dict(contacts=result,bridge_conductance=model.bridge_conductance().tolist(),
                boundary_conductance=model.effective_port_conductance().tolist(),generator=native_learning_drift(model))


def prepare(label,parent,meta):
    m=fork_material(parent)
    observer=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    records=[]
    def condition_event(row):
        observer.native(row);row['conditioned_setup_event']=True;records.append(row)
    m.event_observer=condition_event
    setup=dict(mode=label,condition=None,parameter_intervention=None,events=records)
    if label!='natural':
        setup['condition']=matched_trace_condition(m,observer.state['ids'],observer.carriers['alive'],erase='erase' in label)
    observer.validate(m)
    m.event_observer=None
    return m,observer,setup


def frame(m,observer,reference,index,phase):
    live=cohort(m,observer)
    original=reference['cohort']
    retained={key:value for key,value in live.items() if key in original and value['carrier_id']==original[key]['carrier_id']}
    rebuilt={key:value for key,value in live.items() if key in original and value['carrier_id']!=original[key]['carrier_id']}
    energy=material_energy_components(m)
    return dict(time_ms=m.t,experience_interval=index,phase=phase,nodes=m.n,free_material=int(m.material.sum()),
        internal_bridges=len(m.pre),boundary_contacts=int(m.port_installed.sum()),
        trace=int(m.trace.sum()+m.port_trace.sum()),bound_support=int(m.bound.sum()+m.port_bound.sum()),
        domain_births=m.domain_births-reference['births'],domain_deaths=m.domain_deaths-reference['deaths'],
        carrier_births=observer.carriers['next_id']-reference['carrier_next_id'],
        carrier_deaths=observer.carriers['deaths']-reference['carrier_deaths'],
        original_carriers_retained=len(retained),rebuilt_old_positions=len(rebuilt),
        original_trace_carriers_retained=sum(original[k]['trace'] for k in retained),
        original_trace_carriers_current_trace=sum(original[k]['trace']*v['trace'] for k,v in retained.items()),
        current_cohort=live,energy=energy,source_work=m.work_in-reference['work'],heat_out=m.heat_out-reference['heat'],
        energy_change=energy['total']-reference['energy']['total'],
        maximum_balance_error=max(abs(float(v)) for v in m.balances().values()))


def snapshot(path,m,observer,session,reference,sources,index):
    observer.validate(m)
    assert m.t==reference['start_ms']+4*index
    save_exchange(path,m,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(observer.state),
        carrier_observer=copy.deepcopy(observer.carriers),stage_reference=reference,sources=sources,
        dt_ms=2.,external_interval_ms=4.,experience_protocol=dict(intervals=index,
            actual_text_exposures=index,fresh_exposures=min(index,128),replay_exposures=max(0,index-128),
            next_train_byte=2688+min(index,128),next_validation_byte=256,
            replay_selection=[2560,2688],phase='fresh_F' if index<128 else 'revisit_A')))


def run(label,m,observer,setup,metadata,fresh,replay,out,sources):
    started=time.perf_counter()
    session=copy.deepcopy(metadata['session'])
    session['interventions'].append(dict(time_ms=m.t,conditional_experiment=setup,whole_state_reset=False,
        note='Natural branch has no physical intervention; other copies use declared allowed conditional jumps'))
    reference=dict(start_ms=m.t,work=m.work_in,heat=m.heat_out,energy=material_energy_components(m),
        cohort=cohort(m,observer),carrier_next_id=observer.carriers['next_id'],carrier_deaths=observer.carriers['deaths'],
        births=m.domain_births,deaths=m.domain_deaths,original_reactions=dict(m.reaction_counts))
    atomic_json(out/(label+'_setup.json'),dict(setup=setup,initial_rates=rates(m,observer),reference=reference))
    snapshot(out/(label+'_start.npz'),m,observer,session,reference,sources,0)
    windows=[];rows=[];maximum_error=0.;fork=None
    # Entire unmodified-future sham, including original observer identities.
    if label=='natural':
        sham=fork_material(m);sham_session=copy.deepcopy(session)
        sham_observer=CarrierObservation(sham,observer.state,observer.carriers)
        sham.event_observer,sham.domain_observer=sham_observer.native,sham_observer.domain
    with gzip.open(out/(label+'_events.jsonl.gz'),'wt',encoding='utf-8') as events, \
         gzip.open(out/(label+'_domains.jsonl.gz'),'wt',encoding='utf-8') as domains, \
         (out/(label+'_intervals.jsonl')).open('w',encoding='utf-8') as records:
        def native(row):
            observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):
            observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        m.event_observer,m.domain_observer=native,domain
        for index,target in enumerate(np.r_[fresh,replay]):
            phase='fresh_F' if index<128 else 'revisit_A'
            if index in (0,128):session.update(phase=phase,next_phase_byte=0)
            row=advance_byte(m,session,int(target),phase)
            if label=='natural':
                if index in (0,128):sham_session.update(phase=phase,next_phase_byte=0)
                assert row==advance_byte(sham,sham_session,int(target),phase)
            if fork is not None:
                if index==128:fork_session.update(phase=phase,next_phase_byte=0)
                assert row==advance_byte(fork,fork_session,int(target),phase)
            row.update(experience_interval=index+1,corpus_train_byte=2688+index if index<128 else 2560+index-128,
                       fresh_exposure=index<128,phase=phase)
            rows.append(row);records.write(json.dumps(row,separators=(',',':'))+'\n')
            observer.validate(m)
            maximum_error=max(maximum_error,max(abs(float(v)) for v in m.balances().values()))
            if (index+1)%32==0:
                info=frame(m,observer,reference,index+1,phase)
                info['nll_nats_per_byte']={k:float(np.mean([x['scores'][k] for x in rows[-32:]])) for k in row['scores']}
                windows.append(info);atomic_json(out/(label+'_windows.json'),windows)
                status=dict(arm=label,interval=index+1,phase=phase,time_ms=m.t,nodes=m.n,
                            retained=info['original_carriers_retained'],maximum_error=maximum_error)
                atomic_json(out/'progress.json',dict(status='running',**status));print(json.dumps(status),flush=True)
                events.flush();domains.flush();records.flush()
            if index+1==128:
                path=out/(label+'_mid.npz');snapshot(path,m,observer,session,reference,sources,128)
                fork,meta=load_exchange(path);fork_session=copy.deepcopy(meta['session'])
                fork_observer=CarrierObservation(fork,meta['growth_observer'],meta['carrier_observer'])
                fork.event_observer,fork.domain_observer=fork_observer.native,fork_observer.domain
        same(m.state_dict(),fork.state_dict())
        assert session==fork_session and observer.state==fork_observer.state and observer.carriers==fork_observer.carriers
        if label=='natural':
            same(m.state_dict(),sham.state_dict())
            assert session==sham_session and observer.state==sham_observer.state and observer.carriers==sham_observer.carriers
        assert maximum_error<1e-8
        archive=out/(label+'_complete.npz');snapshot(archive,m,observer,session,reference,sources,256)
    info=frame(m,observer,reference,256,'revisit_A')
    info.update(archive=str(archive),sha256=digest(archive),primary=label=='natural',
        nll_by_phase={phase:{k:float(np.mean([x['scores'][k] for x in rows if x['phase']==phase]))
                            for k in rows[0]['scores']} for phase in ('fresh_F','revisit_A')},
        original_reactions={k:m.reaction_counts.get(k,0)-reference['original_reactions'].get(k,0)
                            for k in set(m.reaction_counts)|set(reference['original_reactions'])},
        next_train_byte=2816,next_validation_byte=256,actual_fresh_exposures=128,actual_replay_exposures=128,
        maximum_error=maximum_error,restored_future_intervals=128,all_future_records_physics_rngs_session_observers_exact=True,
        observer_only_sham_exact=True if label=='natural' else None,wall_seconds=time.perf_counter()-started)
    atomic_json(out/(label+'_summary.json'),info)
    return info


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,meta=load_exchange(args.archive);parent_hash=digest(args.archive)
    assert parent_hash=='d1515a4aafa620adab91a7404cc78ac14ccb9ee5541addb265410ff6a4fba65c'
    assert parent.t==13056. and parent.material_scheme=='symmetric'
    corpus=meta['sources']['corpus'];data=args.data/'train.npy'
    assert digest(data)==corpus['files_sha256']['train.npy']
    values=np.load(data,mmap_mode='r')
    fresh_raw,replay_raw=np.asarray(values[2688:2816]),np.asarray(values[2560:2688])
    assert len(fresh_raw)==len(replay_raw)==128
    if any(np.any((x<4)|(x>=260)) for x in (fresh_raw,replay_raw)):
        raise ValueError('Declare document boundaries before proceeding')
    fresh,replay=np.asarray(fresh_raw-4,np.uint8),np.asarray(replay_raw-4,np.uint8)
    previous=np.load(args.archive.parent/'fresh_train_2432_2688.npy',allow_pickle=False)
    np.testing.assert_array_equal(replay,previous[128:])
    np.save(args.out/'fresh_train_2688_2816.npy',fresh,allow_pickle=False)
    np.save(args.out/'replay_train_2560_2688.npy',replay,allow_pickle=False)
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        fresh_selection=dict(split='train',start=2688,end=2816,token_offset=4),
        replay_selection=dict(split='train',start=2560,end=2688,token_offset=4,
                             verified_against_actual_previous_run=True),
        protocol='128 actual fresh bytes then 128 verified revisited bytes; 4ms each, original dt2ms, full native learning')
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/memory_intervention.py',
        ROOT/'scripts/ib/generative/native_learning_stream.py',ROOT/'scripts/ib/generative/domain_rate_audit.py',
        ROOT/'docs/information_boltzmann/CGB_TRACE_FATE_PREREG_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'))
    paths=list(dict.fromkeys(paths))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        dest=args.out/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(args.out/'sources.json',sources)
    prepared={label:prepare(label,parent,meta) for label in
              ('natural','kept','erase','kept_no_capture','erase_no_capture')}
    checks={}
    for a,b in [('kept','erase'),('kept_no_capture','erase_no_capture')]:
        checks[a+'_'+b]=assert_matched_trace_states(prepared[a][0],prepared[b][0])
    for label,(m,observer,setup) in prepared.items():
        if label.endswith('_no_capture'):
            setup['parameter_intervention']=parameter_quench(m,capture_coupling=0.)
    atomic_json(args.out/'matched_state_checks.json',checks)
    arms={label:run(label,m,observer,setup,meta,fresh,replay,args.out,sources)
          for label,(m,observer,setup) in prepared.items()}
    assert digest(args.archive)==parent_hash
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,source_unchanged=True,
        code_sha256=hashes,matched_state_checks=checks,arms=arms,optimizer_updates=0,
        scope='Current full-generator matched native-memory paths with actual fresh/revisited OWT; not a stable single-path causal effect size or learning-capability conclusion'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_full_step_20261022/symmetric_dt2_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
