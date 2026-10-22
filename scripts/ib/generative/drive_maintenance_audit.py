"""Bounded driven/constant/disconnected comparisons of one complete material."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.energy_observation import material_energy_components
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.learning_observation import native_learning_drift
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.source_protocol import SourceProtocol


def snapshot(path, model, session, observer, protocol, sources, reference):
    observer.validate(model);protocol.validate(model)
    assert model.t == reference['start_ms']+4*protocol.intervals
    save_exchange(path, model, dict(session=copy.deepcopy(session),
        growth_observer=copy.deepcopy(observer.state),carrier_observer=copy.deepcopy(observer.carriers),
        source_protocol=protocol.state_dict(),sources=sources,stage_reference=reference,
        dt_ms=2.,external_interval_ms=4.))


def observations(model, observer, protocol, session, reference):
    observer.validate(model)
    components = material_energy_components(model)
    surviving = sum(uid < reference['carrier_next_id'] for uid in observer.carriers['alive'].values())
    old_positions = set(reference['carrier_keys']) & set(observer.carriers['alive'])
    rebuilt = sum(observer.carriers['alive'][key] >= reference['carrier_next_id'] for key in old_positions)
    return dict(time_ms=model.t,external_intervals=protocol.intervals,
        actual_new_text_exposures=protocol.text_exposures,actual_scored_bytes=session['scored_bytes'],
        nodes=model.n,free_material=int(model.material.sum()),internal_bridges=len(model.pre),
        boundary_contacts=int(model.port_installed.sum()),charged=int(model.charged.sum()),
        free_support=int(model.free.sum()),carried_support=int(model.carried.sum()),
        bound_support=int(model.bound.sum()+model.port_bound.sum()),
        trace=int(model.trace.sum()+model.port_trace.sum()),
        domain_births=model.domain_births-reference['domain_births'],
        domain_deaths=model.domain_deaths-reference['domain_deaths'],
        source_cohort=reference['carrier_count'],source_carriers_surviving=surviving,
        old_positions_occupied=len(old_positions),rebuilt_old_positions=rebuilt,
        carrier_births=observer.carriers['next_id']-reference['carrier_next_id'],
        carrier_deaths=observer.carriers['deaths']-reference['carrier_deaths'],
        energy=components,energy_change=components['total']-reference['energy']['total'],
        component_changes={key:value-reference['energy']['terms'][key] for key,value in components['terms'].items()},
        source_work=model.work_in-reference['work'],heat_out=model.heat_out-reference['heat'],
        source_charge=model.source_charge-reference['source_charge'],
        maximum_balance_error=max(abs(float(value)) for value in model.balances().values()),
        generator=native_learning_drift(model))


def run_arm(label, parent, metadata, block, held_voltage, out, sources):
    model = fork_material(parent)
    same(model.state_dict(), parent.state_dict())
    session = copy.deepcopy(metadata['session'])
    if label == 'text_drive':
        session.update(phase='fresh_E', next_phase_byte=0)
    protocol = SourceProtocol(label, held_voltage if label == 'held_source' else None)
    observer = CarrierObservation(model,metadata['growth_observer'],metadata['carrier_observer'])
    reference = dict(start_ms=model.t,work=model.work_in,heat=model.heat_out,
        source_charge=model.source_charge,energy=material_energy_components(model),
        domain_births=model.domain_births,domain_deaths=model.domain_deaths,
        carrier_count=len(observer.carriers['alive']),carrier_next_id=observer.carriers['next_id'],
        carrier_keys=sorted(observer.carriers['alive']),carrier_deaths=observer.carriers['deaths'],
        original_reactions=dict(model.reaction_counts),text_scored_bytes=session['scored_bytes'])
    snapshot(out/(label+'_start.npz'),model,session,observer,protocol,sources,reference)
    fork=fork_session=fork_observer=fork_protocol=None
    rows=[];windows=[];maximum_error=0.
    with gzip.open(out/(label+'_events.jsonl.gz'),'wt',encoding='utf-8') as events, \
            gzip.open(out/(label+'_domains.jsonl.gz'),'wt',encoding='utf-8') as domains, \
            (out/(label+'_intervals.jsonl')).open('w',encoding='utf-8') as intervals:
        def native(row):
            observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):
            observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        model.event_observer,model.domain_observer=native,domain
        for index in range(len(block)):
            target=int(block[index]) if label=='text_drive' else None
            row=protocol.advance(model,session,target)
            if fork is not None:
                assert row==fork_protocol.advance(fork,fork_session,target)
            if label=='text_drive':row['corpus_train_byte']=1920+index
            rows.append(row);intervals.write(json.dumps(row,separators=(',',':'))+'\n')
            observer.validate(model)
            maximum_error=max(maximum_error,max(abs(float(value)) for value in model.balances().values()))
            if (index+1)%32==0:
                info=observations(model,observer,protocol,session,reference)
                recent=[row['scores'] for row in rows[-32:] if row['scores'] is not None]
                info['nll_nats_per_byte']=({key:float(np.mean([item[key] for item in recent])) for key in recent[0]}
                                             if recent else None)
                windows.append(info);atomic_json(out/(label+'_windows.json'),windows)
                status=dict(arm=label,interval=index+1,time_ms=model.t,nodes=model.n,
                    new_text=protocol.text_exposures,source_work=info['source_work'],
                    energy_change=info['energy_change'],maximum_error=maximum_error)
                atomic_json(out/'progress.json',dict(status='running',**status))
                print(json.dumps(status),flush=True)
                events.flush();domains.flush();intervals.flush()
            if index+1==len(block)//2:
                path=out/(label+'_mid.npz')
                snapshot(path,model,session,observer,protocol,sources,reference)
                fork,meta=load_exchange(path);fork_session=copy.deepcopy(meta['session'])
                fork_observer=CarrierObservation(fork,meta['growth_observer'],meta['carrier_observer'])
                fork_protocol=SourceProtocol(**meta['source_protocol'])
                fork.event_observer,fork.domain_observer=fork_observer.native,fork_observer.domain
        same(model.state_dict(),fork.state_dict())
        assert session==fork_session and observer.state==fork_observer.state and observer.carriers==fork_observer.carriers
        assert protocol.state_dict()==fork_protocol.state_dict()
        assert maximum_error<1e-8
        if label=='source_disconnected':
            assert model.work_in==reference['work'] and model.source_charge==reference['source_charge']
        if label!='text_drive':
            assert session['scored_bytes']==reference['text_scored_bytes'] and protocol.text_exposures==0
        archive=out/(label+'_complete.npz')
        snapshot(archive,model,session,observer,protocol,sources,reference)
    info=observations(model,observer,protocol,session,reference)
    scored=[row['scores'] for row in rows if row['scores'] is not None]
    info.update(archive=str(archive),sha256=digest(archive),
        nll_nats_per_byte={key:float(np.mean([item[key] for item in scored])) for key in scored[0]} if scored else None,
        original_reactions={key:model.reaction_counts.get(key,0)-reference['original_reactions'].get(key,0)
                            for key in set(model.reaction_counts)|set(reference['original_reactions'])},
        restored_future_intervals=len(block)//2,all_records_physics_rngs_session_protocol_and_observers_exact=True,
        primary=label=='text_drive',next_text_cursor=1920+protocol.text_exposures,
        maximum_error=maximum_error)
    atomic_json(out/(label+'_summary.json'),info)
    return info


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    parent,metadata=load_exchange(args.archive);parent_hash=digest(args.archive)
    inherited=metadata['sources']
    while 'corpus' not in inherited:inherited=inherited['original_sources']
    corpus={key:inherited['corpus'][key] for key in ('source','revision','split_rule','tokenizer',
                                                  'manifest_sha256','files_sha256')}
    data=args.data/'train.npy'
    if digest(data)!=corpus['files_sha256']['train.npy']:raise ValueError('Corpus changed')
    array=np.load(data,mmap_mode='r')
    raw=np.asarray(array[1920:2432])
    if len(raw)!=512 or np.any((raw<4)|(raw>=260)):raise ValueError('Declare a document boundary before continuation')
    block=np.asarray(raw-4,np.uint8)
    np.save(args.out/'fresh_train_1920_2432.npy',block,allow_pickle=False)
    last=int(array[1919])-4
    if not 0<=last<256:raise ValueError('Last actual input was not a byte')
    held=[3. if (last>>bit)&1 else -3. for bit in range(8)]
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        text_selection=dict(split='train',start=1920,end=2432,token_offset=4,sha256=__import__('hashlib').sha256(block.tobytes()).hexdigest()),
        held_source_last_actual_byte=last,held_source_voltage=held,
        protocol='512 external intervals of 4ms, original dt2ms; only text_drive exposes new text')
    paths=[Path(__file__),ROOT/'scripts/ib/generative/source_protocol.py',
           ROOT/'information_boltzmann/cgb/energy_observation.py',
           ROOT/'information_boltzmann/cgb/growing_material.py',
           ROOT/'scripts/ib/generative/domain_rate_audit.py',
           ROOT/'docs/information_boltzmann/CGB_DRIVE_MAINTENANCE_PREREG_20261022.md']
    hashes={str(path.relative_to(ROOT)):digest(path) for path in paths}
    folder=args.out/'source_snapshot';folder.mkdir()
    for path in paths:shutil.copy2(path,folder/path.name)
    atomic_json(args.out/'sources.json',sources)
    arms={}
    for label in ('text_drive','held_source','source_disconnected'):
        arms[label]=run_arm(label,parent,metadata,block,held,args.out,sources)
    assert digest(args.archive)==parent_hash
    assert hashes=={str(path.relative_to(ROOT)):digest(path) for path in paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),sources=sources,source_unchanged=True,code_sha256=hashes,
        arms=arms,optimizer_updates=0,reference_temperature=1.,
        scope='Actual bounded source-driving/stock-relaxation comparison. Constant and disconnected controls have zero new text exposures; no forced decay or capability/convergence conclusion.'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_rate_r1_20261022/marginal_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
