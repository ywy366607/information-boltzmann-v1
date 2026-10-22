"""Paid material contact growth at the real streaming sensory boundary."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.learning_observation import native_learning_drift
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.compare_support_transport import digest,same


def project_to_identity(model):
    state=model.state_dict()
    for name in ('port_installed','port_bound','port_trace'):
        state['arrays'].pop(name)
    state.pop('boundary_reactions_enabled');state.pop('boundary_conduction_enabled')
    state.update(kind='identity_mobile_exchange',schema=2 if model.field_scheme=='symmetric_ou' else 1)
    if model.field_scheme=='sequential_ou':state.pop('field_scheme')
    return state


def verify_predecessors(run,symmetric_run):
    data=np.load(run/'stream_blocks.npz',allow_pickle=False)
    reports={}
    for label,archive,rows_path in (
        ('legacy',run/'native_fresh_A_mid.npz',run/'native_scores.jsonl'),
        ('symmetric',symmetric_run/'symmetric_ou_mid.npz',symmetric_run/'symmetric_ou_scores.jsonl')):
        model,meta=load_exchange(archive);session=meta['session']
        phase,cursor=session['phase'],session['next_phase_byte']
        rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
        expected={(row['phase'],row['phase_byte']):row for row in rows}
        for index in range(cursor,cursor+64):
            row=advance_byte(model,session,int(data[phase][index]),phase)
            row.update(phase=phase,phase_byte=index)
            assert row==expected[phase,index]
        result=dict(original_predictions_exact=64,clock_ms=model.t)
        if label=='symmetric':
            final,final_meta=load_exchange(symmetric_run/'symmetric_ou_complete.npz')
            same(model.state_dict(),final.state_dict())
            assert session==final_meta['session']
            result['full_final_state_session_and_three_rngs_exact']=True
        reports[label]=result
    return reports


def causal_probe(archive,blocks,out):
    model,meta=load_exchange(archive)
    session=meta['session'];phase=session['phase'];cursor=session['next_phase_byte']
    if cursor>=len(blocks[phase]):
        return dict(measured=False,reason='No next byte in the stored phase')
    results=[]
    source_state=model.state_dict()
    clones={name:BoundaryIdentityMobileExchange.from_state_dict(source_state) for name in ('full','sham','blocked')}
    clones['blocked'].boundary_conduction_enabled=False
    assert clones['blocked'].energy()==model.energy() and clones['blocked'].free_energy()==model.free_energy()
    for name,current in clones.items():
        state=copy.deepcopy(session)
        work0,heat0=current.work_in,current.heat_out
        row=advance_byte(current,state,int(blocks[phase][cursor]),phase)
        save_exchange(out/('probe_'+name+'.npz'),current,dict(session=state,sources=meta['sources'],
            probe_origin=str(archive),source_state_preserved=True,
            diagnostic_barrier='additional boundary conduction only' if name=='blocked' else 'none'))
        results.append(dict(arm=name,clock_ms=current.t,source_work=current.work_in-work0,
                            heat_out=current.heat_out-heat0,prearrival_scores=row['scores']))
    same(clones['full'].state_dict(),clones['sham'].state_dict())
    return dict(measured=True,source=str(archive),source_sha256=digest(archive),phase=phase,next_byte=cursor,
        start_ms=model.t,end_ms=clones['full'].t,installed_before=model.port_installed.tolist(),
        trace_before=model.port_trace.tolist(),support_before=model.port_bound.tolist(),
        instantaneous_barrier_energy_change=0.,source_memory_erased=False,sham_all_state_and_rng_exact=True,
        brain_charge_difference_norm=float(np.linalg.norm(clones['full'].charge-clones['blocked'].charge)),
        world_charge_difference_norm=float(np.linalg.norm(clones['full'].world_charge-clones['blocked'].world_charge)),
        arms=results,scope='one actual 4ms real byte with all material reactions active; additional-conduction barrier is a diagnostic controller')


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source_hash=digest(args.archive)
    before,metadata=load_exchange(args.archive)
    if digest(args.run/'stream_blocks.npz')!=metadata['sources']['selected_stream_sha256']:
        raise ValueError('Selected real experience changed')
    blocks=np.load(args.run/'stream_blocks.npz',allow_pickle=False)
    predecessor=verify_predecessors(args.run,args.symmetric_run)
    migrated=BoundaryIdentityMobileExchange.from_identity(before)
    same(project_to_identity(migrated),before.state_dict())
    assert migrated.energy()==before.energy() and migrated.free_energy()==before.free_energy()
    phase=metadata['session']['phase'];cursor=metadata['session']['next_phase_byte']
    block=blocks[phase][cursor:cursor+256]
    if len(block)!=256:raise ValueError('Need the declared 256 original future bytes')
    sources=dict(parent=str(args.archive),sha256=source_hash,inherited=metadata['sources'])
    source_paths=[Path(__file__),ROOT/'information_boltzmann/cgb/mobile_exchange.py',
        ROOT/'information_boltzmann/cgb/identity_material.py',ROOT/'information_boltzmann/cgb/boundary_material.py',
        ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',ROOT/'information_boltzmann/cgb/learning_observation.py',
        ROOT/'information_boltzmann/cgb/sensory_channel.py',
        ROOT/'docs/information_boltzmann/CGB_BOUNDARY_CONTACTS_PREREG_20261022.md']
    hashes={str(path.relative_to(ROOT)):digest(path) for path in source_paths}
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for path in source_paths:(snapshot/path.name).write_bytes(path.read_bytes())
    save_exchange(args.out/'before_boundary.npz',migrated,dict(session=metadata['session'],sources=metadata['sources'],
        migration=dict(new_empty_slots=len(migrated.port_nodes),material_created=0,energy_injected=0.,memory_erased=False)))
    initial=migrated.state_dict();arms={};occupied=[]
    for label in ('coupled','boundary_frozen'):
        model=BoundaryIdentityMobileExchange.from_state_dict(initial)
        session=copy.deepcopy(metadata['session'])
        if label=='boundary_frozen':model.boundary_reactions_enabled=False
        session['interventions'].append(dict(time_ms=model.t,boundary_reactions_enabled=model.boundary_reactions_enabled,
            scope='ideal reaction-barrier diagnostic' if label=='boundary_frozen' else 'new paid contacts available',
            material_created=0,instantaneous_work=0.,memory_erased=False))
        baseline=type(before).from_state_dict(before.state_dict()) if label=='boundary_frozen' else None
        baseline_session=copy.deepcopy(metadata['session'])
        events=[]
        def observer(event):
            if event.get('interface',-1)>=0:events.append(event)
        model.event_observer=observer
        scores=[];frames=[];counts=dict(model.reaction_counts)
        errors={key:0. for key in model.balances()};work0,heat0=model.work_in,model.heat_out
        fork=fork_session=None;first_occupied=None;started=time.perf_counter()
        for index,target in enumerate(block):
            row=advance_byte(model,session,int(target),phase)
            row.update(phase=phase,phase_byte=cursor+index)
            scores.append(row)
            if fork is not None:
                other=advance_byte(fork,fork_session,int(target),phase)
                other.update(phase=phase,phase_byte=cursor+index)
                assert row==other
            if baseline is not None:
                other=advance_byte(baseline,baseline_session,int(target),phase)
                other.update(phase=phase,phase_byte=cursor+index)
                assert row==other
            for key,value in model.balances().items():errors[key]=max(errors[key],abs(float(value)))
            if label=='coupled' and first_occupied is None and np.any(model.port_installed):
                first_occupied=args.out/'coupled_first_occupied.npz'
                save_exchange(first_occupied,model,dict(session=copy.deepcopy(session),sources=metadata['sources'],
                    scope='first actual byte boundary with paid contact; complete clock and RNG'))
                occupied.append(first_occupied)
            if (index+1)%32==0:
                frame=dict(time_ms=model.t,next_phase_byte=session['next_phase_byte'],
                    internal_bridges=len(model.pre),port_installed=model.port_installed.tolist(),
                    port_bound=model.port_bound.tolist(),port_trace=model.port_trace.tolist(),
                    effective_port_conductance=model.effective_port_conductance().tolist(),
                    material=model.material.tolist(),charged=model.charged.tolist(),
                    coordinates=model.coordinates.tolist(),pre=model.pre.tolist(),post=model.post.tolist(),
                    learning=native_learning_drift(model))
                frames.append(frame)
                print(json.dumps(dict(arm=label,bytes=index+1,time_ms=model.t,ports=model.port_installed.tolist())),flush=True)
            if index==127:
                middle=args.out/(label+'_mid.npz')
                save_exchange(middle,model,dict(session=copy.deepcopy(session),sources=metadata['sources']))
                fork,restored=load_exchange(middle);fork_session=copy.deepcopy(restored['session'])
                if label=='coupled' and np.any(model.port_installed):occupied.append(middle)
        same(model.state_dict(),fork.state_dict());assert session==fork_session
        if baseline is not None:same(project_to_identity(model),baseline.state_dict())
        assert max(errors.values())<1e-8
        path=args.out/(label+'_complete.npz')
        save_exchange(path,model,dict(session=session,sources=metadata['sources'],restored_future_bytes_exact=128))
        if label=='coupled' and np.any(model.port_installed):occupied.append(path)
        atomic_json(args.out/(label+'_frames.json'),frames)
        for name,rows in (('scores',scores),('events',events)):
            with (args.out/(label+'_'+name+'.jsonl')).open('w',encoding='utf-8') as output:
                for row in rows:output.write(json.dumps(row,separators=(',',':'))+'\n')
        arms[label]=dict(bytes=256,start_ms=before.t,end_ms=model.t,
            seconds_including_resume_and_control=time.perf_counter()-started,
            nll_nats_per_byte={key:float(np.mean([row['scores'][key] for row in scores])) for key in scores[0]['scores']},
            reaction_events={key:model.reaction_counts.get(key,0)-counts.get(key,0) for key in set(model.reaction_counts)|set(counts)},
            actual_boundary_events=len(events),final_ports=model.port_installed.tolist(),
            final_trace=model.port_trace.tolist(),final_support=model.port_bound.tolist(),
            effective_port_conductance=model.effective_port_conductance().tolist(),
            source_work=model.work_in-work0,heat_out=model.heat_out-heat0,maximum_balance_errors=errors,
            extra_physical_array_bytes=model.port_installed.nbytes+model.port_bound.nbytes+model.port_trace.nbytes,
            restored_future_bytes=128,full_state_session_rng_and_scores_exact=True,
            inherited_identity_continuation_exact=baseline is not None,archive_sha256=digest(path))
    probe=causal_probe(occupied[-1],blocks,args.out) if occupied else dict(measured=False,reason='No paid boundary contact installed; not forced')
    assert digest(args.archive)==source_hash
    assert hashes=={str(path.relative_to(ROOT)):digest(path) for path in source_paths}
    report=dict(passed=True,baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=sources,code_sha256=hashes,predecessor_equivalence=predecessor,
        migration=dict(new_empty_slots=len(migrated.port_nodes),material_created=0,energy_injected=0.,all_old_state_preserved=True),
        arms=arms,acute_delivery=probe,optimizer_updates=0,byte_ms=4.,dt_ms=2.,
        scope='Shared native reactions now remodel finite parallel sensory contact; predictive learning/complete body/world closure not claimed')
    atomic_json(args.out/'report.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('sources','code_sha256','arms')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=Path('F:/fly_stream_overflow/cgb_native_learning_20261022'))
    parser.add_argument('--symmetric-run',type=Path,default=Path('F:/fly_stream_overflow/cgb_symmetric_field_20261022'))
    parser.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_symmetric_field_20261022/symmetric_ou_complete.npz'))
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
