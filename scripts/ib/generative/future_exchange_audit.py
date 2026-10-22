"""Continuous real-stream consequences of the existing physical memory path."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.exchange_observation import ExchangeObservation
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.compare_support_transport import digest,same


def focus_ports(model):
    return [port for port,node in enumerate(model.port_nodes)
            if model.port_installed[port] and model.port_trace[port] and model.depleted[node]]


def conditional_pair(source,seed=None):
    ports=focus_ports(source);models={};jumps={}
    for label,kind in (('retained','active'),('reconfigured','write')):
        model=BoundaryIdentityMobileExchange.from_state_dict(source.state_dict())
        if seed is not None:
            for index,name in enumerate(('rng_reaction','rng_motion','rng_field')):
                setattr(model,name,np.random.default_rng(np.random.SeedSequence([seed,index])))
        events=[];model.event_observer=events.append
        for port in ports:
            node=int(model.port_nodes[port])
            event=MaterialEvent(kind,node,node,port,-1,interface=port)
            assert event in model.events()
            model.apply_event(event)
        model.event_observer=None
        models[label]=model;jumps[label]=events
    left,right=models['retained'],models['reconfigured']
    for name in source._arrays:
        if name!='port_trace':np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for name in ('rng_reaction','rng_motion','rng_field'):
        assert getattr(left,name).bit_generator.state==getattr(right,name).bit_generator.state
    return models,jumps,ports


def snapshot(path,model,session,observer,sources,extra=None):
    observer.validate(model)
    save_exchange(path,model,dict(session=copy.deepcopy(session),observer=observer.state_dict(),
        sources=sources,byte_ms=4.,dt_ms=2.,**(extra or {})))


def verify_prior(run,blocks):
    model,meta=load_exchange(run/'coupled_mid.npz')
    state=meta['session'];phase,cursor=state['phase'],state['next_phase_byte']
    observer=ExchangeObservation(model);model.event_observer=observer.observe
    expected={(row['phase'],row['phase_byte']):row for row in
              (json.loads(line) for line in (run/'coupled_scores.jsonl').read_text().splitlines())}
    for index in range(cursor,cursor+128):
        row=advance_byte(model,state,int(blocks[phase][index]),phase)
        row.update(phase=phase,phase_byte=index)
        assert row==expected[phase,index]
    final,final_meta=load_exchange(run/'coupled_complete.npz')
    same(model.state_dict(),final.state_dict());assert state==final_meta['session']
    observer.validate(model)
    return dict(verified_future_bytes=128,physical_state_three_rngs_session_and_predictions_exact=True,
                receipts_close=True)


def run_arm(label,model,session,blocks,out,sources,ports,expected_rows=None,expected_receipts=None):
    observer=ExchangeObservation(model,focus_ports=ports)
    snapshots=[];phases={};frames=[];cursor=session['next_phase_byte']
    assert session['phase']=='fresh_A' and cursor==896
    snapshot(out/(label+'_start.npz'),model,session,observer,sources,
             {'scope':'natural primary' if label=='native' else 'conditional native-transition control'})
    fork=fork_session=fork_observer=None;fork_verified=0
    model.event_observer=observer.observe
    prefix_rows_verified=0;prefix_receipts_verified=0
    with gzip.open(out/(label+'_receipts.jsonl.gz'),'wt',encoding='utf-8') as receipts, \
            (out/(label+'_scores.jsonl')).open('w',encoding='utf-8') as score_file:
        def receive(event):
            nonlocal prefix_receipts_verified
            observer.observe(event)
            encoded=json.dumps(event,separators=(',',':'))
            if expected_receipts is not None and prefix_receipts_verified<len(expected_receipts):
                assert json.loads(encoded)==expected_receipts[prefix_receipts_verified]
                prefix_receipts_verified+=1
            receipts.write(encoded+'\n')
        model.event_observer=receive
        for phase in ('fresh_A','fresh_B','revisit_A'):
            begin=cursor if phase=='fresh_A' else 0
            session['phase']=phase;session['next_phase_byte']=begin
            initial=observer.summary(model);work0,heat0=model.work_in,model.heat_out
            scores=[]
            for index in range(begin,len(blocks[phase])):
                row=advance_byte(model,session,int(blocks[phase][index]),phase)
                row.update(phase=phase,phase_byte=index)
                if expected_rows is not None and (phase,index) in expected_rows:
                    assert row==expected_rows[phase,index]
                    prefix_rows_verified+=1
                score_file.write(json.dumps(row,separators=(',',':'))+'\n');scores.append(row)
                if fork is not None:
                    other=advance_byte(fork,fork_session,int(blocks[phase][index]),phase)
                    other.update(phase=phase,phase_byte=index)
                    assert row==other
                    fork_verified+=1
                if (index+1)%32==0:
                    info=observer.summary(model)
                    info.update(phase=phase,next_phase_byte=index+1,
                        port_installed=model.port_installed.tolist(),port_trace=model.port_trace.tolist(),
                        port_support=model.port_bound.tolist(),charged=model.charged.tolist())
                    frames.append(info)
                    atomic_json(out/(label+'_frames.json'),frames)
                    print(json.dumps(dict(arm=label,phase=phase,next_phase_byte=index+1,time_ms=model.t,
                        original_carriers=info['initial_carriers_surviving'],target_carriers=info['focus_carriers_surviving'])),flush=True)
                if phase=='fresh_B' and index==127:
                    path=out/(label+'_fresh_B_mid.npz')
                    snapshot(path,model,session,observer,sources)
                    fork,meta=load_exchange(path);fork_session=copy.deepcopy(meta['session'])
                    fork_observer=ExchangeObservation(fork,state=meta['observer'])
                    fork.event_observer=fork_observer.observe
            info=observer.summary(model)
            if phase=='fresh_B':
                assert fork_verified==128
                same(model.state_dict(),fork.state_dict());assert session==fork_session
                assert observer.state_dict()==fork_observer.state_dict()
                fork=fork_session=fork_observer=None
            phase_summary=dict(n=len(scores),nll_nats_per_byte={key:float(np.mean([row['scores'][key] for row in scores])) for key in scores[0]['scores']},
                start_ms=initial['time_ms'],end_ms=model.t,source_work=model.work_in-work0,heat_out=model.heat_out-heat0,
                conversion_electrical_delivery=info['conversion_electrical_delivery']-initial['conversion_electrical_delivery'],
                drive_energy_change=info['drive_energy_change']-initial['drive_energy_change'],
                original_carriers_surviving=info['initial_carriers_surviving'],target_carriers_surviving=info['focus_carriers_surviving'],
                births=info['births']-initial['births'],deaths=info['deaths']-initial['deaths'])
            path=out/(label+'_'+phase+'_complete.npz')
            snapshot(path,model,session,observer,sources,{'phase_result':phase_summary})
            snapshots.append(dict(phase=phase,path=str(path),sha256=digest(path)))
            phases[phase]=phase_summary
            receipts.flush();score_file.flush()
    atomic_json(out/(label+'_frames.json'),frames)
    result=dict(phases=phases,snapshots=snapshots,total=observer.summary(model),
        restored_future_bytes=128,full_state_session_observer_and_three_rngs_exact=True,
        maximum_final_balance_error=max(abs(float(v)) for v in model.balances().values()),
        natural_primary=label=='native',recovered_prefix_predictions_exact=prefix_rows_verified,
        recovered_prefix_receipts_exact=prefix_receipts_verified)
    if expected_rows is not None:assert prefix_rows_verified==len(expected_rows)
    if expected_receipts is not None:assert prefix_receipts_verified==len(expected_receipts)
    atomic_json(out/(label+'_summary.json'),result)
    return result


def preserve_completed_native(previous,out):
    # This is a completed natural trajectory. Copy it; do not expose it to
    # the stream twice merely because a separate control's metadata failed.
    for path in previous.glob('native*'):
        if path.is_file():shutil.copy2(path,out/path.name)
    shutil.copytree(previous/'source_snapshot',out/'native_original_source')
    model,meta=load_exchange(out/'native_revisit_A_complete.npz')
    observer=ExchangeObservation(model,state=meta['observer'])
    phases={};snapshots=[]
    for phase in ('fresh_A','fresh_B','revisit_A'):
        path=out/('native_'+phase+'_complete.npz')
        _,data=load_exchange(path);phases[phase]=data['phase_result']
        snapshots.append(dict(phase=phase,path=str(path),sha256=digest(path)))
    return dict(phases=phases,snapshots=snapshots,total=observer.summary(model),restored_future_bytes=128,
        full_state_session_observer_and_three_rngs_exact=True,natural_primary=True,
        maximum_final_balance_error=max(abs(float(v)) for v in model.balances().values()),
        preserved_from=str(previous),natural_experience_reexecuted=False,
        actual_source_snapshot='native_original_source')


def ensemble(source,metadata,blocks,out):
    rows=[];phase=metadata['session']['phase'];cursor=metadata['session']['next_phase_byte']
    for index in range(8):
        seed=20261022+index;models,jumps,ports=conditional_pair(source,seed)
        results={}
        for label,model in models.items():
            session=copy.deepcopy(metadata['session']);observer=ExchangeObservation(model,focus_ports=ports)
            model.event_observer=observer.observe;scores=[]
            for target in blocks[phase][cursor:cursor+64]:
                scores.append(advance_byte(model,session,int(target),phase)['scores']['material'])
            results[label]=dict(nll=float(np.mean(scores)),**observer.summary(model))
        keys=('nll','conversion_electrical_delivery','drive_energy_change','source_work',
              'initial_carriers_surviving','focus_carriers_surviving','live_contacts')
        difference={key:results['retained'][key]-results['reconfigured'][key] for key in keys}
        rows.append(dict(future_noise_seed=seed,bytes=64,source_start_ms=source.t,
                         results=results,retained_minus_reconfigured=difference))
        print(json.dumps(dict(ensemble_seed=seed,difference=difference)),flush=True)
    summary={}
    for key in rows[0]['retained_minus_reconfigured']:
        value=np.array([row['retained_minus_reconfigured'][key] for row in rows],float)
        mean=float(value.mean());sd=float(value.std(ddof=1));radius=float(student_t.ppf(.975,7)*sd/np.sqrt(8))
        summary[key]=dict(paired_mean=mean,sample_sd=sd,descriptive_t95_interval=[mean-radius,mean+radius])
    result=dict(rows=rows,summary=summary,scope='Eight prescribed future-noise pairs, same 64 real bytes; descriptive small-sample t intervals, not convergence or capacity evidence')
    atomic_json(out/'ensemble.json',result)
    return result


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source_hash=digest(args.archive);source,metadata=load_exchange(args.archive)
    data_path=args.run/'stream_blocks.npz'
    if digest(data_path)!=metadata['sources']['selected_stream_sha256']:raise ValueError('Selected stream changed')
    blocks=np.load(data_path,allow_pickle=False)
    prior=verify_prior(args.run,blocks)
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/mobile_exchange.py',
           ROOT/'information_boltzmann/cgb/exchange_observation.py',ROOT/'information_boltzmann/cgb/boundary_material.py',
           ROOT/'docs/information_boltzmann/CGB_FUTURE_EXCHANGE_PREREG_20261022.md']
    hashes={str(path.relative_to(ROOT)):digest(path) for path in paths}
    snapshot_dir=args.out/'source_snapshot';snapshot_dir.mkdir()
    for path in paths:(snapshot_dir/path.name).write_bytes(path.read_bytes())
    pair,jumps,ports=conditional_pair(source)
    models={'native':BoundaryIdentityMobileExchange.from_state_dict(source.state_dict()),**pair}
    arms={}
    if args.recover:
        arms['native']=preserve_completed_native(args.recover,args.out)
        models.pop('native')
    for label,model in models.items():
        session=copy.deepcopy(metadata['session'])
        expected_rows=expected_receipts=None
        if args.recover and label=='retained':
            model,restored=load_exchange(args.recover/'retained_start.npz')
            session=copy.deepcopy(restored['session'])
            previous=[json.loads(line) for line in (args.recover/'retained_scores.jsonl').read_text().splitlines()]
            expected_rows={(row['phase'],row['phase_byte']):row for row in previous}
            with gzip.open(args.recover/'retained_receipts.jsonl.gz','rt',encoding='utf-8') as handle:
                expected_receipts=[json.loads(line) for line in handle]
        elif label!='native':session['interventions'].append(dict(time_ms=model.t,conditional_native_jumps=jumps[label],scope='conditional path, selection controller not modeled'))
        arms[label]=run_arm(label,model,session,blocks,args.out,metadata['sources'],ports,expected_rows,expected_receipts)
    noise_result=ensemble(source,metadata,blocks,args.out)
    assert source_hash==digest(args.archive)
    assert hashes=={str(path.relative_to(ROOT)):digest(path) for path in paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source=str(args.archive),source_sha256=source_hash,source_unchanged=True,code_sha256=hashes,
        observer_equivalence=prior,focus_ports=ports,conditional_initial_jumps=jumps,arms=arms,
        ensemble_summary=noise_result['summary'],fresh_bytes_per_arm=384,revisit_bytes_per_arm=256,
        physical_ms_per_byte=4.,dt_ms=2.,optimizer_updates=0,
        recovery_from=str(args.recover) if args.recover else None,
        claim_scope='Actual native physical consequence path, conditional carrier retention and real prequential reference readout. No extra reward or modified physical learning rule; capability/convergence and complete predictive credit unproven.'))
    print(json.dumps(dict(completed=True,native_final_ms=arms['native']['total']['time_ms'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=Path('F:/fly_stream_overflow/cgb_boundary_contacts_20261022'))
    parser.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_boundary_contacts_20261022/coupled_complete.npz'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover',type=Path)
    main(parser.parse_args())
