"""Real prequential byte stream with native material learning kept continuous.

Diagnostic readouts never write a gradient, reward, or new physical rule.
Every likelihood is formed before the target enters the material/world.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.learning_observation import port_bit_logits,byte_nll,native_learning_drift,parameter_quench
from scripts.ib.generative.compare_support_transport import digest,same


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def byte_segment(array,length,offset=4):
    # No special token is dropped from an already-started experience. Select
    # one boundary-free real document fragment before the run begins.
    panel=np.asarray(array[:min(len(array),1_000_000)])
    bad=np.flatnonzero((panel<offset)|(panel>=offset+256))
    stops=np.r_[-1,bad,len(panel)]
    for lo,hi in zip(stops[:-1]+1,stops[1:]):
        if hi-lo>=length:
            return int(lo),np.asarray(panel[lo:lo+length]-offset,dtype=np.uint8)
    raise ValueError('No sufficiently long boundary-free document fragment in the declared search range')


def prepare_data(folder):
    manifest_path=folder/'manifest.json';tokenizer_path=folder/'tokenizer.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer=json.loads(tokenizer_path.read_text(encoding='utf-8'))
    if tokenizer.get('type')!='utf8_bytes' or tokenizer.get('offset')!=4 or manifest.get('vocab_size')!=260:
        raise ValueError('The recorded byte tokenizer is required, not a BPE stream')
    hashes={}
    for split in ('train','validation'):
        path=folder/(split+'.npy')
        hashes[path.name]=digest(path)
        if hashes[path.name]!=manifest['files'][path.name]:raise ValueError('Corpus differs from retained manifest')
    if digest(tokenizer_path)!=manifest['files']['tokenizer.json']:
        raise ValueError('Tokenizer changed')
    train=np.load(folder/'train.npy',mmap_mode='r')
    validation=np.load(folder/'validation.npy',mmap_mode='r')
    a_start,a=byte_segment(train,1280)
    b_start,b=byte_segment(validation,256)
    blocks={'warm':a[:256],'fresh_A':a[256:1280],'fresh_B':b,'revisit_A':a[:256]}
    locations=dict(warm=dict(split='train',offset=a_start),fresh_A=dict(split='train',offset=a_start+256),
        fresh_B=dict(split='validation',offset=b_start),revisit_A=dict(split='train',offset=a_start,replay=True))
    return blocks,dict(source=manifest['source'],revision=manifest['revision'],split_rule=manifest['split_rule'],
        tokenizer=tokenizer,manifest_sha256=digest(manifest_path),files_sha256=hashes,
        selected_locations=locations,selected_counts={k:len(v) for k,v in blocks.items()},
        selected_sha256={k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in blocks.items()})


def new_session(model,history):
    counts=np.bincount(np.frombuffer(history,dtype=np.uint8),minlength=256)
    bits=((np.frombuffer(history,dtype=np.uint8)[:,None]>>np.arange(8))&1).sum(0)
    return dict(byte_counts=counts.tolist(),bit_ones=bits.tolist(),history_count=len(history),
        fixed_frequency_frozen=False,source_only_voltage=(model.world_charge/model.world_capacitance).tolist(),
        scored_bytes=0,phase=None,next_phase_byte=0,phase_metrics={},interventions=[])


def external_step_count(dt_ms):
    """Resolve the fixed 4ms input interval without shortening its clock."""
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError('Positive finite integration step required')
    count = int(round(4./dt_ms))
    if count < 1 or count*dt_ms != 4.:
        raise ValueError('Integration step must exactly divide the declared 4ms interval')
    return count


def advance_byte(model,session,target,phase,warm=False,*,dt_ms=2.):
    steps = external_step_count(dt_ms)
    # Freeze all forecast objects before any target-dependent physical step.
    logits=port_bit_logits(model)
    source_logits=port_bit_logits(model,voltage=np.asarray(session['source_only_voltage']))
    n=session['history_count'];counts=np.asarray(session['byte_counts']);ones=np.asarray(session['bit_ones'])
    probability=(counts+.5)/(n+128.)
    bit_p=(ones+.5)/(n+1.)
    bit_logits=np.log(bit_p)-np.log1p(-bit_p)
    scores=dict(material=byte_nll(logits,target),fixed_byte=-float(np.log(probability[target])),
                fixed_bits=byte_nll(bit_logits,target),source_filter=byte_nll(source_logits,target))
    source=np.array([3. if (int(target)>>bit)&1 else -3. for bit in range(8)])
    before=model.t
    for _ in range(steps):model.step(dt_ms,source)
    assert model.t==before+4.
    # Exact source-only mean response, with no coupling to a brain and no
    # fitted coefficient. It is a reference filter, not a living controller.
    v=np.asarray(session['source_only_voltage'])
    v=source+(v-source)*np.exp(-4*model.source_conductance/model.world_capacitance)
    session['source_only_voltage']=v.tolist()
    if warm:
        counts[int(target)]+=1;ones+=((int(target)>>np.arange(8))&1)
        session['byte_counts']=counts.tolist();session['bit_ones']=ones.tolist();session['history_count']=n+1
    elif not session['fixed_frequency_frozen']:
        raise ValueError('Freeze reference frequency before comparing active branches')
    row=session['phase_metrics'].setdefault(phase,dict(n=0,sums={k:0. for k in scores}))
    row['n']+=1
    for k,value in scores.items():row['sums'][k]+=value
    session['scored_bytes']+=1;session['next_phase_byte']+=1
    return dict(before_ms=before,after_ms=model.t,target=int(target),scores=scores,
                pre_arrival_logits=logits.tolist(),source='replay' if phase=='revisit_A' else 'fresh')


def run_phase(model,session,block,phase,out,label,source_meta,verify_resume=False,warm=False):
    session['phase']=phase;session['next_phase_byte']=0
    metrics_path=out/(label+'_windows.jsonl')
    samples_path=out/(label+'_scores.jsonl')
    window={k:0. for k in ('material','fixed_byte','fixed_bits','source_filter')}
    started=time.perf_counter();errors={k:0. for k in model.balances()}
    initial_t=model.t;initial_counts=dict(model.reaction_counts);initial_work=model.work_in;initial_heat=model.heat_out
    initial_probe=native_learning_drift(model)
    fork=fork_session=None;fork_bytes=0;resume_verified=False
    with metrics_path.open('a',encoding='utf-8') as metrics,samples_path.open('a',encoding='utf-8') as samples:
        for index,target in enumerate(block):
            row=advance_byte(model,session,int(target),phase,warm)
            if fork is not None:
                other=advance_byte(fork,fork_session,int(target),phase,warm)
                assert row==other
                fork_bytes+=1
                if fork_bytes==16:
                    same(model.state_dict(),fork.state_dict());assert session==fork_session
                    fork=fork_session=None;resume_verified=True
            for k,v in row['scores'].items():window[k]+=v
            row.update(phase=phase,phase_byte=index)
            samples.write(json.dumps(row,separators=(',',':'))+'\n')
            for k,v in model.balances().items():errors[k]=max(errors[k],abs(float(v)))
            if (index+1)%64==0:
                probe=native_learning_drift(model)
                record=dict(arm=label,phase=phase,next_phase_byte=index+1,time_ms=model.t,
                    n=64,nll_nats_per_byte={k:v/64 for k,v in window.items()},learning=probe,
                    bridges=len(model.pre),support=int(model.bound.sum()),trace=int(model.trace.sum()),
                    source_work_since_phase=model.work_in-initial_work,heat_since_phase=model.heat_out-initial_heat)
                metrics.write(json.dumps(record,separators=(',',':'))+'\n');metrics.flush();samples.flush()
                atomic_json(out/'progress.json',dict(status='running',**record))
                print(json.dumps({k:record[k] for k in ('arm','phase','next_phase_byte','time_ms','nll_nats_per_byte')}) ,flush=True)
                window={k:0. for k in window}
            if verify_resume and index+1==len(block)//2:
                path=out/(label+'_'+phase+'_mid.npz')
                save_exchange(path,model,dict(session=copy.deepcopy(session),sources=source_meta,
                    next_input_is_unseen=True,byte_ms=4.,dt_ms=2.))
                fork,meta=load_exchange(path);fork_session=copy.deepcopy(meta['session'])
        path=out/(label+'_'+phase+'_complete.npz')
        save_exchange(path,model,dict(session=copy.deepcopy(session),sources=source_meta,
            byte_ms=4.,dt_ms=2.,next_phase_requires_no_physical_reset=True))
        restored,meta=load_exchange(path);same(model.state_dict(),restored.state_dict());assert meta['session']==session
    assert max(errors.values())<1e-8
    score=session['phase_metrics'][phase]
    return dict(n=score['n'],nll_nats_per_byte={k:v/score['n'] for k,v in score['sums'].items()},
        physical_start_ms=initial_t,physical_end_ms=model.t,seconds=time.perf_counter()-started,
        native_reaction_events={k:model.reaction_counts.get(k,0)-initial_counts.get(k,0)
            for k in set(model.reaction_counts)|set(initial_counts)},
        maximum_balance_errors=errors,source_work=model.work_in-initial_work,heat_out=model.heat_out-initial_heat,
        initial_learning=initial_probe,final_learning=native_learning_drift(model),
        archive=str(path),sha256=digest(path),full_restore_loaded=True,
        restored_future_bytes_tested=16 if resume_verified else 0)


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source_hash=digest(args.archive);model,provenance=load_exchange(args.archive)
    history=bytes.fromhex(provenance['byte_stream_hex'])
    blocks,corpus=prepare_data(args.data)
    np.savez_compressed(args.out/'stream_blocks.npz',**blocks)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/learning_observation.py',
        ROOT/'information_boltzmann/cgb/mobile_exchange.py',ROOT/'information_boltzmann/cgb/identity_material.py',
        ROOT/'information_boltzmann/cgb/exchange_checkpoint.py',
        ROOT/'docs/information_boltzmann/CGB_NATIVE_LEARNING_PREREG_20261022.md']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:(snapshot/p.name).write_bytes(p.read_bytes())
    (args.out/'corpus_manifest_snapshot.json').write_bytes((args.data/'manifest.json').read_bytes())
    (args.out/'tokenizer_snapshot.json').write_bytes((args.data/'tokenizer.json').read_bytes())
    source_meta=dict(archive=str(args.archive),archive_sha256=source_hash,source_time_ms=model.t,
        retained_history_bytes=len(history),genome=model.program.fingerprint(),corpus=corpus,
        selected_stream_sha256=digest(args.out/'stream_blocks.npz'))
    atomic_json(args.out/'sources.json',source_meta)
    session=new_session(model,history)
    warm=run_phase(model,session,blocks['warm'],'warm',args.out,'shared',source_meta,warm=True)
    session['fixed_frequency_frozen']=True
    warm_state=model.state_dict();warm_session=copy.deepcopy(session)
    results={}
    for label in ('native','no_trace_capture','frozen_adaptation'):
        current=type(model).from_state_dict(warm_state);state=copy.deepcopy(warm_session)
        before_state=current.state_dict()
        if label=='no_trace_capture': intervention=parameter_quench(current,capture_coupling=0.)
        elif label=='frozen_adaptation':
            intervention=parameter_quench(current,mobility=0.,assembly_rate=0.,binding_rate=0.,
                transport_rate=0.,release_rate=0.,write_rate=0.,recovery_rate=0.)
        else:
            intervention=parameter_quench(current,capture_coupling=current.params.capture_coupling)
            same(before_state,current.state_dict())
        state['interventions'].append(intervention)
        phases={}
        for phase in ('fresh_A','fresh_B','revisit_A'):
            phases[phase]=run_phase(current,state,blocks[phase],phase,args.out,label,source_meta,
                verify_resume=label=='native' and phase=='fresh_A')
        results[label]=dict(intervention=intervention,phases=phases,params=asdict(current.params))
    assert digest(args.archive)==source_hash
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    report=dict(date='2026-10-22',baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=source_meta,code_sha256=hashes,shared_warm=warm,arms=results,
        byte_ms=4.,dt_ms=2.,primary_units='nats per UTF8 byte, prediction before arrival',
        optimizer_updates=0,learning='actual native reactions; no trained readout head or loss-gradient update',
        scored_fresh_bytes_per_arm=1280,revisit_bytes_per_arm=256,shared_warm_bytes=256,
        fixed_reference_exposure_bytes=len(history)+256,
        readout='declared canonical two-voltage observation model, not the true distribution of arbitrary text',
        convergence='budget-limited first interface; no architecture capacity/convergence conclusion',
        signed_timing_or_predictive_credit='native signed reaction drift and causal cut observed; full E5/utility requires its own evidence',
        main_source_archive_unchanged=True,passed=True)
    atomic_json(args.out/'report.json',report)
    atomic_json(args.out/'progress.json',dict(status='completed',report=str(args.out/'report.json')))
    print(json.dumps(dict(status='completed',report=str(args.out/'report.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_identity_cached_20261022/coupled_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    main(p.parse_args())
