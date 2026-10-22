"""Frozen likelihood instruments on one continuously learning real medium."""
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
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.readout_observation import before_arrival_features,fit_readout,FrozenBitReadout
from information_boltzmann.cgb.learning_observation import byte_nll
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.trace_fate_audit import cohort,frame
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json

FAMILIES=dict(brain_diagonal=('brain',True),source_diagonal=('source',True),
    brain_affine=('brain',False),source_affine=('source',False),previous_affine=('previous',False),
    brain_source=('brain_source',False),source_previous=('source_previous',False),brain_current=('brain_current',False))


def snapshot(path,m,session,observer,reference,sources,index,last_byte,readouts,frequencies,aggregates):
    observer.validate(m)
    assert m.t==reference['start_ms']+4*index
    save_exchange(path,m,dict(session=copy.deepcopy(session),growth_observer=copy.deepcopy(observer.state),
        carrier_observer=copy.deepcopy(observer.carriers),stage_reference=reference,sources=sources,
        dt_ms=2.,external_interval_ms=4.,last_actual_byte=last_byte,
        frozen_external_readouts={name:value.state_dict() for name,value in readouts.items()},
        matched_training_frequencies=frequencies,observed_aggregates=copy.deepcopy(aggregates),
        experience_protocol=dict(intervals=index,actual_text_exposures=index,replay_exposures=0,
            fresh_train_exposures=min(index,1024),fresh_validation_exposures=max(0,index-1024),
            next_train_byte=2816+min(index,1024),next_validation_byte=256+max(0,index-1024))))


def observe(features,readouts,frequencies,target):
    logits={name:value.logits(features[FAMILIES[name][0]]) for name,value in readouts.items()}
    scores={name:byte_nll(value,target) for name,value in logits.items()}
    scores['matched_train_byte_frequency']=-float(np.log(frequencies['byte'][target]))
    scores['matched_train_bit_frequency']=byte_nll(np.asarray(frequencies['bit_logits']),target)
    return logits,scores


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    m,meta=load_exchange(args.archive);parent_hash=digest(args.archive)
    assert parent_hash=='8e359c1da68287d28869b67c59fbca99f0887c1da256d2becb773197d0805029'
    assert m.t==14080. and m.n==52 and m.material_scheme=='symmetric'
    corpus=meta['sources']['corpus']
    blocks={}
    for split,start,end in [('train',2816,3840),('validation',256,1280)]:
        path=args.data/(split+'.npy');assert digest(path)==corpus['files_sha256'][path.name]
        raw=np.asarray(np.load(path,mmap_mode='r')[start:end])
        if len(raw)!=end-start or np.any((raw<4)|(raw>=260)):
            raise ValueError('Declare actual document boundaries before proceeding')
        blocks[split]=np.asarray(raw-4,np.uint8)
        np.save(args.out/(split+'_bytes.npy'),blocks[split],allow_pickle=False)
    previous=[json.loads(x) for x in (args.archive.parent/'natural_intervals.jsonl').read_text().splitlines()]
    last_byte=int(previous[-1]['target'])
    sources=dict(parent_archive=str(args.archive),parent_sha256=parent_hash,corpus=corpus,
        train_selection=dict(split='train',start=2816,end=3840,token_offset=4),
        validation_selection=dict(split='validation',start=256,end=1280,token_offset=4),
        initial_last_actual_byte=last_byte,
        protocol='2048 new real bytes, 4ms each, original native learning; external probes fit train only and frozen before validation')
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/readout_observation.py',
        ROOT/'scripts/ib/generative/native_learning_stream.py',ROOT/'scripts/ib/generative/trace_fate_audit.py',
        ROOT/'scripts/ib/generative/domain_rate_audit.py',
        ROOT/'docs/information_boltzmann/CGB_PREDICTION_OBSERVATION_DERIVATION_20261022.md']
    paths+=sorted((ROOT/'information_boltzmann/cgb').glob('*.py'));paths=list(dict.fromkeys(paths))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        dest=args.out/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_json(args.out/'sources.json',sources)
    observer=CarrierObservation(m,meta['growth_observer'],meta['carrier_observer'])
    session=copy.deepcopy(meta['session'])
    reference=dict(start_ms=m.t,work=m.work_in,heat=m.heat_out,energy=material_energy_components(m),
        cohort=cohort(m,observer),carrier_next_id=observer.carriers['next_id'],carrier_deaths=observer.carriers['deaths'],
        births=m.domain_births,deaths=m.domain_deaths,original_reactions=dict(m.reaction_counts))
    readouts={};frequencies={};aggregates={};windows=[];rows=[];training=[];maximum_error=0.
    snapshot(args.out/'natural_start.npz',m,session,observer,reference,sources,0,last_byte,readouts,frequencies,aggregates)
    sham=fork_material(m);sham_session=copy.deepcopy(session)
    sham_observer=CarrierObservation(sham,observer.state,observer.carriers)
    sham.event_observer,sham.domain_observer=sham_observer.native,sham_observer.domain
    restore=None
    with gzip.open(args.out/'natural_events.jsonl.gz','wt',encoding='utf-8') as events, \
         gzip.open(args.out/'natural_domains.jsonl.gz','wt',encoding='utf-8') as domains, \
         (args.out/'natural_intervals.jsonl').open('w',encoding='utf-8') as records:
        def native(row):
            observer.native(row);events.write(json.dumps(row,separators=(',',':'))+'\n')
        def domain(row):
            observer.domain(row);domains.write(json.dumps(row,separators=(',',':'))+'\n')
        m.event_observer,m.domain_observer=native,domain
        for index,target in enumerate(np.r_[blocks['train'],blocks['validation']]):
            phase='fresh_G' if index<1024 else 'fresh_validation_observation'
            if index in (0,1024):session.update(phase=phase,next_phase_byte=0)
            features=before_arrival_features(m,session,last_byte)
            predictions,scores=observe(features,readouts,frequencies,int(target)) if readouts else ({},None)
            if restore is not None:
                restored_features=before_arrival_features(restore,restored_session,restored_last_byte)
                for key in features:np.testing.assert_array_equal(features[key],restored_features[key])
                other_predictions,other_scores=observe(restored_features,restored_readouts,frequencies,int(target))
                for key in predictions:np.testing.assert_array_equal(predictions[key],other_predictions[key])
                assert scores==other_scores
            row=advance_byte(m,session,int(target),phase)
            if index<128:
                if index==0:sham_session.update(phase=phase,next_phase_byte=0)
                assert row==advance_byte(sham,sham_session,int(target),phase)
            if restore is not None:
                assert row==advance_byte(restore,restored_session,int(target),phase)
                restored_last_byte=int(target)
            if index<1024:training.append(features)
            sums=aggregates.setdefault(phase,dict(n=0,raw={k:0. for k in row['scores']},observed={}))
            sums['n']+=1
            for k,v in row['scores'].items():sums['raw'][k]+=v
            if scores:
                for k,v in scores.items():sums['observed'][k]=sums['observed'].get(k,0.)+v
            row.update(phase=phase,experience_interval=index+1,pre_arrival_features={k:v.tolist() for k,v in features.items()},
                external_frozen_observer_logits={k:v.tolist() for k,v in predictions.items()},
                external_observation_scores=scores,corpus_split='train' if index<1024 else 'validation',
                corpus_token_index=2816+index if index<1024 else 256+index-1024)
            rows.append(row);records.write(json.dumps(row,separators=(',',':'))+'\n')
            last_byte=int(target);observer.validate(m)
            maximum_error=max(maximum_error,max(abs(float(v)) for v in m.balances().values()))
            if (index+1)%64==0:
                info=frame(m,observer,reference,index+1,phase)
                info['nll_nats_per_byte']={k:float(np.mean([x['scores'][k] for x in rows[-64:]])) for k in row['scores']}
                info['external_observer_nll']=({k:float(np.mean([x['external_observation_scores'][k] for x in rows[-64:]])) for k in scores} if scores else None)
                windows.append(info);atomic_json(args.out/'natural_windows.json',windows)
                status=dict(interval=index+1,phase=phase,time_ms=m.t,nodes=m.n,
                            maximum_error=maximum_error,elapsed_wall_s=time.perf_counter()-started)
                atomic_json(args.out/'progress.json',dict(status='running',**status));print(json.dumps(status),flush=True)
                events.flush();domains.flush();records.flush()
            if index+1==128:
                same(m.state_dict(),sham.state_dict())
                assert session==sham_session and observer.state==sham_observer.state and observer.carriers==sham_observer.carriers
                sham=None
            if index+1==1024:
                physical=m.state_dict();old_observer=copy.deepcopy(observer.state);old_carriers=copy.deepcopy(observer.carriers)
                snapshot(args.out/'training_complete.npz',m,session,observer,reference,sources,1024,last_byte,readouts,frequencies,aggregates)
                target_array=blocks['train']
                for name,(feature,diagonal) in FAMILIES.items():
                    readouts[name]=fit_readout(np.array([x[feature] for x in training]),target_array,diagonal=diagonal,ridge=.01)
                counts=np.bincount(target_array,minlength=256)
                bits=((target_array[:,None]>>np.arange(8))&1).sum(axis=0)
                bp=(bits+.5)/(len(target_array)+1.)
                frequencies=dict(byte=((counts+.5)/(len(target_array)+128.)).tolist(),
                                 bit_logits=(np.log(bp)-np.log1p(-bp)).tolist(),samples=1024)
                atomic_json(args.out/'frozen_readouts.json',{k:v.state_dict() for k,v in readouts.items()})
                same(m.state_dict(),physical);assert observer.state==old_observer and observer.carriers==old_carriers
                snapshot(args.out/'natural_mid.npz',m,session,observer,reference,sources,1024,last_byte,readouts,frequencies,aggregates)
                atomic_json(args.out/'fit_observation_checks.json',dict(all_physical_rngs_and_observers_unchanged=True,all_fits_converged=True))
            if index+1==1536:
                path=args.out/'validation_mid.npz'
                snapshot(path,m,session,observer,reference,sources,1536,last_byte,readouts,frequencies,aggregates)
                restore,rmeta=load_exchange(path);restored_session=copy.deepcopy(rmeta['session'])
                restored_observer=CarrierObservation(restore,rmeta['growth_observer'],rmeta['carrier_observer'])
                restore.event_observer,restore.domain_observer=restored_observer.native,restored_observer.domain
                restored_readouts={k:FrozenBitReadout.from_state_dict(v) for k,v in rmeta['frozen_external_readouts'].items()}
                restored_last_byte=rmeta['last_actual_byte']
        same(m.state_dict(),restore.state_dict());assert session==restored_session
        assert observer.state==restored_observer.state and observer.carriers==restored_observer.carriers
        assert maximum_error<1e-8
        archive=args.out/'natural_complete.npz'
        snapshot(archive,m,session,observer,reference,sources,2048,last_byte,readouts,frequencies,aggregates)
    summary=frame(m,observer,reference,2048,'fresh_validation_observation')
    summary.update(archive=str(archive),sha256=digest(archive),primary=True,
        nll_by_phase={k:{metric:v/row['n'] for metric,v in row['raw'].items()} for k,row in aggregates.items()},
        phase_sample_counts={k:row['n'] for k,row in aggregates.items()},
        frozen_external_observer_validation={k:v/1024 for k,v in aggregates['fresh_validation_observation']['observed'].items()},
        actual_fresh_train=1024,actual_fresh_validation=1024,next_train_byte=3840,next_validation_byte=1280,
        maximum_error=maximum_error,restored_future_intervals=512,
        all_physics_rngs_scores_features_readouts_session_observers_exact=True,
        observer_only_sham_bytes=128,observer_only_sham_exact=True,wall_seconds=time.perf_counter()-started)
    assert digest(args.archive)==parent_hash
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    atomic_json(args.out/'natural_summary.json',summary)
    atomic_json(args.out/'report.json',dict(passed=True,protocol_kind='readout_observation',
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=sources,code_sha256=hashes,source_unchanged=True,arms=dict(natural=summary),
        external_readout_fit_info={k:v.fit_info for k,v in readouts.items()},
        body_optimizer_updates=0,readouts_frozen_before_validation=True,
        scope='Independent-split frozen observation fit; physical native learning continues unchanged. Not a newly installed predictive circuit or a full-brain capability conclusion.'))
    atomic_json(args.out/'progress.json',dict(status='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_trace_fate_20261022/natural_complete.npz'))
    p.add_argument('--data',type=Path,default=ROOT/'data/ib_owt_10000')
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
