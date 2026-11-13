"""Continuous real QA: emit, live through unrelated text, receive correction.

Explicit disposable research branches; no mainlife advancement or resets.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('raw','cut','mismatch'), required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=80)
    parser.add_argument('--resume-run-dir',type=Path)
    parser.add_argument('--frozen-mlp-precision',choices=('fp16','row_int8','row_int8_frozen'),default='fp16')
    parser.add_argument('--frozen-weight-placement',choices=('cuda','mapped_cpu'),default='cuda')
    parser.add_argument('--verify-mapped-qa-reference',type=Path)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 80:
        parser.error('use a declared prefix of1..80 episodes')
    if args.frozen_weight_placement=='mapped_cpu' and args.frozen_mlp_precision=='fp16':
        parser.error('mapped placement applies only to explicitly frozen packed weights')
    args.output.mkdir(parents=True, exist_ok=False)
    for name,suffix in [('HF_HOME','/huggingface'),('TRANSFORMERS_CACHE','/transformers'),
                         ('TRITON_CACHE_DIR','/triton'),('TORCHINDUCTOR_CACHE_DIR','/inductor'),('TEMP','/tmp'),('TMP','/tmp')]:
        os.environ[name]='D:/ml_cache'+suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local.delayed_s_learner import DelayedFeedbackLearner, DelayedFeedbackBackend, DelayedFeedbackStream
    from scripts.ib_local.delayed_squad_protocol import digest, answer_scores, validate_recovery
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.native_s_storage import stop_future_joint_archives

    original_source=ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    runtime=argparse.Namespace(source=original_source,
        runtime_manifest=ROOT/'scratch/bounded_causal_tags_20261108/operator_source_manifest.json',
        model=Path('E:/cgb_gdn340_probe/model'), vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        bounded_autotune=True, readonly_checkpoints=True)
    task=json.loads(args.task.read_text(encoding='utf-8'))
    episodes=task['episodes'][:args.episodes]
    report=dict(status='loading', arm=args.arm, task_sha256=digest(args.task), cases=[], windows=[],
                requested_episode_count=args.episodes,
                frozen_mlp_precision=args.frozen_mlp_precision,
                frozen_weight_placement=args.frozen_weight_placement,
                model_source_sha256=digest(runtime.model/'model.safetensors'),
                numeric_runtime={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                main_life_updates=0, state_resets=0, seed=11, extra_credit='delayed correction contribution only; base CE denominator preserved',
                ordinary_teaching='SFT corrected answer targets only',
                scope='approximate local native-S2/13; ordinary old-capsule replay identical in all arms',
                **old.provenance(runtime))
    paths=[Path(__file__),ROOT/'scripts/ib_local/delayed_s_learner.py',ROOT/'scripts/ib_local/delayed_squad_protocol.py',
           ROOT/'scripts/ib_local/frozen_mlp_int8.py']
    paths.extend(Path(p) for p in old.read_json(ROOT/'results/cgb_native_s_credit_20261108/measured_resume64/raw/report.json')['execution_sources'])
    report['execution_sources']={str(p):digest(p) for p in paths}
    def save():old.write_json(args.output/'report.json',report)
    save()
    backend=None
    started=time.monotonic()
    try:
        if digest(task['source_path']) != task['source_sha256']:
            raise ValueError('real annotated data changed')
        if digest(runtime.model/'tokenizer.json') != task['tokenizer_sha256']:
            raise ValueError('task tokenizer differs from model')
        tokenizer=Tokenizer.from_file(str(runtime.model/'tokenizer.json'))
        if args.frozen_mlp_precision!='fp16':
            from scripts.ib_local import pretrained_gdn_stream_load as loading_module
            from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_mlp,load_streamed_int8_frozen
            original=loading_module.load_streamed_pure_gdn
            from functools import partial
            selected_loader=(load_streamed_int8_frozen if args.frozen_mlp_precision=='row_int8_frozen'
                             else load_streamed_int8_mlp)
            loading_module.load_streamed_pure_gdn=partial(selected_loader,
                mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109'
                                 if args.frozen_weight_placement=='mapped_cpu' else None)
            try:_,_,_,model,loading=old.load_runtime(runtime)
            finally:loading_module.load_streamed_pure_gdn=original
        else:_,_,_,model,loading=old.load_runtime(runtime)
        owner=load_readonly_checkpoint(original_source)
        recovery=None
        if args.resume_run_dir is not None:
            prior=old.read_json(args.resume_run_dir/'report.json')
            if prior['status']=='complete':raise ValueError('do not rewind a completed life')
            if (prior['execution_sources']!=report['execution_sources'] or
                    prior['numeric_runtime']!=report['numeric_runtime'] or
                    prior['frozen_mlp_precision']!=report['frozen_mlp_precision'] or
                    prior.get('frozen_weight_placement','cuda')!=report['frozen_weight_placement'] or
                    prior['model_source_sha256']!=report['model_source_sha256']):
                raise ValueError('recovery execution sources changed')
            latest=old.read_json(args.resume_run_dir/'latest.json')
            path=Path(latest['path']).resolve()
            if path.parent!=args.resume_run_dir.resolve() or path.stat().st_size!=latest['bytes'] or digest(path)!=latest['sha256']:
                raise ValueError('recovery archive escaped directory or differs')
            owner=load_readonly_checkpoint(path)
            recovery=validate_recovery(owner,prior,task_sha256=report['task_sha256'],arm=args.arm,episodes=args.episodes)
            report.update(cases=recovery['cases'],windows=recovery['windows'],
                resumed_from=dict(path=str(path),sha256=latest['sha256'],episodes_completed=recovery['completed']))
            for key in ('source_updates','source_seen','source_events','bridge_arrival_tokens','bridge_updates'):
                report[key]=prior[key]
        source=owner['learner']
        learner=DelayedFeedbackLearner(model,layers=source['layers'],capacity=source['capacity'])
        learner.load_state_dict(source)
        if recovery is None:learner.enable_native_s_eligibility(mode=args.arm,rank=2,max_bytes=64*2**20)
        stop_future_joint_archives(learner)
        learner._snapshot_owner=owner
        backend=DelayedFeedbackBackend(learner)
        if args.verify_mapped_qa_reference is not None:
            if recovery is not None or args.frozen_weight_placement!='mapped_cpu':
                raise ValueError('placement calibration requires a fresh mapped research branch')
            reference_dir=args.verify_mapped_qa_reference
            baseline=old.read_json(reference_dir/'int8_report.json')
            if (baseline['source_sha256']!=report['source_sha256'] or
                    baseline['historical_runtime_files']!=report['historical_runtime_files']):
                raise ValueError('placement reference has a different complete source/runtime')
            reference=np.load(reference_dir/'int8_logits.npz')
            ids=episodes[0]['prompt_tokens'][:129]
            if not np.array_equal(reference['qa_target'],np.asarray(ids[1:])):
                raise ValueError('placement calibration prefix differs')
            versions=[p._version for p in learner.model.parameters()]
            seen,updates=learner.cache._seen_tokens,learner.updates
            logits,temporary=learner.predict(torch.tensor([ids[:-1]],device=backend.device))
            actual=logits[0].detach().half().cpu().numpy()
            equal=bool(np.array_equal(actual,reference['qa']))
            report['mapped_placement_calibration']=dict(bitwise_equal_logits=equal,tokens=128,
                reference_logits_sha256=digest(reference_dir/'int8_logits.npz'),
                max_logit_difference=float(np.max(np.abs(actual.astype(np.float32)-reference['qa'].astype(np.float32)))),
                optimizer_updates=0)
            if (not equal or versions!=[p._version for p in learner.model.parameters()] or
                    (seen,updates)!=(learner.cache._seen_tokens,learner.updates)):
                raise AssertionError('mapped placement changed logits or native parameters/live counters')
            del logits,temporary,actual,reference
            save()
        def settled(row):
            record={k:v for k,v in row.items() if k not in ('reads','credit','diagnostics')}
            record['layer_reads']={}
            for layer,read in row.get('reads',{}).items():
                ages,weights=read['ages_tokens'],read['weights']
                old_mass=sum(w for a,w in zip(ages,weights) if a>128)
                record['layer_reads'][str(int(layer)+1)]=dict(
                    query_count=read['query_count'],unique_replays=read['unique_replays'],
                    max_age=max(ages,default=0),outside128_count=sum(a>128 for a in ages),
                    outside128_weight_sum=old_mass,total_weight_sum=sum(weights))
            credit=learner.eligibility.last_credit
            if row.get('native_s',{}).get('teaching_available') and credit is not None:
                record['raw_credit_per_parameter']=credit['per_parameter']
            report['windows'].append(record)
            with (args.output/'windows.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n')
        stream=DelayedFeedbackStream(backend,on_window=settled)
        stream.restore_latest(owner['stream'])
        report.setdefault('source_updates',learner.updates)
        report.setdefault('source_seen',learner.cache._seen_tokens)
        report.setdefault('source_events',stream.events)
        report['weight_loading']=loading
        if recovery:
            with (args.output/'windows.jsonl').open('x',encoding='utf-8') as f:
                for row in recovery['windows']:f.write(json.dumps(row,allow_nan=False)+'\n')
        offset=json.loads((ROOT/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
        corpus=np.load(ROOT/'data/ib_owt_10000/train.npy',mmap_mode='r')
        raw=(corpus[12_000_000:12_393_216].astype(np.int32)-offset).astype(np.uint8).tobytes()
        import hashlib
        report['gap_data']=dict(array_start=12_000_000,array_stop=12_393_216,
            sha256=hashlib.sha256(raw).hexdigest(),revision=json.loads((ROOT/'data/ib_owt_10000/manifest.json').read_text())['revision'])
        traffic=tokenizer.encode(raw.decode('utf-8',errors='replace'),add_special_tokens=False).ids
        bridge=(128-len(stream.inputs)+1) if recovery is None else 0
        needed=sum(e['feedback_delay_tokens']+128 for e in episodes)+bridge
        if len(traffic)<needed:raise ValueError('real gap corpus too short')
        cursor=0 if recovery is None else recovery['cursor']
        def world(count,phase,teach):
            nonlocal cursor
            first=cursor
            scores=stream.receive(traffic[cursor:cursor+count],phase=phase,teach=teach)
            cursor+=count
            return scores,first,cursor
        if recovery is None:
            before_bridge=learner.updates
            world(bridge,'bridge_into_study',False)
            report.update(bridge_arrival_tokens=bridge,bridge_updates=learner.updates-before_bridge)
            if len(stream.inputs)!=1 or learner.eligibility.pending is None:
                raise AssertionError('real prelude did not open the first qualified window')
        feedback_counts={}
        for c in report['cases']:
            if c['correction_provided']:feedback_counts[c['qa_id']]=feedback_counts.get(c['qa_id'],0)+1
        for episode in episodes[0 if recovery is None else recovery['completed']:]:
            if (args.output/'STOP').exists():raise RuntimeError('owned study STOP requested')
            report.update(status='running',in_progress=dict(index=episode['index'],stage=episode['stage'],phase='question'))
            save()
            before=stream.events
            stream.receive(episode['prompt_tokens'],phase='question',teach=False)
            question_end=stream.events
            question_reads=backend.last_live_reads
            first_target=episode['answer_tokens'][0]
            first_nll=float(torch.nn.functional.cross_entropy(stream.next_logits.float().reshape(1,-1),
                torch.tensor([first_target],device=backend.device)))
            generated=stream.generate(eos_token_id=2,max_tokens=16)
            text=tokenizer.decode(generated,skip_special_tokens=True)
            emission_end=stream.events
            report['in_progress'].update(phase='intervening',generated=text,
                generated_tokens=generated,first_answer_token_nll=first_nll)
            save()
            backend.begin_gap()
            _,gap_start,gap_stop=world(episode['feedback_delay_tokens'],'intervening',False)
            survival=backend.gap_survival()
            feedback_started=stream.events
            report['in_progress'].update(phase='feedback',actual_gap=feedback_started-emission_end)
            save()
            scores=[]
            if episode['provide_correction']:
                stream.receive(episode['feedback_prefix_tokens'],phase='feedback_prefix',teach=False)
                feedback_reads=backend.last_live_reads
                scores=stream.receive(episode['answer_tokens'],phase='feedback_answer',teach=True)
            else:feedback_reads=None
            feedback_end=stream.events
            # Actual further language, not synthetic padding or a forced end flush.
            world(128,'after_feedback',False)
            result=dict(index=episode['index'],stage=episode['stage'],group=episode['group'],qa_id=episode['qa_id'],
                study_qa_feedbacks_before=feedback_counts.get(episode['qa_id'],0),
                new_question=episode['new_question'],context_reintroduced=episode['present_context'],
                generated=text, generated_tokens=generated, answers=episode['answers'],
                correction_provided=episode['provide_correction'],
                first_answer_token_nll=first_nll, feedback_preupdate_nll=sum(scores)/len(scores) if scores else None,
                feedback_delay_tokens=feedback_started-emission_end,
                first_corrective_target_distance=(feedback_started-emission_end+len(episode['feedback_prefix_tokens'])+1
                                                 if episode['provide_correction'] else None),
                origin_events=before,emission_end_event=emission_end,feedback_start_event=feedback_started,
                question_end_event=question_end,question_prediction_reads=question_reads,
                first_feedback_prediction_reads=feedback_reads,
                feedback_end_event=feedback_end,end_event=stream.events,
                gap_token_cursor=[gap_start,gap_stop],direct_S_gap_survival=survival,
                updates=learner.updates,seen=learner.cache._seen_tokens,
                **answer_scores(text,episode['answers']))
            if result['feedback_delay_tokens']<=128:raise AssertionError('feedback within BPTT window')
            report['cases'].append(result)
            if episode['provide_correction']:
                feedback_counts[episode['qa_id']]=feedback_counts.get(episode['qa_id'],0)+1
            report.update(status='running',episodes_completed=len(report['cases']),actual_gap_cursor=cursor,
                          end_updates=learner.updates,end_seen=learner.cache._seen_tokens,end_events=stream.events)
            save()
            gc.collect()
            if len(report['cases'])%8==0:
                path=args.output/f'checkpoint_{len(report["cases"]):04d}.pt'
                temporary=path.with_suffix('.tmp.pt')
                import shutil
                owned=list(args.output.parent.glob('*/checkpoint_*.pt'))
                estimate=max([old.SOURCE_BYTES]+[p.stat().st_size for p in owned])
                if (sum(p.stat().st_size for p in owned)+estimate>6*2**30 or
                        shutil.disk_usage(args.output).free<estimate+256*2**20):
                    raise OSError('complete archive plus atomic temporary exceeds6GiB/free disk budget')
                backend.close()
                gc.collect()
                torch.cuda.empty_cache()
                payload=dict(learner=learner.state_dict(),stream=stream.state_dict(),
                    next_case=owner['next_case'],pending=owner.get('pending'),
                    pending_answer_archival_only=True, main_life_updates=0,
                    delayed_squad_driver=dict(episodes_completed=len(report['cases']),gap_cursor=cursor,arm=args.arm,
                                             task_sha256=report['task_sha256'],history_chain=stream.chain,
                                             windows_completed=len(report['windows']),
                                             frozen_mlp_precision=args.frozen_mlp_precision,
                                             frozen_weight_placement=args.frozen_weight_placement,
                                             model_source_sha256=report['model_source_sha256']))
                torch.save(payload,temporary)
                temporary.replace(path)
                del payload
                gc.collect()
                old.write_json(args.output/'latest.json',dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
                generations=sorted(args.output.glob('checkpoint_*.pt'))
                for obsolete in generations[:-2]:
                    if obsolete.resolve().parent != args.output.resolve():
                        raise ValueError('owned checkpoint escaped output directory')
                    with (args.output/'CHECKPOINT_PRUNED.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(path=str(obsolete),bytes=obsolete.stat().st_size,
                            sha256=digest(obsolete),reason='verified newer checkpoint; retain latest two'))+'\n')
                    obsolete.unlink()
        report.update(status='complete',episodes_completed=len(episodes),
                      lifetime_reset_count=0,pending_targets=len(stream.targets),pending_inputs=len(stream.inputs))
    except BaseException as e:
        report.update(status='failed',error=repr(e))
        raise
    finally:
        if backend is not None:backend.close()
        report['seconds']=time.monotonic()-started
        try:old.verify_preserved(runtime,report)
        finally:save()


if __name__=='__main__':main()
