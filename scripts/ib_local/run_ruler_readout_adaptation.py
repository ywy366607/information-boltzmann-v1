"""One continuous shared adaptation life; no delayed-credit arms are launched.

RULER real generator data, ordinary answer-only SFT, existing readout learns.
Independent frozen diagnostics are explicitly separated from the lived stream.
"""
import argparse
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def require_repair_report(path, source, execution_files):
    """Fail before importing Torch/loading a body unless this exact repair passed."""
    if path is None:
        raise RuntimeError('training remains stopped: a verified repair report is required')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ('storage_replay', 'query_conditioned_read', 'tail_native_derivative',
                'causal_and_cached_read', 'source_age_controls', 'frozen_delivery',
                'disk_next_event_restore', 'resource_budget')
    if (report.get('training_ready') is not True
            or any(report.get('checks', {}).get(name) != 'passed' for name in required)):
        raise RuntimeError('repair report has unresolved transfer checks; training remains stopped')
    def digest(file):
        value = hashlib.sha256()
        with Path(file).open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    if report.get('source_checkpoint_sha256') != digest(source):
        raise ValueError('repair evidence belongs to a different source checkpoint')
    for file in execution_files:
        if report.get('execution_sources', {}).get(str(Path(file).resolve())) != digest(file):
            raise ValueError('repair evidence belongs to a different execution version')
    return report


def checkpoint_on_stop(stop_path, completed, checkpoint, report, save):
    """Stop between real examples, saving the latest complete live state first.

    Do not create targets or flush a partial window to manufacture a checkpoint.
    A failed save remains a failure; it must never be reported as a saved stop.
    """
    if not stop_path.exists():
        return False
    checkpoint(completed)
    report.update(status='stopped', stopped_by_user=True, completed=completed,
                  stop_reason=stop_path.read_text(encoding='utf-8-sig').strip(),
                  checkpointed_stop=True)
    save()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repair-report', type=Path)
    parser.add_argument('--runtime-manifest', type=Path,
                        help='separate version manifest for repaired execution; never overwrite the historical one')
    args = parser.parse_args()
    repair_files = [Path(__file__), ROOT/'scripts/ib_local/pretrained_gdn_experience.py',
                    ROOT/'scripts/ib_local/pretrained_gdn_learning.py',
                    ROOT/'scripts/ib_local/native_s_eligibility.py',
                    ROOT/'scripts/ib_local/native_s_learner.py',
                    ROOT/'scripts/ib_local/age_resolved_native_s.py',
                    ROOT/'scripts/ib_local/bounded_source_ages.py',
                    ROOT/'scripts/ib_local/continuous_frozen_case.py']
    require_repair_report(args.repair_report,
                         ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt', repair_files)
    if args.runtime_manifest is None:
        raise RuntimeError('repaired execution requires its own runtime manifest')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    for key, suffix in [('HF_HOME','/huggingface'), ('TRANSFORMERS_CACHE','/transformers'),
                        ('TRITON_CACHE_DIR','/triton'), ('TORCHINDUCTOR_CACHE_DIR','/inductor'),
                        ('TEMP','/tmp'), ('TMP','/tmp')]:
        os.environ[key] = 'D:/ml_cache'+suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.ruler_readout_adaptation import (
        ReadoutAdaptationLearner, scheduled_gap, validate_splits,
    )
    from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream, PretrainedStreamBackend
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.pretrained_gdn_experience import copy_tree
    from scripts.ib_local.native_s_storage import stop_future_joint_archives
    from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
    from scripts.ib_local.continuous_frozen_case import run_continuous_frozen_case
    from scripts.ib_local.probe_continuous_readiness import native_only
    from scripts.ib_local.audit_pretrained_continued_answers import insert_interference
    from scripts.ib_local.analyze_ruler_generation import numeric_scores
    from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
    from scripts.ib_local.checkpoint_digest import state_digest

    runtime = argparse.Namespace(source=ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt',
        runtime_manifest=args.runtime_manifest,
        model=Path('E:/cgb_gdn340_probe/model'),
        vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        bounded_autotune=True, readonly_checkpoints=True)
    data_root = ROOT/'results/cgb_ruler_adaptation_20261109'
    manifests = [data_root/name/'manifest.json' for name in ('data_train','data_validation')]
    datasets = []
    for path in manifests:
        manifest = old.read_json(path)
        cell = manifest['data'][0]
        if manifest['status']!='complete' or old.sha256(cell['path'])!=cell['sha256']:
            raise ValueError('official generated dataset incomplete or changed')
        datasets.append([json.loads(line) for line in Path(cell['path']).read_text().splitlines()])
    training, validation = datasets
    if len(training)!=512 or len(validation)!=8:
        raise ValueError('registered512 adaptation /8 validation samples required')
    validate_splits(training, validation)
    sources = {str(p):old.sha256(p) for p in manifests}
    for manifest_path in manifests:
        for cell in old.read_json(manifest_path)['data']:
            sources[str(Path(cell['path']).resolve())] = cell['sha256']
    report = dict(status='loading', protocol=__doc__, **old.provenance(runtime),
        steps=[], windows=[], evaluations=[], examples_budget=512,
        data_sources=sources, main_life_updates=0, native_parameter_updates=0,
        delayed_qualification_enabled=False, qualification_arms_started=False,
        interpretation='bounded shared readout adaptation, not convergence or credit benefit',
        precision='row_int8_frozen/mapped_cpu', state_resets=0)
    files = [Path(__file__), ROOT/'scripts/ib_local/ruler_readout_adaptation.py',
             ROOT/'scripts/ib_local/frozen_mlp_int8.py',
             ROOT/'scripts/ib_local/continuous_frozen_case.py']
    report['execution_sources'] = {str(p):old.sha256(p) for p in files}
    report['execution_sources'].update({str(p):old.sha256(p) for p in repair_files})
    report['execution_sources'].update(report['historical_runtime_files'])
    started = time.monotonic()
    def save():
        report['seconds'] = time.monotonic()-started
        old.write_json(args.output/'report.json', report)
    prior = old.read_json(args.output/'report.json') if args.resume else None
    if prior and prior['status']=='complete':
        raise ValueError('do not restart a completed adaptation life')
    if prior and (prior['data_sources']!=sources or prior['execution_sources']!=report['execution_sources']):
        raise ValueError('resume inputs or execution changed')
    save()
    backend = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(runtime.model, local_files_only=True)
        original_loader = loading.load_streamed_pure_gdn
        loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
            mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
        try:
            _, _, _, model, weight_loading = old.load_runtime(runtime)
        finally:
            loading.load_streamed_pure_gdn = original_loader
        owner = load_readonly_checkpoint(runtime.source)
        initial = owner['learner']
        loaded = owner
        completed, gap_cursor = 0, 0
        if args.resume:
            latest = old.read_json(args.output/'latest.json')
            path = Path(latest['path']).resolve()
            if path.parent!=args.output.resolve() or old.sha256(path)!=latest['sha256']:
                raise ValueError('resume checkpoint differs or escaped output')
            loaded = load_readonly_checkpoint(path)
            driver = loaded['branch_driver']
            completed, gap_cursor = driver['completed'], driver['gap_cursor']
            if driver['execution_sources']!=report['execution_sources'] or driver['data_sources']!=sources:
                raise ValueError('checkpoint source contract differs')
            report.update(steps=prior['steps'][:completed], windows=prior['windows'][:driver['windows']],
                evaluations=[e for e in prior['evaluations'] if e['example']<=completed],
                resumed_from=latest)
        learner = ReadoutAdaptationLearner(model, layers=initial['layers'], capacity=initial['capacity'])
        learner.load_state_dict(loaded['learner'])
        learner.enable_complete_reactivation()
        stop_future_joint_archives(learner)
        learner._snapshot_owner = loaded
        backend = PretrainedStreamBackend(learner)
        def settled(row):
            record = {key: row[key] for key in (
                'updates','tokens','observed_only','supervised_tokens','ce','sigreg',
                'live_cache_preserved','window','external_teaching_tokens','history_chain',
                'stream_events','lived_preupdate_ce','reexecution_ce_difference',
                'supervised_nll_sum') if key in row}
            record['adaptation_gradient_norm'] = (getattr(learner,'last_adaptation_gradient_norm',None)
                                                   if row['external_teaching_tokens'] else None)
            report['windows'].append(record)
        stream = IrreversibleCreditStream(backend, on_window=settled)
        stream.restore_latest(loaded['stream'])
        report.update(weight_loading=weight_loading, source_updates=initial['updates'],
                      source_seen=initial['cache']['seen_tokens'],
                      reactivation_mode='complete_attention', repair_report=str(args.repair_report))
        offset = old.read_json(ROOT/'data/ib_owt_10000/tokenizer.json')['offset']
        text, raw_sha = read_text(ROOT/'data/ib_owt_10000/train.npy',offset,13_000_000,length=1_500_000)
        traffic = tokenizer.encode(text,add_special_tokens=False)
        required = sum(scheduled_gap(i)+128 for i in range(512))
        if len(traffic)<required+512:
            raise ValueError('disjoint real background corpus too short')
        report['background'] = dict(raw_start=13_000_000,raw_length=1_500_000,sha256=raw_sha)

        def native_preserved():
            for name, parameter in learner.native:
                if not torch.equal(parameter.detach().cpu(), initial['native'][name]):
                    raise AssertionError('shared adaptation changed native parameters')
            stored = initial['optimizer']
            for master, identity in zip(learner.masters, stored['param_groups'][0]['params']):
                actual = learner.optimizer.state.get(master,{})
                expected = stored['state'].get(identity,{})
                if actual.keys()!=expected.keys() or any(
                    not torch.equal(actual[k].cpu(),v.cpu()) if torch.is_tensor(v) else actual[k]!=v
                    for k,v in expected.items()):
                    raise AssertionError('shared adaptation changed native Adam history')

        def evaluate(example):
            backend.close()
            report['status']='evaluating'
            held = dict(memories={i:copy_tree(m.state_dict()) for i,m in learner.memories.items()},
                        cache=clone_cache(learner.cache), updates=learner.updates,tokens=learner.tokens,
                        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(), stream=stream.state_dict())
            before = state_digest(held)['sha256']
            versions = [p._version for p in model.parameters()]
            result = dict(example=example,cases=[],summary={})
            def restore():
                for i,m in learner.memories.items():
                    m.load_state_dict(held['memories'][i])
                learner.cache=clone_cache(held['cache'])
                learner.updates,learner.tokens=held['updates'],held['tokens']
                torch.set_rng_state(held['rng']);torch.cuda.set_rng_state(held['cuda_rng'])
            def restored_digest():
                # Match the snapshot's CPU placement for memory tensors. Native
                # cache stays on its original device in both trees.
                return state_digest(dict(memories={i:copy_tree(m.state_dict())
                    for i,m in learner.memories.items()},cache=learner.cache,
                    updates=learner.updates,tokens=learner.tokens,
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),stream=stream.state_dict()))['sha256']
            restore()
            if restored_digest()!=before:
                raise AssertionError('snapshot roundtrip failed before evaluating any question')
            report['evaluation_in_progress']=result
            try:
                for gap in (0,512):
                    # Validation interference is disjoint from training traffic.
                    background=tokenizer.decode(traffic[-512:],skip_special_tokens=True)
                    for disable in (False,True):
                        samples=[]
                        for row in validation[:4]:
                            report['current_evaluation']=dict(example=example,gap=gap,
                                extra_read_disabled=disable,index=row['index'])
                            save()
                            restore()
                            case=insert_interference(row,background) if gap else row
                            with native_only(learner,disable):
                                item=run_continuous_frozen_case(learner,tokenizer,case,held['stream'],32)
                            item.update(numeric_scores(item['prediction'],item['references']))
                            item.update(extra_gap_tokens=gap,extra_read_disabled=disable)
                            result['cases'].append(item);samples.append(item)
                            save()
                        result['summary'][f'{gap}:{"native" if disable else "full"}']=dict(
                            exact=sum(s['first_number_exact'] for s in samples),cases=len(samples),
                            answer_nll=sum(s['answer_nll'] for s in samples)/len(samples))
            finally:
                restore()
            after = restored_digest()
            if before!=after or versions!=[p._version for p in model.parameters()]:
                raise AssertionError('frozen evaluation did not restore the training state')
            result['training_state_restored']=True
            report['evaluations'].append(result)
            report.pop('evaluation_in_progress',None)
            report.pop('current_evaluation',None)
            save()
            print(json.dumps(dict(kind='evaluation',example=example,summary=result['summary'])),flush=True)
            del held
            gc.collect();torch.cuda.empty_cache()

        def checkpoint(count):
            backend.close();gc.collect();torch.cuda.empty_cache()
            path=args.output/f'checkpoint_{count:04d}.pt'
            latest=old.snapshot(path,learner,backend,stream,owner,dict(completed=count,
                gap_cursor=gap_cursor,windows=len(report['windows']),
                execution_sources=report['execution_sources'],data_sources=sources))
            old.write_json(args.output/'latest.json',latest)
            for obsolete in sorted(args.output.glob('checkpoint_*.pt'))[:-2]:
                old.prune_owned(obsolete,args.output)

        if not args.resume:
            evaluate(0)
        report['admission_passed'] = completed>=2
        for index in range(completed,512):
            if checkpoint_on_stop(args.output/'STOP', index, checkpoint, report, save):
                return
            gap=scheduled_gap(index)
            report.update(status='running',current=dict(example=index+1,gap=gap,phase='prompt'))
            save()
            row=training[index]
            if gap:
                background=tokenizer.decode(traffic[gap_cursor:gap_cursor+gap],skip_special_tokens=True)
                gap_cursor+=gap
                row=insert_interference(row,background)
            prefix,gold=encode_case(tokenizer,row)
            before=stream.events
            stream.receive(prefix,phase='sft_prompt',teach=False)
            report['current']['phase']='answer';save()
            scores=stream.receive(gold+[tokenizer.eos_token_id],phase='sft_answer',teach=True)
            stream.receive(traffic[gap_cursor:gap_cursor+128],phase='real_after_answer',teach=False)
            gap_cursor+=128
            native_preserved()
            item=dict(example=index+1,index=training[index]['index'],gap=gap,
                answer_preupdate_nll=sum(scores)/len(scores),supervised_arrivals=len(gold)+1,
                input_events=stream.events-before,updates=learner.updates,seen=learner.cache._seen_tokens,
                gradient_norm=getattr(learner,'last_adaptation_gradient_norm',None),
                native_unchanged=True,history_chain=stream.chain)
            report['steps'].append(item)
            if index==1:
                if not all(s['gradient_norm'] and s['gradient_norm']>0 for s in report['steps'][-2:]):
                    raise AssertionError('no actual readout gradient in two-case admission')
                report['admission_passed']=True
            if index+1 in (2,64,128,192,256,320,384,448,512):
                checkpoint(index+1)
            if (index+1)%64==0:
                evaluate(index+1)
            report['completed']=index+1
            save();print(json.dumps(dict(kind='adaptation',**item)),flush=True)
            gc.collect();torch.cuda.empty_cache()
        report.update(status='complete',completed=512,budget_exhausted=True,convergence_proven=False)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        if backend is not None:backend.close()
        try:old.verify_preserved(runtime,report)
        finally:save()


if __name__=='__main__':
    main()
