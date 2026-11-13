"""One registered continuous 48-case RULER trial of bounded scalar eligibility."""
import argparse
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import time


ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for key,suffix in [('HF_HOME','huggingface'),('TRANSFORMERS_CACHE','transformers'),
                       ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                       ('TEMP','tmp'),('TMP','tmp')]:
        os.environ[key]='D:/ml_cache/'+suffix
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.scalar_credit_trial import ScalarCreditLearner,ScalarCreditBackend
    from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream
    from scripts.ib_local.native_s_storage import stop_future_joint_archives
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.checkpoint_digest import state_digest
    from scripts.ib_local.analyze_ruler_generation import numeric_scores

    source=ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    curriculum_path=ROOT/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json'
    protocol=ROOT/'docs/information_boltzmann/CGB_WATER_FINAL_DIAGNOSIS_20261109.md'
    sources=[Path(__file__),ROOT/'scripts/ib_local/scalar_credit_trial.py',
             ROOT/'scripts/ib_local/scalar_gate_eligibility.py',
             ROOT/'scripts/ib_local/native_s_learner.py',ROOT/'scripts/ib_local/native_s_eligibility.py',
             ROOT/'scripts/ib_local/irreversible_credit_stream.py']
    data=json.loads(curriculum_path.read_text())
    expected='daf2fc61f04f6482b56f6edf1cedca4a16411312dd4db61658ace750c53576b3'
    if sha(curriculum_path)!=expected or len(data['training'])!=48 or len(data['evaluation'])!=16:
        raise ValueError('registered curriculum differs')
    source_hash=sha(source)
    if source_hash!='514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5':
        raise ValueError('protected source changed')
    report=dict(status='loading',protocol='P-20261109-04',protocol_sha256=sha(protocol),
        source_sha256=source_hash,curriculum_sha256=expected,
        execution_sources={str(p):sha(p) for p in sources},
        main_life_updates=0,main_life_resets=0,optimizer_updates_by_arm={},
        scope='8 A_log coordinates; held-input local-S qualification; not full model credit',
        baseline=None,arms={},budget=dict(cases=48,max_windows=512,
        train_arrival_tokens=data['train_arrival_tokens'],supervised_tokens=data['supervised_tokens']),
        registered_48_case_budget=True,validation_is_previously_inspected_development_set=True)
    started=time.monotonic()
    def save():
        report['seconds']=time.monotonic()-started
        write(args.output/'report.json',report)
    save()
    runtime=argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'),bounded_autotune=True)
    loader=loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn=partial(load_streamed_int8_frozen,
        mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
    try:
        _,_,_,model,weight_loading=old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn=loader
    report['weight_loading']=weight_loading
    tokenizer=AutoTokenizer.from_pretrained(runtime.model,local_files_only=True)
    owner=load_readonly_checkpoint(source)
    origin=owner['learner']
    learner=ScalarCreditLearner(model,layers=origin['layers'],capacity=origin['capacity'])

    @torch.no_grad()
    def score_case(row,pending_length):
        cache=clone_cache(learner.cache)
        ids=row['prefix'];pos=0;window_offset=pending_length
        while pos<len(ids):
            size=min(len(ids)-pos,128-window_offset)
            x=torch.tensor([ids[pos:pos+size]],device='cuda')
            output=model(x,past_key_values=cache,use_cache=True)
            cache=output.past_key_values;pos+=size
            window_offset=(window_offset+size)%128
        logits=output.logits[:,-1].float()
        answer_cache=clone_cache(cache);last=logits;losses=[]
        for token in row['gold']+[data['eos']]:
            label=torch.tensor([token],device='cuda')
            losses.append(float(torch.nn.functional.cross_entropy(last,label)))
            out=model(label.reshape(1,1),past_key_values=answer_cache,use_cache=True)
            answer_cache=out.past_key_values;last=out.logits[:,-1].float()
        generated=[];answer_cache=clone_cache(cache);last=logits
        for _ in range(32):
            token=int(last.argmax())
            generated.append(token)
            out=model(torch.tensor([[token]],device='cuda'),past_key_values=answer_cache,use_cache=True)
            answer_cache=out.past_key_values;last=out.logits[:,-1].float()
            if token==data['eos']:
                break
        text=tokenizer.decode(generated,skip_special_tokens=True)
        stats=numeric_scores(text,row['outputs'])
        return dict(index=row['index'],gap=row['gap'],prediction=text,
            answer_nll=sum(losses[:-1])/len(row['gold']),answer_with_eos_nll=sum(losses)/len(losses),
            exact=bool(stats['first_number_exact']),generated_tokens=len(generated))

    def evaluate(stream):
        before_seen=learner.cache._seen_tokens
        before_updates=learner.updates
        versions=[p._version for p in model.parameters()]
        items=[score_case(row,len(stream.inputs)) for row in data['evaluation']]
        if before_seen!=learner.cache._seen_tokens or before_updates!=learner.updates or versions!=[p._version for p in model.parameters()]:
            raise AssertionError('evaluation mutated the live research branch')
        return dict(cases=items,summary={str(gap):dict(exact=sum(x['exact'] for x in items if x['gap']==gap),
            count=sum(x['gap']==gap for x in items),mean_nll=sum(x['answer_nll'] for x in items if x['gap']==gap)/8)
            for gap in (0,256)},scope='frozen researcher forks; same inherited native S, not primary life history')

    def nonselected_digest():
        values={}
        for master,(name,param) in zip(learner.masters,learner.native):
            if name not in learner.selected_masters:
                values[name]=dict(param=param.detach(),master=master.detach(),adam=learner.optimizer.state.get(master,{}))
        return state_digest(values)['sha256']

    def setup(mode):
        learner.load_state_dict(origin)
        migration=stop_future_joint_archives(learner)
        learner.initialize_trial(mode)
        backend=ScalarCreditBackend(learner)
        stream=IrreversibleCreditStream(backend)
        stream.restore_latest(owner['stream'])
        archived=sum(stream.learn_mask)
        stream.learn_mask=[False]*len(stream.learn_mask)
        if len(data['training'][0]['prefix']) <= 128-len(stream.inputs):
            raise ValueError('first answer could reach the uncaptured inherited partial')
        return backend,stream,migration,archived

    def run_case(stream,index):
        row=data['training'][index]
        before=stream.events
        stream.receive(row['prefix'],phase='ruler_prompt',teach=False)
        scores=stream.receive(row['gold']+[data['eos']],phase='ruler_feedback',teach=True)
        stream.receive(row['tail'],phase='real_after_feedback',teach=False)
        return dict(example=index+1,index=row['index'],events=stream.events-before,
                    supervised_arrivals=len(row['gold'])+1,preupdate_nll=sum(scores)/len(scores),
                    updates=learner.updates,seen=learner.cache._seen_tokens,history_chain=stream.chain)

    try:
        backend,stream,migration,archived=setup('raw')
        report.update(status='baseline',archived_pending_labels_masked_in_research_fork=archived,
                      migration=migration)
        report['baseline']=evaluate(stream);save()
        short=report['baseline']['summary']['0']['exact']
        long=report['baseline']['summary']['256']['exact']
        report['baseline_gate']=dict(short=short,long=long,required_short=4,
                                    passed=short>=4 and long<short)
        backend.close()
        if not report['baseline_gate']['passed']:
            report.update(status='baseline_gate_failed',decision='task lacks registered competence/interference range')
            return
        for mode in ('raw','cut_old','mismatch_old'):
            backend,stream,migration,archived=setup(mode)
            initial=nonselected_digest();windows=[];steps=[]
            report['arms'][mode]=dict(status='running',windows=windows,steps=steps,
                evaluations={},initial_nonselected_digest=initial,migration=migration)
            report['status']='running';report['current_arm']=mode;save()
            def on_window(result):
                windows.append(result)
                if len(windows)>data['max_actual_windows']:
                    raise RuntimeError('registered per-arm window budget exceeded')
                if result.get('reexecution_ce_difference') is not None and abs(result['reexecution_ce_difference'])>.01:
                    raise RuntimeError('actual vs reexecution CE exceeds registered .01')
                if not result.get('live_cache_preserved',False):
                    raise RuntimeError('actual native cache was replaced')
                capture=learner.eligibility
                # Upper bound on explicit arrays, not process RSS or model memory.
                max_head=max(s['key_dim']*s['value_dim'] for s in capture.specs.values())
                physical=2*sum(s['heads']*s['key_dim']*s['value_dim']*8 for s in capture.specs.values())
                capture_bytes=result.get('scalar_advance',{}).get('capture_bytes_at_commit',capture.nbytes)
                explicit=(learner.scalar.nbytes+capture_bytes+capture.estimated_temporary_bytes+
                          physical+8*max_head*8)
                result['scalar_resource_accounting']=dict(scalar_bytes=learner.scalar.nbytes,
                    capture_bytes=capture_bytes,capture_workspace_bytes=capture.estimated_temporary_bytes,
                    physical_and_head_workspace_bound=physical+8*max_head*8,
                    conservative_explicit_bytes=explicit,scope='excludes model, optimizer, Python, GPU/RSS')
                if explicit>64*2**20:
                    raise MemoryError('scalar plus capture explicit state/workspace exceeds 64 MiB')
                save()
            stream.on_window=on_window
            for i in range(48):
                if i==2 and mode=='raw':
                    # Case 3 was already run after restoring the admission checkpoint.
                    continue
                if (args.output/'STOP').exists():
                    raise RuntimeError('explicit user STOP')
                steps.append(run_case(stream,i));save()
                if i==1 and mode=='raw':
                    available=[w['native_scalar'] for w in windows if 'native_scalar' in w]
                    active=[w for w in available if any(abs(v)>1e-12 for arr in w['old'].values() for v in arr)]
                    if not active:
                        raise RuntimeError('two-case admission: no measurable strictly-old teaching')
                    shadow=active[-1]['shadow_fp32_delta']
                    difference=max(abs(a-b) for n in shadow['raw'] for a,b in zip(shadow['raw'][n],shadow['cut_old'][n]))
                    if difference==0:
                        raise RuntimeError('two-case admission: old teaching causes no FP32 Adam displacement')
                    actual=active[-1]['parameter_delta']
                    shadow_error=max(abs(a-b) for n in actual for a,b in zip(actual[n],shadow['raw'][n]))
                    if shadow_error>1e-8:
                        raise AssertionError('actual update differs from matching original-Adam shadow')
                    report['admission_signal']=dict(strict_old_updates=len(active),max_shadow_fp32_difference=difference)
                    # Save complete state before repeating a factual next case.
                    checkpoint=args.output/'admission_checkpoint.pt'
                    info=old.snapshot(checkpoint,learner,backend,stream,owner,dict(completed=2))
                    report['admission_checkpoint']=info;save()
                    # First pass is discarded, second is the actual third case.
                    temporary=[];stream.on_window=temporary.append
                    trial=run_case(stream,2)
                    first=state_digest(dict(learner=learner.state_dict(),stream=stream.state_dict()))['sha256']
                    del temporary;gc.collect()
                    saved=load_readonly_checkpoint(checkpoint)
                    backend.close();learner.load_state_dict(saved['learner'])
                    backend=ScalarCreditBackend(learner);stream=IrreversibleCreditStream(backend)
                    stream.restore_latest(saved['stream']);stream.on_window=on_window
                    trial2=run_case(stream,2)
                    second=state_digest(dict(learner=learner.state_dict(),stream=stream.state_dict()))['sha256']
                    if first!=second or trial!=trial2:
                        raise AssertionError('checkpoint did not reproduce next full actual case')
                    steps.append(trial2)
                    report['admission_recovery']=dict(passed=True,full_state_sha256=first,
                        next_case=3,duplicate_research_execution_discarded=True)
                    save()
                if i+1 in (16,48):
                    backend.close()
                    report['arms'][mode]['evaluations'][str(i+1)]=evaluate(stream)
                    save()
                if i+1==48:
                    break
            backend.close()
            end=nonselected_digest()
            if end!=initial:
                raise AssertionError('unselected native masters or Adam moments changed')
            report['arms'][mode].update(status='complete',completed_cases=48,
                final_nonselected_digest=end,updates=learner.updates-524,
                seen=learner.cache._seen_tokens,events=stream.events,
                actual_windows=len(windows),scalar_persistent_bytes=learner.scalar.nbytes)
            report['optimizer_updates_by_arm'][mode]=learner.updates-524
            save();gc.collect();torch.cuda.empty_cache()
        evaluations={m:report['arms'][m]['evaluations']['48']['cases'] for m in report['arms']}
        def long_values(mode):
            return {x['index']:x for x in evaluations[mode] if x['gap']==256}
        raw=long_values('raw');comparisons={}
        for mode in ('cut_old','mismatch_old'):
            other=long_values(mode)
            comparisons[mode]=dict(raw_better_nll=sum(raw[k]['answer_nll']<other[k]['answer_nll'] for k in raw),
                mean_raw_minus_other=sum(raw[k]['answer_nll']-other[k]['answer_nll'] for k in raw)/len(raw),
                raw_exact=sum(x['exact'] for x in raw.values()),other_exact=sum(x['exact'] for x in other.values()))
        passed=all(x['raw_better_nll']>=6 and x['mean_raw_minus_other']<0 and x['raw_exact']>=x['other_exact']
                   for x in comparisons.values())
        report.update(status='complete',comparisons=comparisons,registered_decision_passed=passed,
            source_unchanged=sha(source)==source_hash,
            execution_sources_unchanged=all(sha(p)==report['execution_sources'][str(p)] for p in sources),
            peak_cuda_allocated=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status='stopped' if (args.output/'STOP').exists() else 'failed',error=repr(exc))
        raise
    finally:
        try: backend.close()
        except (NameError,AttributeError): pass
        save()


if __name__=='__main__':
    main()
