"""Same-body held/full native replay: bounded real RULER requests, all four layers.

The body observes the prompt without answer labels, committing local native
participation after each128-byte chunk. Only answer-containing chunks update
parameters. Two arms differ only in the derivative through old native replay.
This is a fixed-budget learning-path experiment, not a RULER benchmark score.
"""
import argparse
import copy
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import random
from dataclasses import fields, is_dataclass

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_experience import configure_small_replay
from scripts.ib_local.small_gdn_ruler import encode_case, sha256, fingerprint, request_branch, run_case
from scripts.ib_local.train_small_gdn_ruler import eligible_rows
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local.audit_small_native_failure import probe
from scripts.ib_local.predictive_responsibility import predictive_mixture


def continuation_digest(engine):
    """Actual continuation state, including Adam, native boundaries and RNG."""
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            h.update(str((value.dtype,tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,np.ndarray):
            h.update(str((value.dtype,value.shape)).encode());h.update(value.tobytes())
        elif is_dataclass(value):
            for f in fields(value):visit(f.name);visit(getattr(value,f.name))
        elif isinstance(value,dict):
            for key in sorted(value,key=str):visit(key);visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:h.update(repr(value).encode())
    visit(engine.model.state_dict());visit(engine.optimizer.state_dict())
    visit(engine.recurrent);visit(engine.pending);visit(engine.world.state_dict())
    visit(torch.get_rng_state());visit(torch.cuda.get_rng_state_all())
    visit(np.random.get_state());visit(random.getstate())
    h.update(fingerprint(engine.model,engine.table).encode())
    return h.hexdigest()


def read_manifest(path):
    manifest=json.loads(path.read_text())
    rows=[]
    for item in manifest['data']:
        if item['task'] != 'niah_single_1':
            continue
        assert sha256(item['path']) == item['sha256']
        rows.extend(json.loads(line) for line in Path(item['path']).read_text().splitlines())
    return rows,manifest


def old_gradients(engine, ce):
    result=[]
    for i,(layer,wrapper) in enumerate(zip(engine.model.base.layers,engine.model.local_replay.memories)):
        responses=[r for r in wrapper.replayed if r.requires_grad]
        if not responses:
            result.append(dict(layer=i+1,old_native_norms={}))
            continue
        teaching=torch.autograd.grad(ce,responses,retain_graph=True,allow_unused=True)
        active=[(r,g) for r,g in zip(responses,teaching) if g is not None]
        named=list(layer.named_parameters())
        gradients=torch.autograd.grad([r for r,g in active],[p for n,p in named],
            grad_outputs=[g.detach() for r,g in active],retain_graph=True,allow_unused=True)
        result.append(dict(layer=i+1,old_native_norms={n:float(g.norm()) if g is not None else 0.
            for (n,p),g in zip(named,gradients)}))
    return result


def question_step(engine,row,diagnose=False):
    bank=engine.model.local_replay
    prefix,gold=encode_case(row)
    raw=np.frombuffer(bytes([engine.pending])+prefix+gold+b'\n',dtype=np.uint8)
    reports=[]
    for start in range(0,len(raw)-1,128):
        end=min(start+128,len(raw)-1)
        part=raw[start:end+1]
        _,prob,_=engine.table.run_credit(part,max_window=128)
        ids,targets,logp=engine._inputs(part,prob)
        mask=torch.tensor(np.arange(start+1,end+1)>=1+len(prefix),device=engine.device)[None]
        with bank.observation(engine.optimizer.param_groups[-1],end-start):
            if not bool(mask.any()):
                with torch.no_grad():
                    _,state,_,features,read=engine.model.predict_observed(ids,engine.recurrent,logp)
                    engine.model.commit_observed(features,read)
                    engine.recurrent=state.detach()
                reports.append(dict(observed=end-start,supervised=0,layers=bank.diagnostics()))
            else:
                loss,state,features,diag=engine.model.objective(ids,targets,engine.recurrent,logp,
                    torch.randn(64,32,device=engine.device),score_mask=mask)
                report=dict(observed=end-start,supervised=int(mask.sum()),ce=float(diag['ce']),
                            layers=bank.diagnostics())
                if bank.schema in (3, 4):
                    forecast=bank.forecast
                    report['responsibility']=forecast.report(targets,mask)
                    if diagnose:
                        ce=F.cross_entropy(diag['logits'][mask],targets[mask])
                        derivative,=torch.autograd.grad(ce,forecast.prior_scores,retain_graph=True)
                        expected=(forecast.log_prior.exp()-forecast.posterior(targets))
                        expected=expected*mask.reshape(-1,1)/mask.sum()
                        error=float((derivative-expected).abs().max())
                        report['responsibility']['prior_gradient_max_error']=error
                        assert error<2e-6
                        if bank.schema == 4:
                            derivatives=torch.autograd.grad(ce,[m.view_logit for m in bank.memories],
                                                            retain_graph=True,allow_unused=True)
                            report['view_prior_gradients']=[None if g is None else float(g)
                                                            for g in derivatives]
                if diagnose:
                    ce=F.cross_entropy(diag['logits'][mask],targets[mask])
                    report['old_gradients']=old_gradients(engine,ce)
                    if bank.schema == 5:
                        derivatives=torch.autograd.grad(ce,[m.view_logit for m in bank.memories],
                                                        retain_graph=True,allow_unused=True)
                        report['view_prior_gradients']=[None if g is None else float(g.detach())
                                                       for g in derivatives]
                        matrices=[m.inlet.projection.output_weight for m in bank.memories
                                  if m.inlet.projection.output_weight is not None]
                        if matrices:
                            derivatives=torch.autograd.grad(ce,matrices,retain_graph=True,allow_unused=True)
                            report['native_output_gradient_norms']=[None if g is None else float(g.norm())
                                                                    for g in derivatives]
                report['gradient_norm']=engine._advance(loss,features,diag,state)
                reports.append(report)
            bank.settle_and_commit()
        engine.pending=int(raw[end])
    return dict(source_index=row['index'],input_bytes=len(raw)-1,windows=reports)


@torch.no_grad()
def evaluate(engine,rows,valid,generation_bytes=32):
    before=fingerprint(engine.model,engine.table)
    # Keep evaluation's local participation interval equal to training's128.
    import scripts.ib_local.small_gdn_ruler as api
    original=api.consume_prompt
    def prompt(model,table,prefix,chunk=128):
        return original(model,table,prefix,128)
    api.consume_prompt=prompt
    try:
        cases=[run_case(engine,row,generation_bytes) for row in rows]
        chunk=128 if engine.model.local_replay.schema in (3, 4, 5) else 512
        language=validate(engine.model,engine.table,valid,chunk=chunk)
    finally:
        api.consume_prompt=original
    assert fingerprint(engine.model,engine.table)==before
    return dict(cases=cases,exact=sum(c['first_number_exact'] for c in cases),
        answer_nll=sum(c['answer_nll'] for c in cases)/len(cases),owt_nll=language)


def config_of(checkpoint):
    with gzip.open(checkpoint,'rb') as stream:
        payload=torch.load(stream,map_location='cpu',weights_only=False)
    return dict(payload['config'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','train_manifest','test_manifest','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--long-test-manifest',type=Path)
    p.add_argument('--examples',type=int,default=128)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--replay-schema',type=int,choices=(2,3,4,5),default=2)
    p.add_argument('--native-output-projection',action='store_true',
                   help='Schema5: learn one native-space output matrix, neutral zero initialization')
    p.add_argument('--test-start',type=int,default=0)
    p.add_argument('--test-count',type=int,default=8)
    p.add_argument('--generation-bytes',type=int,default=32)
    args=p.parse_args()
    if args.native_output_projection and args.replay_schema != 5:
        p.error('--native-output-projection requires --replay-schema 5')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);gdn_fused.enable()
    training,train_manifest=read_manifest(args.train_manifest)
    testing,test_manifest=read_manifest(args.test_manifest)
    if args.test_start<0 or args.test_count<1 or args.examples<9 or args.generation_bytes<1:
        raise ValueError('nonnegative test offset, positive test count and >=9 examples required')
    training=training[:args.examples]
    testing=testing[args.test_start:args.test_start+args.test_count]
    assert len(training)==args.examples and len(testing)==args.test_count
    longrows=[] if args.long_test_manifest is None else read_manifest(args.long_test_manifest)[0][:8]
    assert not {v for r in training for v in r['outputs']}&{v for r in testing+longrows for v in r['outputs']}
    report=dict(status='running',protocol=__doc__,source_sha256=sha256(args.checkpoint),
        bptt_bytes=128,capacity_per_layer=16,local_replay_bytes=128,replay_schema=args.replay_schema,
        native_output_projection=args.native_output_projection,
        train_examples=args.examples,test_start=args.test_start,test_count=args.test_count,arms={},audit={},
        generation_bytes=args.generation_bytes,
        source_files={},train_manifest_sha256=sha256(args.train_manifest),
        test_manifest_sha256=sha256(args.test_manifest),
        long_test_manifest_sha256=None if args.long_test_manifest is None else sha256(args.long_test_manifest))
    files={Path(m.__file__) for n,m in sys.modules.copy().items()
           if n.startswith('scripts.ib_local.') and getattr(m,'__file__',None)}|{Path(__file__)}
    report['source_files']={str(f):sha256(f) for f in sorted(files)}
    source_dir=args.output/'executed_sources';source_dir.mkdir()
    import shutil
    for f in files: shutil.copy2(f,source_dir/f.name)
    out=args.output/'report.json'
    def write():out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    start=time.time();write()
    try:
        engine,raw,offset,source=build(args.checkpoint)
        cfg=config_of(args.checkpoint)
        selected=np.asarray(raw[:2*(262144+2049)])
        valid=(selected[selected>=offset]-offset)[262144:262144+2049].astype(np.uint8)
        del raw,selected
        # Neutral migration checked against the complete current body.
        ids=torch.arange(128,device=engine.device)[None]
        logp=torch.full((1,128,256),-np.log(256),device=engine.device)
        with torch.no_grad(): before=engine.model.predict(ids,engine.recurrent,logp)[0]
        bank=configure_small_replay(engine,dict(schema=args.replay_schema,
            **({'native_output_projection':True} if args.native_output_projection else {})))
        cfg['small_native_replay']=bank.config
        with torch.no_grad(): after=engine.model.predict(ids,engine.recurrent,logp)[0]
        report['audit']['neutral_max']=float((before-after).abs().max())
        assert report['audit']['neutral_max']==0
        report['neural_parameters']=sum(p.numel() for p in engine.model.parameters())
        report['native_output_parameters']=sum(
            m.inlet.projection.output_weight.numel() for m in bank.memories
            if getattr(m.inlet.projection,'output_weight',None) is not None)
        common=args.output/'common';common.mkdir()
        save(engine,common,source['cycle'],cfg)
        report['common_checkpoint']=str(common/'last.pt.gz')
        # Eight real task updates open the neutral inlet through ordinary loss.
        # This audit branch is discarded; both experiment arms start at common.
        report['audit']['warmup']=[question_step(engine,r,diagnose=i==7) for i,r in enumerate(training[:8])]
        with torch.no_grad():
            bank.set_held(False); full=engine.model.predict(ids,engine.recurrent,logp)[0]
            bank.set_held(True); held=engine.model.predict(ids,engine.recurrent,logp)[0]
        report['audit']['held_full_forward_max']=float((full-held).abs().max())
        assert report['audit']['held_full_forward_max']==0
        audit_windows=report['audit']['warmup'][-1]['windows']
        last=next(w for w in reversed(audit_windows) if w['supervised'])
        for layer in last['old_gradients']:
            norms=layer['old_native_norms']
            assert norms['alpha_proj.weight']>0 and norms['beta_proj.weight']>0
            assert norms['mlp.up.weight']>0
        if args.native_output_projection:
            assert len(last['native_output_gradient_norms'])==len(bank.memories)
            assert all(v is not None and np.isfinite(v) and v>0
                       for v in last['native_output_gradient_norms'])
        assert all(r['mass_older_than_window']>0 for r in last['layers'])
        bank.set_held(False)
        # Responsibility-specific admission: the earlier migration only
        # checked arbitrary old-native gradients, missing this relation.
        report['audit']['responsibility']=[]
        for row in testing:
            alt=next(r['outputs'][0] for r in testing
                     if r['outputs'][0][0]!=row['outputs'][0][0])
            before=fingerprint(engine.model,engine.table)
            factual=probe(engine,row,alt,differentiate=True)
            removed=probe(engine,row,alt,cut='needle_record')
            assert fingerprint(engine.model,engine.table)==before
            for item in factual['layers']:
                assert any(r['overlap_digits']==len(row['outputs'][0]) for r in item['records'])
                assert item['direct_digit_position_mass']>0
                assert item['direct_digit_input_grad_norm']>0
            report['audit']['responsibility'].append(dict(case=row['index'],
                layers=[{k:item[k] for k in ('layer','direct_digit_position_mass',
                        'direct_digit_input_grad_norm')} for item in factual['layers']],
                digit_record_removal_nll_delta=removed['first_digit_nll']-factual['first_digit_nll']))
        restored_dir=args.output/'audit_resume';restored_dir.mkdir()
        save(engine,restored_dir,source['cycle'],cfg)
        # Real next update, then reconstruct from disk and repeat it.
        first=question_step(engine,training[8])
        first_fingerprint=continuation_digest(engine)
        del engine,bank;gc.collect();torch.cuda.empty_cache()
        engine,_,_,_=build(restored_dir/'last.pt.gz')
        second=question_step(engine,training[8])
        report['audit']['resume_next_update_exact']=(first==second and
            first_fingerprint==continuation_digest(engine))
        assert report['audit']['resume_next_update_exact']
        report['audit']['passed']=True
        write();print('AUDIT PASSED',time.time()-start,flush=True)
        del engine;gc.collect();torch.cuda.empty_cache()
        if not args.audit_only:
            for arm in ('held','full'):
                engine,_,_,_=build(common/'last.pt.gz')
                engine.model.local_replay.set_held(arm=='held')
                result=dict(steps=[],evaluations=[]);report['arms'][arm]=result
                initial=dict(example=0,short=evaluate(engine,testing,valid,args.generation_bytes))
                if longrows:initial['long']=evaluate(engine,longrows,valid,args.generation_bytes)
                result['evaluations'].append(initial)
                for i,row in enumerate(training,1):
                    if (args.output/'STOP').exists():raise InterruptedError('STOP requested')
                    result['steps'].append(question_step(engine,row,diagnose=arm=='full' and i in (16,args.examples)))
                    if i in (1,32,64,128,256,512,args.examples):
                        measured=dict(example=i,short=evaluate(engine,testing,valid,args.generation_bytes))
                        if longrows:measured['long']=evaluate(engine,longrows,valid,args.generation_bytes)
                        result['evaluations'].append(measured)
                        print(arm,i,measured['short']['exact'],measured['short']['answer_nll'],flush=True)
                        folder=args.output/arm;folder.mkdir(exist_ok=True)
                        cfg['small_native_credit_run']=dict(arm=arm,examples=i,window=128)
                        save(engine,folder,source['cycle'],cfg)
                    report['seconds']=time.time()-start;write()
                del engine;gc.collect();torch.cuda.empty_cache()
        report.update(status='complete',seconds=time.time()-start,
            source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            executed_sources_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert report['source_unchanged'] and report['executed_sources_unchanged']
    except Exception as exc:
        report.update(status='failed',error=repr(exc),seconds=time.time()-start)
        raise
    finally:
        write()


if __name__=='__main__':main()
