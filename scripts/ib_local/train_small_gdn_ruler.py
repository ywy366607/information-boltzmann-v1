"""Full-parameter short RULER task learning on the restored small body.

Task-format admission precedes old-native replay experiments. Every selected
question and answer fits the existing512-byte local window. This phase does
not claim an effect from gradients beyond that window.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.run_language_navigation_v2 import validate
from scripts.ib_local.small_gdn_ruler import (
    encode_case, fingerprint, full_parameter_audit, run_case, sha256,
)


def training_window(row,pending,window=512):
    prefix,gold=encode_case(row)
    raw=bytes([pending])+prefix+gold+b'\n'
    if len(raw)-1>window:
        raise ValueError('short task must fit the whole local window')
    positions=np.arange(1,len(raw))
    return np.frombuffer(raw,dtype=np.uint8),positions>=1+len(prefix)


def eligible_rows(manifest,limit,window=512):
    candidates=[]
    for data in manifest['data']:
        if data['task']!='niah_single_1':
            continue
        assert sha256(data['path'])==data['sha256']
        for line in Path(data['path']).read_text().splitlines():
            row=json.loads(line)
            prefix,gold=encode_case(row)
            if len(prefix)+len(gold)+1<=window:
                candidates.append(row)
    if len(candidates)<limit:
        raise ValueError(f'only{len(candidates)} length-eligible examples, expected{limit}')
    return candidates[:limit],len(candidates)


@torch.no_grad()
def evaluate(engine,rows,valid,max_new_bytes=128):
    before=fingerprint(engine.model,engine.table)
    cases=[run_case(engine,row,max_new_bytes) for row in rows]
    language=validate(engine.model,engine.table,valid)
    assert fingerprint(engine.model,engine.table)==before
    return dict(samples=cases,first_number_exact_count=sum(s['first_number_exact'] for s in cases),
        contains_count=sum(s['official_contains'] for s in cases),
        answer_nll=sum(s['answer_nll']*s['answer_bytes'] for s in cases)/sum(s['answer_bytes'] for s in cases),
        owt_validation_nll=language,state_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','train_manifest','test_manifest','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--examples',type=int,default=1024)
    p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--eval-limit',type=int,default=8)
    p.add_argument('--validation-bytes',type=int,default=8192)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    with gzip.open(args.checkpoint,'rb') as stream:
        payload=torch.load(stream,map_location='cpu',weights_only=False)
    config=dict(payload['config']);del payload
    engine,raw_validation,offset,source=build(args.checkpoint)
    train_manifest=json.loads(args.train_manifest.read_text())
    test_manifest=json.loads(args.test_manifest.read_text())
    if train_manifest['seed']==test_manifest['seed']:
        raise ValueError('different task data seeds required')
    training,available=eligible_rows(train_manifest,args.examples)
    testing,test_available=eligible_rows(test_manifest,args.eval_limit)
    train_targets={v for r in training for v in r['outputs']}
    test_targets={v for r in testing for v in r['outputs']}
    if train_targets&test_targets:
        raise ValueError('random target overlap; refuse rather than select outcomes')
    selected=np.asarray(raw_validation[:2*(262144+args.validation_bytes+1)])
    valid=(selected[selected>=offset]-offset)[262144:262144+args.validation_bytes+1].astype(np.uint8)
    del raw_validation,selected
    if len(valid)!=args.validation_bytes+1:
        raise ValueError('insufficient independent language validation')
    report=dict(status='running',source=source,protocol=__doc__,examples=args.examples,
        bptt_window_bytes=512,train_seed=train_manifest['seed'],test_seed=test_manifest['seed'],
        length_eligible_train=available,length_eligible_test=test_available,
        train_manifest_sha256=sha256(args.train_manifest),test_manifest_sha256=sha256(args.test_manifest),
        parameter_audit=full_parameter_audit(engine),
        supervision='answer UTF-8 bytes plus actual newline; prompts masked; no EOS added to256-byte vocabulary',
        update='existing model.objective and engine._advance; existing Adam states/LRs/group clipping/read-credit settlement',
        steps=[],evaluations=[])
    files={Path(m.__file__) for n,m in sys.modules.copy().items()
           if n.startswith('scripts.ib_local.') and getattr(m,'__file__',None)}
    files.add(Path(__file__))
    report['source_files']={str(f):sha256(f) for f in sorted(files)}
    frozen=args.output/'executed_sources';frozen.mkdir()
    for f in files:shutil.copy2(f,frozen/f.name)
    path=args.output/'report.json'
    def write():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    write()
    start=time.time()
    try:
        report['evaluations'].append(dict(example=0,**evaluate(engine,testing,valid)))
        write()
        print('BASELINE',report['evaluations'][-1]['first_number_exact_count'],flush=True)
        for index,row in enumerate(training,1):
            if (args.output/'STOP').exists():
                raise InterruptedError('controlled stop before next task update')
            raw,mask=training_window(row,engine.pending)
            _,prob,_=engine.table.run_credit(raw,max_window=512)
            ids,targets,logp=engine._inputs(raw,prob)
            directions=torch.randn(64,32,device=engine.device)
            loss,next_state,features,diag=engine.model.objective(
                ids,targets,engine.recurrent,logp,directions,
                score_mask=torch.tensor(mask,device=engine.device,dtype=torch.float32)[None])
            norm=engine._advance(loss,features,diag,next_state)
            engine.pending=int(raw[-1])
            report['steps'].append(dict(example=index,source_index=row['index'],
                input_bytes=len(raw)-1,supervised_bytes=int(mask.sum()),answer_ce=float(diag['ce']),
                loss=float(loss.detach()),gradient_norm=norm))
            if index%16==0:
                write();print('STEP',index,float(diag['ce']),flush=True)
            if index%args.eval_every==0 or index==len(training):
                result=dict(example=index,**evaluate(engine,testing,valid))
                report['evaluations'].append(result)
                print('EVAL',index,result['first_number_exact_count'],result['answer_nll'],result['owt_validation_nll'],flush=True)
                config['ruler_short_adaptation']=dict(examples=index,bptt_bytes=512,
                    source=source,train_manifest=str(args.train_manifest),test_manifest=str(args.test_manifest))
                last=args.output/'last.pt.gz'
                if last.exists():shutil.copy2(last,args.output/'previous.pt.gz')
                save(engine,args.output,source['cycle'],config)
                report['checkpoint']=str(last)
                report['seconds']=time.time()-start
                write()
        report.update(status='complete',seconds=time.time()-start,
            source_checkpoint_unchanged=sha256(args.checkpoint)==source['sha256'],
            source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
            optimizer_steps=len(report['steps']),peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert report['source_checkpoint_unchanged'] and report['source_files_unchanged']
    except Exception as exc:
        report.update(status='failed',error=repr(exc),seconds=time.time()-start)
        raise
    finally:
        write()


if __name__=='__main__':
    main()
