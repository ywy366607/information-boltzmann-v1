"""Restore the continuing small body and assess official RULER in UTF-8 bytes.

This is an independent, frozen request sidecar, not an organism reset. Native
state starts empty per request; the learned lexical and causal memories remain.
Their temporary assimilation is rolled back after each request. Generation has
a declared byte cap: this256-byte vocabulary has no EOS token.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local import gdn_fused
from scripts.ib_local.analyze_ruler_generation import numeric_scores
from scripts.ib_local.language_action import peek
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.table_branch import FIELDS, table_branch
from scripts.ib_local.train_gdn_fast_table import FLOOR


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_case(row):
    prefix=(row['input']+row.get('answer_prefix','')).encode('utf-8')
    gold=(' '+', '.join(row['outputs'])).encode('utf-8')
    if not prefix or not row['outputs']:
        raise ValueError('nonempty prompt and answer required')
    return prefix,gold


@contextmanager
def request_branch(engine, *, reset_history=True):
    """Preserve actual model memory and lexical rows, including pending queues."""
    old=engine.model.memory
    bank=getattr(engine.model,'local_replay',None)
    native=copy.deepcopy(bank.state_dict()) if bank is not None else None
    engine.model.memory=copy.deepcopy(old)
    devices=[engine.device.index or 0] if engine.device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), table_branch(engine.table) as table:
            if reset_history:
                table.history=np.uint64(0)
            yield table
    finally:
        engine.model.memory=old
        if bank is not None:
            bank.load_state_dict(native)


def probability_tensor(model, probability):
    device=next(model.parameters()).device
    return torch.tensor(np.log(np.maximum(probability,FLOOR)),device=device)[None]


@torch.no_grad()
def consume_prompt(model,table,prefix,chunk=512):
    """Execute every observed byte once; never fabricate a last-byte target."""
    device=next(model.parameters()).device
    state=model.initial_state(1,device=device)
    bank=getattr(model,'local_replay',None)
    if bank is not None and bank.schema in (2, 3, 4, 5):
        chunk=min(chunk,bank.config['replay_tokens'])
    for start in range(0,len(prefix)-1,chunk):
        raw=np.frombuffer(prefix[start:min(start+chunk+1,len(prefix))],dtype=np.uint8)
        _,prob,_=table.run_credit(raw,max_window=chunk)
        ids=torch.tensor(raw[:-1].astype(np.int64),device=device)[None]
        bank=getattr(model,'local_replay',None)
        if bank is None:
            logits,state,_,features,read=model.predict_observed(ids,state,probability_tensor(model,prob))
            model.commit_observed(features,read)
        else:
            # This request's actual prompt must create the same local records
            # as training. request_branch restores every slot afterwards.
            group=dict(lr=0.,betas=(.9,.999),eps=1e-8)
            with bank.observation(group,ids.shape[1]):
                logits,state,_,features,read=model.predict_observed(ids,state,probability_tensor(model,prob))
                model.commit_observed(features,read)
                bank.settle_and_commit()
    last=prefix[-1]
    logp=probability_tensor(model,peek(table,last)[None])
    logits,state,_,features,read=model.predict_observed(torch.tensor([[last]],device=device),state,logp)
    model.commit_observed(features,read)
    return logits[0,-1],state,last


@torch.no_grad()
def advance(model,table,state,pending,observed):
    """Only after a byte arrives, settle its predecessor's lexical transition."""
    table.run_credit(np.array([pending,observed],dtype=np.uint8),max_window=1)
    logp=probability_tensor(model,peek(table,observed)[None])
    logits,state,_=model.predict(torch.tensor([[observed]],device=state.device),state,logp)
    return logits[0,-1],state,observed


@torch.no_grad()
def run_case(engine,row,max_new_bytes):
    if max_new_bytes<=0:
        raise ValueError('positive, gold-independent generation budget required')
    prefix,gold=encode_case(row)
    with request_branch(engine) as table:
        first,state,pending=consume_prompt(engine.model,table,prefix)
        # Score gold on a lexical branch. predict() does not commit causal
        # episodes; the recurrent result is local and never replaces state.
        with table_branch(table) as teacher:
            current,local,last=first,state,pending
            loss=0.;byte_nll=[]
            for i,byte in enumerate(gold):
                value=float(-current.float().log_softmax(-1)[byte])
                loss+=value;byte_nll.append(value)
                if i+1<len(gold):
                    current,local,last=advance(engine.model,teacher,local,last,byte)
        current=first
        generated=[]
        for i in range(max_new_bytes):
            byte=int(current.argmax())
            generated.append(byte)
            if i+1<max_new_bytes:
                current,state,pending=advance(engine.model,table,state,pending,byte)
    text=bytes(generated).decode('utf-8',errors='replace')
    needle=row['input'].find(row['outputs'][0])
    distance=None if needle<0 else len(prefix)-len(row['input'][:needle].encode('utf-8'))
    return dict(index=row['index'],references=row['outputs'],prediction=text,
        prompt_bytes=len(prefix),answer_distance_bytes=distance,
        answer_nll=loss/len(gold),answer_byte_nll=byte_nll,
        first_digit_nll=next((byte_nll[i] for i,b in enumerate(gold) if 48<=b<=57),None),
        answer_bytes=len(gold),generated_bytes=len(generated),
        generation_cap_bytes=max_new_bytes,has_eos=False,**numeric_scores(text,row['outputs']))


def full_parameter_audit(engine):
    owned={id(p) for group in engine.optimizer.param_groups for p in group['params']}
    named=list(engine.model.named_parameters())
    missing=[n for n,p in named if id(p) not in owned]
    frozen=[n for n,p in named if not p.requires_grad]
    if missing or frozen:
        raise AssertionError(dict(missing_optimizer=missing,frozen=frozen))
    base=engine.model.base
    return dict(neural_parameters=sum(p.numel() for _,p in named),
        trainable_parameters=sum(p.numel() for _,p in named if p.requires_grad),
        all_parameters_owned_by_optimizer=True,architecture=base.config_dict(),
        mlp_present_per_layer=[layer.mlp is not None for layer in base.layers],
        note='All parameters are available for learning; task-inactive navigation heads need not receive language gradients.')


def gradient_audit(engine,row,window=128):
    prefix,gold=encode_case(row)
    sequence=prefix+gold
    boundary=len(sequence)-1-window
    if boundary<1:
        raise ValueError('insufficient real input for the declared gradient window')
    with request_branch(engine) as table:
        _,state,pending=consume_prompt(engine.model,table,prefix[:boundary])
        # Complete the observed boundary transition before executing the tail.
        table.run_credit(np.array([pending,sequence[boundary]],dtype=np.uint8),max_window=1)
        tail=np.frombuffer(sequence[boundary:],dtype=np.uint8)
        _,prob,_=table.run_credit(tail,max_window=window)
        device=state.device
        ids=torch.tensor(tail[:-1].astype(np.int64),device=device)[None]
        target=torch.tensor(tail[1:].astype(np.int64),device=device)
        logits,_,_,_,_=engine.model.predict_observed(ids,state,probability_tensor(engine.model,prob))
        positions=torch.arange(boundary+1,len(sequence),device=device)
        mask=positions>=len(prefix)
        loss=F.cross_entropy(logits[0,mask].float(),target[mask])
        named=list(engine.model.named_parameters())
        gradients=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
        norms={n:None if g is None else float(g.float().norm()) for (n,_),g in zip(named,gradients)}
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError('nonfinite task gradient')
    layers=[]
    for i in range(engine.model.base.num_layers):
        stem=f'body.base.layers.{i}.'
        selected={n:v for n,v in norms.items() if n.startswith(stem)}
        if not selected:
            raise AssertionError('unknown restored parameter namespace')
        gdn={n:v for n,v in selected.items() if '.mlp' not in n}
        mlp={n:v for n,v in selected.items() if '.mlp.' in n}
        layers.append(dict(layer=i+1,gdn_gradient_norm=sum((v or 0)**2 for v in gdn.values())**.5,
            mlp_gradient_norm=sum((v or 0)**2 for v in mlp.values())**.5))
    assert all(x['gdn_gradient_norm']>0 and x['mlp_gradient_norm']>0 for x in layers)
    return dict(window_bytes=window,answer_nll=float(loss.detach()),optimizer_steps=0,layers=layers,
        embedding_decoder_tied_gradient_norm=norms['body.base.source.embedding.weight'])


def fingerprint(model,table):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode())
        if torch.is_tensor(value):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode())
    for name in FIELDS:
        h.update(memoryview(getattr(table,name)).cast('B'))
    h.update(str((int(table.history),table.events)).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','official','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--length',type=int,default=512,help='Original generator token budget, not this body byte length')
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--max-new-bytes',type=int,default=128)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    engine,_,_,source=build(args.checkpoint)
    manifest=json.loads(args.manifest.read_text())
    data=next(d for d in manifest['data'] if d['task']=='niah_single_1' and d['length']==args.length)
    assert sha256(data['path'])==data['sha256']
    rows=[json.loads(x) for x in Path(data['path']).read_text().splitlines()][:args.limit]
    spec=importlib.util.spec_from_file_location('ruler_official',args.official/'scripts/eval/synthetic/constants.py')
    scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
    report=dict(status='running',source=source,scope=__doc__,original_generator_budget=args.length,
        manifest_sha256=sha256(args.manifest),data_sha256=data['sha256'],
        official_revision=manifest['official_revision'],optimizer_steps=0,samples=[])
    path=args.output/'report.json'
    def write(): path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    write()
    start=time.time()
    initial=fingerprint(engine.model,engine.table)
    try:
        report['parameter_audit']=full_parameter_audit(engine)
        report['gradient_audit']=gradient_audit(engine,rows[0])
        write()
        print('GRADIENT',json.dumps(report['gradient_audit']),flush=True)
        for row in rows:
            result=run_case(engine,row,args.max_new_bytes)
            report['samples'].append(result)
            write()
            print('CASE',json.dumps(result),flush=True)
        report.update(status='complete',seconds=time.time()-start,
            ruler_score=scoring.TASKS['niah']['metric_fn']([x['prediction'] for x in report['samples']],
                                                       [x['references'] for x in report['samples']]),
            first_number_exact_count=sum(x['first_number_exact'] for x in report['samples']),
            model_and_tables_unchanged=fingerprint(engine.model,engine.table)==initial,
            source_checkpoint_unchanged=sha256(args.checkpoint)==source['sha256'])
        assert report['model_and_tables_unchanged'] and report['source_checkpoint_unchanged']
        # Exact full-individual copy, including original optimizer/world/RNG.
        selected=args.output/'selected';selected.mkdir()
        shutil.copy2(args.checkpoint,selected/'last.pt.gz')
        assert sha256(selected/'last.pt.gz')==source['sha256']
        report['selected_checkpoint']=str(selected/'last.pt.gz')
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        write()
        print('COMPLETE',json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));write();raise


if __name__=='__main__':
    main()
