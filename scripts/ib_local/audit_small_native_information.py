"""Locate content attenuation inside actual retained native computations.

No body updates. The researcher swaps equal-length historical observations
while holding the present S/table fixed. Digit locations label ONLY this audit.
Anagram interventions distinguish sequence sensitivity from digit frequencies.
Frozen-feature byte decoders are diagnostic readers, never installed in body.
"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.audit_small_native_content import collect_alternative
from scripts.ib_local.audit_small_native_failure import provenance
from scripts.ib_local.language_action import peek
from scripts.ib_local.pretrained_gdn_experience import LocalExperience
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import (
    consume_prompt,encode_case,fingerprint,probability_tensor,request_branch,sha256,
)


def distance(a,b):
    delta=(a-b).double()
    return dict(rms=float(delta.square().mean().sqrt()),
        relative_norm=float(delta.norm()/((a.double().norm()+b.double().norm())/2).clamp_min(1e-12)))


@contextmanager
def capture_reads(bank):
    original=[];captures={}
    for index,m in enumerate(bank.memories):
        old=m.retrieve_context;original.append((m,old))
        def read(query,block,index=index,m=m,old=old):
            result=old(query,block)
            pr=m.position_read
            if pr is not None:
                records={};offset=0
                for j,(slot,length) in enumerate(zip(pr['slots'].tolist(),pr['lengths'])):
                    response=m.replayed_by_slot[slot][0].detach().cpu()
                    encoded=m.encoder(response).detach()
                    mass=pr['weights'][0,offset:offset+length]
                    weights=mass/mass.sum().clamp_min(1e-30)
                    pooled=pr['conditional_values'][j,0].detach().cpu()
                    injection=m.inlet.projection(pooled.to(query.device)).detach().cpu()
                    records[slot]=dict(inputs=m.capsules[slot].inputs[0].detach().cpu(),
                        response=response,encoded=encoded,weights=weights.detach().cpu(),
                        pooled=pooled,injection=injection)
                    offset+=length
                captures[index]=records
            return result
        m.retrieve_context=read
    try:yield captures
    finally:
        for m,old in original:m.retrieve_context=old


@torch.no_grad()
def case_measure(engine,row,alternative,kind,*,collect_features):
    model=engine.model;bank=model.local_replay
    alternatives=collect_alternative(engine,row,alternative)
    prefix,_=encode_case(row);value=row['outputs'][0]
    needle=prefix.index(value.encode())
    features=[]
    with request_branch(engine) as table,provenance(bank) as metadata:
        starts=[int(m.memory.clock) for m in bank.memories]
        _,state,pending=consume_prompt(model,table,prefix,128)
        table.run_credit(np.array([pending,32],dtype=np.uint8),max_window=1)
        logp=probability_tensor(model,peek(table,32)[None])
        ids=torch.tensor([[32]],device=engine.device)
        with capture_reads(bank) as captured:
            factual=model.predict(ids,state,logp)[0][0,0]
            first=copy.deepcopy(captured)
            pairs=[]
            for i,m in enumerate(bank.memories):
                for slot,p in metadata[i].items():
                    lo=p['start']-starts[i];hi=p['end']-starts[i]
                    if collect_features and 0<=lo<hi<=len(prefix):
                        record=first[i][slot]
                        labels=torch.tensor(list(prefix[lo:hi]),dtype=torch.long)
                        target_mask=(torch.arange(lo,hi)>=needle)&(torch.arange(lo,hi)<needle+len(value))
                        features.append(dict(case=row['index'],layer=i+1,labels=labels,
                            target_mask=target_mask,
                            **{key:record[key] for key in ('inputs','response','encoded')}))
                    if (lo,hi) in alternatives[i]:
                        c=alternatives[i][lo,hi]
                        pairs.append((i,slot,lo,hi))
                        m.capsules[slot]=LocalExperience(c.inputs,c.boundary,
                            int(m.memory.read_credit_version[slot]))
            changed=model.predict(ids,state,logp)[0][0,0]
            second=copy.deepcopy(captured)
        layers=[]
        for i,slot,lo,hi in pairs:
            f,c=first[i][slot],second[i][slot]
            digit=(torch.arange(lo,hi)>=needle)&(torch.arange(lo,hi)<needle+len(value))
            w=f['weights'];wc=c['weights'];ef=f['encoded'];ec=c['encoded']
            # Exact decomposition into changed values and changed routing.
            content=w@(ef-ec);routing=(w-wc)@ec
            pooled=f['pooled']-c['pooled']
            layers.append(dict(layer=i+1,slot=slot,span=[lo,hi],
                digits=int(digit.sum()),positions=hi-lo,
                stages={key:distance(f[key],c[key]) for key in
                        ('inputs','response','encoded','pooled','injection')},
                within_record_digit_mass=float(w[digit].sum()),
                uniform_digit_mass=float(digit.float().mean()),
                effective_positions=float(w.square().sum().reciprocal()),
                uniform_pool_delta_rms=float((ef.mean(0)-ec.mean(0)).square().mean().sqrt()),
                value_change_rms=float(content.square().mean().sqrt()),
                routing_change_rms=float(routing.square().mean().sqrt()),
                decomposition_max_error=float((pooled-content-routing).abs().max())))
        a,b=ord(value[0]),ord(alternative[0])
        result=dict(case=row['index'],kind=kind,value=value,alternative=alternative,layers=layers,
            factual_digit_nll=float(-factual.log_softmax(-1)[a]),
            changed_digit_nll=float(-changed.log_softmax(-1)[a]),
            stored_value_contrast=float((factual[a]-factual[b])-(changed[a]-changed[b])),
            max_logit_change=float((factual-changed).abs().max()))
    return result,features


def frozen_decoders(features,mode='all'):
    """4 group-held-out folds; fit real observed byte identity, not answers.

    Training rows never include the held-out question, including its distractor
    positions. Fixed trace-scaled ridge (1e-3), fitted intercept, no test tuning.
    This tests decodability at retained positions, not retrieval/task ability.
    """
    cases=sorted({f['case'] for f in features});reports=[]
    for layer in range(1,5):
        rows=[f for f in features if f['layer']==layer]
        for key in ('inputs','response','encoded'):
            total=correct=digits=digit_correct=0;folds=[]
            for fold in range(4):
                held={c for j,c in enumerate(cases) if j%4==fold}
                train=[r for r in rows if r['case'] not in held]
                test=[r for r in rows if r['case'] in held]
                x=torch.cat([r[key] for r in train]).double()
                labels=torch.cat([r['labels'] for r in train])
                present=torch.bincount(labels,minlength=256)>0
                if mode=='digits_only':
                    chosen=(labels>=48)&(labels<=57)
                    x=x[chosen];labels=labels[chosen]
                y=F.one_hot(labels,256).double()
                if mode=='balanced':
                    frequency=torch.bincount(labels,minlength=256).double()
                    weights=1/frequency[labels]
                else:weights=x.new_ones(len(x))
                normalized=weights/weights.sum()
                center=(x*normalized[:,None]).sum(0)
                target=(y*normalized[:,None]).sum(0)
                centered=x-center;gram=centered.T@centered
                gram=centered.T@(weights[:,None]*centered)
                penalty=1e-3*gram.diag().mean().clamp_min(1e-12)
                weight=torch.linalg.solve(gram+penalty*torch.eye(gram.shape[0]),
                                          centered.T@(weights[:,None]*(y-target)))
                tx=torch.cat([r[key] for r in test]).double()
                ty=torch.cat([r['labels'] for r in test])
                mask=torch.cat([r['target_mask'] for r in test])
                if mode=='digits_only':
                    selected=(ty>=48)&(ty<=57)
                    tx=tx[selected];ty=ty[selected];mask=mask[selected]
                prediction=((tx-center)@weight+target).argmax(-1)
                hit=prediction==ty
                total+=len(hit);correct+=int(hit.sum())
                digits+=int(mask.sum());digit_correct+=int(hit[mask].sum())
                folds.append(dict(fold=fold,held_cases=sorted(held),
                    all_accuracy=float(hit.float().mean()),digit_correct=int(hit[mask].sum()),
                    digits=int(mask.sum()),target_digits_unseen_in_train=int((~present[ty[mask]]).sum())))
            reports.append(dict(layer=layer,stage=key,reader_mode=mode,all_correct=correct,all_positions=total,
                all_accuracy=correct/total,digit_correct=digit_correct,digits=digits,
                digit_accuracy=digit_correct/digits,folds=folds))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.device=='cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    engine,_,_,_=build(args.checkpoint,device=args.device)
    assert engine.model.local_replay.schema==3
    before=fingerprint(engine.model,engine.table);checksum=sha256(args.checkpoint)
    rows=[json.loads(line) for line in args.data.read_text().splitlines()][:8]
    records=[];features=[];start=time.time()
    for row in rows:
        value=row['outputs'][0]
        other=next(r['outputs'][0] for r in rows if r['outputs'][0][0]!=value[0])
        # Cyclic permutation preserves the digit multiset and changes first
        # digit; old records come from actual counterfactual observations.
        rotate=next(value[j:]+value[:j] for j in range(1,len(value)) if value[j]!=value[0])
        for kind,alternative in (('different_digits',other),('same_digits_different_order',rotate)):
            record,observed=case_measure(engine,row,alternative,kind,
                                        collect_features=kind=='different_digits')
            records.append(record);features.extend(observed)
            print(kind,row['index'],record['stored_value_contrast'],flush=True)
    decoders=[row for mode in ('all','balanced','digits_only')
              for row in frozen_decoders(features,mode)]
    feature_path=args.output.with_suffix('.features.pt')
    if feature_path.exists():raise FileExistsError(feature_path)
    torch.save(features,feature_path)
    result=dict(status='complete',body_optimizer_updates=0,
        probe='Frozen-feature real byte diagnostic; readers not installed in body',
        checkpoint=str(args.checkpoint),checkpoint_sha256=checksum,device=args.device,
        records=records,frozen_byte_decoders=decoders,seconds=time.time()-start,
        body_unchanged=fingerprint(engine.model,engine.table)==before,
        checkpoint_unchanged=sha256(args.checkpoint)==checksum,
        source_files={str(Path(__file__)):sha256(Path(__file__))},
        frozen_features_path=str(feature_path),frozen_features_sha256=sha256(feature_path))
    assert result['body_unchanged'] and result['checkpoint_unchanged']
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
