"""NumPy-only address geometry and retrospective fact-window rank diagnosis.

No model loading, no CUDA, no body/optimizer/qualification update. Current
encoder reencodes actually observed features; this is not historical replay
under the historical encoder or a language-quality intervention.
"""
import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def covariance_model(x, shrink=True):
    """Classic LW finite-sample plug-in, treated as descriptive for this stream."""
    n,d=x.shape
    mean=x.mean(0)
    centered=x-mean
    c=centered.T@centered/n
    mu=float(np.trace(c)/d)
    delta=float(np.square(c-mu*np.eye(d)).sum())
    beta=max(0.,float(np.mean(np.square(np.sum(centered**2,axis=1)))-np.square(c).sum())/n)
    strength=min(1.,beta/delta) if delta>0 else 1.
    actual=strength if shrink else 0.
    cleaned=(1-actual)*c+actual*mu*np.eye(d)
    values,vectors=np.linalg.eigh(cleaned)
    floor=max(mu*1e-8,1e-30)
    transform=(vectors/np.sqrt(np.maximum(values,floor)))@vectors.T
    return dict(mean=mean,transform=transform,covariance=c,cleaned=cleaned,
                strength=strength,eigenvalues=np.maximum(values,0))


def spectral_summary(x):
    gram=(x@x.T if x.shape[0]<x.shape[1] else x.T@x)/len(x)
    values=np.maximum(np.linalg.eigvalsh(gram),0)[::-1]
    p=values/max(float(values.sum()),1e-30)
    nonzero=p[p>1e-15]
    return dict(rank=int(np.linalg.matrix_rank(x)),top1_energy=float(p[0]),
        top4_energy=float(p[:4].sum()),participation_rank=float(1/max(float(p@p),1e-30)),
        entropy_rank=float(np.exp(-np.sum(nonzero*np.log(nonzero)))))


def transform_keys(keys,queries,method):
    if method=='cosine':return normalize(keys),normalize(queries),None
    model=covariance_model(keys,shrink=method=='lw')
    k,q=keys-model['mean'],queries-model['mean']
    if method in ('lw','unshrunk_whiten'):
        k,q=k@model['transform'],q@model['transform']
    return normalize(k),normalize(q),model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--training-report',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads((args.snapshot/'snapshot.json').read_text())
    arrays=np.load(args.snapshot/'addresses.npz')
    training=json.loads(args.training_report.read_text())
    rows=[json.loads(l) for l in Path('results/cgb_ruler_adaptation_20261109/data_train/512/niah_single_1/test.jsonl').read_text().splitlines()]
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('E:/cgb_gdn340_probe/model/tokenizer.json')
    episodes=[]
    completed=meta['driver']['completed']
    for step,row in zip(training['steps'][:completed],rows):
        text=row['input']+row.get('answer_prefix','')
        encoded=tokenizer.encode(text,add_special_tokens=False)
        answer=row['outputs'][0]
        start=text.find(answer)
        if start<0 or text.find(answer,start+1)>=0:raise ValueError('nonunique fact')
        span=[i for i,(a,b) in enumerate(encoded.offsets) if a<start+len(answer) and b>start]
        if not span:raise ValueError('fact token mapping missing')
        prompt_len=step['input_events']-step['supervised_arrivals']-128
        delta=prompt_len-len(encoded.ids)
        if step['gap']==0 and delta!=0:raise ValueError('native tokenizer disagrees with actual short prompt')
        if step['gap'] and not 500<=delta<=525:raise ValueError('interference length reconstruction differs')
        question=text.rfind('\nWhat is ')
        if question<0:raise ValueError('official question boundary missing')
        qpos=next(i for i,(a,b) in enumerate(encoded.offsets) if b>question)
        incoming=step['seen']-step['input_events']
        episodes.append(dict(example=step['example'],fact_start=incoming+min(span),
            fact_stop=incoming+max(span)+1,question_start=incoming+qpos+delta,
            question_stop=incoming+prompt_len))
    report=dict(scope=__doc__,snapshot=str(args.snapshot),checkpoint_sha256=meta['checkpoint_sha256'],
        completed_adaptation_examples=completed,model_updates=0,layers={},
        assumptions='IID LW formula is descriptive: dictionary entries are serially correlated. No random-matrix noise edge assumed.',
        rank_probe='current encoder on real source-index64/history-index127 features observed during question, before answer arrival; only already committed keys eligible',
        language_benefit_tested=False)
    methods=('cosine','centered_cosine','unshrunk_whiten','lw')
    for layer,cfg in meta['layers'].items():
        prefix=f'l{layer}_'
        occupied=arrays[prefix+'occupied'].astype(bool)
        ids=np.flatnonzero(occupied)
        raw=arrays[prefix+'source'][occupied].astype(np.float64)
        weight=arrays[prefix+'encoder_weight'].astype(np.float64)
        bias=arrays[prefix+'encoder_bias'].astype(np.float64)
        keys=raw@weight.T+bias
        histories=arrays[prefix+'history'].astype(np.float64)
        query_features=np.concatenate((raw,histories))
        birth=arrays[prefix+'birth'][occupied].astype(np.int64)+cfg['coordinate_offset']
        times=np.concatenate((birth,arrays[prefix+'history_clock']-1+cfg['coordinate_offset']))
        queries=query_features@weight.T+bias
        model=covariance_model(keys)
        unit=normalize(keys)
        off=(unit@unit.T)[~np.eye(len(keys),dtype=bool)]
        entry=dict(occupied=len(keys),dimensions=keys.shape[1],
            maximum_centered_rank=len(keys)-1,uncentered=spectral_summary(keys),
            centered=spectral_summary(keys-keys.mean(0)),
            mean_direction_energy=float(np.sum(keys.mean(0)**2)/np.mean(np.sum(keys**2,axis=1))),
            key_pair_cosine=dict(median=float(np.median(off)),p90=float(np.quantile(off,.9))),
            lw_strength=model['strength'],cleaned_condition=float(model['eigenvalues'][-1]/max(model['eigenvalues'][0],1e-30)),
            history_retrieval={},fact_query_count=0,fact_rank_cases=[],fact_summary={})
        entry['raw_source_spectrum']=spectral_summary(raw)
        entry['raw_source_mean_energy']=float(np.sum(raw.mean(0)**2)/np.mean(np.sum(raw**2,axis=1)))
        if prefix+'reference_encoder_weight' in arrays:
            reference_keys=raw@arrays[prefix+'reference_encoder_weight'].T+arrays[prefix+'reference_encoder_bias']
            entry['same_features_pre_adaptation_encoder']=dict(
                uncentered=spectral_summary(reference_keys),
                centered=spectral_summary(reference_keys-reference_keys.mean(0)),
                mean_direction_energy=float(np.sum(reference_keys.mean(0)**2)/np.mean(np.sum(reference_keys**2,axis=1))))
        h=histories@weight.T+bias
        for method in methods:
            k,q,_=transform_keys(keys,h,method)
            scores=q@k.T
            winners=scores.argmax(1)
            counts=np.bincount(winners,minlength=len(keys))
            entry['history_retrieval'][method]=dict(top1_unique=int(np.count_nonzero(counts)),
                largest_slot_share=float(counts.max()/len(h)),
                top1_cosine_median=float(np.median(scores.max(1))))
        # The live reader's selected identities are independently exported via
        # its actual code, including learned preferences and null competition.
        entry['actual_reader']=dict(top1_unique=int(len(np.unique(arrays[prefix+'baseline_slots'][:,0]))),
            null_weight_mean=float(np.mean(1-arrays[prefix+'baseline_weights'].sum(1))))
        for feature,time in zip(queries,times):
            episode=next((e for e in episodes if e['question_start']<=time<e['question_stop']),None)
            if episode is None:continue
            available=birth+64<=time
            relevant=(birth-64<=episode['fact_start'])&(birth+64>=episode['fact_stop'])&available
            if available.sum()<2 or not relevant.any():continue
            item=dict(example=episode['example'],query_time=int(time),available=int(available.sum()),
                target_slots=ids[relevant].tolist(),age_min=int(time-birth[relevant].max()),methods={})
            item['fact_inside_first65_of_target_replay']=bool(
                np.any((birth[relevant]+1)>=episode['fact_stop']))
            for method in methods:
                k,q,_=transform_keys(keys[available],feature[None],method)
                scores=(q@k.T)[0]
                order=np.argsort(-scores,kind='stable')
                targets=np.flatnonzero(relevant[available])
                rank=min(int(np.flatnonzero(order==i)[0])+1 for i in targets)
                item['methods'][method]=dict(rank=rank,top1=rank==1,top2=rank<=2)
            entry['fact_rank_cases'].append(item)
        entry['fact_query_count']=len(entry['fact_rank_cases'])
        if entry['fact_query_count']:
            for method in methods:
                cases=[x['methods'][method] for x in entry['fact_rank_cases']]
                entry['fact_summary'][method]=dict(top1=sum(x['top1'] for x in cases),
                    top2=sum(x['top2'] for x in cases),mean_rank=float(np.mean([x['rank'] for x in cases])))
        report['layers'][str(cfg['human_layer'])]=entry
    path=args.snapshot/'spectrum_report.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
