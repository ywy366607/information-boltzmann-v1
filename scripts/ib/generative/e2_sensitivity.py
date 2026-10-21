"""Matched causal sensitivity audit of an immutable historical E2 individual.

Uses its original code and complete physical state. No optimizer/BPTT or new
individual is started. This is a frozen-parameter diagnostic, not a learning
capability or a replacement for the primary continuous-individual protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''): h.update(chunk)
    return h.hexdigest()


def schedule(n,window,minimum,maximum,seed):
    rng=np.random.default_rng(seed);result=[]
    for start in range(0,n,window):
        mean=float(np.exp(rng.uniform(np.log(minimum),np.log(maximum)))) if 0<minimum<maximum else maximum
        result.extend(rng.geometric(1/max(mean,1),size=min(window,n-start)).tolist())
    return np.asarray(result,int)


def exposure_tokens(tokens, offset, words, lag=0):
    """Keep targets fixed; a lag uses strictly earlier, never future, inputs."""
    if words <= 0 or lag < 0 or offset-lag-1 < 0 or offset+words > len(tokens):
        raise ValueError('Invalid exposure range or input lag')
    targets=np.asarray(tokens[offset:offset+words],np.int64)
    inputs=np.asarray(tokens[offset-lag-1:offset-lag-1+words],np.int64)
    return inputs,targets


def main(args):
    start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=True)
    if any(k.startswith('ZS_') or k in ('INTRINSIC_KAPPA','INTRINSIC_PATTERN') for k in os.environ):
        raise RuntimeError('Unrequested zero-shot environment overrides must be cleared')
    os.environ['FLY_REUSE_TRANSMISSION']='0'
    sys.path.insert(0,str(args.source_root))
    import torch
    import torch.nn.functional as F
    import pandas as pd
    torch.set_num_threads(4)
    if torch.cuda.memory_allocated(): raise RuntimeError('Start audit in a fresh process')
    loader=args.source_root/'scripts/ib/fly_interface/world_eval_e2.py'
    scope={'__name__':'e2_historical_loader'}
    exec(compile(loader.read_text(encoding='utf-8').split('def evaluate(fork):')[0],str(loader),'exec'),scope)
    if scope['ROOT'].resolve()!=args.source_root.resolve(): raise ValueError('Historical loader root mismatch')
    checkpoint=args.checkpoint/'last.pt';original_hash=digest(checkpoint)
    model,cfg,L=scope['load'](args.checkpoint)
    fl,move_state=scope['fl'],scope['move_state']
    print(json.dumps({'stage':'loaded','allocated_mib':torch.cuda.memory_allocated()/2**20}),flush=True)
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    # Restore a historical non-tensor flag omitted by the old convenience loader.
    # It selects the site of learning modulation, not a new intervention.
    if cfg.get('dopamine_modulation'):
        model.dopamine_site=cfg.get('dopamine_site') or 'post'
    restored=dict(model.state_dict());parameter_mismatches=[]
    # Fast-edge arrays are intentionally non-persistent module buffers; the
    # trainer stores them explicitly alongside the ordinary state_dict.
    for key in ('edge_weight_e','edge_weight_i'):
        restored[key]=getattr(model,key)
    for key,value in saved['model'].items():
        if key not in restored or not torch.equal(restored[key].detach().cpu(),value):
            parameter_mismatches.append(key)
    if parameter_mismatches:
        detail={}
        for key in parameter_mismatches:
            if key in restored:
                a=restored[key].detach().cpu();b=saved['model'][key]
                detail[key]={'shape':list(a.shape),'dtype':[str(a.dtype),str(b.dtype)],
                    'changed':int((a!=b).sum()),'absmax':float((a.float()-b.float()).abs().max())}
        (args.out/'restore_failure.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
        raise RuntimeError('Incomplete historical parameter restore: '+str(detail))
    graph=args.source_root/cfg['graph'];ann=pd.read_feather(graph.parent/'body-annotations.feather').drop_duplicates('bodyId').set_index('bodyId')
    ids=np.load(graph,allow_pickle=False)['neuron_body_ids'];ann=ann.reindex(ids)
    cls,typ=ann['class'].astype(str).to_numpy(),ann['type'].astype(str).to_numpy()
    sensory=np.zeros(len(ids),dtype=bool)
    sensory[model.topographic_writer.injection_index.detach().cpu().numpy()]=True
    sensory_comp_edges=(int(sensory[model.comp_pre.detach().cpu().numpy()].sum())
                        if hasattr(model,'comp_pre') else 0)
    if 'sensory_cut' in args.arms and sensory_comp_edges:
        raise ValueError('Sensory cut would miss separate compartment edges; extend the intervention explicitly')
    mb_pre=np.isin(cls,['Kenyon_Cell','ALPN','MBON'])|np.isin(typ,['APL','DPM'])
    mb_post=np.isin(cls,['Kenyon_Cell','MBON'])|np.isin(typ,['APL','DPM'])
    cx=cls=='CX';mb_cells=np.isin(cls,['Kenyon_Cell','MBON'])|np.isin(typ,['APL','DPM'])
    regions={k:torch.as_tensor(np.flatnonzero(v),device='cuda') for k,v in [('mb',mb_cells),('cx',cx)]}
    if not len(regions['cx']): raise ValueError('No annotated CX cells in graph')
    val_path=args.source_root/cfg['data']/'validation.npy';val=np.load(val_path,mmap_mode='r')
    inputs,targets=exposure_tokens(val,args.offset,args.words)
    lag_inputs,_=exposure_tokens(val,args.offset,args.words,lag=32)
    ticks=schedule(len(inputs),int(cfg['window']),float(cfg.get('stream_pace_min') or 0),
                   float(cfg.get('stream_pace_mean') or cfg['stream_ticks']),20261021)
    np.savez(args.out/'exposure.npz',inputs=inputs,lag_inputs=lag_inputs,targets=targets,ticks=ticks)
    bias_logits=model.decoder.bias.detach().float()
    bias_nll=(-torch.log_softmax(bias_logits,0)[torch.as_tensor(targets,device='cuda')]).cpu().numpy()
    np.savez(args.out/'bias_baseline.npz',nll=bias_nll)
    constant_info=None
    if args.constant_readout is not None:
        with np.load(args.constant_readout,allow_pickle=False) as calibration:
            constant=np.asarray(calibration['readout']).mean(axis=0,keepdims=True)
        vector=torch.as_tensor(constant,device='cuda',dtype=model.decoder.weight.dtype)
        logits=model.decoder(model.read_norm(vector)).float()[0]
        constant_nll=(-torch.log_softmax(logits,0)[torch.as_tensor(targets,device='cuda')]).cpu().numpy()
        np.savez(args.out/'constant_baseline.npz',nll=constant_nll,readout=constant)
        constant_info={'mean_nll':float(constant_nll.mean()),'calibration':str(args.constant_readout),
                       'calibration_sha256':digest(args.constant_readout),'method':'Mean calibration readout, then unchanged read_norm and decoder'}
    edge_names=['edge_weight_e','edge_weight_i']+(['comp_weight'] if hasattr(model,'comp_weight') else [])
    originals={key:getattr(model,key).detach().cpu().clone() for key in edge_names}
    opts=dict(base_rates=model.get_decay_rates(),thresholds=model.get_thresholds(),conductance_gains=model.get_conductance_gains(),
              alif_params=model.get_alif_params(),stp_params=model.get_stp_params())
    reference=None;rows={}
    for arm in args.arms.split(','):
        for name,value in originals.items(): getattr(model,name).copy_(value.to('cuda'))
        changed={}
        if arm not in ('real','sham','input_lag'):
            for tag in ('e','i'):
                pre=getattr(model,'edge_pre_'+tag).long();post=getattr(model,'edge_post_'+tag).long()
                if arm=='all_cut': mask=torch.ones_like(pre,dtype=torch.bool)
                elif arm=='cx_cut': mask=torch.as_tensor(cx,device='cuda')[pre]
                elif arm=='mb_cut': mask=torch.as_tensor(mb_pre,device='cuda')[pre]&torch.as_tensor(mb_post,device='cuda')[post]
                elif arm=='sensory_cut': mask=torch.as_tensor(sensory,device='cuda')[pre]
                else: raise ValueError(arm)
                w=getattr(model,'edge_weight_'+tag);changed[tag]=int(((w!=0)&mask).sum());w[mask]=0
            if 'comp_weight' in originals and arm in ('mb_cut','all_cut'):
                changed['compartment']=int((model.comp_weight!=0).sum());model.comp_weight.zero_()
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state(saved['rng_cuda'])
        s=move_state(fl.FlyPhysicalState(**L['physical']),'cuda')
        scores=[];features=[];activity={k:[] for k in regions};t0=time.perf_counter()
        with torch.no_grad():
            branch_inputs=lag_inputs if arm=='input_lag' else inputs
            for i,(token,target,dt) in enumerate(zip(branch_inputs,targets,ticks)):
                if i%int(cfg['window'])==0 and getattr(model,'dopamine_sensitivity',None) is not None:
                    opts['dopamine_s_class']=fl.dopamine_class_sensitivity(model,s.h)
                s,latent=fl.advance_fly_token_stream(model,s,torch.tensor([int(token)],device='cuda'),
                    ticks=int(dt),hold_input=bool(cfg.get('hold_input',True)),
                    writer_baseline_clock=L.get('writer_baseline_clock','physical'),**opts)
                latent=fl.read_fly_motor(model,s)
                score=F.cross_entropy(model.decoder(model.read_norm(latent)).float(),torch.tensor([int(target)],device='cuda'))
                scores.append(float(score));features.append(latent.detach().cpu().numpy()[0])
                for k,indices in regions.items(): activity[k].append(float((s.ring[0][0].index_select(0,indices)>0).float().mean()))
                s=fl.expect_next(model,s,latent)
                if (i+1)%32==0: print(json.dumps({'stage':arm,'words':i+1,'seconds':time.perf_counter()-t0}),flush=True)
        scores,features=np.asarray(scores),np.asarray(features)
        arrays={'nll':scores,'readout':features,**{k+'_terminal_tick_activity':np.asarray(v) for k,v in activity.items()}}
        final={k:(tuple(t.cpu().numpy() for t in v) if k=='ring' else v.cpu().numpy()) for k,v in s.state_dict().items()}
        row={'words':len(scores),'physical_ticks':int(ticks.sum()),'mean_nll':float(scores.mean()),
             'changed_nonzero_edges':changed,'seconds':time.perf_counter()-t0,
             'terminal_tick_activity':{k:float(np.mean(v)) for k,v in activity.items()}}
        if reference is None:
            reference=(scores.copy(),features.copy(),final)
        else:
            delta=scores-reference[0]
            row.update(paired_nll_delta=float(delta.mean()),paired_nll_delta_second_half=float(delta[len(delta)//2:].mean()),
                       max_nll_difference=float(np.abs(delta).max()),readout_rms_difference=float(np.sqrt(np.mean((features-reference[1])**2))))
            state_diff={}
            for k,v in final.items():
                old=reference[2][k]
                if k=='ring': difference=max(float(np.abs(a-b).max()) if a.size else 0. for a,b in zip(v,old))
                else: difference=float(np.abs(v.astype(float)-old.astype(float)).max()) if v.size else 0.
                state_diff[k]=difference
            row['final_physical_absmax_difference']=state_diff
        rows[arm]=row
        np.savez_compressed(args.out/(arm+'.npz'),**arrays)
        (args.out/(arm+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps({'completed':arm,**{k:v for k,v in row.items() if k!='final_physical_absmax_difference'}},indent=2),flush=True)
        del s,features,final;gc.collect();torch.cuda.empty_cache()
    sham=rows.get('sham',{})
    report={'scope':'Historical frozen-parameter acute causal diagnostic; local physical plasticity runs, no optimizer/BPTT, not E2 functional acceptance',
      'source_root':str(args.source_root),'checkpoint':str(checkpoint),'checkpoint_sha256':original_hash,
      'checkpoint_unchanged':digest(checkpoint)==original_hash,'historical_train_tokens':saved.get('bptt_train_tokens'),
      'saved_model_tensors_restored_exactly':True,'restored_dopamine_site':getattr(model,'dopamine_site',None),
      'decoder_bias_only_nll':float(bias_nll.mean()),
      'constant_readout_baseline':constant_info,'input_lag_tokens':32 if 'input_lag' in rows else None,
      'sensory_injection_cells':int(sensory.sum()),'sensory_compartment_edges':sensory_comp_edges,
      'original_train_cursor':saved['train_cursor'],'offset':args.offset,'exposure_sha256':digest(args.out/'exposure.npz'),
      'dataset':str(val_path),'schedule':'Window log-uniform mean, per-token geometric stopping; shared new diagnostic clock',
      'nll_read':'Last physical tick before next token arrival; not historical training weighted-hazard objective',
      'source_hashes':{str(p):digest(p) for p in (Path(__file__),loader,args.source_root/'information_boltzmann/core/fly_bptt_learning.py')},
      'arms':rows,'sham_pass':bool(sham and sham['max_nll_difference']<1e-4 and max(sham['final_physical_absmax_difference'].values())<1e-4),
      'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'seconds':time.perf_counter()-start}
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--words',type=int,default=128)
    p.add_argument('--offset',type=int,default=60000);p.add_argument('--arms',default='real,mb_cut,cx_cut,all_cut,sham')
    p.add_argument('--constant-readout',type=Path,default=None)
    args=p.parse_args()
    import torch
    with torch.no_grad(): main(args)
