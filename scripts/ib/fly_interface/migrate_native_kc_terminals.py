"""Zero-exposure KC site extension of one complete continuous individual."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.core.fly_kc_terminals import (
    attach_kc_terminals,KC_TERMINAL_PARAMETER_NAMES,KC_TERMINAL_ID_NAMES,
)
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.native_causal_audit import sha256


def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    return a==b


def migrate(checkpoint,geometry,output):
    if output.exists():raise ValueError('Use a new isolated candidate, never overwrite a continuing individual')
    if shutil.disk_usage(output.parent).free<2*checkpoint.stat().st_size+1024**3:
        raise OSError('Need a full candidate and atomic replacement reserve')
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if source['config'].get('kc_terminal'):raise ValueError('Already has local terminal state')
    with np.load(geometry,allow_pickle=False) as packed:
        owner=torch.from_numpy(packed['owner'].copy()).long()
        site_owner=torch.from_numpy(packed['site_owner'].copy()).long()
        kc=torch.from_numpy(packed['kc_index'].copy()).long()
        edge_kc=torch.from_numpy(packed['comp_edge_kc'].copy()).long()
    holder=torch.nn.Module();holder.register_buffer('edge_pre_e',torch.empty(0,dtype=torch.long),persistent=False)
    manifest=attach_kc_terminals(holder,geometry,metadata_only=True)
    additions={n:getattr(holder,n).detach() for n in KC_TERMINAL_PARAMETER_NAMES+KC_TERMINAL_ID_NAMES}
    old=source['learner'];physical=old['physical']
    local={
        'kc_terminal_gaba':physical['mb_terminal_gaba'].index_select(1,owner),
        'kc_terminal_5ht':physical['mb_dpm_concentration'].index_select(1,owner),
        'kc_terminal_receptor':physical['mb_dpm_receptor'].index_select(1,owner),
        'kc_terminal_packets':torch.cat([p.index_select(1,kc).index_select(1,site_owner) for p in physical['ring']],0),
        'kc_terminal_eligibility':physical['comp_trace'].index_select(1,edge_kc),
        'kc_terminal_eligibility_credit':physical['credit_comp_trace'].index_select(1,edge_kc)}
    states={name:old['optimizer']['state'].get(i) for g in old['optimizer']['param_groups'] for name,i in zip(g['parameter_names'],g['params'])}
    optimizer={'state':{},'param_groups':[]};offset=0
    for group in old['optimizer']['param_groups']:
        names=list(group['parameter_names'])
        if 'apl_log_axial' in names:names+=list(KC_TERMINAL_PARAMETER_NAMES)
        indices=list(range(offset,offset+len(names)));offset+=len(names)
        optimizer['param_groups'].append({**group,'params':indices,'parameter_names':names})
        optimizer['state'].update({i:states[n] for n,i in zip(names,indices) if states.get(n) is not None})
    names=list(old['adam_names']);index=names.index('apl_log_axial')+1
    learner={**old,'physical':{**physical,**local},'optimizer':optimizer,
        'adam_names':names[:index]+list(KC_TERMINAL_PARAMETER_NAMES)+names[index:]}
    payload={**source,'model':{**source['model'],**additions},'learner':learner,
        'config':{**source['config'],'kc_terminal':True,'kc_terminal_version':1,'kc_terminal_geometry':str(geometry.resolve())}}
    output.mkdir(parents=True);atomic_save(output/'last.pt',payload)
    del payload,source,old,learner,local,physical,holder,additions,optimizer,states
    gc.collect()
    before=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    after=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    assert all(same(v,after['model'][k]) for k,v in before['model'].items())
    assert all(same(v,after['learner']['physical'][k]) for k,v in before['learner']['physical'].items())
    assert all(same(v,after['learner'][k]) for k,v in before['learner'].items() if k not in ['physical','optimizer','adam_names'])
    for key in before:
        if key not in ['model','learner','config']:assert same(before[key],after[key]),key
    def named(s):return {n:s['state'].get(i) for g in s['param_groups'] for n,i in zip(g['parameter_names'],g['params'])}
    b=named(before['learner']['optimizer']);a=named(after['learner']['optimizer'])
    assert all(same(v,a[k]) for k,v in b.items())
    report={'source':str(checkpoint),'source_sha256':sha256(checkpoint),'candidate_sha256':sha256(output/'last.pt'),
        'trained':before['bptt_train_tokens'],'new_inputs':0,'new_optimizer_updates':0,
        'old_weights_physical_optimizer_credit_world_rng_exact':True,
        'spatial_initialization':'Uniform prolongation of old per-KC chemistry, actual delayed packets, compartment traces and their carried derivative. No past spatial pattern invented.',
        'geometry':{k:v for k,v in manifest.items() if k!='tree_counts'},'status':'zero_exposure_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();migrate(a.checkpoint,a.geometry,a.output)
