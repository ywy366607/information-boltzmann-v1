"""Add local DAN-KC signal coordinates without replay, reset, or lost old state."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.core.fly_kc_dopamine import KC_DOPAMINE_VERSION
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.native_causal_audit import sha256


def named(optimizer):
    return {name:optimizer['state'].get(index) for group in optimizer['param_groups']
            for name,index in zip(group['parameter_names'],group['params'])}


def migrate(checkpoint,geometry,output):
    if output.exists():
        raise ValueError('Use a new isolated candidate')
    if shutil.disk_usage(output.parent).free < 2*checkpoint.stat().st_size+1024**3:
        raise OSError('Need complete candidate plus atomic replacement reserve')
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not source['config'].get('kc_contact_memory') or source['config'].get('kc_dopamine'):
        raise ValueError('Require contact-memory source without spatial DAN extension')
    with np.load(source['config']['kc_terminal_geometry'],allow_pickle=False) as packed:
        owner=torch.from_numpy(packed['owner'].copy()).long()
        kc=torch.from_numpy(packed['kc_index'].copy()).long()
    target=kc[owner]
    old=source['learner']; physical=old['physical']
    local={'kc_dopamine_signal':physical['dopamine'].index_select(1,target),
           'kc_dopamine_credit':physical['credit_dopamine'].index_select(1,target)
             if physical['credit_dopamine'].numel() else torch.zeros(3,target.numel())}
    additions={'kc_dopamine_log_transport':source['model']['kc_terminal_log_diffusion'][1].clone(),
               'kc_dopamine_digest':torch.tensor(list(hashlib.sha256(geometry.read_bytes()).digest()),dtype=torch.uint8)}
    by_name=named(old['optimizer'])
    optimizer={'state':{},'param_groups':[]}
    offset=0;insert_after=None
    for group in old['optimizer']['param_groups']:
        names=list(group['parameter_names'])
        if 'kc_terminal_log_diffusion' in names:
            insert_after=names[-1]
            names += ['kc_dopamine_log_transport']
        indices=list(range(offset,offset+len(names)));offset+=len(names)
        optimizer['param_groups'].append({**group,'params':indices,'parameter_names':names})
        optimizer['state'].update({i:by_name[n] for n,i in zip(names,indices) if by_name.get(n) is not None})
    if insert_after is None:
        raise ValueError('Missing original native physiology optimizer group')
    names=list(old['adam_names']);position=names.index(insert_after)+1
    payload={**source,'model':{**source['model'],**additions},
        'learner':{**old,'physical':{**physical,**local},'optimizer':optimizer,
            'adam_names':names[:position]+['kc_dopamine_log_transport']+names[position:]},
        'config':{**source['config'],'kc_dopamine':True,'kc_dopamine_geometry':str(geometry.resolve()),
            'kc_dopamine_version':KC_DOPAMINE_VERSION,'window':1,'checkpoint_dir':str(output.resolve())}}
    output.mkdir(parents=True)
    atomic_save(output/'last.pt',payload)
    state_bytes=sum(v.numel()*v.element_size() for v in local.values())
    del payload,source,old,physical,local,additions,optimizer,by_name
    gc.collect()
    before=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    after=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    assert all(same(v,after['model'][k]) for k,v in before['model'].items())
    assert all(same(v,after['learner']['physical'][k]) for k,v in before['learner']['physical'].items())
    assert all(same(v,after['learner'][k]) for k,v in before['learner'].items() if k not in ('physical','optimizer','adam_names'))
    assert all(same(v,after[k]) for k,v in before.items() if k not in ('model','learner','config'))
    a,b=named(after['learner']['optimizer']),named(before['learner']['optimizer'])
    assert all(same(v,a[k]) for k,v in b.items())
    report={'source':str(checkpoint),'source_sha256':sha256(checkpoint),'candidate_sha256':sha256(output/'last.pt'),
        'trained':before['bptt_train_tokens'],'new_inputs':0,'new_optimizer_updates':0,'window_tokens':1,
        'old_model_physical_optimizer_credit_world_rng_exact':True,'new_state_bytes':state_bytes,
        'initialization':'Prolong existing KC fast/slow signals and conditional clock sensitivities uniformly over its morphology; no reconstructed spatial past. Contact phi/history remains exact.',
        'teacher_scope':'Actual DAN-KC synapses, local effective receptor-driving signal transport; no molecular receptor/extracellular volume or local DAN cable claim',
        'status':'zero_exposure_spatial_dopamine_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();migrate(args.checkpoint,args.geometry,args.output)
