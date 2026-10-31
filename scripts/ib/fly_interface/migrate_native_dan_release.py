"""Extend a continuous individual with local DAN source coordinates, no replay."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.core.fly_dan_release import initial_dan_parameters,DAN_RELEASE_VERSION,DAN_RELEASE_PARAMETER_NAMES
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.migrate_native_kc_dopamine import named
from scripts.ib.fly_interface.native_causal_audit import sha256


def migrate(checkpoint,geometry,output):
    if output.exists():raise ValueError('Use a new isolated full-state candidate')
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not source['config'].get('kc_dopamine') or source['config'].get('dan_local_release'):
        raise ValueError('Require spatial KC teacher source without local DAN extension')
    with np.load(geometry,allow_pickle=False) as data:
        index=torch.from_numpy(data['dan_index'].copy()).long()
        site_owner=torch.from_numpy(data['site_owner'].copy()).long();node_count=len(data['owner'])
    old=source['learner'];physical=old['physical'];sites=site_owner.numel()
    initial={'dan_local_receptor':torch.zeros(1,node_count),'dan_local_calcium':torch.zeros(1,node_count),
        'dan_local_resources':torch.ones(1,sites),
        'dan_local_packets':torch.cat([v.index_select(1,index[site_owner]) for v in physical['ring']],0)}
    state_bytes=sum(v.numel()*v.element_size() for v in initial.values())
    if shutil.disk_usage(output.parent).free<2*(checkpoint.stat().st_size+state_bytes)+1024**3:
        raise OSError('Need full candidate and atomic replacement reserve')
    holder=SimpleNamespace(dopamine_logit_fast=source['model']['dopamine_logit_fast'],
        dopamine_logit_slow=source['model']['dopamine_logit_slow'])
    additions=initial_dan_parameters(holder,index)
    additions['dan_release_digest']=torch.tensor(list(hashlib.sha256(geometry.read_bytes()).digest()),dtype=torch.uint8)
    by_name=named(old['optimizer']);optimizer={'state':{},'param_groups':[]};offset=0;insert_after=None
    for group in old['optimizer']['param_groups']:
        names=list(group['parameter_names'])
        if 'kc_dopamine_log_transport' in names:
            insert_after=names[-1];names+=list(DAN_RELEASE_PARAMETER_NAMES)
        indices=list(range(offset,offset+len(names)));offset+=len(names)
        optimizer['param_groups'].append({**group,'params':indices,'parameter_names':names})
        optimizer['state'].update({i:by_name[n] for n,i in zip(names,indices) if by_name.get(n) is not None})
    if insert_after is None:raise ValueError('Missing original native physiology optimizer group')
    names=list(old['adam_names']);position=names.index(insert_after)+1
    payload={**source,'model':{**source['model'],**additions},
        'learner':{**old,'physical':{**physical,**initial},'optimizer':optimizer,
            'adam_names':names[:position]+list(DAN_RELEASE_PARAMETER_NAMES)+names[position:]},
        'config':{**source['config'],'dan_local_release':True,'dan_release_version':DAN_RELEASE_VERSION,
            'dan_release_geometry':str(geometry.resolve()),'window':1,'checkpoint_dir':str(output.resolve()),
            'credit':'1-token BPTT with all realized physical ticks; existing carried conditional credit retained'}}
    output.mkdir(parents=True);atomic_save(output/'last.pt',payload)
    del payload,source,old,physical,initial,additions,optimizer,by_name;gc.collect()
    a=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    assert all(same(v,b['model'][k]) for k,v in a['model'].items())
    assert all(same(v,b['learner']['physical'][k]) for k,v in a['learner']['physical'].items())
    assert all(same(v,b['learner'][k]) for k,v in a['learner'].items() if k not in ('physical','optimizer','adam_names'))
    assert all(same(v,b[k]) for k,v in a.items() if k not in ('model','learner','config'))
    after,before=named(b['learner']['optimizer']),named(a['learner']['optimizer'])
    assert all(same(v,after[k]) for k,v in before.items())
    report={'source':str(checkpoint),'source_sha256':sha256(checkpoint),
        'candidate_sha256':sha256(output/'last.pt'),'trained':a['bptt_train_tokens'],
        'new_inputs':0,'new_optimizer_updates':0,'window_tokens':1,'new_state_bytes':state_bytes,
        'old_model_physical_optimizer_credit_world_rng_exact':True,
        'initialization':'Existing somatic AP queue copied to actual DAN release sites; newly represented receptor/calcium coordinates start resting and resources available. No spatial chemical/vesicle past invented; all old brain/world/optimizer state retained.',
        'scope':'Local reciprocal chemical source and calcium-related release; existing somatic AP drive retained, not active axon/VGCC molecular reconstruction',
        'status':'zero_exposure_local_DAN_source_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();migrate(a.checkpoint,a.geometry,a.output)
