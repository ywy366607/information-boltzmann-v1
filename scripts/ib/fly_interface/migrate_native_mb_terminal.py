"""Activate anatomical MB terminal chemistry without discarding learned state."""
import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import ensure_gpu_idle, restore_native, sha256
from scripts.ib.fly_interface.action_loop_v2_audit import cpu_copy, state_digest
from scripts.ib.fly_interface.migrate_native_amines import named_optimizer
from information_boltzmann.core.fly_mb_terminal import (
    MB_TERMINAL_PARAMETER_NAMES, MB_TERMINAL_ID_NAMES, MB_TERMINAL_STATE_FIELDS,
    quarantine_mb_pooled_credit, attach_mb_terminal,
)
from information_boltzmann.core.fly_native_motor_port import checkpoint_model_weights
from scripts.ib.train_fly_bptt_stream import atomic_save


def containers(value):
    if isinstance(value,dict):
        return {k:containers(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(containers(v) for v in value)
    return value


def expected_interaction(source):
    expected=containers(source)
    dummy=SimpleNamespace(model=SimpleNamespace(mb_pooled_quarantined=True),
                          consequence_state=expected.get('consequence_state'))
    quarantine_mb_pooled_credit(dummy)
    return expected


def check(source, weights, current):
    changed_weights=[k for k,v in source['model'].items() if state_digest(weights.get(k))!=state_digest(v)]
    extra=set(weights)-set(source['model'])
    old=source['learner']
    changed_learner=[k for k,v in old.items() if k not in ('physical','optimizer','adam_names','interaction')
                     and state_digest(current.get(k))!=state_digest(v)]
    changed_physical=[k for k,v in old['physical'].items()
                      if state_digest(current['physical'].get(k))!=state_digest(v)]
    extra_physical=set(current['physical'])-set(old['physical'])
    a,b=named_optimizer(old['optimizer']),named_optimizer(current['optimizer'])
    changed_moments=[k for k,v in a.items() if b.get(k)!=v]
    interaction_ok=state_digest(current['interaction'])==state_digest(expected_interaction(old['interaction']))
    if (changed_weights or changed_learner or changed_physical or changed_moments or not interaction_ok
            or extra!=set(MB_TERMINAL_PARAMETER_NAMES+MB_TERMINAL_ID_NAMES)
            or extra_physical-set(MB_TERMINAL_STATE_FIELDS)
            or any(current['physical'][k].numel() for k in extra_physical)
            or set(current['adam_names'])-set(old['adam_names'])!=set(MB_TERMINAL_PARAMETER_NAMES)):
        raise RuntimeError(f'MB activation changed existing state: weights={changed_weights}, '
                           f'learner={changed_learner}, physical={changed_physical}, '
                           f'moments={changed_moments}, interaction={interaction_ok}, extra={extra}')
    return {'old_model_exact':True,'old_named_adam_moments_exact':True,'old_physical_fields_exact':True,
            'old_other_learner_fields_exact':True,'pending_credit_preserved_with_proxy_archive':interaction_ok,
            'new_empty_physical_fields':sorted(extra_physical), 'new_model_entries':sorted(extra)}


def verify_saved(checkpoint, output):
    source_hash,destination_hash=sha256(checkpoint),sha256(output/'last.pt')
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    saved=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    verification=check(source,saved['model'],saved['learner'])
    changed_top=[k for k,v in source.items() if k not in ('config','model','learner')
                 and state_digest(saved[k])!=state_digest(v)]
    if changed_top:
        raise RuntimeError('World/RNG/cursors/meters changed during zero-exposure migration: '+str(changed_top))
    report=json.loads((output/'migration_pending.json').read_text(encoding='utf8'))
    if source_hash!=report['source_sha256']:
        raise RuntimeError('Migration source changed')
    report.update(verification,checkpoint=str(output/'last.pt'),sha256=destination_hash,
                  status='full_state_activation_verified',independent_saved_payload_verification=True)
    (output/'migration.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)


def migrate(checkpoint,output):
    if output.exists():
        raise ValueError('Use a new isolated output directory')
    if shutil.disk_usage(output.parent).free < 2*checkpoint.stat().st_size+512*2**20:
        raise OSError('Need a full state plus atomic-replacement reserve')
    fingerprint=sha256(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved['config'].get('mb_terminal_projection'):
        raise ValueError('This is a recorded first activation, not an existing-mechanism restore')
    # Construct ONLY new anatomical metadata/parameters on CPU, not a second
    # full brain plus GPU/Adam copies. The real full trainer restore verifies
    # this migration afterwards; this holder never runs neural dynamics.
    holder=torch.nn.Module()
    holder.synapse_model='coba'
    holder.register_buffer('edge_pre_i',torch.empty(0,dtype=torch.long),persistent=False)
    for name in ('amine_log_clearance_5ht','amine_log_affinity_5ht','amine_log_off_5ht'):
        setattr(holder,name,torch.nn.Parameter(saved['model'][name].detach().clone()))
    graph=Path(saved['config']['graph'])
    manifest=attach_mb_terminal(holder,graph,graph.parent/'body-annotations.feather',
                                graph.parent/'dendritic_share_Kenyon_Cell.feather')
    weights={**saved['model'],**{k:getattr(holder,k).detach().clone()
                                for k in MB_TERMINAL_PARAMETER_NAMES+MB_TERMINAL_ID_NAMES}}
    old=saved['learner']
    current={**old,'physical':{**old['physical'],**{k:torch.empty(0) for k in MB_TERMINAL_STATE_FIELDS}},
             'interaction':expected_interaction(old['interaction'])}
    # Preserve semantic group order/options and every old moment. Reindex
    # integer handles canonically, just as the real constructor will do.
    old_by_name={name:old['optimizer']['state'].get(index)
                 for g in old['optimizer']['param_groups']
                 for name,index in zip(g['parameter_names'],g['params'])}
    optimizer={'state':{},'param_groups':[]}
    next_index=0
    inserted=False
    for group in old['optimizer']['param_groups']:
        names=list(group['parameter_names'])
        if 'mb_dpm_logit_decay' in names:
            names+=list(MB_TERMINAL_PARAMETER_NAMES)
            inserted=True
        indices=list(range(next_index,next_index+len(names)))
        next_index+=len(names)
        optimizer['param_groups'].append({**group,'params':indices,'parameter_names':names})
        optimizer['state'].update({i:old_by_name[n] for n,i in zip(names,indices)
                                   if old_by_name.get(n) is not None})
    if not inserted:
        raise ValueError('Expected existing named MB optimizer group')
    names=list(old['adam_names'])
    insertion=names.index('mb_apl_feedback')+1
    current.update(optimizer=optimizer,adam_names=names[:insertion]+list(MB_TERMINAL_PARAMETER_NAMES)+names[insertion:])
    verification=check(saved,weights,current)
    report={'at':datetime.now(timezone.utc).isoformat(),'source':str(checkpoint),'source_sha256':fingerprint,
            'trained':int(saved['bptt_train_tokens']),**verification,
            'contacts':manifest,
            'migration_construction':'CPU metadata only; full brain is not simulated during zero-exposure migration',
            'changed_dynamics':'Pooled proxy archived; measured axonal GABA and DPM5HT regulate actual per-KC output',
            'spatial_scope':'one axonal output per KC; APL cable and individual bouton fields still unresolved',
            'scope':'zero-exposure state migration, not learning/consolidation benefit'}
    payload={**saved,'model':weights,'learner':current,
             'config':{**saved['config'],'mb_terminal_projection':True,'mb_terminal_version':1}}
    output.mkdir(parents=True)
    (output/'migration_pending.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    atomic_save(output/'last.pt',payload)
    del payload,saved,current,weights,holder,old,old_by_name,optimizer
    gc.collect()
    torch.cuda.empty_cache()
    verify_saved(checkpoint,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-saved',action='store_true')
    args=parser.parse_args()
    (verify_saved if args.verify_saved else migrate)(args.checkpoint,args.output)
