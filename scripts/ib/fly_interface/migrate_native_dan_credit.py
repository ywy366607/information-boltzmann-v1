"""Explicit zero-exposure credit boundary; retain every existing learned state."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_dan_credit import DAN_CREDIT_FIELDS, DAN_CREDIT_VERSION
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.native_causal_audit import sha256


def migrate(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new isolated candidate')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = saved['config']
    if not cfg.get('dan_local_release') or not cfg.get('carried_credit') or cfg.get('dan_carried_credit'):
        raise ValueError('Need an existing local DAN source with carried credit, without this extension')
    if any(saved['learner']['physical'].get(n, torch.empty(0)).numel() for n in DAN_CREDIT_FIELDS):
        raise ValueError('Refuse to reset existing DAN historical derivatives')
    with np.load(cfg['dan_release_geometry'], allow_pickle=False) as data:
        nodes, sites = len(data['owner']), len(data['site_owner'])
    shapes = ((2, nodes), (4, nodes), (7, sites), (4, 7, sites))
    added = {n: saved['learner']['physical']['h'].new_zeros(shape) for n, shape in zip(DAN_CREDIT_FIELDS, shapes)}
    size = sum(v.numel()*v.element_size() for v in added.values())
    if shutil.disk_usage(output.parent).free < 2*(checkpoint.stat().st_size+size)+1024**3:
        raise OSError('Need full candidate, atomic replacement and1GiB reserve')
    payload = {**saved, 'learner': {**saved['learner'], 'physical': {**saved['learner']['physical'], **added}},
        'config': {**cfg, 'dan_carried_credit': True, 'dan_credit_version': DAN_CREDIT_VERSION,
                   'window': 1, 'checkpoint_dir': str(output.resolve())}}
    output.mkdir(parents=True)
    atomic_save(output/'last.pt', payload)
    del saved, payload, added
    gc.collect()
    before = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    after = torch.load(output/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert same(before['model'], after['model'])
    assert all(same(v, after['learner']['physical'][k]) for k,v in before['learner']['physical'].items() if k not in DAN_CREDIT_FIELDS)
    assert all(same(v, after['learner'][k]) for k,v in before['learner'].items() if k!='physical')
    assert all(same(v, after[k]) for k,v in before.items() if k not in ('model', 'learner', 'config'))
    report = {'source': str(checkpoint), 'source_sha256': sha256(checkpoint),
        'candidate_sha256': sha256(output/'last.pt'), 'trained': before['bptt_train_tokens'],
        'physical_ticks': before['learner'].get('physical_ticks'), 'new_inputs':0, 'new_optimizer_updates':0,
        'additional_state_bytes':size, 'window_tokens':1, 'existing_complete_state_exact':True,
        'initialization':'New conditional material sensitivities start at this explicit boundary; no unrecorded past derivatives reconstructed. All old physical/optimizer/RNG/world/pending credit retained.',
        'scope':'Full spatial local source recursion conditional on incoming KC/AP packets; not full recurrent/downstream RTRL',
        'status':'zero_exposure_DAN_credit_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    migrate(args.checkpoint,args.output)
