"""Prolong the old lumped APL state onto its real cable without deleting history."""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import sha256
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from scripts.ib.fly_interface.migrate_native_amines import named_optimizer
from scripts.ib.train_fly_bptt_stream import atomic_save
from information_boltzmann.core.fly_apl_cable import (
    attach_apl_cable, APL_CABLE_PARAMETER_NAMES, APL_CABLE_ID_NAMES, APL_CABLE_STATE_FIELDS,
)


def prolonged_physical(source, data):
    old = source['learner']['physical']
    index = torch.from_numpy(data['apl_index'])
    owner = torch.from_numpy(data['node_owner'])
    values = {name: old[field].index_select(1, index).index_select(1, owner)
              for name, field in zip(APL_CABLE_STATE_FIELDS[:3], ('h', 'ge', 'gi'))}
    values['apl_packets'] = torch.cat([p.index_select(1, index).index_select(1, owner) for p in old['ring']], 0)
    return values


def verify(source_path, output):
    source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=False)
    saved = torch.load(output / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    data = np.load(Path(source['config']['graph']).parent / 'spatial/apl_cable_native_v1.npz')
    expected = prolonged_physical(source, data)
    old, current = source['learner'], saved['learner']
    bad_model = [k for k, value in source['model'].items() if state_digest(value) != state_digest(saved['model'].get(k))]
    bad_physical = [k for k, value in old['physical'].items() if state_digest(value) != state_digest(current['physical'].get(k))]
    # Old archives predating this schema may already have empty new placeholders.
    bad_physical = [k for k in bad_physical if k not in APL_CABLE_STATE_FIELDS or old['physical'][k].numel()]
    bad_extension = [k for k, value in expected.items() if not torch.equal(value, current['physical'][k])]
    a, b = named_optimizer(old['optimizer']), named_optimizer(current['optimizer'])
    bad_moments = [k for k, value in a.items() if b.get(k) != value]
    bad_learner = [k for k, value in old.items() if k not in ('physical', 'optimizer', 'adam_names') and state_digest(value) != state_digest(current.get(k))]
    bad_top = [k for k, value in source.items() if k not in ('model', 'learner', 'config') and state_digest(value) != state_digest(saved.get(k))]
    if bad_model or bad_physical or bad_extension or bad_moments or bad_learner or bad_top:
        raise RuntimeError(f'APL prolongation changed history: {bad_model,bad_physical,bad_extension,bad_moments,bad_learner,bad_top}')
    report = json.loads((output / 'migration_pending.json').read_text(encoding='utf8'))
    report.update(status='full_state_prolongation_verified', source_sha256=sha256(source_path),
        checkpoint_sha256=sha256(output / 'last.pt'), old_model_exact=True, old_physical_exact=True,
        old_optimizer_moments_exact=True, old_other_learner_exact=True, old_world_rng_cursors_exact=True,
        new_spatial_values_match_explicit_prolongation=True)
    (output / 'migration.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


def migrate(source_path, output):
    if output.exists():
        raise ValueError('Choose a new isolated candidate directory')
    if shutil.disk_usage(output.parent).free < 2 * source_path.stat().st_size + 512 * 2**20:
        raise OSError('Need full candidate plus atomic-save reserve')
    source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=False)
    if source['config'].get('apl_cable'):
        raise ValueError('This is a first spatial activation, not an existing-cable restore')
    geometry = Path(source['config']['graph']).parent / 'spatial/apl_cable_native_v1.npz'
    data = np.load(geometry)
    holder = torch.nn.Module()
    holder.register_buffer('edge_pre_i', torch.empty(0, dtype=torch.long), persistent=False)
    manifest = attach_apl_cable(holder, geometry, metadata_only=True)
    new_model = {k: getattr(holder, k).detach().clone() for k in APL_CABLE_PARAMETER_NAMES + APL_CABLE_ID_NAMES}
    old = source['learner']
    by_name = {name: old['optimizer']['state'].get(index)
               for group in old['optimizer']['param_groups']
               for name, index in zip(group['parameter_names'], group['params'])}
    optimizer = {'state': {}, 'param_groups': []}
    offset = 0
    for group in old['optimizer']['param_groups']:
        names = list(group['parameter_names'])
        if 'mb_dpm_log_off' in names:
            names += list(APL_CABLE_PARAMETER_NAMES)
        indices = list(range(offset, offset + len(names)))
        offset += len(names)
        optimizer['param_groups'].append({**group, 'params': indices, 'parameter_names': names})
        optimizer['state'].update({i: by_name[n] for n, i in zip(names, indices) if by_name.get(n) is not None})
    names = list(old['adam_names'])
    insertion = names.index('mb_dpm_log_off') + 1
    learner = {**old, 'physical': {**old['physical'], **prolonged_physical(source, data)},
        'optimizer': optimizer, 'adam_names': names[:insertion] + list(APL_CABLE_PARAMETER_NAMES) + names[insertion:]}
    payload = {**source, 'learner': learner, 'model': {**source['model'], **new_model},
        'config': {**source['config'], 'apl_cable': True, 'apl_cable_version': 1}}
    output.mkdir(parents=True)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'source': str(source_path),
        'source_sha256': sha256(source_path), 'trained': source['bptt_train_tokens'],
        'geometry': manifest, 'initial_material_parameter': 'log axial conductance zero (one model unit), positive and learnable, no length/time cap',
        'initial_spatial_state': 'uniform prolongation of each old APL membrane/conductance and each old delayed packet; no past spatial pattern invented',
        'scope': 'zero-exposure state extension; full trainer restore and native routing checks required before learning'}
    (output / 'migration_pending.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    atomic_save(output / 'last.pt', payload)
    del payload, source, holder, learner, new_model, old, by_name, optimizer
    gc.collect()
    verify(source_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.checkpoint, args.output)
