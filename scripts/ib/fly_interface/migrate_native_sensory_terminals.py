"""Archive a zero-exposure sensory-terminal extension of a full individual.

New per-cell materials start at the neutral values (transport scale 1, terminal
gains 1); terminal conductance starts at rest; site packets already in flight
are the cells' archived pulses at each of their actual sites, so the first
delivery equals the point delivery. Every old value is kept byte-identical.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_sensory_terminals import (
    SENSORY_TERMINAL_VERSION, SENSORY_TERMINAL_PARAMETER_NAMES, SENSORY_TERMINAL_STATE_FIELDS,
    initial_parameters, file_digest,
)
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest

# The learner's native spatial group order ends ... KC lateral, DAN release, sensory terminal.
from information_boltzmann.core.fly_kc_lateral import KC_LATERAL_PARAMETER_NAMES
from information_boltzmann.core.fly_dan_release import DAN_RELEASE_PARAMETER_NAMES
PRECEDING = KC_LATERAL_PARAMETER_NAMES + DAN_RELEASE_PARAMETER_NAMES


def insertion(names):
    present = [i for i, n in enumerate(names) if n in PRECEDING]
    if not present:
        raise ValueError('Expected native spatial physiology names are missing')
    return max(present) + 1


def named(optimizer):
    return {name: optimizer['state'].get(index) for group in optimizer['param_groups']
            for name, index in zip(group['parameter_names'], group['params'])}


def extend(source, geometry):
    config = source['config']
    if config.get('sensory_terminal'):
        raise ValueError('Archive already carries sensory terminal history')
    old = source['learner']
    if any(old['physical'].get(n, torch.empty(0)).numel() for n in SENSORY_TERMINAL_STATE_FIELDS):
        raise ValueError('Do not overwrite sensory terminal history')
    with np.load(geometry, allow_pickle=False) as z:
        cells = z['cell_index'].astype(np.int64)
        nodes = len(z['owner'])
        site_cell = torch.as_tensor(cells[z['site_owner']])
    ring = old['physical']['ring']
    if len(ring) != 4 or ring[0].shape[0] != 1:
        raise ValueError('Unexpected archived pulse ring')
    initial = {'sensory_terminal_signal': torch.zeros(2, nodes, dtype=ring[0].dtype),
               'sensory_terminal_packets': torch.cat([p.index_select(1, site_cell) for p in ring], 0).contiguous()}
    additions = initial_parameters(len(cells))
    additions['sensory_terminal_digest'] = torch.tensor(list(file_digest(geometry)), dtype=torch.uint8)
    by_name = named(old['optimizer'])
    optimizer = {'state': {}, 'param_groups': []}
    offset, found = 0, False
    for group in old['optimizer']['param_groups']:
        names = list(group['parameter_names'])
        if 'kc_lateral_log_camp_gain' in names:
            if found:
                raise ValueError('Duplicated native physiology group')
            pos = insertion(names)
            names[pos:pos] = SENSORY_TERMINAL_PARAMETER_NAMES
            found = True
        indices = list(range(offset, offset + len(names)))
        offset += len(names)
        optimizer['param_groups'].append({**group, 'params': indices, 'parameter_names': names})
        optimizer['state'].update({i: by_name[n] for n, i in zip(names, indices) if by_name.get(n) is not None})
    if not found:
        raise ValueError('Expected native spatial physiology group is missing')
    names = list(old['adam_names'])
    pos = insertion(names)
    names[pos:pos] = SENSORY_TERMINAL_PARAMETER_NAMES
    return {**source, 'model': {**source['model'], **additions},
            'learner': {**old, 'physical': {**old['physical'], **initial}, 'optimizer': optimizer,
                        'adam_names': names},
            'config': {**config, 'sensory_terminal': True, 'sensory_terminal_version': SENSORY_TERMINAL_VERSION,
                       'sensory_terminal_geometry': str(geometry.resolve())}}


def verify(source_path, migrated_path, geometry):
    old = torch.load(source_path, map_location='cpu', mmap=True, weights_only=False)
    new = torch.load(migrated_path, map_location='cpu', mmap=True, weights_only=False)
    problems = []
    for key in old:
        if key in ('model', 'learner', 'config'):
            continue
        if state_digest(old[key]) != state_digest(new[key]):
            problems.append(key)
    for key, value in old['model'].items():
        if state_digest(value) != state_digest(new['model'][key]):
            problems.append('model.' + key)
    added_model = set(new['model']) - set(old['model'])
    if added_model != set(SENSORY_TERMINAL_PARAMETER_NAMES) | {'sensory_terminal_digest'}:
        problems.append('unexpected model additions ' + str(sorted(added_model)))
    for key, value in old['learner'].items():
        if key in ('optimizer', 'adam_names'):
            continue
        if key == 'physical':
            for field, tensor in value.items():
                if state_digest(tensor) != state_digest(new['learner']['physical'][field]):
                    problems.append('physical.' + field)
            continue
        if state_digest(value) != state_digest(new['learner'][key]):
            problems.append('learner.' + key)
    if [n for n in new['learner']['adam_names'] if n not in SENSORY_TERMINAL_PARAMETER_NAMES] != old['learner']['adam_names']:
        problems.append('adam_names order')
    old_named, new_named = named(old['learner']['optimizer']), named(new['learner']['optimizer'])
    for name, value in old_named.items():
        if state_digest(value) != state_digest(new_named.get(name)):
            problems.append('optimizer.state.' + name)
    for a, b in zip(old['learner']['optimizer']['param_groups'], new['learner']['optimizer']['param_groups']):
        stripped = {**b, 'parameter_names': [n for n in b['parameter_names'] if n not in SENSORY_TERMINAL_PARAMETER_NAMES]}
        if {k: v for k, v in a.items() if k != 'params'} != {k: v for k, v in stripped.items() if k != 'params'}:
            problems.append('optimizer group hyperparameters/names')
    if any(new_named.get(n) is not None for n in SENSORY_TERMINAL_PARAMETER_NAMES):
        problems.append('new parameters must start without optimizer history')
    added_config = {k for k in new['config'] if k not in old['config']} | {
        k for k in old['config'] if old['config'][k] != new['config'].get(k)}
    if added_config - {'sensory_terminal', 'sensory_terminal_version', 'sensory_terminal_geometry', 'checkpoint_dir'}:
        problems.append('config changes ' + str(sorted(added_config)))
    digest = bytes(new['model']['sensory_terminal_digest'].tolist())
    if digest != file_digest(geometry):
        problems.append('geometry digest')
    physical = new['learner']['physical']
    with np.load(geometry, allow_pickle=False) as z:
        site_cell = torch.as_tensor(z['cell_index'].astype(np.int64)[z['site_owner']])
    ring = old['learner']['physical']['ring']
    expected = torch.cat([p.index_select(1, site_cell) for p in ring], 0)
    if not torch.equal(physical['sensory_terminal_packets'], expected) or physical['sensory_terminal_signal'].abs().sum() != 0:
        problems.append('initial terminal state')
    report = {'old_fields_changed_or_problems': problems,
              'new_parameters': int(sum(new['model'][n].numel() for n in SENSORY_TERMINAL_PARAMETER_NAMES)),
              'new_state_bytes': int(sum(physical[n].numel() * physical[n].element_size()
                                         for n in SENSORY_TERMINAL_STATE_FIELDS)),
              'trained': int(new['bptt_train_tokens'])}
    return report


def main(args):
    from scripts.ib.train_fly_bptt_stream import atomic_save
    from scripts.ib.fly_interface.native_causal_audit import sha256
    if args.output.exists():
        raise ValueError('Use a new isolated archive directory')
    reserve = args.checkpoint.stat().st_size + 512 * 1024 ** 2 + 1024 ** 3
    if shutil.disk_usage(args.output.parent).free < reserve:
        raise OSError('Insufficient atomic archive reserve')
    source_sha = sha256(args.checkpoint)
    source = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    payload = extend(source, args.geometry)
    payload['config']['checkpoint_dir'] = str(args.output.resolve())
    args.output.mkdir(parents=True)
    atomic_save(args.output / 'last.pt', payload)
    del payload, source
    gc.collect()
    report = verify(args.checkpoint, args.output / 'last.pt', args.geometry)
    report.update(source=str(args.checkpoint), source_sha256=source_sha, source_sha256_after=sha256(args.checkpoint),
                  candidate=str(args.output / 'last.pt'), candidate_sha256=sha256(args.output / 'last.pt'),
                  geometry=str(args.geometry), new_formal_inputs=0, new_formal_updates=0,
                  status=('zero_exposure_sensory_terminal_extension_verified'
                          if not report['old_fields_changed_or_problems'] else 'not_accepted'),
                  initialization='terminal conductance at rest; in-flight pulses copied to each actual site; '
                                 'neutral transport/gain materials without optimizer history',
                  training_adoption_approved=False)
    (args.output / 'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1))
    if report['old_fields_changed_or_problems']:
        raise RuntimeError('Migration verification failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
