"""Add local APL calcium/resource coordinates without replaying experience."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_apl_local_release import (
    APL_LOCAL_RELEASE_VERSION, APL_LOCAL_RELEASE_PARAMETER_NAMES,
    APL_LOCAL_RELEASE_STATE_FIELDS, initial_parameters,
)
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.migrate_native_kc_dopamine import named
from scripts.ib.fly_interface.native_causal_audit import sha256


def extend_payload(source):
    if not source['config'].get('apl_cable') or source['config'].get('apl_local_release'):
        raise ValueError('Require native APL cable without the new release history')
    model, old = source['model'], source['learner']
    if source['config'].get('mother_native_generative') or (old.get('interaction') or {}).get('native_generative_state'):
        raise ValueError('Active native hypotheses need an explicit common reference-axis extension before APL migration')
    physical = old['physical']
    voltage = physical['apl_voltage']
    if voltage.ndim != 2 or voltage.shape[0] != 1 or not voltage.numel():
        raise ValueError('Native spatial APL voltage must already be represented')
    if any(physical.get(n, torch.empty(0)).numel() for n in APL_LOCAL_RELEASE_STATE_FIELDS):
        raise ValueError('Do not replace existing local calcium/resource history')
    additions = initial_parameters(model['apl_log_axial'].numel(), dtype=model['apl_log_axial'].dtype)
    initial = dict(zip(APL_LOCAL_RELEASE_STATE_FIELDS,
                       (torch.zeros_like(voltage), torch.zeros_like(voltage), torch.ones_like(voltage))))
    by_name = named(old['optimizer'])
    optimizer = {'state': {}, 'param_groups': []}
    offset, inserted = 0, False
    for group in old['optimizer']['param_groups']:
        names = list(group['parameter_names'])
        if 'apl_log_axial' in names:
            if inserted:
                raise ValueError('Duplicated APL optimizer parameter')
            pos = names.index('apl_log_axial') + 1
            names[pos:pos] = APL_LOCAL_RELEASE_PARAMETER_NAMES
            inserted = True
        indices = list(range(offset, offset + len(names)))
        offset += len(names)
        optimizer['param_groups'].append({**group, 'params': indices, 'parameter_names': names})
        optimizer['state'].update({i: by_name[n] for n, i in zip(names, indices)
                                  if by_name.get(n) is not None})
    if not inserted:
        raise ValueError('Missing native APL physiology optimizer group')
    names = list(old['adam_names'])
    pos = names.index('apl_log_axial') + 1
    names[pos:pos] = APL_LOCAL_RELEASE_PARAMETER_NAMES
    return {**source, 'model': {**model, **additions},
            'learner': {**old, 'physical': {**physical, **initial}, 'optimizer': optimizer, 'adam_names': names},
            'config': {**source['config'], 'apl_local_release': True,
                       'apl_local_release_version': APL_LOCAL_RELEASE_VERSION}}


def migrate(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new isolated candidate directory')
    if shutil.disk_usage(output.parent).free < 2 * checkpoint.stat().st_size + 1024**3:
        raise OSError('Need full candidate plus atomic replacement reserve')
    before_sha = sha256(checkpoint)
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    payload = extend_payload(source)
    state_bytes = sum(payload['learner']['physical'][n].numel() * payload['learner']['physical'][n].element_size()
                      for n in APL_LOCAL_RELEASE_STATE_FIELDS)
    payload['config']['checkpoint_dir'] = str(output.resolve())
    output.mkdir(parents=True)
    atomic_save(output / 'last.pt', payload)
    del payload, source
    gc.collect()
    a = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    b = torch.load(output / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert all(same(v, b['model'][k]) for k, v in a['model'].items())
    assert all(same(v, b['learner']['physical'][k]) for k, v in a['learner']['physical'].items()
               if k not in APL_LOCAL_RELEASE_STATE_FIELDS)
    assert all(same(v, b['learner'][k]) for k, v in a['learner'].items()
               if k not in ('physical', 'optimizer', 'adam_names'))
    before, after = named(a['learner']['optimizer']), named(b['learner']['optimizer'])
    assert all(same(v, after[k]) for k, v in before.items())
    assert all(same(v, b[k]) for k, v in a.items() if k not in ('model', 'learner', 'config'))
    report = {'source': str(checkpoint), 'source_sha256': before_sha,
              'candidate': str(output / 'last.pt'), 'candidate_sha256': sha256(output / 'last.pt'),
              'source_unchanged': before_sha == sha256(checkpoint),
              'trained': int(a['bptt_train_tokens']), 'new_inputs': 0, 'new_optimizer_updates': 0,
              'old_model_physical_optimizer_credit_world_rng_exact': True,
              'new_state_bytes': state_bytes,
              'initialization': 'New calcium/gates start resting, resource pool available; all already emitted delayed packets retained. No missing calcium past invented.',
              'scope': 'Local effective calcium-dependent nonspiking release, not identified VGCC/Na/K physiology or molecular priming',
              'status': 'zero_exposure_APL_local_release_extension_verified'}
    (output / 'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.checkpoint, args.output)
