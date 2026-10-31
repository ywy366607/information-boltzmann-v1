"""Add site resources at a full-state boundary; do not replay or reset history."""
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
from information_boltzmann.core.fly_kc_site_stp import (
    attach_kc_site_stp, KC_SITE_STP_PARAMETER_NAMES, KC_SITE_STP_VERSION,
)
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.native_causal_audit import sha256


def migrate(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new isolated candidate')
    if shutil.disk_usage(output.parent).free < 2 * checkpoint.stat().st_size + 1024**3:
        raise OSError('Need candidate plus full atomic-save reserve')
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if not source['config'].get('kc_terminal') or source['config'].get('kc_site_stp'):
        raise ValueError('Source must have native sites without site-STP extension')
    with np.load(source['config']['kc_terminal_geometry'], allow_pickle=False) as geometry:
        kc_index = torch.from_numpy(geometry['kc_index'].copy()).long()
        owner = torch.from_numpy(geometry['site_owner'].copy()).long()
    holder = torch.nn.Module()
    holder.register_buffer('kct_index', kc_index, persistent=False)
    holder.register_buffer('kct_site_owner', owner, persistent=False)
    holder.use_stp = False
    attach_kc_site_stp(holder)
    additions = {n: getattr(holder, n).detach() for n in KC_SITE_STP_PARAMETER_NAMES}
    local = {'kc_site_resources': torch.ones(1, owner.numel()),
             'kc_site_utilization': torch.sigmoid(additions['kc_site_logit_u0'])[owner].unsqueeze(0)}
    old = source['learner']
    by_name = {n: old['optimizer']['state'].get(i) for g in old['optimizer']['param_groups']
               for n, i in zip(g['parameter_names'], g['params'])}
    optimizer = {'state': {}, 'param_groups': []}
    offset = 0
    for group in old['optimizer']['param_groups']:
        names = list(group['parameter_names'])
        if 'kc_terminal_log_diffusion' in names:
            names += list(KC_SITE_STP_PARAMETER_NAMES)
        ids = list(range(offset, offset + len(names)))
        offset += len(names)
        optimizer['param_groups'].append({**group, 'params': ids, 'parameter_names': names})
        optimizer['state'].update({i: by_name[n] for n, i in zip(names, ids) if by_name.get(n) is not None})
    names = list(old['adam_names'])
    index = names.index('kc_terminal_log_diffusion') + 1
    payload = {**source, 'model': {**source['model'], **additions},
               'learner': {**old, 'physical': {**old['physical'], **local}, 'optimizer': optimizer,
                           'adam_names': names[:index] + list(KC_SITE_STP_PARAMETER_NAMES) + names[index:]},
               'config': {**source['config'], 'kc_site_stp': True,
                          'kc_site_stp_version': KC_SITE_STP_VERSION, 'window': 1}}
    output.mkdir(parents=True)
    atomic_save(output / 'last.pt', payload)
    del payload, source, old, holder, local, additions, optimizer, by_name
    gc.collect()
    before = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    after = torch.load(output / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert all(same(v, after['model'][k]) for k, v in before['model'].items())
    assert all(same(v, after['learner']['physical'][k]) for k, v in before['learner']['physical'].items())
    assert all(same(v, after['learner'][k]) for k, v in before['learner'].items()
               if k not in ('physical', 'optimizer', 'adam_names'))
    for key in before:
        if key not in ('model', 'learner', 'config'):
            assert same(before[key], after[key]), key
    def named(saved):
        return {n: saved['state'].get(i) for g in saved['param_groups']
                for n, i in zip(g['parameter_names'], g['params'])}
    a, b = named(after['learner']['optimizer']), named(before['learner']['optimizer'])
    assert all(same(v, a[k]) for k, v in b.items())
    report = {'source': str(checkpoint), 'source_sha256': sha256(checkpoint),
              'candidate_sha256': sha256(output / 'last.pt'), 'trained': before['bptt_train_tokens'],
              'new_inputs': 0, 'new_optimizer_updates': 0, 'future_window_tokens': 1,
              'old_weights_physical_optimizer_credit_world_rng_exact': True,
              'new_state_scope': 'Previously unrepresented resource coordinates initialized as rested; no past resource history invented. Existing packets, phi and all old state unchanged.',
              'status': 'zero_exposure_site_resource_extension_verified'}
    (output / 'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.checkpoint, args.output)
