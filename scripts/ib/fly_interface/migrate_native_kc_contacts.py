"""Prolong a complete individual's memory to actual plastic contact coordinates."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_kc_contact_memory import (
    attach_kc_contact_memory, KC_CONTACT_STATE_FIELDS, KC_CONTACT_VERSION,
)
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.native_causal_audit import sha256


def migrate(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new isolated candidate')
    if shutil.disk_usage(output.parent).free < 2 * checkpoint.stat().st_size + 1024**3:
        raise OSError('Need full candidate and atomic replacement reserve')
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if source['config'].get('kc_contact_memory') or not source['config'].get('kc_terminal'):
        raise ValueError('Require native site source without contact extension')
    with np.load(source['config']['kc_terminal_geometry'], allow_pickle=False) as packed:
        data = {n: packed[n] for n in ('comp_pre', 'comp_post', 'fast_pre', 'fast_post',
                'fast_site', 'fast_delay', 'fast_fraction')}
    holder = torch.nn.Module()
    holder.kct_index = True
    holder.comp_pre = torch.from_numpy(data['comp_pre'])
    holder.comp_post = torch.from_numpy(data['comp_post'])
    n = source['config']['n_neurons']
    key = data['comp_pre'] * n + data['comp_post']
    order = np.argsort(key)
    sorted_key = key[order]
    query = data['fast_pre'] * n + data['fast_post']
    positions = np.searchsorted(sorted_key, query)
    valid = positions < len(sorted_key)
    valid &= sorted_key[np.minimum(positions, len(sorted_key)-1)] == query
    for lag in range(1, 5):
        select = valid & (data['fast_delay'] == lag)
        for name, value in [('edge', order[positions[select]]),
            ('site', data['fast_site'][select]), ('fraction', data['fast_fraction'][select])]:
            holder.register_buffer(f'kct_comp_{lag}_{name}', torch.as_tensor(
                value, dtype=torch.float32 if name == 'fraction' else torch.long), persistent=False)
    attach_kc_contact_memory(holder)
    edge = holder.kcm_edge
    old = source['learner']['physical']
    sources = ('comp_memory', 'kc_terminal_eligibility', 'comp_da_trace', 'comp_p_stats',
               'comp_camp', 'credit_comp', 'kc_terminal_eligibility_credit', 'credit_comp_camp')
    rows = (1, 1, 1, 2, 1, 0, 1, 0)
    local = {name: old[src].index_select(1, edge) if old[src].numel()
             else old['h'].new_zeros(r, edge.numel())
             for name, src, r in zip(KC_CONTACT_STATE_FIELDS, sources, rows)}
    payload = {**source, 'model': {**source['model'], 'kc_contact_ids': holder.kc_contact_ids},
        'learner': {**source['learner'], 'physical': {**old, **local}},
        'config': {**source['config'], 'kc_contact_memory': True,
                   'kc_contact_version': KC_CONTACT_VERSION, 'window': 1,
                   'checkpoint_dir': str(output.resolve())}}
    output.mkdir(parents=True)
    atomic_save(output / 'last.pt', payload)
    contact_count, site_count = edge.numel(), int(torch.unique(holder.kcm_site).numel())
    local_bytes = sum(v.numel() * v.element_size() for v in local.values())
    del payload, source, old, local, holder, data, edge
    gc.collect()
    before = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    after = torch.load(output / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert all(same(v, after['model'][k]) for k, v in before['model'].items())
    assert all(same(v, after['learner']['physical'][k]) for k, v in before['learner']['physical'].items())
    assert all(same(v, after['learner'][k]) for k, v in before['learner'].items() if k != 'physical')
    assert all(same(v, after[k]) for k, v in before.items() if k not in ('model', 'learner', 'config'))
    report = {'source': str(checkpoint), 'source_sha256': sha256(checkpoint),
        'candidate_sha256': sha256(output/'last.pt'), 'trained': before['bptt_train_tokens'],
        'new_inputs': 0, 'new_optimizer_updates': 0, 'window_tokens': 1,
        'contacts': contact_count, 'physical_release_positions': site_count,
        'new_state_bytes': local_bytes,
        'old_model_physical_optimizer_credit_world_rng_exact': True,
        'extension': 'Each contact inherits its prior connection phi, local eligibility, dopamine trace, background/cAMP and conditional sensitivities. Old fields retained as inactive history. No new experiences or missing gradients invented.',
        'teacher_scope': 'Existing MBON-compartment dopamine teacher; no spatial bouton dopamine reconstruction',
        'status': 'zero_exposure_contact_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.checkpoint, args.output)
