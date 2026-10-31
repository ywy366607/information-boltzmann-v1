"""Archive a zero-exposure KC lateral-receptor extension of a full individual."""
import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_kc_lateral import (
    KC_LATERAL_VERSION, KC_LATERAL_PARAMETER_NAMES, KC_LATERAL_STATE_FIELDS, initial_parameters,
)


def named(optimizer):
    return {name: optimizer['state'].get(index) for group in optimizer['param_groups']
            for name, index in zip(group['parameter_names'], group['params'])}


def extend(source, geometry):
    if not source['config'].get('kc_contact_memory') or source['config'].get('kc_lateral'):
        raise ValueError('Require native contacts without new lateral history')
    old = source['learner']
    if source['config'].get('mother_native_generative') or (old.get('interaction') or {}).get('native_generative_state'):
        raise ValueError('Active native hypotheses require a recorded common reference-axis extension')
    with np.load(geometry, allow_pickle=False) as z:
        count = len(z['kc_index'])
    nodes = old['physical']['kc_terminal_gaba'].shape[1]
    if any(old['physical'].get(n, torch.empty(0)).numel() for n in KC_LATERAL_STATE_FIELDS):
        raise ValueError('Do not overwrite lateral receptor history')
    additions = initial_parameters(count)
    additions['kc_lateral_digest'] = torch.tensor(list(hashlib.sha256(geometry.read_bytes()).digest()), dtype=torch.uint8)
    initial = {name: torch.zeros(1, nodes) for name in KC_LATERAL_STATE_FIELDS}
    by_name = named(old['optimizer'])
    optimizer = {'state': {}, 'param_groups': []}
    offset, found = 0, False
    for group in old['optimizer']['param_groups']:
        names = list(group['parameter_names'])
        if 'kc_dopamine_log_transport' in names:
            if found:
                raise ValueError('Duplicated native physiology group')
            pos = names.index('kc_dopamine_log_transport') + 1
            names[pos:pos] = KC_LATERAL_PARAMETER_NAMES
            found = True
        indices = list(range(offset, offset + len(names)))
        offset += len(names)
        optimizer['param_groups'].append({**group, 'params': indices, 'parameter_names': names})
        optimizer['state'].update({i: by_name[n] for n, i in zip(names, indices) if by_name.get(n) is not None})
    if not found:
        raise ValueError('Expected native spatial physiology group is missing')
    names = list(old['adam_names'])
    pos = names.index('kc_dopamine_log_transport') + 1
    names[pos:pos] = KC_LATERAL_PARAMETER_NAMES
    return {**source, 'model': {**source['model'], **additions},
        'learner': {**old, 'physical': {**old['physical'], **initial}, 'optimizer': optimizer, 'adam_names': names},
        'config': {**source['config'], 'kc_lateral': True, 'kc_lateral_version': KC_LATERAL_VERSION,
                   'kc_lateral_geometry': str(geometry.resolve())}}


def migrate(checkpoint, geometry, output, archive_only=False):
    from scripts.ib.train_fly_bptt_stream import atomic_save
    if output.exists():
        raise ValueError('Use a new isolated archive')
    # A new read-only archive needs one atomic temporary, not two full copies.
    # Training adoption still needs reserve for replacing an existing archive.
    reserve = checkpoint.stat().st_size * (1 if archive_only else 2) + 128*1024**2 + 1024**3
    if shutil.disk_usage(output.parent).free < reserve:
        raise OSError('Insufficient atomic archive reserve')
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    payload = extend(source, geometry)
    payload['config']['checkpoint_dir'] = str(output.resolve())
    output.mkdir(parents=True)
    atomic_save(output/'last.pt', payload)
    del payload, source
    gc.collect()
    # On Windows, two simultaneous private mappings can exhaust commit even
    # without copying tensor payloads. Fingerprint one archive per subprocess.
    verifier = ROOT/'scripts/ib/fly_interface/verify_kc_lateral_archive.py'
    record = output/'source_fingerprints.json'
    subprocess.run([sys.executable,str(verifier),'--checkpoint',str(checkpoint),
                    '--source-record',str(record),'--capture-source'],check=True)
    subprocess.run([sys.executable,str(verifier),'--checkpoint',str(output/'last.pt'),
                    '--source-record',str(record)],check=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args()
    migrate(args.checkpoint,args.geometry,args.output,args.archive_only)
