"""Whole-batch numerical admission of native contact/transport operands.

No world, spike model, reward, optimizer or animal is instantiated. These are
conservation and endpoint checks on real supplied anatomy, not learning tests.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.core.fly_apl_cable import diffuse
from information_boltzmann.core.fly_kc_terminals import CellwiseGeometry
from link_native_sensory_geometry import atomic_json


def main(args):
    manifest = json.loads((args.geometry / 'manifest.json').read_text(encoding='utf-8'))
    paths = sorted(args.geometry.glob('geometry_*.npz'))
    if manifest['status'] != 'complete' or len(paths) != manifest['completed_batches']:
        raise ValueError('Require a complete, stable geometry set before unified admission')
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            volume = data['volume']
            owner = data['owner']
            first, second, g = data['u'], data['v'], data['g']
            contacts, sites = data['incoming_node'], data['site_node']
            cells = len(data['cell_body_ids'])
            n = len(volume)
            if len(data['node_ids']) != n or np.any(owner[first] != owner[second]):
                raise ValueError('Native morphology identity or intercell axial link violated')
            if np.any(contacts < 0) or np.any(contacts >= n) or np.any(sites < 0) or np.any(sites >= n):
                raise ValueError('Invalid local endpoint')
            recipient = data['incoming_source_site_ids']  # only used for contact count below
            if len(recipient) != len(contacts):
                raise ValueError('Contact identity was pooled away')
        geometry = CellwiseGeometry(volume, first, second, g, owner)
        deposited = np.bincount(contacts, minlength=n).astype(np.float64)
        values = torch.from_numpy(np.stack((np.ones(n), deposited / volume)))
        # Zero is log(1), a numerical transport fixture, not a claimed fitted
        # sensory-material parameter or biological diffusion time.
        with torch.no_grad():
            result = diffuse(values, torch.zeros(cells, dtype=torch.float64), geometry).numpy()
        constant_error = float(np.max(np.abs(result[0] - 1)))
        before = np.bincount(owner, weights=deposited, minlength=cells)
        after = np.bincount(owner, weights=result[1] * volume, minlength=cells)
        mass_error = float(np.max(np.abs(after - before) / np.maximum(before, 1)))
        if constant_error > 1e-8 or mass_error > 1e-8 or result[1].min(initial=0) < -1e-8:
            raise ValueError('Native conservative transport numerical admission failed')
        # Static root labels expose supplied disconnected trees instead of
        # inventing bridges so that every input can reach every output.
        root = np.arange(n, dtype=np.int32)
        for node in geometry.order:
            parent = geometry.parent[node]
            if parent >= 0:
                root[node] = root[parent]
        input_roots = np.unique(root[contacts])
        output_roots = np.unique(root[sites])
        connected_contacts = int(np.isin(root[contacts], output_roots).sum())
        connected_sites = int(np.isin(root[sites], input_roots).sum())
        record = {'batch': path.stem, 'cells': cells, 'nodes': n, 'sites': len(sites),
                  'contacts': len(contacts), 'constant_error': constant_error,
                  'per_cell_mass_relative_error': mass_error,
                  'contacts_in_a_supplied_tree_with_output_sites': connected_contacts,
                  'output_sites_in_a_supplied_tree_with_input_contacts': connected_sites,
                  'field_output_finite': bool(np.isfinite(result).all())}
        rows.append(record)
        print(json.dumps(record), flush=True)
        del geometry, values, result
    report = {'at': datetime.now(timezone.utc).isoformat(), 'blocks_checked': len(rows),
              'passed': True, 'totals': {k: sum(r[k] for r in rows) for k in ('cells', 'nodes', 'sites', 'contacts',
                  'contacts_in_a_supplied_tree_with_output_sites', 'output_sites_in_a_supplied_tree_with_input_contacts')},
              'maximum_constant_error': max(r['constant_error'] for r in rows),
              'maximum_per_cell_mass_relative_error': max(r['per_cell_mass_relative_error'] for r in rows),
              'per_block': rows, 'full_brain_runtime_checked': False, 'learning_checked': False,
              'scope': 'All supplied sensory geometry blocks: original endpoint validity, distinct cells/roots, positive conservative tree transport. Unit log-diffusion is a numerical fixture, not measured biophysics. Not receptor/effector validation, full-state restore, actual world-clock circuit delivery or functional learning evidence.'}
    atomic_json(args.output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
