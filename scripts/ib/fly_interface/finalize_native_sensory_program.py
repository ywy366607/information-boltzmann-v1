"""Bind cross-batch native release sites and audit complete pair conservation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from compile_native_connectome_program import checksum
from link_native_sensory_geometry import atomic_json, positions


def contact_keys(values):
    return np.ascontiguousarray(values, dtype='<i8').view(
        np.dtype([(str(i), '<i8') for i in range(4)])).reshape(-1)


def main(args):
    manifest = json.loads((args.geometry / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Anatomical linker must finish with complete contact coverage')
    files = sorted(args.geometry.glob('geometry_*.npz'))
    if len(files) != manifest['completed_batches']:
        raise ValueError('Geometry file count differs from manifest')
    site_tables, sizes, identities = [], [], []
    for path in files:
        note = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if checksum(path) != note['sha256']:
            raise ValueError('Geometry digest mismatch')
        with np.load(path, allow_pickle=False) as data:
            sites = data['site_ids']
            site_tables.append(sites)
            sizes.append(len(sites))
        identities.append(note['sha256'])
    offsets = np.r_[0, np.cumsum(sizes)]
    sites = np.concatenate(site_tables)
    del site_tables
    key = contact_keys(sites)
    order = np.argsort(key)
    sorted_keys = key[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError('Physical release site assigned to multiple blocks')
    del sites, key
    bodies = np.asarray(manifest['bodies'], dtype=np.int64)
    output_dir = args.geometry / 'bindings'
    output_dir.mkdir(exist_ok=True)
    total_native_contacts = 0
    total_external_contacts = 0
    for block, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            source_ids = data['incoming_source_site_ids']
        native = positions(bodies, source_ids[:, 0]) >= 0
        at = positions(sorted_keys, contact_keys(source_ids[native]))
        if np.any(at < 0):
            raise ValueError(f'Native contact in block {block} lacks its actual output release site')
        global_site = order[at]
        source_block = np.full(len(source_ids), -1, dtype=np.int16)
        source_site = np.full(len(source_ids), -1, dtype=np.int32)
        native_block = np.searchsorted(offsets, global_site, side='right') - 1
        source_block[native] = native_block
        source_site[native] = global_site - offsets[native_block]
        target = output_dir / path.name
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, incoming_source_block=source_block,
                                incoming_source_site=source_site,
                                geometry_sha256=np.frombuffer(bytes.fromhex(identities[block]), dtype=np.uint8))
        temporary.replace(target)
        total_native_contacts += int(native.sum())
        total_external_contacts += int((~native).sum())
    with np.load(args.master, allow_pickle=False) as master:
        if checksum(args.master) != manifest['master_sha256']:
            raise ValueError('Master graph changed')
        ids = master['neuron_body_ids']
        pre = master['edge_pre']
        expected_outputs = int(np.isin(pre, np.flatnonzero(np.isin(ids, bodies))).sum())
    actual_outputs = manifest['totals']['outgoing_pair_count']
    report = {'at': datetime.now(timezone.utc).isoformat(), 'geometry': str(args.geometry.resolve()),
              'master_sha256': manifest['master_sha256'], 'geometry_sha256_by_block': identities,
              'required_return_pairs': manifest['required_pairs'],
              'localized_return_pairs': manifest['covered_pairs'],
              'native_source_output_pairs_expected': expected_outputs,
              'native_source_output_pairs_localized': actual_outputs,
              'native_source_contacts_bound_to_actual_release_sites': total_native_contacts,
              'external_source_contacts_retaining_original_cell_identity': total_external_contacts,
              'native_release_site_identity_misses': 0,
              'all_expected_output_pairs_localized': actual_outputs == expected_outputs,
              'totals': manifest['totals'],
              'runtime_active': False, 'checkpoint_migrated': False,
              'scope': 'Whole-batch anatomical binding and conservation only. Native source contacts reference actual release sites across blocks; external sources retain original cell routing. Receptor/effector kinetics, full-state integration and functional causal acceptance remain unbound.'}
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'geometry_sha256_by_block'}), flush=True)
    if not report['all_expected_output_pairs_localized']:
        raise ValueError('Some native source output pairs lack contact evidence; inspect report')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--master', type=Path, default=Path('data/malecns_v1/fly_reservoir_delayed.npz'))
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
