"""Compare every old field of a source archive with a zero-exposure migrated archive.

Both archives are memory-mapped and hashed one entry at a time, so the check
fits beside other work on a 16 GB host. Reports added fields separately.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from scripts.ib.fly_interface.native_causal_audit import sha256


def compare(source: Path, migrated: Path, report_path: Path):
    old = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    new = torch.load(migrated, map_location='cpu', mmap=True, weights_only=False)
    unequal, added = [], []
    for key in old:
        if key not in new:
            unequal.append(key + ' (missing)')
            continue
        if isinstance(old[key], dict) and key in ('model', 'learner', 'config'):
            for name, value in old[key].items():
                actual = new[key].get(name)
                if name == 'physical' and key == 'learner':
                    for field, tensor in value.items():
                        if field not in actual or state_digest(tensor) != state_digest(actual[field]):
                            unequal.append(f'learner.physical.{field}')
                    added += [f'learner.physical.{f}' for f in actual if f not in value]
                elif state_digest(value) != state_digest(actual):
                    unequal.append(f'{key}.{name}')
                gc.collect()
            added += [f'{key}.{n}' for n in new[key] if n not in old[key]]
        elif state_digest(old[key]) != state_digest(new[key]):
            unequal.append(key)
    added += [k for k in new if k not in old]
    report = {'source': str(source), 'source_sha256': sha256(source),
              'migrated': str(migrated), 'migrated_sha256': sha256(migrated),
              'migrated_bytes': migrated.stat().st_size,
              'old_fields_changed': unequal, 'fields_added': added,
              'status': 'old_fields_exact' if not unequal else 'not_accepted'}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1))
    if unequal:
        raise RuntimeError('Migrated archive changed old fields')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--migrated', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    compare(a.source, a.migrated, a.report)
