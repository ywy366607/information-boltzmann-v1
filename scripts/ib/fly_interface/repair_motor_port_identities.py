"""Explicit zero-exposure repair of the first motor-port checkpoint exporter.

The first exporter retained parameters but omitted the ordered body-ID buffer.
Recover it from the checkpoint's graph and annotations, save a separate complete
branch, and prove that every pre-existing payload field stayed identical.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import sha256
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest


def repair(source: Path, output: Path, identity_reference: Path):
    if output.exists():
        raise FileExistsError('Repair requires a new branch directory')
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if not saved['config'].get('native_motor_port') or 'native_motor_body_ids' in saved['model']:
        raise ValueError('Only the initial identity-omitting motor-port export is repairable here')
    graph_path = Path(saved['config']['graph'])
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    ann_path = graph_path.parent / 'body-annotations.feather'
    graph = np.load(graph_path, allow_pickle=False)
    ann = pd.read_feather(ann_path).set_index('bodyId').reindex(graph['neuron_body_ids'])
    index = np.flatnonzero(ann['superclass'].isin(['cb_motor', 'vnc_motor']).to_numpy())
    reference = json.loads(identity_reference.read_text(encoding='utf-8'))
    prior_ids = reference.get('motor_terminal_body_ids')
    if prior_ids != graph['neuron_body_ids'][index].tolist():
        raise ValueError('Recorded original channel identities do not match the current graph')
    if saved['model']['motor_port_projection.weight'].shape[1] != len(index):
        raise ValueError('Projection and anatomical source have different channel counts')
    before_sha = sha256(source)
    before = state_digest(saved)
    old_origin = saved['origin']
    saved['model'] = dict(saved['model'])
    saved['model']['native_motor_body_ids'] = torch.as_tensor(graph['neuron_body_ids'][index].copy())
    migration = {'at': datetime.now(timezone.utc).isoformat(), 'source': str(source),
                 'source_sha256': before_sha, 'graph_sha256': sha256(graph_path),
                 'annotations_sha256': sha256(ann_path), 'new_exposure': 0,
                 'identity_reference': str(identity_reference),
                 'identity_reference_sha256': sha256(identity_reference),
                 'reason': 'parameter-only exporter omitted ordered motor body IDs',
                 'scope': 'recover from the same graph; not anatomical remapping'}
    saved['origin'] = copy.deepcopy(old_origin)
    saved['origin']['native_motor_identity_repair'] = migration
    output.mkdir(parents=True)
    target = output / 'last.pt'
    temporary = output / 'last.pt.tmp'
    torch.save(saved, temporary)
    temporary.replace(target)
    reloaded = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
    actual_ids = reloaded['model']['native_motor_body_ids']
    assert torch.equal(actual_ids, saved['model']['native_motor_body_ids'])
    check = dict(reloaded)
    check['model'] = {k:v for k,v in check['model'].items() if k != 'native_motor_body_ids'}
    check['origin'] = old_origin
    if before != state_digest(check) or sha256(source) != before_sha:
        raise RuntimeError('Repair failed exact full-state/source preservation')
    report = {**migration, 'candidate': str(target), 'candidate_sha256': sha256(target),
              'trained': int(saved['bptt_train_tokens']), 'motor_cells': len(index),
              'preexisting_payload_exactly_preserved': True, 'source_unchanged': True,
              'status': 'identity_export_repaired_restore_pending'}
    (output/'checkpoint_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity-reference', type=Path, required=True,
                        help='Pre-export audit containing the original ordered motor_terminal_body_ids')
    args = parser.parse_args()
    repair(args.checkpoint, args.output, args.identity_reference)
