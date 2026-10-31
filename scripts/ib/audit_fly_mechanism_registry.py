"""Run the registry-level whole-brain mechanism audit.

This intentionally reports four distinct layers. A node being present in the
connectome is never promoted to a claim that its biological role is active.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("data/malecns_v1/fly_reservoir_coba_r2.npz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/malecns_v1/body-annotations.feather"))
    parser.add_argument("--registry", type=Path, default=Path("data/malecns_v1/fly_mechanism_registry.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help='Optional actual checkpoint; distinguish implemented features from enabled features')
    parser.add_argument("--system-matrix", type=Path, default=None,
                        help='Optional matched-clock matrix for this exact checkpoint')
    args = parser.parse_args()

    graph = np.load(args.graph, allow_pickle=False)
    annotations = pd.read_feather(args.annotations).set_index("bodyId").reindex(graph["neuron_body_ids"])
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    config = None
    checkpoint_info = None
    matrix = None
    if args.checkpoint:
        import torch
        saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
        config = dict(saved['config'])
        checkpoint_info = {'path': str(args.checkpoint), 'trained': int(saved['bptt_train_tokens'])}
        del saved
    if args.system_matrix:
        if not args.checkpoint:
            parser.error('--system-matrix requires --checkpoint')
        matrix = json.loads(args.system_matrix.read_text(encoding='utf-8'))
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024**2), b''):
                digest.update(block)
        if (matrix.get('sha256') != digest.hexdigest()
                or matrix.get('trained') != checkpoint_info['trained']):
            raise ValueError('Matrix is stale or belongs to a different checkpoint')
    types = annotations["type"].fillna("").astype(str)
    classes = annotations["class"].fillna("").astype(str)
    superclasses = annotations["superclass"].fillna("untyped").astype(str)
    rows = []
    for system in registry["systems"]:
        masks = []
        for pattern in system.get("cell_classes", []):
            masks.append((classes == pattern).to_numpy())
            masks.append(types.str.startswith(pattern).to_numpy())
        for pattern in system.get("cell_superclasses", []):
            masks.append((superclasses == pattern).to_numpy())
        mask = np.logical_or.reduce(masks) if masks else np.zeros(len(types), dtype=bool)
        row = {
            "id": system["id"],
            "neurons_by_annotation": int(mask.sum()),
            "anatomical_graph": "present_by_full_graph" if mask.any() else "no_matching_annotation",
            "full_dynamics": "generic_connectome_update" if mask.any() else "unmeasured",
            "specialized_role": system["current_status"],
            "specialized_fidelity_gap": system.get("specialized_fidelity_gap"),
            "evidence_scope": system.get("evidence_scope"),
            "closed_loop": "unmeasured",
            "next_unified_mechanism": system["next_unified_mechanism"],
        }
        features = system.get('runtime_features', {})
        row['runtime_features'] = ({name: bool(config.get(flag, False)) for name, flag in features.items()}
                                   if config is not None else 'checkpoint_not_inspected')
        row['acute_causal_delivery'] = 'unmeasured'
        if matrix:
            row['acute_causal_delivery'] = {
                'controls_passed': matrix.get('status') == 'controls_passed',
                'arms': {key: matrix['arms'][key] for key in system.get('causal_matrix_arms', [])
                         if key in matrix['arms']},
                'learning_or_behavior_gain': 'unmeasured',
            }
        rows.append(row)
    result = {
        "graph": str(args.graph),
        "neurons": int(len(graph["neuron_body_ids"])),
        "edges": int(len(graph["edge_pre"])),
        "registry_schema_version": registry["schema_version"],
        "checkpoint": checkpoint_info,
        "systems": rows,
        "interpretation": "generic full dynamics and specialized biological function are separate statuses",
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
