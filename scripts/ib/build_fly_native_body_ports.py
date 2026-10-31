"""Export actual native endpoints; never invent a muscle or receptor pairing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inventory(graph_path, annotation_path):
    graph = np.load(graph_path, allow_pickle=False)
    ids = graph['neuron_body_ids']
    if len(set(ids)) != len(ids):
        raise ValueError('Native body IDs must be unique')
    ann = pd.read_feather(annotation_path).set_index('bodyId').reindex(ids)
    superclass = ann['superclass'].fillna('')
    groups = {
        'motor_neuron': superclass.isin(['cb_motor', 'vnc_motor']).to_numpy(),
        'proprioceptor': (ann['class'].fillna('') == 'mechanosensory_proprioceptive').to_numpy(),
        'descending': (superclass == 'descending_neuron').to_numpy(),
        'ascending': (superclass == 'ascending_neuron').to_numpy(),
    }
    columns = ['type', 'superclass', 'subclass', 'somaSide', 'rootSide',
               'entryNerve', 'exitNerve', 'receptorType', 'mancBodyid', 'mancType']
    ports, counts = [], {}
    for role, mask in groups.items():
        selected = ann.loc[mask, columns].copy()
        selected.insert(0, 'bodyId', ids[mask])
        selected.insert(1, 'graph_index', np.flatnonzero(mask))
        selected.insert(2, 'role', role)
        ports.extend(json.loads(selected.to_json(orient='records')))
        counts[role] = {'neurons': int(mask.sum()),
                       'known_entry_nerve': int(selected.entryNerve.notna().sum()),
                       'known_exit_nerve': int(selected.exitNerve.notna().sum()),
                       'matched_manc_body_id': int(selected.mancBodyid.notna().sum())}
    pre, post = graph['edge_pre'], graph['edge_post']
    contacts = {}
    for source, target in [('descending', 'motor_neuron'), ('proprioceptor', 'motor_neuron'),
                           ('proprioceptor', 'ascending'), ('motor_neuron', 'ascending')]:
        select = groups[source][pre] & groups[target][post]
        contacts[f'{source}_to_{target}'] = {
            'edges': int(select.sum()), 'graph_weight': float(graph['edge_weight'][select].sum())}
    return {'schema': 'native-body-endpoints-v1', 'graph': str(graph_path),
            'annotations': str(annotation_path),
            'neuron_id_fingerprint': hashlib.sha256(ids.tobytes()).hexdigest(),
            'counts': counts, 'ports': ports, 'raw_anatomical_contacts': contacts,
            'scope': 'Native cell identities and CNS contacts, not a peripheral connection map or functional proof',
            'missing': ['motor-neuron-to-effector mapping with explicit source evidence',
                        'effector state to actual sensor-transduction mapping',
                        'actual peripheral dynamics and measured return',
                        'native motor execution into the language world without non-motor bypass']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    parser.add_argument('--annotations', type=Path, default=Path('data/malecns_v1/body-annotations.feather'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.graph, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'contacts': report['raw_anatomical_contacts']}))
