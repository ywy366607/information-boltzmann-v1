"""Align real external event receipts with same-event native head flow changes.

Modes use only the first chronological half of committed quiet samples.  This
is a descriptive alignment audit: no intervention or claim of causation.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    dirs = sorted(args.work.glob('collect_observations*'))
    quiet, event_arrays, receipts = [], {}, []
    for seg, directory in enumerate(dirs):
        with gzip.open(directory / 'operations.jsonl.gz', 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] != 'event':
                    continue
                data = row['data']
                if data.get('autonomous_event', {}).get('kind') != 'external':
                    continue
                q = data.get('shared_unknown') or {}
                joint = data.get('joint_relation_update') or {}
                receipts.append({'clock': (seg, int(row['event'])), 'unknown': q.get('identity'),
                    'revision': q.get('revision') or {}, 'uses': joint.get('variable_uses') or [],
                    'cycle': row.get('external_cycle'), 'relation': q.get('relation_identity')})
        for path in sorted(directory.glob('physical_*.npz')):
            with np.load(path, allow_pickle=False) as data:
                for i, kind in enumerate(data['kind']):
                    clock = (seg, int(data['event'][i]))
                    if kind == 'quiet_after':
                        quiet.append((clock, data['edge_flux'][i].astype(np.float64)))
                    if kind in ('event_before', 'event_after'):
                        event_arrays.setdefault(clock, {})[str(kind)] = {
                            'flux': data['edge_flux'][i].astype(np.float64),
                            'weight': data['edge_weight'][i].astype(np.float64)}
    split = len(quiet)//2
    train = np.stack([q[1] for q in quiet[:split]])
    modes = np.stack([np.linalg.svd(train[:, layer] - train[:, layer].mean(0),
                                  full_matrices=False)[2][0] for layer in range(train.shape[1])])
    cut = quiet[split-1][0]
    committed = {}
    aligned = []
    for rec in receipts:
        clock = rec['clock']
        matched = [u for u in rec['uses'] if (u.get('identity'), u.get('version')) in committed]
        if clock > cut and set(event_arrays.get(clock, {})) == {'event_before', 'event_after'}:
            before, after = event_arrays[clock]['event_before'], event_arrays[clock]['event_after']
            delta = after['flux']-before['flux']
            total = np.square(delta).sum()
            score = float(np.square((delta*modes).sum(-1)).sum()/total) if total else None
            layer_energy = np.square(delta).sum(-1)
            layer_score = np.divide(np.square((delta*modes).sum(-1)), layer_energy,
                                    out=np.zeros_like(layer_energy), where=layer_energy>1e-30)
            valid = layer_energy > 1e-30
            aligned.append({'clock': clock, 'cycle': rec['cycle'], 'relation': rec['relation'],
                'prior_committed_versions_used': len(matched), 'matching_unknowns': [u['identity'] for u in matched],
                'fraction_flow_delta_in_prior_modes': score,
                'equal_layer_mean_fraction': float(layer_score[valid].mean()) if valid.any() else None,
                'largest_layer_energy_share': float(layer_energy.max()/total) if total else None,
                'largest_layer_zero_based': int(np.argmax(layer_energy)) if total else None,
                'weight_delta_l2': float(np.linalg.norm(after['weight']-before['weight']))})
        revision = rec['revision']
        if revision.get('committed'):
            committed[(rec['unknown'], revision.get('version_after'))] = clock
    vals = [r for r in aligned if r['fraction_flow_delta_in_prior_modes'] is not None]
    yy = np.array([r['fraction_flow_delta_in_prior_modes'] for r in vals])
    labels = np.array([r['prior_committed_versions_used'] > 0 for r in vals])
    def contrast(mask):
        return float(yy[mask].mean()-yy[~mask].mean()) if mask.any() and (~mask).any() else None
    observed = contrast(labels)
    shifted = [contrast(np.roll(labels, i)) for i in range(1, len(labels))] if len(labels) else []
    out = {'mode_training_quiet_samples': split, 'mode_training_end_clock': cut,
           'evaluation_external_events': len(aligned), 'events': aligned,
           'use_vs_no_use_difference': observed, 'cyclic_label_shift_controls': shifted,
           'note': 'Same-event association, not causal; cyclic shifts preserve label order but sample size and drift limit interpretation. A receipt includes current selected relation, not an isolated memory intervention.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
