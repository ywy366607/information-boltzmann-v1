"""Analyze recorded native energy flows and relation versions without replay.

This reports descriptive modes and recorded dependencies, not causal effects.
Each segment has its own event clock; physical before/after pairs never cross it.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('work', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    dirs = sorted(a.work.glob('collect_observations*'))
    flow, harmonic_flow, quiet_segment, pairs, counts = [], [], [], collections.defaultdict(list), collections.Counter()
    revisions, mechanisms, worlds, update_records, snapshots, real_receipts = [], [], [], [], [], []
    for segment, directory in enumerate(dirs):
        if not (directory / 'operations.jsonl.gz').exists():
            continue
        pending = {}
        for path in sorted(directory.glob('physical_*.npz')):
            with np.load(path, allow_pickle=False) as data:
                for i, kind in enumerate(data['kind']):
                    event = int(data['event'][i])
                    key = (event, str(kind).removesuffix('_before').removesuffix('_after'))
                    f = np.asarray(data['edge_flux'][i], dtype=np.float64)
                    w = np.asarray(data['edge_weight'][i], dtype=np.float64)
                    if kind == 'quiet_after':
                        flow.append(f)
                        harmonic_flow.append(np.asarray(data['edge_graph_harmonic'][i], dtype=np.float64))
                        quiet_segment.append(segment)
                    if kind in ('quiet_before', 'update_before', 'event_before'):
                        pending[key] = (f, w)
                    elif kind in ('quiet_after', 'update_after', 'event_after') and key in pending:
                        oldf, oldw = pending.pop(key)
                        pairs[str(kind)].append((np.linalg.norm(f-oldf), np.linalg.norm(w-oldw)))
        with gzip.open(directory / 'operations.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                kind, data = row['kind'], row['data']
                clock = (segment, int(row['event']))
                counts[kind] += 1
                if kind == 'relation_revision':
                    revisions.append((clock, data.get('identity'), data.get('version_before'),
                                      data.get('version_after'), data.get('observed'), data.get('kind')))
                elif kind == 'mechanism':
                    mechanisms.append((clock, data.get('identity'), data.get('generation'),
                                       data.get('version'), data.get('kind')))
                elif kind == 'world':
                    worlds.append(clock)
                elif kind == 'structural_relation_dependencies':
                    snapshots.append((clock, {tuple(e) for e in data.get('edges', [])}))
                elif kind == 'event':
                    update_records.append((clock, data.get('autonomous_event', {}).get('kind'), data.get('updates', 0)))
                    if data.get('autonomous_event', {}).get('kind') == 'external':
                        q = data.get('shared_unknown') or {}
                        joint = data.get('joint_relation_update') or {}
                        real_receipts.append(dict(clock=clock, unknown=q.get('identity'),
                            recalled=q.get('recalled'), relation=q.get('relation_identity'),
                            revision=q.get('revision') or {}, uses=joint.get('variable_uses') or []))
    x = np.asarray(flow)
    if not len(x):
        raise ValueError('No committed quiet observations')
    # PCA is performed separately within each real layer, on its measured 28
    # head-pair fluxes. Singular values describe variance, not causal modes.
    layer_modes, heldout_modes, continuity, cycle_fraction = [], [], [], []
    harmonic_flow = np.asarray(harmonic_flow)
    for layer in range(x.shape[1]):
        z = x[:, layer, :] - x[:, layer, :].mean(axis=0)
        singular = np.linalg.svd(z, compute_uv=False)
        energy = singular ** 2
        layer_modes.append(float(energy[0] / energy.sum()) if energy.sum() else None)
        # Estimate the principal fluctuation on the first half and score it on
        # the second half, with its center fixed from the first half.
        split = len(x) // 2
        first, second = x[:split, layer, :], x[split:, layer, :]
        center = first.mean(axis=0)
        _, _, vh = np.linalg.svd(first-center, full_matrices=False)
        held = second-center
        denom = np.square(held).sum()
        heldout_modes.append(float(np.square(held @ vh[0]).sum()/denom) if denom else None)
        same_segment = np.array(quiet_segment[1:]) == np.array(quiet_segment[:-1])
        left, right = x[:-1, layer, :][same_segment], x[1:, layer, :][same_segment]
        norms = np.linalg.norm(left, axis=1)*np.linalg.norm(right, axis=1)
        good = norms > 1e-20
        continuity.append(float(np.median((left[good]*right[good]).sum(1)/norms[good])) if good.any() else None)
        d = np.square(x[:,layer,:]).sum()
        cycle_fraction.append(float(np.square(harmonic_flow[:,layer,:]).sum()/d) if d else None)
    changes = [dict(added=len(b-a), removed=len(a-b))
               for (_, a), (_, b) in zip(snapshots, snapshots[1:])]
    structural_changes = []
    for (_, old), (_, new) in zip(snapshots, snapshots[1:]):
        # relation id, unknown id, port, generation; last value is seen version
        old_edges = {entry[:4] for entry in old}
        new_edges = {entry[:4] for entry in new}
        structural_changes.append(dict(added=len(new_edges-old_edges), removed=len(old_edges-new_edges)))
    committed = [r for r in real_receipts if r['revision'].get('committed')]
    commit_reuse = []
    for r in committed:
        version = r['revision'].get('version_after')
        later = [dict(clock=s['clock'], relation=s['relation'], use=u)
                 for s in real_receipts if s['clock'] > r['clock']
                 for u in s['uses'] if u.get('identity') == r['unknown'] and u.get('version') == version]
        commit_reuse.append(dict(clock=r['clock'], unknown=r['unknown'], version=version, later_uses=later))
    summary = {
        'scope': 'committed native observations from uninterrupted segments; no training or intervention',
        'segments': len(dirs), 'committed_quiet_samples': len(x), 'world_events': len(worlds),
        'event_kinds': dict(collections.Counter(k for _, k, _ in update_records)),
        'recorded_optimizer_updates': sum(v for _, _, v in update_records),
        'flow_pca_top_variance_by_layer': layer_modes,
        'flow_pca_first_half_to_second_fraction_by_layer': heldout_modes,
        'flow_adjacent_quiet_cosine_median_by_layer': continuity,
        'graph_cycle_energy_fraction_by_layer': cycle_fraction,
        'physical_pairs': {k: {'count': len(v), 'median_flux_change_l2': float(np.median(np.array(v)[:, 0])),
                               'median_weight_change_l2': float(np.median(np.array(v)[:, 1]))}
                           for k, v in pairs.items() if v},
        'relation_revision_records': len(revisions),
        'revision_observed_flags': dict(collections.Counter(str(r[4]) for r in revisions)),
        'dependency_snapshots': len(snapshots),
        'dependency_changes_between_snapshots': changes,
        'dependency_edge_changes_ignoring_seen_version': structural_changes,
        'real_question_receipts': {'total': len(real_receipts), 'recalled': sum(bool(r['recalled']) for r in real_receipts),
                                   'committed': len(committed), 'committed_then_later_used': sum(bool(r['later_uses']) for r in commit_reuse),
                                   'commits': commit_reuse},
        'caveat': 'U revise returns a functional preview and does not commit or increment version. Mechanism records include candidate previews. Dependency differences include version metadata; they do not prove actual reuse. PCA is descriptive.',
        'counts': dict(counts),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
