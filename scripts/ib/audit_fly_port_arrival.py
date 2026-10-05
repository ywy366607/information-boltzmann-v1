"""Structural port-to-output arrival bounds from the actual delayed connectome.

No neural forward or learning. Positive-delay min-plus propagation gives exact
earliest *possible* arrival within the reported horizon. Spike thresholds,
synaptic filters and learned weights can delay or suppress actual information;
topological reachability is not measured predictive contribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def earliest_arrival(n_neurons, source, edges, horizon):
    if horizon < 1:
        raise ValueError('Positive arrival horizon required')
    distance = np.full(n_neurons, horizon+1, dtype=np.int32)
    distance[source] = 0
    for _ in range(horizon):
        previous = distance
        distance = previous.copy()
        for pre, post, delay in edges:
            if np.any(delay < 1):
                raise ValueError('Chemical delay bounds require positive delays')
            np.minimum.at(distance, post, previous[pre]+delay)
        np.minimum(distance, horizon+1, out=distance)
        if np.array_equal(distance, previous):
            break
    return distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--partitions', type=Path, default=Path('data/malecns_v1/sensory_partitions.npz'))
    parser.add_argument('--output', type=Path, default=Path('results/published/fly_port_arrival_bounds.json'))
    args = parser.parse_args()
    with np.load(args.graph) as graph, np.load(args.partitions) as parts:
        names = graph['superclass_names'].tolist()
        ids = graph['superclass_id']
        output_names = ('cb_motor', 'vnc_motor', 'descending_neuron',
                        'cb_efferent', 'vnc_efferent', 'efferent_ascending',
                        'efferent_descending', 'cb_endocrine', 'vnc_endocrine')
        output = np.flatnonzero(np.isin(ids, [names.index(n) for n in output_names if n in names]))
        source = np.concatenate([parts[k] for k in ('visual_idx', 'chemo_idx', 'mechano_idx')])
        edges = [(graph['edge_pre_'+s], graph['edge_post_'+s], graph['edge_delay_'+s])
                 for s in ('e', 'i')]
        horizon = max(int(delay.max()) for _, _, delay in edges)
        distance = earliest_arrival(len(ids), source, edges, horizon)
        rows = []
        for ticks in range(1, horizon+1):
            reached = int((distance[output] <= ticks).sum())
            rows.append({'quiet_delay_ticks': ticks, 'output_nodes_reachable': reached,
                         'output_fraction_reachable': reached/len(output)})
        report = {
            'graph': str(args.graph), 'partitions': str(args.partitions),
            'input_nodes': len(source), 'output_nodes': len(output),
            'input_output_overlap': len(np.intersect1d(source, output)),
            'edges': sum(len(pre) for pre, _, _ in edges),
            'horizon_basis': 'maximum configured single-edge delay; all paths within horizon included',
            'earliest_possible_arrival': rows,
            'scope': ('Static anatomical reachability with positive delays, independent of learned '
                      'weights/thresholds/filters. Actual functional arrival can be later or absent. '
                      'The input pulse creates a ring entry; delay d requires d subsequent quiet ticks. '
                      'This report does not prescribe a new token interval or claim causal language value.')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
