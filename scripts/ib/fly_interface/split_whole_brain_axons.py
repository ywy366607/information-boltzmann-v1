"""Axon/dendrite split of every traced cell by synapse flow centrality (Schneider-Mizell et al. 2016).

For each cell's supplied skeleton (every node kept), inputs = postsynaptic
contacts on the cell, outputs = its presynaptic contacts (each polyadic
partner counted, as in the connectome weights). For the edge between a node
and its parent the centrifugal flow is
    in_sub * (OUT - out_sub) + out_sub * (IN - in_sub)
(number of input->output paths through the edge). The split edge maximizes
it; the side with the larger output fraction is the axon. The segregation
index SI = 1 - H_split / H_cell (entropy of input/output labels) measures how
polarized the cell is; the threshold for treating a cell as polarized is a
reported choice, not a biological constant.

Output per shard: node_is_axon (bool per node), and per cell: split node,
segregation index, inputs/outputs on each side.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from numba import njit


@njit(cache=True)
def _entropy(i, o):
    n = i + o
    if n == 0 or i == 0 or o == 0:
        return 0.0
    p = i / n
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@njit(cache=True)
def split_cells(parent, counts, inputs, outputs, axon):
    """parent: cell-local parent per node (-1 roots); counts: nodes per cell. Writes axon flags in place.
    Returns per cell: split node (local, -1 none), SI, inputs axon, outputs axon, inputs dend, outputs dend."""
    n_cells = len(counts)
    result = np.zeros((n_cells, 6))
    start = 0
    for c in range(n_cells):
        n = counts[c]
        result[c, 0] = -1
        if n == 0:
            continue
        par = parent[start:start + n]
        # children lists via counting sort, BFS order from every root (forest)
        child_count = np.zeros(n + 1, np.int64)
        for v in range(n):
            if par[v] >= 0:
                child_count[par[v] + 1] += 1
        for v in range(n):
            child_count[v + 1] += child_count[v]
        children = np.empty(max(child_count[n], 1), np.int64)
        fill = child_count[:n].copy()
        for v in range(n):
            p = par[v]
            if p >= 0:
                children[fill[p]] = v
                fill[p] += 1
        order = np.empty(n, np.int64)
        end = 0
        for v in range(n):
            if par[v] < 0:
                order[end] = v
                end += 1
        cursor = 0
        while cursor < end:
            v = order[cursor]
            for k in range(child_count[v], child_count[v + 1]):
                order[end] = children[k]
                end += 1
            cursor += 1
        ins = inputs[start:start + n].astype(np.float64)
        outs = outputs[start:start + n].astype(np.float64)
        in_sub = ins.copy()
        out_sub = outs.copy()
        for j in range(n - 1, -1, -1):
            v = order[j]
            p = par[v]
            if p >= 0:
                in_sub[p] += in_sub[v]
                out_sub[p] += out_sub[v]
        IN = ins.sum()
        OUT = outs.sum()
        best, best_v = -1.0, -1
        for v in range(n):
            if par[v] < 0:
                continue
            f = in_sub[v] * (OUT - out_sub[v]) + out_sub[v] * (IN - in_sub[v])
            if f > best:
                best, best_v = f, v
        if best_v < 0 or best <= 0 or IN == 0 or OUT == 0:
            result[c, 3] = IN
            result[c, 4] = OUT
            start += n
            continue
        # side A = subtree of best_v, side B = rest
        a_in, a_out = in_sub[best_v], out_sub[best_v]
        b_in, b_out = IN - a_in, OUT - a_out
        frac_a = a_out / (a_in + a_out) if a_in + a_out > 0 else 0.0
        frac_b = b_out / (b_in + b_out) if b_in + b_out > 0 else 0.0
        subtree_is_axon = frac_a >= frac_b
        # mark subtree of best_v
        inside = np.zeros(n, np.bool_)
        inside[best_v] = True
        for j in range(n):
            v = order[j]
            p = par[v]
            if p >= 0 and inside[p]:
                inside[v] = True
        for v in range(n):
            axon[start + v] = inside[v] if subtree_is_axon else not inside[v]
        h_cell = _entropy(IN, OUT)
        total = IN + OUT
        h_split = ((a_in + a_out) * _entropy(a_in, a_out) + (b_in + b_out) * _entropy(b_in, b_out)) / total
        si = 1.0 - h_split / h_cell if h_cell > 0 else 0.0
        result[c, 0] = best_v
        result[c, 1] = si
        if subtree_is_axon:
            result[c, 2], result[c, 3], result[c, 4], result[c, 5] = a_in, a_out, b_in, b_out
        else:
            result[c, 2], result[c, 3], result[c, 4], result[c, 5] = b_in, b_out, a_in, a_out
        start += n
    return result


def main(args):
    skeletons = sorted(args.skeletons.glob('skeletons_*.npz'))
    contact_files = sorted((args.contacts / 'contacts').glob('contacts_*.npz'))
    n_shards = len(skeletons)
    counts = []
    for path in skeletons:
        with np.load(path) as z:
            counts.append(z['node_count'].astype(np.int64))
    node_totals = [int(c.sum()) for c in counts]
    inputs = [np.zeros(t, np.int32) for t in node_totals]
    outputs = [np.zeros(t, np.int32) for t in node_totals]
    for path in contact_files:
        s = int(path.stem.split('_')[1])
        with np.load(path) as z:
            post_cell, post_node = z['post_cell'].astype(np.int64), z['post_node_local'].astype(np.int64)
            site, site_node = z['site'].astype(np.int64), z['site_node_local'].astype(np.int64)
        ok = post_node >= 0
        shard_of = post_cell[ok] // args.shard
        for t in np.unique(shard_of):
            sel = shard_of == t
            np.add.at(inputs[t], post_node[ok][sel], 1)
        contact_site_node = site_node[site]
        good = contact_site_node >= 0
        np.add.at(outputs[s], contact_site_node[good], 1)
        print(json.dumps({'counted_contacts_file': s}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, path in enumerate(skeletons):
        with np.load(path) as z:
            parent, bodies = z['parent'], z['body_ids']
        axon = np.zeros(node_totals[s], np.bool_)
        started = time.perf_counter()
        result = split_cells(parent.astype(np.int64), counts[s], inputs[s], outputs[s], axon)
        np.savez(args.output / f'split_{s:04d}.npz', node_is_axon=axon, body_ids=bodies,
                 split_node=result[:, 0].astype(np.int64), segregation=result[:, 1].astype(np.float32),
                 axon_in=result[:, 2], axon_out=result[:, 3], dend_in=result[:, 4], dend_out=result[:, 5],
                 node_inputs=inputs[s], node_outputs=outputs[s])
        rows.append(result)
        print(json.dumps({'split_shard': s, 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    allr = np.concatenate(rows)
    si = allr[:, 1]
    split = allr[:, 0] >= 0
    report = {'at': datetime.now(timezone.utc).isoformat(), 'cells': int(len(allr)),
              'cells_with_split': int(split.sum()),
              'segregation_index': {q: float(np.quantile(si[split], p)) for q, p in
                                    (('min', 0), ('p25', .25), ('median', .5), ('mean', None), ('p75', .75), ('max', 1))
                                    if p is not None} | {'mean': float(si[split].mean())},
              'cells_SI_above': {str(t): int((si > t).sum()) for t in (0.05, 0.1, 0.2, 0.3, 0.5)},
              'axon_side_inputs_total': float(allr[:, 2].sum()), 'dendrite_side_inputs_total': float(allr[:, 4].sum()),
              'nodes_total': int(sum(node_totals)),
              'scope': 'Flow-centrality split on supplied coarse skeletons; polarization threshold is a choice'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--skeletons', type=Path, default=Path('F:/fly_stream_overflow/whole_brain_skeletons_v1'))
    p.add_argument('--contacts', type=Path, default=Path('E:/fly_stream_overflow/whole_brain_contacts_v1'))
    p.add_argument('--output', type=Path, default=Path('F:/fly_stream_overflow/whole_brain_split_v1'))
    p.add_argument('--shard', type=int, default=1024)
    main(p.parse_args())
