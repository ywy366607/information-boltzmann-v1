"""Synaptic efficacy by the receiving cell's own membrane (2026-10-13).  A point neuron with one leak conductance for
every cell makes a big neuron's drive grow with its synapse count: at birth the total excitatory weight onto a cell
scales as size^0.93 (r 0.71; motor neurons 26 thresholds, descending neurons 21, optic-lobe intrinsic 2.5), so the large
output neurons sit near saturation and are the ones that latch.  In a real neuron the leak conductance grows with the
membrane (thin neurites: area ~ volume), so each synapse moves a big cell less.  Here every synapse onto cell i is scaled
by median(size) / size_i (cells without a size measurement: 1), excitatory and inhibitory alike (E/I balance kept), the
median cell unchanged.  Writes a new graph npz next to the source with the same arrays.
Usage: python size_normalize_graph.py <source.npz> <out.npz>
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
g = dict(np.load(src, allow_pickle=True))
ids = g['neuron_body_ids']
size = pd.read_feather(src.parent / 'neuron_sizes.feather').set_index('bodyId').reindex(ids)['size'].values.astype(float)
ok = np.isfinite(size) & (size > 0)
scale = np.ones(len(ids))
scale[ok] = np.median(size[ok]) / size[ok]
for key, post in (('edge_weight_e', 'edge_post_e'), ('edge_weight_i', 'edge_post_i'), ('edge_weight', 'edge_post')):
    g[key] = (g[key] * scale[g[post].astype(np.int64)]).astype(g[key].dtype)
N = len(ids)
E = np.bincount(g['edge_post_e'].astype(np.int64), weights=g['edge_weight_e'], minlength=N) / 0.1
sc = np.array(g['superclass_names'])[g['superclass_id']].astype(str)
summary = pd.Series(E).groupby(sc).median().sort_values(ascending=False).round(2).head(12).to_dict()
q = np.quantile(scale[ok], [0.01, 0.1, 0.5, 0.9, 0.99])
g['size_normalization'] = np.array(json.dumps({'rule': 'w_ij *= median(size) / size_i', 'cells_with_size': int(ok.sum()),
                                               'scale_q01_q10_q50_q90_q99': [round(float(x), 3) for x in q]}))
np.savez(out, **g)
print('scale quantiles 1/10/50/90/99%:', np.round(q, 3))
print('total excitatory input after (thresholds), median by superclass:', summary)
