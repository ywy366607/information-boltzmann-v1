"""Is there a route for a corollary discharge (static, signed, absolute birth weights)?

The speaking act is read from the output cells (descending, motor, efferent).  A corollary discharge needs their
spikes to reach the auditory pathway: Johnston's organ (JO-A/B) -> its first-order targets (AMMC / WED cells with
>= MIN_SHARE of their excitatory input from JO-A/B) -> second-order.  The full CNS graph has the loop
DN -> VNC -> ascending neurons -> brain.  Signed transfer v_{k+1} = W^T v_k from the output cells (unit spike each),
summed onto the auditory sets per hop, against 20 random cell sets of the same size and superclass mix.
Usage: python efference_route_probe.py [hops=4]"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp

HOPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MIN_SHARE = 0.05
g = np.load('data/malecns_v1/fly_reservoir_coba.npz', allow_pickle=True)
ids = g['neuron_body_ids']; n = len(ids)
ann = pd.read_feather('data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(ids)
typ = ann['type'].astype(str).values; sup = ann['superclass'].astype(str).values
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]); post = np.concatenate([g['edge_post_e'], g['edge_post_i']])
w = np.concatenate([np.abs(g['edge_weight_e']), -np.abs(g['edge_weight_i'])]).astype(np.float64)
W = sp.csr_matrix((w, (pre, post)), shape=(n, n))          # W[i, j]: i -> j, signed
We = sp.csr_matrix((np.abs(g['edge_weight_e']).astype(np.float64), (g['edge_pre_e'], g['edge_post_e'])), shape=(n, n))
jo = np.flatnonzero(pd.Series(typ).str.match('^(JO-A|JO-B)').values)
exc_in = np.asarray(We.sum(0)).ravel()
from_jo = np.asarray(We[jo].sum(0)).ravel()
share = np.where(exc_in > 0, from_jo / np.maximum(exc_in, 1e-30), 0)
first = np.setdiff1d(np.flatnonzero(share >= MIN_SHARE), jo)
from_first = np.asarray(We[first].sum(0)).ravel()
second = np.setdiff1d(np.flatnonzero(np.where(exc_in > 0, from_first / np.maximum(exc_in, 1e-30), 0) >= MIN_SHARE),
                      np.concatenate([jo, first]))
out_cls = ('descending_neuron', 'vnc_motor', 'cb_motor', 'vnc_efferent')
src = np.flatnonzero(np.isin(sup, out_cls))
print(f'JO-A/B {jo.size}; auditory first-order {first.size} (top types {pd.Series(typ[first]).value_counts().head(6).to_dict()})')
print(f'second-order {second.size}; output cells {src.size}')
WT = W.T.tocsr()

def transfer(cells):
    v = np.zeros(n); v[cells] = 1.0
    rows = []
    for k in range(1, HOPS + 1):
        v = WT @ v
        rows.append((v[first].sum(), v[second].sum(), v[jo].sum()))
    return np.array(rows)

real = transfer(src)
rng = np.random.default_rng(0)
ctrl = []
for _ in range(20):
    pick = np.concatenate([rng.choice(np.flatnonzero(sup == s), (sup[src] == s).sum(), replace=False)
                           for s in np.unique(sup[src])])
    ctrl.append(transfer(rng.choice(np.flatnonzero(~np.isin(sup, out_cls)), src.size, replace=False)))
ctrl = np.array(ctrl)
print('hop | first-order: output cells  [random non-output sets: min | median | max] | second-order: output  [random]')
for k in range(HOPS):
    c1, c2 = ctrl[:, k, 0], ctrl[:, k, 1]
    print(f'{k+1} | {real[k,0]:+.3e} [{c1.min():+.2e} | {np.median(c1):+.2e} | {c1.max():+.2e}] | '
          f'{real[k,1]:+.3e} [{c2.min():+.2e} | {np.median(c2):+.2e} | {c2.max():+.2e}]')
# which routes carry it: the classes of the cells that pass the output cells' signal to the first-order set at hop 2
v1 = WT @ (lambda v: (v.__setitem__(src, 1.0), v)[1])(np.zeros(n))
contrib = sp.diags(v1) @ W[:, first]
by = pd.Series(np.asarray(contrib.sum(1)).ravel(), index=sup).groupby(level=0).sum().sort_values()
print('hop-2 relay into first-order, by superclass (signed):', {k: f'{v:+.2e}' for k, v in by.items() if abs(v) > 0})
asc = np.flatnonzero(sup == 'ascending_neuron')
print(f'ascending neurons -> first-order direct (signed sum): {np.asarray(W[asc][:, first].sum()):+.3e}; '
      f'output cells -> ascending (hop 1..2 reach): {transfer(src)[0,0]:+.2e}')
