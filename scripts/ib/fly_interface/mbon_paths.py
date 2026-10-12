"""Where can the mushroom body act?  Connectome only (MaleCNS graph, birth weights), no model.

Influence: A[post, pre] = w(pre -> post) / total input weight of post (each cell's input shares; signed by the
presynaptic transmitter).  The share of a target's input that comes from a source set after exactly h steps is
(A^h)[target, source] summed over the source set (unsigned: |A|; signed: A).  Reported for hops 1..4:
  MBON -> descending neurons (the act surface the language read already uses), per MBON valence class
  MBON -> DAN (the feedback that lets dopamine carry a prediction error)
the minimal cumulative delay (ticks) from any MBON to each descending neuron, and the descending neurons with the
largest MBON influence within 2 hops (type, sign).
Usage: python mbon_paths.py
"""
import json, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from pathlib import Path

ROOT = Path('D:/information-boltzmann-v1')
graph = ROOT / 'data/malecns_v1/fly_reservoir_coba.npz'
g = np.load(graph, allow_pickle=True)
N = len(g['neuron_body_ids'])
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, typ = ann['class'].astype(str).values, ann['type'].astype(str).values
names = list(g['superclass_names']); sup = np.array(names)[g['superclass_id']]
mbon = np.flatnonzero(cls == 'MBON')
dn = np.flatnonzero(sup == 'descending_neuron')
dan = g['dan_indices'].astype(np.int64)
pam = dan[[t.startswith('PAM') for t in typ[dan]]]
ppl1 = dan[[t.startswith('PPL1') for t in typ[dan]]]

pre, post, delay = g['edge_pre'].astype(np.int64), g['edge_post'].astype(np.int64), g['edge_delay']
w = np.abs(g['edge_weight'])                                # magnitudes; the sign comes from the transmitter
sign = g['nt_sign'][pre].astype(np.float32)
W = sp.csr_matrix((w, (post, pre)), shape=(N, N))
inp = np.asarray(W.sum(1)).ravel(); inp[inp == 0] = 1
A_abs = sp.diags(1 / inp) @ W
A_sig = sp.diags(1 / inp) @ sp.csr_matrix((w * sign, (post, pre)), shape=(N, N))

# MBON valence: dopamine input from PAM (appetitive) vs PPL1 (aversive) through the modulatory edges
dpre, dpost, dw = g['dan_edge_pre'].astype(np.int64), g['dan_edge_post'].astype(np.int64), g['dan_edge_weight']
P = np.bincount(dpost[np.isin(dpre, pam)], dw[np.isin(dpre, pam)], N)
Q = np.bincount(dpost[np.isin(dpre, ppl1)], dw[np.isin(dpre, ppl1)], N)
val = (Q - P) / np.maximum(Q + P, 1e-12)                     # > 0: PPL1-dominated (learned avoidance), < 0: PAM
groups = {'all_MBON': mbon, 'PPL1_dominated': mbon[(Q + P)[mbon] > 0][val[mbon][(Q + P)[mbon] > 0] > 0],
          'PAM_dominated': mbon[(Q + P)[mbon] > 0][val[mbon][(Q + P)[mbon] > 0] < 0]}


def reach(A, source, targets, hops=4):
    x = np.zeros(N, np.float64); x[source] = 1.0
    out = []
    for _ in range(hops):
        x = A @ x
        out.append(x[targets])
    return out


rep = {'n_MBON': len(mbon), 'n_DN': len(dn), 'n_PAM': len(pam), 'n_PPL1': len(ppl1),
       'n_PPL1_dominated_MBON': len(groups['PPL1_dominated']), 'n_PAM_dominated_MBON': len(groups['PAM_dominated'])}
for name, src in groups.items():
    ua, sa = reach(A_abs, src, dn), reach(A_sig, src, dn)
    rep[f'{name}_to_DN'] = [{'hop': h + 1, 'mean_input_share': float(u.mean()), 'max_input_share': float(u.max()),
                             'DN_with_share_over_1pct': int((u > 0.01).sum()), 'signed_mean': float(s.mean())}
                            for h, (u, s) in enumerate(zip(ua, sa))]
# feedback for a prediction error: MBON -> DAN through the main (fast) edges (DAN -> MBON is the modulatory list)
for name, src, tgt in (('MBON_to_PAM', mbon, pam), ('MBON_to_PPL1', mbon, ppl1)):
    ua, sa = reach(A_abs, src, tgt, 2), reach(A_sig, src, tgt, 2)
    rep[name] = [{'hop': h + 1, 'mean_input_share': float(u.mean()), 'max_input_share': float(u.max()),
                  'targets_over_1pct': int((u > 0.01).sum()), 'signed_mean': float(s.mean())}
                 for h, (u, s) in enumerate(zip(ua, sa))]
# minimal cumulative delay MBON -> each DN (edges as unit-weight hops with their delay as length)
D = sp.csr_matrix((delay.astype(np.float32), (pre, post)), shape=(N, N))
dist = dijkstra(D, indices=mbon, min_only=True)[dn]
fin = dist[np.isfinite(dist)]
rep['MBON_to_DN_min_delay_ticks'] = {'reachable': int(len(fin)), 'min': float(fin.min()), 'median': float(np.median(fin)),
                                     'max': float(fin.max()), 'histogram': {int(k): int((fin == k).sum()) for k in np.unique(fin)}}
# the descending neurons most under MBON influence within two hops
u1, u2 = reach(A_abs, mbon, dn, 2)
s1, s2 = reach(A_sig, mbon, dn, 2)
top = np.argsort(-(u1 + u2))[:25]
rep['top_DN'] = [{'type': typ[dn[i]], 'hop1': round(float(u1[i]), 4), 'hop2': round(float(u2[i]), 4),
                  'signed_1plus2': round(float(s1[i] + s2[i]), 4)} for i in top]
print(json.dumps(rep, indent=1))
out = ROOT / 'results/published/fly_consequence_20261012/mbon_paths.json'
out.write_text(json.dumps(rep, indent=1))
