"""Structural audit of the maleCNS connectome for the circuits that can hold activity without input
(no simulation): positive-feedback loops (latch / ignition candidates) and cells that must be
driven from inside (spontaneous-activity candidates).

Graph: the repository's signed r2 graph (data/malecns_v1/fly_reservoir_coba_r2.npz, the fly model's
own wiring: Shiu scale, transmitter overrides applied). Synapse counts are recovered as
weight / smallest weight per sign (Kenyon claw edges carry the 1.8 claw scale, so their counts are
overstated by that factor). A strong edge has at least `--min_syn` synapses.

Latch candidates (positive feedback, the sign product of the loop is +):
  E<->E  reciprocal excitatory pairs (recurrent excitation);
  I<->I  reciprocal inhibitory pairs (mutual inhibition: a flip-flop);
  excitatory strongly connected components of the strong excitatory graph.
Validation: the head-direction ring (EPG <-> PEN, Seelig & Jayaraman 2015; Turner-Evans 2017) must
appear among the E<->E pairs and in one excitatory component.

Spontaneous candidates: the wiring cannot show intrinsic currents, so two structural readings:
  annotated clock neurons (pacemakers in the fly: s-LNv, l-LNv, LNd, DN1, DN2, LPN, 5th s-LNv);
  central (non-sensory) cells with strong output and almost no synaptic input: whatever activity
  they have must come from inside the cell or from neuromodulation.

Usage: python scripts/ib/connectome_persistence_motifs.py <out.json> [--min_syn 5]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as F
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
CLOCK = ("s-LNv", "l-LNv", "LNd_b", "LNd_c", "5thsLNv_LNd6", "DN1a", "DN1pA", "DN1pB", "LN-DN1", "LN-DN2",
         "LPN_a", "LPN_b")


def dist(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    return dict(min=float(x.min()), median=float(np.median(x)), mean=float(x.mean()), max=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--graph", default="data/malecns_v1/fly_reservoir_coba_r2.npz")
    ap.add_argument("--min_syn", type=int, default=5)
    a = ap.parse_args()
    z = np.load(ROOT / a.graph)
    N = len(z["neuron_body_ids"])
    sc_names = [str(s) for s in z["superclass_names"]]
    sc = np.array([sc_names[i] for i in z["superclass_id"]])
    ann = F.read_table(ROOT / "data/malecns_v1/body-annotations.feather", columns=["bodyId", "type"]).to_pandas()
    tmap = dict(zip(ann["bodyId"].to_numpy(), ann["type"].fillna("").to_numpy()))
    typ = np.array([tmap.get(int(b), "") for b in z["neuron_body_ids"]], dtype=object)

    mats = {}
    for s in ("e", "i"):
        w = z["edge_weight_" + s]
        n = np.round(w / w.min()).astype(np.int32)
        mats[s] = sp.csr_matrix((n, (z["edge_pre_" + s], z["edge_post_" + s])), shape=(N, N))
        mats[s].sum_duplicates()
    allsyn = mats["e"] + mats["i"]
    syn_in = np.asarray(allsyn.sum(0)).ravel()
    syn_out = np.asarray(allsyn.sum(1)).ravel()

    out = dict(graph=a.graph, min_syn=a.min_syn, neurons=N)

    def strong(M):
        S = M.copy()
        S.data = (S.data >= a.min_syn).astype(np.int8)
        S.eliminate_zeros()
        S.setdiag(0)
        S.eliminate_zeros()
        return S

    def pairs(S):
        R = S.multiply(S.T).tocoo()
        m = R.row < R.col
        return R.row[m], R.col[m]

    def summarize_pairs(i, j, M):
        st = Counter(tuple(sorted((typ[x] or "?", typ[y] or "?"))) for x, y in zip(i, j))
        ss = Counter(tuple(sorted((sc[x], sc[y]))) for x, y in zip(i, j))
        loop = np.minimum(np.asarray(M[i, j]).ravel(), np.asarray(M[j, i]).ravel())
        return dict(pairs=int(len(i)), cells=int(len(np.unique(np.concatenate([i, j])))),
                    weaker_direction_synapses=dist(loop),
                    top_type_pairs=[[f"{p[0]} <-> {p[1]}", c] for p, c in st.most_common(25)],
                    superclass_pairs=[[f"{p[0]} <-> {p[1]}", c] for p, c in ss.most_common(12)])

    Se, Si = strong(mats["e"]), strong(mats["i"])
    ie, je = pairs(Se)
    ii, ji = pairs(Si)
    out["EE_reciprocal"] = summarize_pairs(ie, je, mats["e"])
    out["II_reciprocal"] = summarize_pairs(ii, ji, mats["i"])

    ncomp, lab = connected_components(Se, directed=True, connection="strong")
    sizes = np.bincount(lab)
    big = np.argsort(-sizes)[:8]
    comps = []
    for c in big:
        if sizes[c] < 2:
            break
        mem = np.nonzero(lab == c)[0]
        comps.append(dict(size=int(sizes[c]), superclasses=Counter(sc[mem]).most_common(8),
                          types=Counter(typ[mem]).most_common(15)))
    out["excitatory_SCC"] = dict(nontrivial=int((sizes >= 2).sum()), cells_in_nontrivial=int(sizes[sizes >= 2].sum()),
                                 size=dist(sizes[sizes >= 2]), largest=comps)

    # validation: the head-direction ring
    epg = np.isin(typ, ["EPG"])
    pen = np.isin(typ, ["PEN_a(PEN1)", "PEN_b(PEN2)"])
    ring_pairs = int(sum((epg[x] and pen[y]) or (pen[x] and epg[y]) for x, y in zip(ie, je)))
    ring_lab = Counter(lab[epg | pen])
    out["validation_head_direction_ring"] = dict(EPG=int(epg.sum()), PEN=int(pen.sum()), EPG_PEN_reciprocal_pairs=ring_pairs,
                                                 components_holding_EPG_PEN=[[int(k), int(v), int(sizes[k])] for k, v in ring_lab.most_common(3)])

    # spontaneous candidates
    clock = np.isin(typ, CLOCK)
    out["clock_neurons"] = dict(cells=int(clock.sum()), types=Counter(typ[clock]).most_common(),
                                synapses_in=dist(syn_in[clock]), synapses_out=dist(syn_out[clock]),
                                in_EE_reciprocal=int(np.isin(np.nonzero(clock)[0], np.concatenate([ie, je])).sum()))
    sensory = np.char.find(sc.astype(str), "sensory") >= 0
    central = ~sensory
    low = central & (syn_out >= 100) & (syn_in < 0.02 * syn_out)
    out["inputless_central"] = dict(rule="non-sensory, >= 100 output synapses, input < 2% of output", cells=int(low.sum()),
                                    superclasses=Counter(sc[low]).most_common(10), types=Counter(typ[low]).most_common(25),
                                    synapses_in=dist(syn_in[low]), synapses_out=dist(syn_out[low]))
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                           encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in
                     ("pairs", "cells", "nontrivial", "cells_in_nontrivial", "EPG_PEN_reciprocal_pairs",
                      "components_holding_EPG_PEN", "types", "superclasses")}) for k, v in out.items()},
                     default=str)[:6000])


if __name__ == "__main__":
    main()
