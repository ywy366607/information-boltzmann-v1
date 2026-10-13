"""Build pre-partitioned Conductance-Based (COBA) connectome graph npz.

Partitions 25.32M edges into:
- Excitatory edges (ACh): 15,519,257 edges (positive conductance weight W_E >= 0)
- Inhibitory edges (GABA, Glu, His): 9,802,138 edges (positive conductance weight W_I = |W| >= 0)
Both partitioned and sorted by axonal delay (1..4 ms) with corresponding delay splits.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np


def build_coba_graph(
    source_npz: str | Path = "data/malecns_v1/fly_reservoir_biological.npz",
    output_npz: str | Path = "data/malecns_v1/fly_reservoir_coba.npz",
    excitatory_scale: float = 1.0,
    inhibitory_scale: float = 1.0,
    no_sensory_input: bool = False,
    nt_overrides: str | Path | None = None,
    kenyon_claw_scale: float = 1.0,
    dan_to_gaba_types: tuple = (),
) -> None:
    """excitatory_scale / inhibitory_scale multiply the synapse weights (the source graph is calibrated for E_E 1.0:
    x1.82 keeps the EPSP at 0.617% of threshold under E_E 0.55; inhibition x3.67 more makes an IPSP at rest as large
    in area as an EPSP, as in the validated whole-brain LIF of Shiu et al. 2024 -- with E_I -0.15 alone it is 3.7x
    smaller and the brain ignites).  no_sensory_input drops every synapse onto a sensory neuron: those inputs sit on
    the axon terminals (presynaptic modulation); somatic in a point model, they make receptors excite each other
    (bitter GRNs fire 169 Hz on a 100 Hz drive and turn the proboscis motor neuron MN9 on).  Audit 2026-10-13."""
    source_path = Path(source_npz)
    output_path = Path(output_npz)
    print(f"Loading {source_path}...")
    t0 = time.time()
    data = dict(np.load(source_path, allow_pickle=False))

    pre = data["edge_pre"].astype(np.int64)
    post = data["edge_post"].astype(np.int64)
    weight = data["edge_weight"].astype(np.float32)
    delay = data["edge_delay"].astype(np.int32)
    sign = data["nt_sign"].copy()
    if nt_overrides is not None:
        # literature corrections of the predicted transmitter (data/malecns_v1/nt_overrides.csv: class, type regex,
        # which predicted transmitter it applies to, the corrected one, the source)
        import pandas as pd
        folder = source_path.parent
        ann = pd.read_feather(folder / "body-annotations.feather").set_index("bodyId").reindex(data["neuron_body_ids"])
        cls_a, typ_a = ann["class"].astype(str).values, ann["type"].astype(str).values
        nt = pd.read_feather(folder / "body-neurotransmitters.feather").set_index("body").reindex(data["neuron_body_ids"])
        nt_a = nt["consensus_nt"].fillna("unclear").astype(str).values
        signs = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
        applied = []
        for r in pd.read_csv(nt_overrides).itertuples():
            m = (cls_a == r[1]) & pd.Series(typ_a).str.contains(r[2], regex=True).values
            if r[3] != "any":
                m &= nt_a == r[3]
            changed = m & (sign != signs[r[4]])
            sign[m] = signs[r[4]]
            applied.append(f"{r[1]} {r[2]} ({r[3]}) -> {r[4]}: {int(m.sum())} cells, {int(changed.sum())} sign changes")
            print(applied[-1])
        data["nt_sign"] = sign
        data["nt_overrides_applied"] = np.array("; ".join(applied))
    if dan_to_gaba_types:
        # cells the transmitter prediction put in the dopamine list that are not dopaminergic in the fly (DPM:
        # serotonergic and GABAergic, Lee et al. 2011, Haynes et al. 2015): their edges leave the dopamine list and
        # become fast inhibition, and they leave dan_indices
        import pandas as pd
        typ_d = pd.read_feather(source_path.parent / "body-annotations.feather").set_index("bodyId").reindex(
            data["neuron_body_ids"])["type"].astype(str).values
        move_cells = np.nonzero(np.isin(typ_d, list(dan_to_gaba_types)))[0]
        dpre = data["dan_edge_pre"].astype(np.int64)
        mv = np.isin(dpre, move_cells)
        pre = np.concatenate([pre, dpre[mv]]); post = np.concatenate([post, data["dan_edge_post"].astype(np.int64)[mv]])
        weight = np.concatenate([weight, -np.abs(data["dan_edge_weight"][mv]).astype(np.float32)])
        delay = np.concatenate([delay, data["dan_edge_delay"].astype(np.int32)[mv]])
        sign[move_cells] = -1
        keep = ~mv
        order = np.lexsort((data["dan_edge_pre"][keep], data["dan_edge_delay"][keep]))
        for k in ("dan_edge_pre", "dan_edge_post", "dan_edge_weight", "dan_edge_delay"):
            data[k] = data[k][keep][order]
        data["dan_delay_splits"] = np.array([0] + [int((data["dan_edge_delay"] <= d).sum()) for d in (1, 2, 3, 4)], dtype=np.int32)
        data["dan_indices"] = np.setdiff1d(data["dan_indices"], move_cells).astype(np.int32)
        tot = np.bincount(data["dan_edge_post"], weights=data["dan_edge_weight"], minlength=len(sign))
        data["dan_scale"] = np.float32(np.mean(tot[tot > 0]))      # same definition as the biological builder
        print(f"Dopamine list -> fast GABA: {list(dan_to_gaba_types)} ({len(move_cells)} cells, {int(mv.sum()):,} edges)")
    pre_sign = sign[pre]

    exc_mask = (pre_sign > 0)
    inh_mask = (pre_sign < 0)
    if no_sensory_input:
        names = np.array(data["superclass_names"]).astype(str)[data["superclass_id"]]
        sensory = np.char.find(names, "sensory") >= 0
        onto = sensory[post]
        print(f"Dropping {int(onto.sum()):,} edges onto {int(sensory.sum()):,} sensory neurons (axo-axonal)")
        exc_mask &= ~onto
        inh_mask &= ~onto

    print(f"Total edges: {len(pre):,}")
    print(f"Excitatory edges (ACh): {int(exc_mask.sum()):,}")
    print(f"Inhibitory edges (GABA, Glu, His): {int(inh_mask.sum()):,}")

    def partition_and_sort(mask, is_abs: bool):
        p_pre = pre[mask]
        p_post = post[mask]
        p_w = np.abs(weight[mask]) if is_abs else weight[mask]
        p_delay = delay[mask]

        # Stable sort by delay 1..4
        order = np.argsort(p_delay, kind="stable")
        p_pre = p_pre[order].astype(np.int32)
        p_post = p_post[order].astype(np.int32)
        p_w = p_w[order].astype(np.float32)
        p_delay = p_delay[order].astype(np.int32)

        splits = [0]
        for d in (1, 2, 3, 4):
            splits.append(int(np.searchsorted(p_delay, d, side="right")))
        return p_pre, p_post, p_w, p_delay, np.array(splits, dtype=np.int64)

    print("Partitioning and sorting excitatory edges...")
    pre_e, post_e, w_e, delay_e, splits_e = partition_and_sort(exc_mask, is_abs=False)
    w_e = (w_e * excitatory_scale).astype(np.float32)
    if kenyon_claw_scale != 1.0:
        # the Kenyon cells' dendritic claws: one PN spike gives ~2-4 mV on a 15-20 mV rest-to-threshold distance
        # (~15-20%; Gruntman 2013 thesis / Gruntman & Turner 2013, which also find >= 3 active claws are needed to
        # fire reliably); the uniform 0.617% per synapse gives ~10% for a median claw (16 synapses): x1.8
        import pandas as pd
        cls_k = pd.read_feather(source_path.parent / "body-annotations.feather").set_index("bodyId").reindex(
            data["neuron_body_ids"])["class"].astype(str).values == "Kenyon_Cell"
        claw = cls_k[post_e] & ~cls_k[pre_e]
        w_e[claw] *= kenyon_claw_scale
        print(f"Kenyon claw inputs x{kenyon_claw_scale}: {int(claw.sum()):,} edges")

    print("Partitioning and sorting inhibitory edges...")
    pre_i, post_i, w_i, delay_i, splits_i = partition_and_sort(inh_mask, is_abs=True)
    w_i = (w_i * excitatory_scale * inhibitory_scale).astype(np.float32)
    data["coba_conversion"] = np.array(f"excitatory_scale={excitatory_scale} inhibitory_scale={inhibitory_scale} "
                                       f"no_sensory_input={no_sensory_input} kenyon_claw_scale={kenyon_claw_scale} dan_to_gaba={list(dan_to_gaba_types)}")

    data["edge_pre_e"] = pre_e
    data["edge_post_e"] = post_e
    data["edge_weight_e"] = w_e
    data["edge_delay_e"] = delay_e
    data["delay_splits_e"] = splits_e

    data["edge_pre_i"] = pre_i
    data["edge_post_i"] = post_i
    data["edge_weight_i"] = w_i
    data["edge_delay_i"] = delay_i
    data["delay_splits_i"] = splits_i

    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, **data)
    elapsed = time.time() - t0
    print(f"Saved {output_path} ({output_path.stat().st_size / 1e6:.1f} MB) in {elapsed:.2f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/malecns_v1/fly_reservoir_biological.npz')
    ap.add_argument('--output', default='data/malecns_v1/fly_reservoir_coba.npz')
    ap.add_argument('--excitatory-scale', type=float, default=1.0)
    ap.add_argument('--inhibitory-scale', type=float, default=1.0)
    ap.add_argument('--no-sensory-input', action='store_true')
    ap.add_argument('--nt-overrides', default=None, help='CSV of literature transmitter corrections (data/malecns_v1/nt_overrides.csv)')
    ap.add_argument('--kenyon-claw-scale', type=float, default=1.0)
    ap.add_argument('--dan-to-gaba-types', default='', help='comma-separated types moved from the dopamine list to fast GABA (DPM)')
    a = ap.parse_args()
    build_coba_graph(a.source, a.output, a.excitatory_scale, a.inhibitory_scale, a.no_sensory_input, a.nt_overrides,
                     a.kenyon_claw_scale, tuple(t for t in a.dan_to_gaba_types.split(',') if t))
