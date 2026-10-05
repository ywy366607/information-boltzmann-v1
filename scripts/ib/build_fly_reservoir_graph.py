"""Build the full-neuron signed sparse graph for the level-0 fly reservoir.

Unlike build_malecns_graph.py (which coarsens 165k neurons into 256 parcels),
this builder keeps EVERY traced neuron as a node and stores the signed sparse
connectome directly: one LIF neuron per traced body, synaptic weights signed
by the presynaptic neuron's consensus neurotransmitter.  Output is an npz
consumed by information_boltzmann/core/fly_reservoir.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix

# Fast neurotransmitter sign convention (v1, documented simplification).
# gaba/glutamate/histamine are treated as inhibitory in the adult fly CNS;
# acetylcholine is the main excitatory transmitter; monoamines are grouped
# positive by default.  This table is data, not dogma - it is stored in the
# output metadata so a later arm can revise it.
NT_SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "unclear": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_full.npz"))
    args = parser.parse_args()
    data = args.data

    annotations = feather.read_table(
        data / "body-annotations.feather",
        columns=["bodyId", "status", "superclass"]).to_pandas()
    traced = annotations[annotations.status.eq("Traced")]
    body_ids = traced.bodyId.to_numpy(np.int64)
    order = np.argsort(body_ids)
    sorted_ids = body_ids[order]
    superclasses = (
        traced.superclass.fillna("unknown").astype(str).to_numpy()[order])
    n_neurons = sorted_ids.size
    print(json.dumps({"traced_neurons": int(n_neurons)}), flush=True)

    nt = feather.read_table(data / "body-neurotransmitters.feather",
                            columns=["body", "consensus_nt"]).to_pandas()
    nt_map = pd.Series(nt.consensus_nt.to_numpy(), index=nt.body.to_numpy())
    nt_names = nt_map.reindex(body_ids).fillna("unclear").to_numpy()
    nt_sign = np.array([NT_SIGN.get(name, 1) for name in nt_names], dtype=np.int8)

    lookup_type = np.argsort(sorted_ids)
    pre_list, post_list, weight_list = [], [], []
    source = feather.read_table(data / "connectome-weights.feather",
                                columns=["body_pre", "body_post", "weight"])
    batches = source.to_batches(max_chunksize=8_000_000)
    total_rows = kept_rows = 0
    for batch in batches:
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False)
        total_rows += batch.num_rows
        pre_index = np.searchsorted(sorted_ids, pre)
        post_index = np.searchsorted(sorted_ids, post)
        pre_index_clipped = np.minimum(pre_index, n_neurons - 1)
        post_index_clipped = np.minimum(post_index, n_neurons - 1)
        valid = (pre_valid := sorted_ids[pre_index_clipped] == pre) & \
                (sorted_ids[post_index_clipped] == post) & (weight > 0) & \
                (pre != post)
        if valid.any():
            pre_list.append(pre_index_clipped[valid].astype(np.int32))
            post_list.append(post_index_clipped[valid].astype(np.int32))
            weight_list.append(weight[valid].astype(np.float32))
            kept_rows += int(valid.sum())
        if (len(pre_list) and kept_rows and (len(pre_list) % 8) == 0):
            print(json.dumps({"rows": total_rows, "kept": kept_rows}), flush=True)

    pre = np.concatenate(pre_list)
    post = np.concatenate(post_list)
    weight = np.concatenate(weight_list)
    print(json.dumps({"kept_rows": int(kept_rows)}), flush=True)

    # Aggregate duplicate (pre, post) pairs via lexicographic sort + reduceat.
    keys = pre.astype(np.int64) * n_neurons + post.astype(np.int64)
    sort_order = np.argsort(keys, kind="stable")
    keys = keys[sort_order]
    weight = weight[sort_order]
    sign = nt_sign[pre[sort_order]].astype(np.float32)
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [keys.size]))
    unique_keys = keys[starts]
    agg_weight = np.add.reduceat(weight, starts)
    # Weighted-mean sign over the presynaptic neuron's transmitter (constant
    # per presynaptic neuron, so reduceat of sign*weight / weight = sign).
    agg_sign = np.add.reduceat(weight * sign, starts) / np.maximum(agg_weight, 1e-12)
    edge_pre = (unique_keys // n_neurons).astype(np.int32)
    edge_post = (unique_keys % n_neurons).astype(np.int32)
    raw_synapses = agg_weight.astype(np.float32)
    sign = agg_sign.astype(np.float32)

    # Flyvis In-degree Normalization: W_ij = (count_ij * sign_i) / sqrt(max(K_in(j), 1)) * g
    k_in = np.bincount(edge_post, minlength=n_neurons)
    in_degree_norm = np.sqrt(np.maximum(k_in[edge_post], 1.0)).astype(np.float32)
    w_norm = (raw_synapses * sign) / in_degree_norm

    matrix = csr_matrix((w_norm, (edge_pre, edge_post)), shape=(n_neurons, n_neurons))
    vector = np.random.default_rng(11).standard_normal(n_neurons)
    vector /= np.linalg.norm(vector)
    for _ in range(25):
        vector = matrix @ vector
        norm = np.linalg.norm(vector)
        vector /= max(norm, 1e-12)
    rho_unscaled = float(norm)

    g = 1.0 / max(rho_unscaled, 1e-12)
    edge_weight = (w_norm * g).astype(np.float32)
    print(json.dumps({
        "normalization": "Flyvis in-degree sqrt(K_in)",
        "unscaled_spectral_radius": rho_unscaled,
        "scaling_factor_g": g,
        "target_spectral_radius": 1.0,
    }), flush=True)

    superclass_names = sorted(set(superclasses.tolist()))
    superclass_id = np.searchsorted(np.array(superclass_names), superclasses)
    digest = hashlib.sha256(
        str(edge_weight.tobytes()).encode("utf-8")).hexdigest()[:16]
    meta = {
        "source": "MaleCNS v1.0 connectome-weights.feather",
        "license": "CC-BY",
        "neurons": int(n_neurons),
        "edges": int(edge_weight.size),
        "raw_rows": int(total_rows),
        "spectral_radius_before_scaling": radius,
        "sign_convention": NT_SIGN,
        "superclasses": superclass_names,
        "weight_digest": digest,
        "protocol": "level-0 frozen LIF reservoir, broadcast input, learned I/O",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        edge_pre=edge_pre, edge_post=edge_post, edge_weight=edge_weight,
        neuron_body_ids=body_ids, nt_sign=nt_sign,
        superclass_id=superclass_id.astype(np.int8),
        superclass_names=np.array(superclass_names),
        meta=np.array(json.dumps(meta)),
    )
    print(json.dumps({"written": str(args.output), "meta": meta}), flush=True)


if __name__ == "__main__":
    main()
