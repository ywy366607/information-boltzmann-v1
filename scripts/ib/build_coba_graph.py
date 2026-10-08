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
) -> None:
    source_path = Path(source_npz)
    output_path = Path(output_npz)
    print(f"Loading {source_path}...")
    t0 = time.time()
    data = dict(np.load(source_path, allow_pickle=False))

    pre = data["edge_pre"].astype(np.int64)
    post = data["edge_post"].astype(np.int64)
    weight = data["edge_weight"].astype(np.float32)
    delay = data["edge_delay"].astype(np.int32)
    sign = data["nt_sign"]
    pre_sign = sign[pre]

    exc_mask = (pre_sign > 0)
    inh_mask = (pre_sign < 0)

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

    print("Partitioning and sorting inhibitory edges...")
    pre_i, post_i, w_i, delay_i, splits_i = partition_and_sort(inh_mask, is_abs=True)

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
    build_coba_graph()
