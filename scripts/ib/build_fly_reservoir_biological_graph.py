"""Build the biological fly reservoir graph with Scheme B continuous leak and dopamine modulation.

This script upgrades fly_reservoir_delayed.npz into fly_reservoir_biological.npz:
1. Calculates Scheme B continuous membrane time constant tau_m(j) and baseline leak lambda_0(j)
   purely from connectome topological in-degree (K_in), without piecewise artificial step-functions.
2. Identifies the 392 Dopamine Neurons (DANs) from body-neurotransmitters.feather.
3. Separates the 25.56M edges into:
   - 25.32M fast ionotropic transmission edges (ACh, GABA, Glu, Histamine) -> I_syn
   - 241,701 neuromodulatory dopamine edges -> M_j(t) gating membrane leak lambda_j(t).
4. Strictly preserves (delay, pre) sorting and multi-delay split boundaries for Triton kernels.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "malecns_v1"

def build():
    t0 = time.perf_counter()
    print("Loading fly_reservoir_delayed.npz...")
    delayed_npz = DATA_DIR / "fly_reservoir_delayed.npz"
    if not delayed_npz.exists():
        raise FileNotFoundError(f"Missing {delayed_npz}")
    packed = np.load(delayed_npz)

    bids = pd.Series(packed["neuron_body_ids"])
    n_neurons = len(bids)
    print(f"Total neurons: {n_neurons}")

    pre = packed["edge_pre"]
    post = packed["edge_post"]
    weight = packed["edge_weight"]
    delay = packed["edge_delay"]

    # 1. Scheme B continuous baseline leak from connectome topological in-degree
    print("Computing Scheme B continuous membrane time constants tau_m and lambda_0...")
    k_in = np.bincount(post, minlength=n_neurons).astype(np.float32)
    k_mean = float(np.mean(k_in))
    print(f"  Connectome mean in-degree K_mean = {k_mean:.2f}, median = {float(np.median(k_in)):.2f}")

    # Physical cable theory parameters (Drosophila physiology):
    # tau_min = 3.0 ms (fast primary sensory refractory limit)
    # tau_max = 78.0 ms (Kenyon cell / associative high-impedance integration limit)
    tau_min = 3.0
    tau_max = 78.0
    tau_j = tau_min + (tau_max - tau_min) * (k_in / (k_in + k_mean))
    lambda_0 = np.exp(-1.0 / tau_j).astype(np.float32)
    print(f"  lambda_0 min: {lambda_0.min():.4f}, mean: {lambda_0.mean():.4f}, max: {lambda_0.max():.4f}")

    # 2. Identify Dopamine Neurons (DANs)
    print("Identifying Dopamine Neurons from body-neurotransmitters.feather...")
    nt_path = DATA_DIR / "body-neurotransmitters.feather"
    nt = pd.read_feather(nt_path)
    nt_map = nt.set_index("body")["consensus_nt"].to_dict()
    neuron_nts = np.array([nt_map.get(int(b), "unclear") for b in bids])
    dan_mask = (neuron_nts == "dopamine")
    dan_indices = np.flatnonzero(dan_mask).astype(np.int32)
    print(f"  Found {len(dan_indices)} Dopamine Neurons (DANs).")

    # 3. Separate fast ionotropic edges vs modulatory DAN edges
    print("Separating fast ionotropic edges vs neuromodulatory DAN edges...")
    is_dan_edge = np.isin(pre, dan_indices)

    fast_pre = pre[~is_dan_edge]
    fast_post = post[~is_dan_edge]
    fast_weight = weight[~is_dan_edge]
    fast_delay = delay[~is_dan_edge]

    dan_pre = pre[is_dan_edge]
    dan_post = post[is_dan_edge]
    dan_weight = weight[is_dan_edge]
    dan_delay = delay[is_dan_edge]

    def compute_splits(d_arr):
        splits = [0]
        for d in (1, 2, 3, 4):
            splits.append(splits[-1] + int(np.sum(d_arr == d)))
        return np.array(splits, dtype=np.int32)

    fast_splits = compute_splits(fast_delay)
    dan_splits = compute_splits(dan_delay)
    print(f"  Fast edges: {len(fast_pre)}, splits: {fast_splits.tolist()}")
    print(f"  DAN edges:  {len(dan_pre)}, splits: {dan_splits.tolist()}")

    # Verify sorting preservation
    for d in (1, 2, 3, 4):
        mask_f = (fast_delay == d)
        assert np.all(np.diff(fast_pre[mask_f]) >= 0), f"fast_pre not sorted for delay {d}"
        mask_d = (dan_delay == d)
        assert np.all(np.diff(dan_pre[mask_d]) >= 0), f"dan_pre not sorted for delay {d}"

    # Characteristic biological dopamine sensitivity scale (mean non-zero target input)
    dan_in_weight_sum = np.bincount(dan_post, weights=dan_weight, minlength=n_neurons)
    dan_scale = float(np.mean(dan_in_weight_sum[dan_in_weight_sum > 0]))
    print(f"  Biological DAN sensitivity scale: {dan_scale:.6f}")

    output_path = DATA_DIR / "fly_reservoir_biological.npz"
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        # Fast ionotropic transmission graph
        edge_pre=fast_pre,
        edge_post=fast_post,
        edge_weight=fast_weight,
        edge_delay=fast_delay,
        delay_splits=fast_splits,
        # Neuromodulatory dopamine graph
        dan_edge_pre=dan_pre,
        dan_edge_post=dan_post,
        dan_edge_weight=dan_weight,
        dan_edge_delay=dan_delay,
        dan_delay_splits=dan_splits,
        dan_indices=dan_indices,
        dan_scale=np.float32(dan_scale),
        # Topological Scheme B baseline leak
        lambda_0=lambda_0,
        tau_m=tau_j,
        # Metadata and coordinates
        coords_um=packed["coords_um"],
        neuron_body_ids=packed["neuron_body_ids"],
        nt_sign=packed["nt_sign"],
        superclass_id=packed["superclass_id"],
        superclass_names=packed["superclass_names"],
        meta="fly_reservoir_biological_v1: Scheme B topological continuous leak + 241k delayed dopamine modulation synapses",
    )
    print(f"Successfully generated {output_path} ({output_path.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    build()
