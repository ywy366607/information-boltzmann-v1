"""Build the delay-aware connectome graph with physical axonal transmission delays.

Extracts 3D spatial soma coordinates from MaleCNS v1.0 body annotations (8nm EM space),
completes missing coordinates via 1-hop neighbor harmonic centroids, computes physical
axonal cable distances (in micrometers), and calculates realistic biological conduction
delays based on unmyelinated Drosophila axon conduction velocity (0.3 m/s = 300 um/ms)
and synaptic delay (0.8 ms).

Edges are grouped and sorted by delay d in {1, 2, 3, 4} ms and presynaptic index for
maximum GPU memory coalescing and zero-divergence fused execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-graph", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_full.npz"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/malecns_v1/body-annotations.feather"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/malecns_v1/fly_reservoir_delayed.npz"))
    parser.add_argument("--velocity-um-per-ms", type=float, default=300.0,
                        help="Unmyelinated axon conduction velocity in um/ms (0.3 m/s)")
    parser.add_argument("--synaptic-delay-ms", type=float, default=0.8,
                        help="Chemical synaptic transmission delay in ms")
    parser.add_argument("--max-delay-steps", type=int, default=4,
                        help="Maximum discrete delay steps")
    args = parser.parse_args()

    print(f"Loading base graph from {args.base_graph}...")
    base = np.load(args.base_graph)
    body_ids = base["neuron_body_ids"]
    pre = base["edge_pre"]
    post = base["edge_post"]
    weights = base["edge_weight"]
    n_neurons = len(body_ids)
    n_edges = len(pre)
    print(f"Base graph: {n_neurons} neurons, {n_edges} edges")

    print(f"Loading annotations from {args.annotations}...")
    ann = feather.read_table(args.annotations).to_pandas()
    ann_dict = ann.set_index("bodyId").to_dict("index")

    # 1. 3D Coordinates (8nm EM resolution -> um)
    coords = np.full((n_neurons, 3), np.nan, dtype=np.float32)
    for i, bid in enumerate(body_ids):
        row = ann_dict.get(bid)
        if row and row.get("somaLocation") is not None and len(row["somaLocation"]) == 3:
            coords[i] = row["somaLocation"]

    known = ~np.isnan(coords[:, 0])
    coords_um = coords * 0.008  # 1 voxel = 8nm = 0.008 um
    print(f"Known exact somas: {np.sum(known)} / {n_neurons} ({np.sum(known)/n_neurons:.1%})")

    # Harmonic completion for neurons without traced soma
    neighbor_sum = np.zeros_like(coords_um)
    neighbor_count = np.zeros(n_neurons, dtype=np.float32)

    mask_post_known = known[post]
    np.add.at(neighbor_sum, pre[mask_post_known], coords_um[post[mask_post_known]])
    np.add.at(neighbor_count, pre[mask_post_known], 1.0)

    mask_pre_known = known[pre]
    np.add.at(neighbor_sum, post[mask_pre_known], coords_um[pre[mask_pre_known]])
    np.add.at(neighbor_count, post[mask_pre_known], 1.0)

    has_neighbors = (neighbor_count > 0) & (~known)
    coords_um[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors, None]
    still_missing = np.isnan(coords_um[:, 0])
    global_mean = np.nanmean(coords_um, axis=0)
    coords_um[still_missing] = global_mean
    print(f"Completed coordinates: {np.sum(has_neighbors)} by 1-hop neighbors, {np.sum(still_missing)} by global mean")

    # 2. Euclidean / physical cable distance between pre and post neurons
    diff = coords_um[pre] - coords_um[post]
    dist_um = np.sqrt(np.sum(diff ** 2, axis=1)).astype(np.float32)
    print(f"Distances (um): min = {dist_um.min():.1f}, median = {np.median(dist_um):.1f}, mean = {dist_um.mean():.1f}, max = {dist_um.max():.1f}")

    # 3. Physical delays
    delays_ms = dist_um / args.velocity_um_per_ms + args.synaptic_delay_ms
    delays_steps = np.clip(np.round(delays_ms), 1, args.max_delay_steps).astype(np.int32)

    print("Delay distribution:")
    for d in range(1, args.max_delay_steps + 1):
        cnt = int(np.sum(delays_steps == d))
        print(f"  Delay {d} ms: {cnt} edges ({cnt / n_edges:.1%})")

    # 4. Sort edges by (delay, pre, post) for optimal memory coalescing and segmented execution
    print("Sorting edges by (delay, pre, post)...")
    # Composite sort key: delay * (N^2) + pre * N + post
    sort_order = np.lexsort((post, pre, delays_steps))
    sorted_pre = pre[sort_order]
    sorted_post = post[sort_order]
    sorted_weights = weights[sort_order]
    sorted_delays = delays_steps[sort_order]

    # Calculate split boundaries for each delay
    delay_splits = [0]
    for d in range(1, args.max_delay_steps + 1):
        count = int(np.sum(sorted_delays == d))
        delay_splits.append(delay_splits[-1] + count)
    delay_splits = np.array(delay_splits, dtype=np.int32)
    print(f"Delay split indices: {delay_splits.tolist()}")

    # 5. Save enhanced delayed graph
    meta = json.loads(str(base["meta"])) if "meta" in base else {}
    meta["physical_delays"] = {
        "velocity_um_per_ms": args.velocity_um_per_ms,
        "synaptic_delay_ms": args.synaptic_delay_ms,
        "max_delay_steps": args.max_delay_steps,
        "delay_splits": delay_splits.tolist(),
        "mean_distance_um": float(dist_um.mean()),
        "median_distance_um": float(np.median(dist_um)),
    }

    print(f"Saving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        edge_pre=sorted_pre,
        edge_post=sorted_post,
        edge_weight=sorted_weights,
        edge_delay=sorted_delays,
        delay_splits=delay_splits,
        coords_um=coords_um.astype(np.float32),
        neuron_body_ids=body_ids,
        nt_sign=base["nt_sign"],
        superclass_id=base["superclass_id"],
        superclass_names=base["superclass_names"],
        meta=np.array(json.dumps(meta)),
    )
    print(f"Successfully generated {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
