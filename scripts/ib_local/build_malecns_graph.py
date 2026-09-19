"""Coarsen the official MaleCNS v1.0 connectome into a trainable graph field.

The raw edge table contains roughly 152 million neuron-to-neuron records.  This
script streams the Feather record batches and only materializes the final
parcel graph.  Parcels are fitted in physical soma coordinates; connectivity
then determines graph neighborhoods, Laplacian modes and collision matchings.
"""
import os
import sys

# Prevent scripts/ib_local/types.py from shadowing the standard library.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.cluster import MiniBatchKMeans


NT_NAMES = (
    "acetylcholine", "gaba", "glutamate", "histamine",
    "dopamine", "octopamine", "serotonin", "unclear",
)

INPUT_PORT_NAMES = (
    "central_brain_sensory", "optic_lobe_sensory",
    "vnc_sensory", "sensory_relay",
)
OUTPUT_PORT_NAMES = ("motor", "efferent", "endocrine", "other_exit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lookup(sorted_ids: np.ndarray, values: np.ndarray):
    positions = np.searchsorted(sorted_ids, values)
    clipped = np.minimum(positions, len(sorted_ids) - 1)
    valid = (positions < len(sorted_ids)) & (sorted_ids[clipped] == values)
    return clipped, valid


def _collision_matchings(adjacency, coordinates, layers):
    parcels = adjacency.shape[0]
    symmetric = adjacency + adjacency.T
    np.fill_diagonal(symmetric, 0)
    rows, columns = np.triu_indices(parcels, 1)
    scores = symmetric[rows, columns]
    useful = scores > 0
    rows, columns, scores = rows[useful], columns[useful], scores[useful]
    base_order = np.argsort(scores)[::-1]
    previously_used = set()
    all_pairs, all_features = [], []
    for layer in range(layers):
        # Rotate equal-strength ties between layers while retaining strong edges.
        tie = ((rows * 1315423911 + columns * 2654435761 + layer * 97531)
               & 0xffff) / 65536.
        order = np.lexsort((tie, -scores))
        used, pairs = np.zeros(parcels, dtype=bool), []
        for index in order:
            i, j = int(rows[index]), int(columns[index])
            edge = (i, j)
            if edge in previously_used or used[i] or used[j]:
                continue
            used[i] = used[j] = True
            pairs.append(edge)
            previously_used.add(edge)
            if len(pairs) == parcels // 2:
                break
        # A dense parcel graph normally gives a perfect matching.  Coordinate
        # neighbors are a deterministic fallback for isolated parcels.
        remaining = np.flatnonzero(~used).tolist()
        while len(remaining) >= 2:
            i = remaining.pop(0)
            distances = np.square(coordinates[remaining] - coordinates[i]).sum(1)
            take = int(np.argmin(distances))
            j = remaining.pop(take)
            pairs.append((i, j))
        pairs = np.asarray(pairs, dtype=np.int64)
        features = []
        for i, j in pairs:
            forward, backward = adjacency[i, j], adjacency[j, i]
            total = forward + backward
            direction = coordinates[j] - coordinates[i]
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            features.append((np.log1p(total), *direction,
                             (forward - backward) / max(total, 1.)))
        all_pairs.append(pairs)
        all_features.append(np.asarray(features, dtype=np.float32))
    edge_features = np.stack(all_features)
    edge_features[..., 0] /= max(float(edge_features[..., 0].max()), 1e-8)
    return np.stack(all_pairs), edge_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parcels", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--collision-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.modes % 2 or args.modes >= args.parcels:
        raise ValueError("modes must be even and smaller than parcels")
    output = args.output or args.data / f"malecns_parcels_{args.parcels}.npz"
    annotations_path = args.data / "body-annotations.feather"
    nt_path = args.data / "body-neurotransmitters.feather"
    edges_path = args.data / "connectome-weights.feather"

    annotations = pd.read_feather(
        annotations_path,
        columns=["bodyId", "status", "somaLocation", "superclass",
                 "class", "entryNerve", "exitNerve"])
    traced = annotations[annotations.status.eq("Traced")].copy()
    has_position = traced.somaLocation.notna().to_numpy()
    spatial = traced.loc[has_position]
    coordinates_raw = np.stack(spatial.somaLocation).astype(np.float64)
    low, high = np.quantile(coordinates_raw, [.005, .995], axis=0)
    coordinates_normalized = np.clip(
        (coordinates_raw - low) / np.maximum(high - low, 1.), 0, 1)

    clustering = MiniBatchKMeans(
        args.parcels, random_state=args.seed, batch_size=8192,
        n_init=3, max_iter=200, reassignment_ratio=.005)
    labels_spatial = clustering.fit_predict(coordinates_normalized)
    body_ids = spatial.bodyId.to_numpy(np.int64)
    order = np.argsort(body_ids)
    sorted_ids, sorted_labels = body_ids[order], labels_spatial[order]

    superclass = traced.superclass.fillna("").astype(str)
    input_group = np.full(len(traced), -1, dtype=np.int64)
    input_group[superclass.eq("cb_sensory").to_numpy()] = 0
    input_group[superclass.eq("ol_sensory").to_numpy()] = 1
    input_group[superclass.eq("vnc_sensory").to_numpy()] = 2
    input_group[superclass.isin(
        ("sensory_ascending", "sensory_descending")).to_numpy()] = 3
    input_rows = np.flatnonzero(input_group >= 0)
    input_ids = traced.bodyId.to_numpy(np.int64)[input_rows]
    input_groups = input_group[input_rows]
    input_order = np.argsort(input_ids)
    sorted_input_ids = input_ids[input_order]
    sorted_input_groups = input_groups[input_order]

    adjacency = np.zeros((args.parcels, args.parcels), dtype=np.float64)
    input_port_weights = np.zeros(
        (len(INPUT_PORT_NAMES), args.parcels), dtype=np.float64)
    included_edges = included_weight = total_rows = 0
    source = pa.memory_map(str(edges_path), "r")
    reader = ipc.RecordBatchFileReader(source)
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False)
        pre_index, pre_valid = _lookup(sorted_ids, pre)
        post_index, post_valid = _lookup(sorted_ids, post)
        input_index, input_valid = _lookup(sorted_input_ids, pre)
        input_boundary = input_valid & post_valid
        if input_boundary.any():
            group = sorted_input_groups[input_index[input_boundary]]
            target = sorted_labels[post_index[input_boundary]]
            flat_input = group * args.parcels + target
            input_port_weights += np.bincount(
                flat_input, weights=weight[input_boundary],
                minlength=input_port_weights.size).reshape(
                    input_port_weights.shape)
        valid = pre_valid & post_valid
        if valid.any():
            source_parcel = sorted_labels[pre_index[valid]]
            target_parcel = sorted_labels[post_index[valid]]
            flat = source_parcel * args.parcels + target_parcel
            adjacency += np.bincount(
                flat, weights=weight[valid],
                minlength=args.parcels ** 2).reshape(adjacency.shape)
            included_edges += int(valid.sum())
            included_weight += int(weight[valid].sum())
        total_rows += batch.num_rows
        if (batch_index + 1) % 250 == 0:
            print(json.dumps({"batches": batch_index + 1,
                              "total_batches": reader.num_record_batches,
                              "rows": total_rows}), flush=True)

    counts = np.bincount(labels_spatial, minlength=args.parcels).astype(np.float64)
    parcel_coordinates = np.stack([
        np.bincount(labels_spatial, weights=coordinates_normalized[:, axis],
                    minlength=args.parcels) / np.maximum(counts, 1)
        for axis in range(3)
    ], axis=1)

    nt = pd.read_feather(nt_path, columns=["body", "consensus_nt"])
    nt_map = pd.Series(nt.consensus_nt.to_numpy(), index=nt.body.to_numpy())
    spatial_nt = nt_map.reindex(body_ids).fillna("unclear").to_numpy()
    nt_distribution = np.stack([
        np.bincount(labels_spatial, weights=(spatial_nt == name),
                    minlength=args.parcels)
        for name in NT_NAMES
    ], axis=1).astype(np.float64)
    nt_distribution /= np.maximum(nt_distribution.sum(1, keepdims=True), 1)

    spatial_superclass = spatial.superclass.fillna("").astype(str).to_numpy()
    spatial_exit = spatial.exitNerve.notna().to_numpy()
    output_group = np.full(len(spatial), -1, dtype=np.int64)
    output_group[np.isin(spatial_superclass, ("cb_motor", "vnc_motor"))] = 0
    output_group[np.isin(
        spatial_superclass,
        ("cb_efferent", "vnc_efferent", "efferent_ascending",
         "efferent_descending"))] = 1
    output_group[np.isin(
        spatial_superclass, ("cb_endocrine", "vnc_endocrine"))] = 2
    output_group[(output_group < 0) & spatial_exit] = 3
    output_port_weights = np.stack([
        np.bincount(labels_spatial, weights=(output_group == group),
                    minlength=args.parcels)
        for group in range(len(OUTPUT_PORT_NAMES))
    ]).astype(np.float64)

    no_self = adjacency.copy()
    np.fill_diagonal(no_self, 0)
    symmetric = no_self + no_self.T
    degree = symmetric.sum(1)
    normalized = symmetric / np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1.))
    laplacian = np.eye(args.parcels) - normalized
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    basis = eigenvectors[:, 1:args.modes + 1]
    mode_values = eigenvalues[1:args.modes + 1]

    neighbor_indices = np.argsort(symmetric, axis=1)[:, -args.neighbors:][:, ::-1]
    neighbor_weights = np.take_along_axis(symmetric, neighbor_indices, axis=1)
    neighbor_weights /= np.maximum(neighbor_weights.sum(1, keepdims=True), 1.)
    pairs, edge_features = _collision_matchings(
        adjacency, parcel_coordinates, args.collision_layers)

    inbound, outbound = no_self.sum(0), no_self.sum(1)
    structural = np.stack((np.log1p(inbound), np.log1p(outbound),
                           np.log1p(counts)), axis=1)
    structural /= np.maximum(structural.max(0, keepdims=True), 1e-8)
    node_features = np.concatenate(
        (parcel_coordinates, structural, nt_distribution,
         np.log1p(input_port_weights.sum(0))[:, None],
         np.log1p(output_port_weights.sum(0))[:, None]), axis=1).astype(np.float32)
    node_features[:, -2:] /= np.maximum(
        node_features[:, -2:].max(0, keepdims=True), 1e-8)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coordinates=parcel_coordinates.astype(np.float32),
        coordinate_scale=np.maximum(high - low, 1.).astype(np.float32),
        node_features=node_features,
        adjacency=adjacency.astype(np.float32),
        input_port_weights=input_port_weights.astype(np.float32),
        output_port_weights=output_port_weights.astype(np.float32),
        neighbor_indices=neighbor_indices.astype(np.int64),
        neighbor_weights=neighbor_weights.astype(np.float32),
        laplacian_basis=basis.astype(np.float32),
        laplacian_eigenvalues=mode_values.astype(np.float32),
        collision_pairs=pairs,
        collision_features=edge_features,
    )
    metadata = {
        "source": "MaleCNS v1.0",
        "license": "CC-BY",
        "parcels": args.parcels,
        "neighbors": args.neighbors,
        "modes": args.modes,
        "collision_layers": args.collision_layers,
        "input_port_names": list(INPUT_PORT_NAMES),
        "output_port_names": list(OUTPUT_PORT_NAMES),
        "input_port_nonzero_parcels": [
            int((row > 0).sum()) for row in input_port_weights],
        "output_port_nonzero_parcels": [
            int((row > 0).sum()) for row in output_port_weights],
        "traced_neurons": int(len(traced)),
        "spatial_traced_neurons": int(len(spatial)),
        "raw_edges": int(total_rows),
        "included_edges": included_edges,
        "included_synapse_weight": included_weight,
        "input_port_names": INPUT_PORT_NAMES,
        "output_port_names": OUTPUT_PORT_NAMES,
        "input_port_synapse_weight": input_port_weights.sum(1).tolist(),
        "output_port_neuron_count": output_port_weights.sum(1).astype(int).tolist(),
        "coordinate_quantiles": {"low": low.tolist(), "high": high.tolist()},
        "files_sha256": {
            path.name: _sha256(path)
            for path in (annotations_path, nt_path, edges_path)
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
