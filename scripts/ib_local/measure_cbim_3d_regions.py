"""Measure emergent spatial specialization in a frozen CBIM-3D checkpoint.

The probe uses held-out OpenWebText only. It measures persistent spatial
heterogeneity, write/read routing diversity, and readout-only causal lesions of
the eight torus octants without changing the recurrent dynamics.
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
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.cbim_operator3d import CBIMOperator3D


def normalized_entropy(values: torch.Tensor) -> float:
    probability = values.double().clamp_min(0)
    probability = probability / probability.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy / math.log(probability.numel()))


def token_category(token_id: int, encoding) -> str:
    if encoding is None:
        return "all"
    value = encoding.decode_single_token_bytes(token_id)
    if value.isspace():
        return "space"
    if value.isalpha():
        return "alpha"
    if value.isdigit():
        return "numeric"
    if all(not chr(byte).isalnum() and not chr(byte).isspace() for byte in value):
        return "punctuation"
    return "mixed"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=32768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=2048)
    parser.add_argument("--readout-batch", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    shape = tuple(config["shape"])
    channels = int(config["channels"])
    if shape != (4, 4, 4):
        parser.error("The current octant probe expects a 4x4x4 checkpoint")
    model = CBIMOperator3D(shape=shape, d=channels).cuda().eval()
    model.load_state_dict(saved["model"])
    stream = np.load(args.data / "validation.npy", mmap_mode="r")
    end = args.start + args.burn_in + args.score_tokens + 1
    if end > len(stream):
        parser.error("Requested held-out window exceeds validation stream")

    try:
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
    except ImportError:
        encoding = None

    ids_cpu = np.asarray(stream[args.start:end - 1], dtype=np.int64)
    targets_cpu = np.asarray(stream[args.start + 1:end], dtype=np.int64)
    ids = torch.as_tensor(ids_cpu, device="cuda")
    multiplier, _ = model.transport.multiplier()
    position = model.readout.position_encoding()
    state = torch.zeros(1, *model.state_shape, device="cuda")
    snapshots = []

    for index, token in enumerate(ids):
        state, _ = model.source(state, token.view(1))
        state = model.transport.apply_multiplier(state, multiplier)
        state, _ = model.scattering(state)
        if index >= args.burn_in:
            snapshots.append(state[0].clone())
    states = torch.stack(snapshots)
    targets = torch.as_tensor(targets_cpu[args.burn_in:], device="cuda")
    grid_points = math.prod(shape)
    flat_states = states.reshape(args.score_tokens, grid_points, channels)

    energy_maps = 0.5 * flat_states.square().sum(-1)
    mean_energy = energy_maps.mean(0)
    temporal_cv = energy_maps.std(0, unbiased=False) / mean_energy.clamp_min(1e-8)
    centered = energy_maps - energy_maps.mean(1, keepdim=True)
    neighbor = torch.zeros_like(states[..., 0])
    scalar_energy = 0.5 * states.square().sum(-1)
    for axis in (1, 2, 3):
        neighbor += torch.roll(scalar_energy, 1, axis) + torch.roll(scalar_energy, -1, axis)
    neighbor /= 6
    spatial_neighbor_correlation = torch.corrcoef(torch.stack((
        scalar_energy.flatten(), neighbor.flatten())))[0, 1]

    lag_correlations = {}
    for lag in (1, 8, 64, 256):
        left, right = centered[:-lag].flatten(1), centered[lag:].flatten(1)
        numerator = (left * right).sum(1)
        denominator = left.norm(dim=1) * right.norm(dim=1)
        lag_correlations[str(lag)] = float(
            (numerator / denominator.clamp_min(1e-8)).mean())

    embeddings = model.source.embedding(ids[args.burn_in:])
    centers = torch.sigmoid(model.source.address(embeddings))
    cell_indices = torch.floor(centers * torch.tensor(shape, device="cuda")).long()
    linear_indices = ((cell_indices[:, 0] * shape[1] + cell_indices[:, 1])
                      * shape[2] + cell_indices[:, 2])
    write_counts = torch.bincount(linear_indices, minlength=grid_points)

    encoded = model.readout.norm(flat_states)
    keys = model.readout.key(encoded + position)
    queries = model.readout.query.expand(args.score_tokens, -1, -1)
    heads = model.readout.heads
    width = channels // heads
    keys = keys.reshape(args.score_tokens, grid_points, heads, width).transpose(1, 2)
    queries = queries.reshape(args.score_tokens, -1, heads, width).transpose(1, 2)
    attention = torch.softmax(queries @ keys.transpose(-2, -1) / math.sqrt(width), -1)
    mean_attention = attention.mean(0)
    attention_entropies = [[normalized_entropy(mean_attention[h, q])
                            for q in range(model.readout.queries)]
                           for h in range(heads)]
    distributions = mean_attention.reshape(-1, grid_points)
    mean_distribution = distributions.mean(0)
    query_js = (distributions * (
        distributions.clamp_min(1e-12).log()
        - mean_distribution.clamp_min(1e-12).log())).sum(1).mean()

    masks = []
    octant_names = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                mask = torch.ones(*shape, 1, device="cuda")
                mask[x * 2:(x + 1) * 2,
                     y * 2:(y + 1) * 2,
                     z * 2:(z + 1) * 2] = 0
                masks.append(mask)
                octant_names.append(f"x{x}y{y}z{z}")
    masks = torch.stack(masks)
    loss_sums = torch.zeros(9, device="cuda", dtype=torch.float64)
    category_sums = {}
    category_counts = {}
    for start in range(0, args.score_tokens, args.readout_batch):
        stop = min(start + args.readout_batch, args.score_tokens)
        current = states[start:stop]
        arms = torch.cat((current[None], current[None] * masks[:, None]), 0)
        arms = arms.reshape(-1, *shape, channels)
        logits = model.decoder(model.readout(arms, position))
        logits = logits.reshape(9, stop - start, -1)
        target = targets[start:stop]
        losses = F.cross_entropy(
            logits.flatten(0, 1), target.repeat(9), reduction="none"
        ).reshape(9, stop - start).double()
        loss_sums += losses.sum(1)
        for offset, token_id in enumerate(targets_cpu[args.burn_in + start:
                                                       args.burn_in + stop]):
            category = token_category(int(token_id), encoding)
            category_sums.setdefault(category, torch.zeros(9, dtype=torch.float64))
            category_sums[category] += losses[:, offset].cpu()
            category_counts[category] = category_counts.get(category, 0) + 1

    mean_losses = loss_sums / args.score_tokens
    baseline = float(mean_losses[0])
    lesion_delta = mean_losses[1:] - mean_losses[0]
    category_results = {}
    for category, sums in category_sums.items():
        losses = sums / category_counts[category]
        category_results[category] = {
            "count": category_counts[category],
            "baseline_nll": float(losses[0]),
            "octant_delta_nll": {
                name: float(value) for name, value in zip(octant_names, losses[1:] - losses[0])
            },
        }

    top_energy = torch.topk(mean_energy, 8)
    top_write = torch.topk(write_counts, 8)
    report = {
        "checkpoint_step": int(saved["step"]),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "protocol": (
            "Frozen CBIM-3D on held-out OpenWebText. Persistent dynamics are unchanged; "
            "octant lesions are applied only to the state copy presented to the readout."
        ),
        "window": {"start": args.start, "burn_in": args.burn_in,
                   "score_tokens": args.score_tokens},
        "spatial_state": {
            "mean_energy_cv_across_cells": float(mean_energy.std(unbiased=False)
                                                   / mean_energy.mean()),
            "mean_temporal_cv_across_cells": float(temporal_cv.mean()),
            "neighbor_energy_correlation": float(spatial_neighbor_correlation),
            "energy_map_lag_correlation": lag_correlations,
            "top_energy_cells": [
                {"cell": int(index), "mean_energy": float(value)}
                for value, index in zip(top_energy.values, top_energy.indices)
            ],
        },
        "write_routing": {
            "normalized_entropy": normalized_entropy(write_counts),
            "occupied_cells": int((write_counts > 0).sum()),
            "top_cells": [
                {"cell": int(index), "count": int(value)}
                for value, index in zip(top_write.values, top_write.indices)
            ],
        },
        "read_routing": {
            "normalized_entropy_by_head_query": attention_entropies,
            "mean_js_from_global_attention": float(query_js),
            "mean_attention_by_head_query_cell": mean_attention.cpu().tolist(),
        },
        "readout_octant_lesion": {
            "baseline_nll": baseline,
            "delta_nll": {name: float(value)
                          for name, value in zip(octant_names, lesion_delta)},
            "delta_std": float(lesion_delta.std(unbiased=False)),
            "delta_range": float(lesion_delta.max() - lesion_delta.min()),
            "by_target_category": category_results,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
