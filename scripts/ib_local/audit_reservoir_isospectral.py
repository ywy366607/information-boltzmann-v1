"""Read-only CPU audit: a global Brockett flow changes basis, not retrieval.

This uses actual occupied episode features and the checkpoint's current encoder.
It does not run the body, optimizer, world, or CUDA. Rotated queries and keys
are compared together; rotating only one would change their coordinate contract.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summary(x):
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return {
        "mean_energy_fraction": float(x.mean(0).square().sum() / x.square().sum(1).mean()),
        "centered_trace": float(eig.sum()),
        "centered_top1_energy": float(eig[-1] / eig.sum()),
        "centered_participation_rank": float(eig.sum().square() / eig.square().sum()),
        "off_diagonal_squared_energy_fraction": float(
            (covariance.square().sum() - covariance.diag().square().sum())
            / covariance.square().sum()
        ),
    }, covariance


@torch.no_grad()
def basis_audit(x, steps):
    x = x.double()
    before, covariance = summary(x)
    initial = covariance.clone()
    dim = x.shape[1]
    identity = torch.eye(dim, dtype=x.dtype)
    total_rotation = identity.clone()
    # Ascend Tr(N C): Cdot=[[N,C],C]. N is a numerical sorting reference,
    # not a target learned from actions or consequences.
    target = torch.diag(torch.linspace(-1, 1, dim, dtype=x.dtype))
    step = 0.25 / torch.linalg.matrix_norm(covariance, ord=2).clamp_min(1e-12)
    for _ in range(steps):
        omega = target @ covariance - covariance @ target
        rotation = torch.linalg.solve(identity - step * omega / 2,
                                      identity + step * omega / 2)
        covariance = rotation @ covariance @ rotation.T
        total_rotation = rotation @ total_rotation
    rotated = x @ total_rotation.T
    after, recomputed = summary(rotated)
    # Real recorded source queries spread across the occupied reservoir.
    indices = torch.linspace(0, len(x) - 1, min(64, len(x))).long()
    original_scores = F.normalize(x[indices], dim=-1) @ F.normalize(x, dim=-1).T
    rotated_scores = F.normalize(rotated[indices], dim=-1) @ F.normalize(rotated, dim=-1).T
    first = original_scores.topk(min(8, len(x)), dim=-1).indices.sort(-1).values
    second = rotated_scores.topk(min(8, len(x)), dim=-1).indices.sort(-1).values
    return dict(
        before=before, after=after, steps=steps,
        max_cosine_change=float((original_scores - rotated_scores).abs().max()),
        top8_sets_equal=int((first == second).all(-1).sum()),
        queries=len(indices),
        max_eigenvalue_change=float((torch.linalg.eigvalsh(initial)
                                    - torch.linalg.eigvalsh(recomputed)).abs().max()),
        rotation_orthogonality_error=float((total_rotation @ total_rotation.T - identity).abs().max()),
        matrix_vs_rotated_features_error=float((covariance - recomputed).abs().max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sha_before = digest(args.checkpoint)
    with gzip.open(args.checkpoint, "rb") as stream:
        saved = torch.load(stream, map_location="cpu", weights_only=False)
    model = saved["model"]
    occupied = model["memory.occupied"]
    source = model["memory.source"][occupied].double()
    addresses = source @ model["encoder.weight"].double().T
    counts = model["memory.count"][occupied]
    boundary_valid = (occupied & model["memory.internal_valid"]
                      & (model["memory.count"] == 1)
                      & (model["memory.internal_stamp"] == model["memory.last_write"]))
    boundary_source = model["memory.source"][boundary_valid].double()
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha_before,
        script_sha256=digest(Path(__file__)), cycle=saved["cycle"],
        occupied=int(occupied.sum()), capacity=len(occupied),
        single_episode_slots=int((counts == 1).sum()),
        merged_slots=int((counts > 1).sum()),
        valid_unmerged_boundary_pairs=int(boundary_valid.sum()),
        valid_boundary_source_summary=summary(boundary_source)[0],
        memory_configuration=model["memory._extra_state"],
        scope="read-only coordinate diagnostic; no language or causal-utility evaluation",
        dtype="float64", device="cpu",
    )
    del saved, model, occupied, counts
    gc.collect()
    report["source128"] = basis_audit(source, args.steps)
    report["current_address64"] = basis_audit(addresses, args.steps)
    report["checkpoint_unchanged"] = digest(args.checkpoint) == sha_before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
