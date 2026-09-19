"""Train anatomical-port, geometric-transport CBIM on continuous OWT."""
import os
import sys

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
import shutil
import time

import numpy as np
import torch

from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
from scripts.ib_local.cbim_malecns_port_transport import (
    CBIMMaleCNSPortTransport,
)


def atomic_json(path, data, required=False):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    payload = json.dumps(data, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return True
        except PermissionError:
            time.sleep(.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Unable to update required file: {path}")
    return False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def state_spectrum(field):
    singular = torch.linalg.svdvals(field.float())
    square = singular.square()
    total = square.sum().clamp_min(1e-12)
    probability = square / total
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum())
    stable_rank = total / square.max().clamp_min(1e-12)
    node_energy = field.square().sum(-1)
    node_probability = node_energy / node_energy.sum().clamp_min(1e-12)
    effective_nodes = torch.exp(-(
        node_probability * node_probability.clamp_min(1e-12).log()).sum())
    return {
        "channel_effective_rank": float(effective_rank),
        "channel_stable_rank": float(stable_rank),
        "spatial_effective_nodes": float(effective_nodes),
        "max_node_energy_share": float(node_probability.max()),
    }


def prepare_monitor(output, graph_path, model):
    """Write the static anatomical layout and the matching live dashboard."""
    graph = np.load(graph_path)
    coordinates = np.asarray(graph["coordinates"], dtype=np.float32)
    adjacency = np.asarray(graph["adjacency"], dtype=np.float32)
    symmetric = adjacency + adjacency.T
    np.fill_diagonal(symmetric, 0)
    upper = np.triu(symmetric, 1)
    flat = upper.ravel()
    # A maximum-weight spanning tree keeps weak anatomical bridges visible;
    # strongest remaining edges add local structure without turning the view
    # into an unreadable rendering of all ~21k parcel pairs.
    nodes = len(coordinates)
    seen = np.zeros(nodes, dtype=bool)
    seen[int(np.argmax(symmetric.sum(1)))] = True
    best = np.where(seen[:, None], symmetric, 0).max(0)
    parent = np.where(seen[:, None], symmetric, 0).argmax(0)
    tree = []
    for _ in range(nodes - 1):
        candidate = np.where(seen, -1, best)
        target = int(candidate.argmax())
        if candidate[target] <= 0:
            raise ValueError("monitor graph is disconnected")
        source = int(parent[target])
        tree.append([source, target])
        seen[target] = True
        improve = symmetric[target] > best
        best[improve] = symmetric[target, improve]
        parent[improve] = target
    tree_keys = {tuple(sorted(edge)) for edge in tree}
    extra_count = min(512 - len(tree), int(np.count_nonzero(flat)))
    indices = np.argpartition(flat, -extra_count)[-extra_count:]
    sources, targets = np.unravel_index(indices, upper.shape)
    extras = [[int(source), int(target)] for source, target in zip(sources, targets)
              if upper[source, target] > 0
              and (int(source), int(target)) not in tree_keys]
    edges = tree + extras[:512 - len(tree)]

    # Highlight the strongest edges crossing the largest anatomical coordinate
    # gap (for MaleCNS this is the brain--VNC/abdomen separation).
    sorted_coordinates = np.sort(coordinates, axis=0)
    gap_axis = int(np.diff(sorted_coordinates, axis=0).max(0).argmax())
    axis_values = np.sort(coordinates[:, gap_axis])
    gap_index = int(np.diff(axis_values).argmax())
    split = .5 * (axis_values[gap_index] + axis_values[gap_index + 1])
    side = coordinates[:, gap_axis] <= split
    crossing = np.where(side[:, None] != side[None, :], upper, 0)
    bridge_count = min(24, int(np.count_nonzero(crossing)))
    bridge_flat = crossing.ravel()
    bridge_indices = np.argpartition(bridge_flat, -bridge_count)[-bridge_count:]
    bridge_sources, bridge_targets = np.unravel_index(
        bridge_indices, crossing.shape)
    bridge_edges = [[int(source), int(target)]
                    for source, target in zip(bridge_sources, bridge_targets)
                    if crossing[source, target] > 0]
    input_strength = model.boundary.input_gate_masks.amax(0).detach().cpu().numpy()
    output_strength = model.boundary.output_gate.detach().cpu().numpy()
    atomic_json(output / "graph_layout.json", {
        "coordinates": coordinates.tolist(),
        "edges": edges,
        "bridge_edges": bridge_edges,
        "input_strength": input_strength.tolist(),
        "output_strength": output_strength.tolist(),
        "velocity_vectors": model.transport.velocity_vectors.detach().cpu().tolist(),
        "input_port_names": model.boundary.input_port_names,
        "output_port_names": model.boundary.output_port_names,
    }, required=True)
    dashboard = Path(__file__).resolve().parents[2] / "present" / (
        "cbim_malecns_port_transport_live.html")
    shutil.copyfile(dashboard, output / "index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path,
                        default=Path(
                            "data/malecns_v1/malecns_parcels_256_ports.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path(
                            "results/cbim_malecns_port_transport_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--velocities", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=8)
    parser.add_argument("--input-topk", type=int, default=16)
    parser.add_argument("--output-topk", type=int, default=16)
    parser.add_argument("--stream-fraction", type=float, default=.5)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(.48)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and args.resume is None:
        raise ValueError("Existing run requires --resume or a new output directory")
    train = np.load(args.data / "train.npy", mmap_mode="r")
    validation = np.load(args.data / "validation.npy", mmap_mode="r")
    if args.steps * args.tokens + 1 > len(train):
        raise ValueError("Training stream is shorter than requested budget")
    if args.validation_tokens % args.tokens:
        raise ValueError("validation-tokens must be divisible by tokens")

    graph_meta = json.loads(args.graph.with_suffix(".json").read_text())
    source_files = [
                    "scripts/ib_local/cbim_malecns_port_transport.py",
                    "scripts/ib_local/geometric_transport.py",
                    "scripts/ib_local/cbim_cuda_graph.py",
                    "present/cbim_malecns_port_transport_live.html", __file__]
    model = CBIMMaleCNSPortTransport(
        args.graph, velocities=args.velocities,
        content_dim=args.content_dim, input_topk=args.input_topk,
        output_topk=args.output_topk,
        stream_fraction=args.stream_fraction).cuda()
    prepare_monitor(args.output, args.graph, model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "architecture": model.architecture,
        "data": str(args.data), "graph": str(args.graph),
        "graph_sha256": sha256(args.graph), "graph_metadata": graph_meta,
        "output": str(args.output), "steps": args.steps,
        "tokens": args.tokens, "velocities": args.velocities,
        "content_dim": args.content_dim, "channels": model.d,
        "state_channels": model.d, "nodes": model.L,
        "input_ports": model.boundary.input_ports,
        "output_ports": model.boundary.output_ports,
        "input_topk": args.input_topk, "output_topk": args.output_topk,
        "stream_fraction": args.stream_fraction,
        "parameter_count": parameter_count, "precision": "FP32", "seed": 11,
        "lr": 3e-4, "validate_every": args.validate_every,
        "validation_tokens": args.validation_tokens,
        "stopping": f"fixed {args.steps}-update budget; convergence not established",
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {name: sha256(name) for name in source_files},
    }
    atomic_json(args.output / "config.json", config, required=True)
    atomic_json(args.output / "progress.json", {
        "status": "capturing", "step": 0, "target_steps": args.steps})

    runner = CBIMGraphTrainer(model, tokens=args.tokens)
    step, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("architecture", "graph_sha256", "velocities",
                    "content_dim", "tokens"):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(saved["model"])
        runner.optimizer.load_state_dict(saved["optimizer"])
        runner.state.copy_(saved["state"])
        step, best = saved["step"], saved["best_validation_nll"]

    def batch(data, offset):
        return tuple(torch.as_tensor(
            np.array(data[offset + shift:offset + shift + args.tokens]),
            dtype=torch.long, device="cuda")[None] for shift in (0, 1))

    def save(name):
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save({"model": model.state_dict(),
                    "optimizer": runner.optimizer.state_dict(),
                    "state": runner.state.detach(), "step": step,
                    "events": step * args.tokens,
                    "best_validation_nll": best, "config": config}, temporary)
        os.replace(temporary, args.output / name)

    def log(row):
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite metric {name}: {value}")
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        state = model.initial_state(1, device="cuda")
        total = 0.0
        for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
            ids, targets = batch(validation, offset)
            loss, state, _ = model(ids, targets, state)
            if offset:
                total += float(loss) * args.tokens
        return total / args.validation_tokens

    try:
        if args.resume is None:
            best = validate()
            log({"kind": "validation", "step": 0, "validation_nll": best})
            save("BBest.pt")
            save("last.pt")
        while step < args.steps:
            ids, targets = batch(train, step * args.tokens)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, state, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                raise FloatingPointError("Non-finite training loss")
            if elapsed > 1.5 and step > 2:
                raise RuntimeError(
                    f"Training step exceeded 1.5 seconds: {elapsed:.3f}")
            if step == 1 or step % 10 == 0 or step == args.steps:
                field = state.detach()
                spectrum = state_spectrum(field[0])
                row = {
                    "kind": "train", "step": step,
                    "events": step * args.tokens, "nll": loss_value,
                    "seconds": elapsed,
                    "energy": float(diagnostics["final_energy"]),
                    "incident_energy": float(diagnostics["incident_energy"]),
                    "token_out_energy": float(diagnostics["token_out_energy"]),
                    "bath_out_energy": float(diagnostics["bath_out_energy"]),
                    "write_balance_residual": float(
                        diagnostics["write_balance_residual"]),
                    "bath_balance_residual": float(
                        diagnostics["bath_balance_residual"]),
                    "write_angle_abs_mean": float(
                        diagnostics["write_angle_abs_mean"]),
                    "bath_angle_abs_mean": float(
                        diagnostics["bath_angle_abs_mean"]),
                    "bath_angle_abs_max": float(
                        diagnostics["bath_angle_abs_max"]),
                    "input_port_entropy": float(
                        diagnostics["input_port_entropy"]),
                    "transport_norm_residual": float(
                        diagnostics["transport_norm_residual"]),
                    "transport_directional_alignment": float(
                        diagnostics["transport_directional_alignment"]),
                    "transport_energy_change": float(
                        diagnostics["transport_energy_change"]),
                    "collision_angle_abs_mean": float(
                        diagnostics["collision_angle_abs_mean"]),
                    "grad_norm": float(runner.grad_norm),
                    "spatial_variance": float(field.var(1, unbiased=False).mean()),
                    **spectrum,
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(args.output / "progress.json", {
                    "status": "running", "target_steps": args.steps, **row})
                atomic_json(args.output / "live_state.json", {
                    **row,
                    "node_values": field[0].square().mean(-1).sqrt().cpu().tolist(),
                    "node_velocity_energy": field[0].reshape(
                        model.L, model.velocities, model.content_dim
                    ).square().mean(-1).cpu().tolist(),
                })
            if step % args.validate_every == 0 or step == args.steps:
                score = validate()
                improved = score < best
                if improved:
                    best = score
                log({"kind": "validation", "step": step,
                     "validation_nll": score, "best_validation_nll": best})
                if improved:
                    save("BBest.pt")
                save("last.pt")
                save(f"age_{step:06d}.pt")
        atomic_json(args.output / "progress.json", {
            "status": "complete", "step": step,
            "target_steps": args.steps, "best_validation_nll": best})
    except BaseException as error:
        atomic_json(args.output / "progress.json", {
            "status": "failed", "step": step, "error": str(error)})
        raise


if __name__ == "__main__":
    main()

