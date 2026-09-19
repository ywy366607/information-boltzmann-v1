"""Train self-organizing internal-time CBIM on continuous OWT."""
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
from scripts.ib_local.cbim_malecns_internal_time import CBIMMaleCNSInternalTime


class TruncatedInternalTimeGraphTrainer:
    """Capture one recurrent chunk; clip lexical and dynamics groups separately."""

    def __init__(self, model, tokens=128, chunk_tokens=16, lr=3e-4):
        if tokens % chunk_tokens:
            raise ValueError("tokens must be divisible by chunk_tokens")
        self.model, self.tokens, self.chunk_tokens = model, tokens, chunk_tokens
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.state = model.initial_state(1, device="cuda")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
        named = list(model.named_parameters())
        lexical_names = {"source.embedding.weight", "decoder.weight", "decoder.bias"}
        self.gradient_group_names = ("lexical", "dynamics")
        self.gradient_groups = (
            [p for n, p in named if n in lexical_names],
            [p for n, p in named if n not in lexical_names],
        )
        originals = [p.detach().clone() for p in model.parameters()]
        initial_state = self.state.detach().clone()
        def backward_chunk():
            loss, next_state, diagnostics = model(self.ids, self.targets, self.state)
            (loss / (tokens // chunk_tokens)).backward()
            return loss, next_state, diagnostics
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True); backward_chunk()
        torch.cuda.current_stream().wait_stream(stream); torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.next_state, self.diagnostics = backward_chunk()
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), originals): parameter.copy_(original)
            self.state.copy_(initial_state)
        self.optimizer.zero_grad(set_to_none=False)

    def step(self, ids, targets):
        self.optimizer.zero_grad(set_to_none=False)
        loss_sum, diagnostics = 0.0, None
        for start in range(0, self.tokens, self.chunk_tokens):
            self.ids.copy_(ids[:, start:start+self.chunk_tokens])
            self.targets.copy_(targets[:, start:start+self.chunk_tokens])
            self.graph.replay(); loss_sum += self.loss.detach().clone()
            diagnostics = {k: v.detach().clone() for k, v in self.diagnostics.items()}
            with torch.no_grad(): self.state.copy_(self.next_state.detach())
        self.group_grad_norms = torch.stack(tuple(
            torch.nn.utils.clip_grad_norm_(group, 1.0, foreach=True)
            for group in self.gradient_groups))
        self.grad_norm = self.group_grad_norms.square().sum().sqrt()
        if not torch.isfinite(self.grad_norm): raise FloatingPointError("Non-finite gradient norm")
        self.optimizer.step()
        return loss_sum / (self.tokens // self.chunk_tokens), self.state, diagnostics

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
    # Functional write/read locations are learned during training; the live
    # state itself reveals them instead of painting fixed anatomical ports.
    input_strength = np.zeros(nodes, dtype=np.float32)
    output_strength = np.zeros(nodes, dtype=np.float32)
    atomic_json(output / "graph_layout.json", {
        "coordinates": coordinates.tolist(),
        "edges": edges,
        "bridge_edges": bridge_edges,
        "input_strength": input_strength.tolist(),
        "output_strength": output_strength.tolist(),
        "velocity_vectors": model.transport.velocity_vectors.detach().cpu().tolist(),
        "input_port_names": ["learned token boundary"],
        "output_port_names": ["learned attention observation"],
    }, required=True)
    dashboard = Path(__file__).resolve().parents[2] / "present" / (
        "cbim_malecns_internal_time_live.html")
    shutil.copyfile(dashboard, output / "index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path,
                        default=Path(
                            "data/malecns_v1/malecns_parcels_256_ports.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path(
                            "results/cbim_malecns_internal_time_k4_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--velocities", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=8)
    parser.add_argument("--micro-steps", type=int, default=4)
    parser.add_argument("--spectral-transport", action="store_true")
    parser.add_argument("--checkpoint-tokens", type=int, default=1)
    parser.add_argument("--bptt-chunk-tokens", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(.85)
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
                    "scripts/ib_local/cbim_malecns_internal_time.py",
                    "scripts/ib_local/geometric_transport.py",
                    "scripts/ib_local/cbim_cuda_graph.py",
                    "present/cbim_malecns_port_transport_live.html", __file__]
    model = CBIMMaleCNSInternalTime(
        args.graph, velocities=args.velocities,
        content_dim=args.content_dim, micro_steps=args.micro_steps,
        checkpoint_tokens=args.checkpoint_tokens,
        spectral_transport=args.spectral_transport).cuda()
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
        "micro_steps": args.micro_steps,
        "spectral_transport": args.spectral_transport,
        "checkpoint_tokens": args.checkpoint_tokens,
        "bptt_chunk_tokens": args.bptt_chunk_tokens,
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

    runner = TruncatedInternalTimeGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=args.bptt_chunk_tokens)
    step, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("architecture", "graph_sha256", "velocities",
                    "content_dim", "tokens", "micro_steps"):
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
                with torch.no_grad():
                    head_query_maps = model.readout.attention_weights(field)[0]
                    query_maps = head_query_maps.mean(0)
                    read_entropy = -(
                        query_maps * query_maps.clamp_min(1e-12).log()
                    ).sum(-1).mean()
                    read_max_weight = query_maps.amax(-1).mean()
                    read_logit_scale = model.readout.logit_scales().mean()
                    monitor_embedding = model.source.embedding(ids[:, -1])
                    monitor_width = torch.nn.functional.softplus(
                        model.source.width(monitor_embedding))
                row = {
                    "kind": "train", "step": step,
                    "events": step * args.tokens, "nll": loss_value,
                    "seconds": elapsed,
                    "energy": float(diagnostics["final_energy"]),
                    "incident_energy": float(diagnostics["incident_energy"]),
                    "token_out_energy": float(diagnostics["reflected_energy"]),
                    "bath_out_energy": float(diagnostics["bath_out_energy"]),
                    "write_balance_residual": float(
                        diagnostics["write_balance_residual"]),
                    "write_angle_abs_mean": float(
                        diagnostics["write_angle_abs_mean"]),
                    "bath_angle_abs_mean": float(
                        diagnostics["bath_angle_abs_mean"]),
                    "bath_active_fraction": float(
                        diagnostics["bath_active_fraction"]),
                    "transport_norm_residual": float(
                        diagnostics["transport_norm_residual"]),
                    "collision_angle_abs_mean": float(
                        diagnostics["collision_angle_abs_mean"]),
                    "read_attention_entropy": float(read_entropy),
                    "read_attention_max_weight": float(read_max_weight),
                    "read_logit_scale_mean": float(read_logit_scale),
                    "write_width_mean": float(monitor_width.mean()),
                    "grad_norm": float(runner.grad_norm),
                    **{
                        f"grad_norm_{name}": float(value)
                        for name, value in zip(
                            runner.gradient_group_names,
                            runner.group_grad_norms)
                    },
                    "spatial_variance": float(field.var(1, unbiased=False).mean()),
                    **spectrum,
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(args.output / "progress.json", {
                    "status": "running", "target_steps": args.steps, **row})
                with torch.no_grad():
                    last_token = ids[:, -1]
                    embedding = model.source.embedding(last_token)
                    center = torch.sigmoid(model.source.address(embedding))[:, None]
                    width = torch.nn.functional.softplus(
                        model.source.width(embedding))[:, None]
                    distance = (model.source.coordinates - center) / width
                    write_map = torch.exp(-.5 * distance.square().sum(-1))[0]
                    query_maps = model.readout.attention_map(field)[0]
                atomic_json(args.output / "live_state.json", {
                    **row,
                    "node_values": field[0].square().mean(-1).sqrt().cpu().tolist(),
                    "node_velocity_energy": field[0].reshape(
                        model.L, model.velocities, model.content_dim
                    ).square().mean(-1).cpu().tolist(),
                    "write_map": write_map.cpu().tolist(),
                    "read_map": query_maps.mean(0).cpu().tolist(),
                    "read_query_maps": query_maps.cpu().tolist(),
                    "read_head_maps": head_query_maps.mean(1).cpu().tolist(),
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



