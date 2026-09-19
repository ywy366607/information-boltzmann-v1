"""Train the CBIM-MaleCNS Unified V6 model on continuous OpenWebText."""
import os
import sys

# Prevent scripts/ib_local/types.py from shadowing the standard library types.
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
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.cbim_malecns_unified import CBIMMaleCNSUnifiedV6
from scripts.ib_local.geometric_transport import cube_velocities


class TruncatedUnifiedGraphTrainer:
    """Replay one captured recurrent chunk and update once per token budget.

    Capturing all chunks in one graph pins every Woodbury activation address
    and exceeds a 4 GiB device.  A single chunk graph is replayed while its
    parameter gradients accumulate; AdamW still steps exactly once per 128
    tokens and the persistent state is carried across every replay.
    """

    def __init__(self, model, tokens=128, chunk_tokens=8, lr=3e-4):
        if tokens % chunk_tokens:
            raise ValueError("tokens must be divisible by chunk_tokens")
        self.model = model
        self.tokens = int(tokens)
        self.chunk_tokens = int(chunk_tokens)
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.state = model.initial_state(1, device="cuda")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
        originals = [parameter.detach().clone() for parameter in model.parameters()]
        initial_state = self.state.detach().clone()

        def chunk_backward():
            loss, next_state, diagnostics = model(
                self.ids, self.targets, self.state)
            (loss / (tokens // chunk_tokens)).backward()
            return loss, next_state, diagnostics

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                chunk_backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.next_state, self.diagnostics = chunk_backward()
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), originals):
                parameter.copy_(original)
            self.state.copy_(initial_state)
        self.optimizer.zero_grad(set_to_none=False)

    def step(self, ids, targets):
        self.optimizer.zero_grad(set_to_none=False)
        loss_sum = 0.0
        diagnostic_sums = None
        chunks = self.tokens // self.chunk_tokens
        for start in range(0, self.tokens, self.chunk_tokens):
            self.ids.copy_(ids[:, start:start + self.chunk_tokens])
            self.targets.copy_(targets[:, start:start + self.chunk_tokens])
            self.graph.replay()
            loss_sum = loss_sum + self.loss.detach().clone()
            current_diagnostics = {
                name: value.detach().clone()
                for name, value in self.diagnostics.items()
            }
            if diagnostic_sums is None:
                diagnostic_sums = current_diagnostics
            else:
                for name, value in current_diagnostics.items():
                    diagnostic_sums[name].add_(value)
            with torch.no_grad():
                self.state.copy_(self.next_state.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0, foreach=True)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        self.optimizer.step()
        self.grad_norm = grad_norm.detach()
        diagnostics = {name: value / chunks
                       for name, value in diagnostic_sums.items()}
        return loss_sum / chunks, self.state, diagnostics


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
    effective_rank = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
    stable_rank = total / square.max().clamp_min(1e-12)
    node_energy = field.square().sum(-1)
    node_probability = node_energy / node_energy.sum().clamp_min(1e-12)
    effective_nodes = torch.exp(-(node_probability * node_probability.clamp_min(1e-12).log()).sum())
    return {
        "channel_effective_rank": float(effective_rank),
        "channel_stable_rank": float(stable_rank),
        "spatial_effective_nodes": float(effective_nodes),
        "max_node_energy_share": float(node_probability.max()),
    }


def prepare_monitor(output, graph_path, model):
    graph = np.load(graph_path)
    coordinates = np.asarray(graph["coordinates"], dtype=np.float32)
    adjacency = np.asarray(graph["adjacency"], dtype=np.float32)
    symmetric = adjacency + adjacency.T
    np.fill_diagonal(symmetric, 0)
    upper = np.triu(symmetric, 1)

    nodes = len(coordinates)
    seen = np.zeros(nodes, dtype=bool)
    seen[int(np.argmax(symmetric.sum(1)))] = True
    best = np.where(seen[:, None], symmetric, 0).max(0)
    parent = np.where(seen[:, None], symmetric, 0).argmax(0)
    tree = []
    for _ in range(nodes - 1):
        candidate = np.where(seen, -1, best)
        target = int(np.argmax(candidate))
        tree.append([int(parent[target]), target])
        seen[target] = True
        improved = symmetric[target] > best
        best = np.where(improved, symmetric[target], best)
        parent = np.where(improved, target, parent)

    remaining_count = min(128, int(np.count_nonzero(upper) - len(tree)))
    if remaining_count > 0:
        in_tree = np.zeros_like(upper, dtype=bool)
        for source, target in tree:
            in_tree[min(source, target), max(source, target)] = True
        residual = np.where(in_tree, 0, upper)
        flat = residual.ravel()
        indices = np.argpartition(flat, -remaining_count)[-remaining_count:]
        sources, targets = np.unravel_index(indices, residual.shape)
        edges = tree + [[int(s), int(t)] for s, t in zip(sources, targets) if residual[s, t] > 0]
    else:
        edges = tree

    sorted_coordinates = np.sort(coordinates, axis=0)
    gap_axis = int(np.diff(sorted_coordinates, axis=0).max(0).argmax())
    axis_values = np.sort(coordinates[:, gap_axis])
    gap_index = int(np.diff(axis_values).argmax())
    split = 0.5 * (axis_values[gap_index] + axis_values[gap_index + 1])
    side = coordinates[:, gap_axis] <= split
    crossing = np.where(side[:, None] != side[None, :], upper, 0)
    bridge_count = min(24, int(np.count_nonzero(crossing)))
    bridge_flat = crossing.ravel()
    bridge_indices = np.argpartition(bridge_flat, -bridge_count)[-bridge_count:]
    bridge_sources, bridge_targets = np.unravel_index(bridge_indices, crossing.shape)
    bridge_edges = [[int(s), int(t)] for s, t in zip(bridge_sources, bridge_targets) if crossing[s, t] > 0]

    input_strength = np.zeros(nodes, dtype=np.float32)
    output_strength = np.zeros(nodes, dtype=np.float32)
    atomic_json(output / "graph_layout.json", {
        "coordinates": coordinates.tolist(),
        "edges": edges,
        "bridge_edges": bridge_edges,
        "input_strength": input_strength.tolist(),
        "output_strength": output_strength.tolist(),
        "velocity_vectors": cube_velocities().tolist(),
        "input_port_names": ["full-rank contextual write"],
        "output_port_names": ["multi-query qk-norm readout"],
    }, required=True)
    dashboard = Path(__file__).resolve().parents[2] / "present" / "cbim_malecns_unified_live.html"
    shutil.copyfile(dashboard, output / "index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path, default=Path("data/malecns_v1/malecns_parcels_256_ports.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/cbim_malecns_unified_v6_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--velocities", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=8)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--collision-rank", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(11)
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
        "scripts/ib_local/cbim_malecns_unified.py",
        __file__
    ]
    model = CBIMMaleCNSUnifiedV6(
        args.graph,
        velocities=args.velocities,
        content_dim=args.content_dim,
        modes=args.modes,
        collision_rank=args.collision_rank,
    ).cuda()

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {
        "architecture": model.architecture,
        "data": str(args.data),
        "graph": str(args.graph),
        "graph_sha256": sha256(args.graph),
        "graph_metadata": graph_meta,
        "output": str(args.output),
        "steps": args.steps,
        "tokens": args.tokens,
        "velocities": args.velocities,
        "content_dim": args.content_dim,
        "channels": model.d,
        "nodes": model.L,
        "modes": args.modes,
        "collision_rank": args.collision_rank,
        "parameter_count": parameter_count,
        "precision": "FP32",
        "seed": 11,
        "lr": 3e-4,
        "validate_every": args.validate_every,
        "validation_tokens": args.validation_tokens,
        "stopping": f"fixed {args.steps}-update budget; convergence not established",
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {name: sha256(name) for name in source_files},
    }
    atomic_json(args.output / "config.json", config, required=True)
    prepare_monitor(args.output, args.graph, model)
    atomic_json(args.output / "progress.json", {
        "status": "capturing", "step": 0, "target_steps": args.steps
    }, required=True)

    runner = TruncatedUnifiedGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=args.chunk_tokens, lr=3e-4
    )
    step, best = 0, float("inf")

    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
        runner.optimizer.load_state_dict(checkpoint_data["optimizer"])
        runner.state.copy_(checkpoint_data["state"])
        step = checkpoint_data["step"]
        best = checkpoint_data["best_validation_nll"]

    def batch(data, offset):
        return tuple(
            torch.as_tensor(
                np.array(data[offset + shift:offset + shift + args.tokens]),
                dtype=torch.long, device="cuda"
            )[None]
            for shift in (0, 1)
        )

    def save(name):
        temp = args.output / f"{name}.tmp"
        payload = {
            "model": model.state_dict(),
            "optimizer": runner.optimizer.state_dict(),
            "state": runner.state.detach(),
            "step": step,
            "events": step * args.tokens,
            "best_validation_nll": best,
            "config": config,
        }
        torch.save(payload, temp)
        os.replace(temp, args.output / name)

    def log(row):
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        field_state = model.initial_state(1, device="cuda")
        total = 0.0
        for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
            ids, targets = batch(validation, offset)
            loss, field_state, _ = model(ids, targets, field_state)
            if offset:
                total += float(loss) * args.tokens
        return total / args.validation_tokens

    try:
        if not args.resume:
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
            spectrum = state_spectrum(state[0])
            row = {
                "kind": "train",
                "step": step,
                "events": step * args.tokens,
                "nll": float(loss),
                "seconds": elapsed,
                "energy": float(diagnostics["field_energy"]),
                "incident_energy": float(diagnostics["incident_energy"]),
                "reflected_energy": float(diagnostics["reflected_energy"]),
                "absorbed_power": float(diagnostics["absorbed_power"]),
                "write_angle_abs_mean": float(diagnostics["write_angle_abs_mean"]),
                "write_balance_residual": float(diagnostics["write_balance_residual"]),
                "write_envelope_mean": float(diagnostics["envelope_mean"]),
                "bath_power": float(diagnostics["bath_power"]),
                "transport_edge_angle_mean": float(diagnostics["transport_edge_angle_mean"]),
                "transport_spectral_angle_mean": float(diagnostics["transport_spectral_angle_mean"]),
                "collision_generator_norm": float(diagnostics["collision_generator_norm"]),
                "collision_norm_residual": float(diagnostics["collision_norm_residual"]),
                "occupation_ratio_mean": float(diagnostics["occupation_ratio_mean"]),
                "occupation_ratio_max": float(diagnostics["occupation_ratio_max"]),
                "cooling_sin2_mean": float(diagnostics["cooling_sin2_mean"]),
                "transport_norm_residual": float(diagnostics["transport_norm_residual"]),
                "grad_norm": float(runner.grad_norm),
                **spectrum,
                "allocated_mib": torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mib": torch.cuda.memory_reserved() / (1024 ** 2),
            }
            log(row)

            if step % 10 == 0:
                pos = model.readout.position_encoding()
                _, attn = model.readout(state, pos)
                attn_vis = attn.mean(1)[0].detach().cpu().numpy()  # [H, N]
                read_entropy = -(attn * attn.clamp_min(1e-12).log()).sum(-1).mean()
                token = model.boundary_write.embedding(ids[:, -1])
                center = torch.sigmoid(model.boundary_write.address(token))[:, None]
                width = torch.nn.functional.softplus(
                    model.boundary_write.width(token)).clamp_min(1e-4)[:, None]
                dist = (model.boundary_write.coordinates - center) / width
                write_map = torch.exp(-0.5 * dist.square().sum(-1))[0]
                velocity_energy = state.reshape(
                    state.shape[0], model.L, model.velocities, -1
                ).square().sum(-1)[0]
                live_state = {
                    "step": step,
                    "tokens": step * args.tokens,
                    "field_energy": float(diagnostics["field_energy"]),
                    "absorbed_power": float(diagnostics["absorbed_power"]),
                    "bath_power": float(diagnostics["bath_power"]),
                    "incident_energy": float(diagnostics["incident_energy"]),
                    "reflected_energy": float(diagnostics["reflected_energy"]),
                    "loss": float(loss),
                    "best_validation_nll": best,
                    "node_amplitudes": state[0].norm(dim=-1).detach().cpu().tolist(),
                    "node_velocity_energy": velocity_energy.detach().cpu().tolist(),
                    "write_map": write_map.detach().cpu().tolist(),
                    "readout_attention_heads": attn_vis.tolist(),
                    "read_attention_entropy": float(read_entropy.detach()),
                    "read_logit_scales": torch.exp(
                        model.readout.head_log_scale).flatten().detach().cpu().tolist(),
                }
                atomic_json(args.output / "live_state.json", live_state)
                atomic_json(args.output / "progress.json", {
                    "status": "running", "target_steps": args.steps, **row
                })

            if step % args.validate_every == 0:
                current_val = validate()
                is_best = current_val < best
                if is_best:
                    best = current_val
                log({
                    "kind": "validation",
                    "step": step,
                    "validation_nll": current_val,
                    "best_validation_nll": best,
                })
                save("last.pt")
                save(f"age_{step:06d}.pt")
                if is_best:
                    save("BBest.pt")
                atomic_json(args.output / "progress.json", {
                    "status": "running",
                    "step": step,
                    "target_steps": args.steps,
                    "validation_nll": current_val,
                    "best_validation_nll": best,
                })

        atomic_json(args.output / "progress.json", {
            "status": "complete",
            "step": step,
            "target_steps": args.steps,
            "best_validation_nll": best,
        })
    except Exception as exc:
        atomic_json(args.output / "progress.json", {
            "status": "failed", "step": step, "error": str(exc)
        })
        raise


if __name__ == "__main__":
    main()
