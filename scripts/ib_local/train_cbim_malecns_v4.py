"""Train passive-port MaleCNS kinetic CBIM v4 on continuous OWT."""
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
import time

import numpy as np
import torch

from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer
from scripts.ib_local.cbim_malecns_v4 import CBIMMaleCNSV4


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--graph", type=Path,
                        default=Path("data/malecns_v1/malecns_parcels_256.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cbim_malecns_v4_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--velocities", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=8)
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
    source_files = ["scripts/ib_local/cbim_malecns_v4.py",
                    "scripts/ib_local/cbim_cuda_graph.py", __file__]
    model = CBIMMaleCNSV4(
        args.graph, velocities=args.velocities,
        content_dim=args.content_dim).cuda()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "architecture": model.architecture,
        "data": str(args.data), "graph": str(args.graph),
        "graph_sha256": sha256(args.graph), "graph_metadata": graph_meta,
        "output": str(args.output), "steps": args.steps,
        "tokens": args.tokens, "velocities": args.velocities,
        "content_dim": args.content_dim, "channels": model.d,
        "state_channels": model.d, "nodes": model.L,
        "queries": 4, "heads": 4,
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
                    "outgoing_energy": float(diagnostics["outgoing_energy"]),
                    "boundary_work": float(diagnostics["boundary_work"]),
                    "boundary_balance_residual": float(
                        diagnostics["boundary_balance_residual"]),
                    "coupling_angle_abs_mean": float(
                        diagnostics["coupling_angle_abs_mean"]),
                    "reflection_ratio": float(diagnostics["reflection_ratio"]),
                    "transport_angle_abs_mean": float(
                        diagnostics["transport_angle_abs_mean"]),
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

