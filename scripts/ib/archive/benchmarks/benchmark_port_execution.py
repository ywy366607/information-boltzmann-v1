"""Measure complete captured OWT updates without saving a training checkpoint.

Use the same checkpoint/config/data offset for execution comparisons. Timing
includes chunk replay, state continuation, gradient clipping and AdamW; eager
component proportions are not substituted for measured production times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D
from scripts.ib.train_q8_port_agents import TruncatedBeliefGraphTrainer, _batch


def load_benchmark_model(saved, bath_type=None):
    """Load unchanged weights, or explicitly replace only the benchmark bath.

    A replacement bath starts from its constructor defaults. All other weights,
    field and precision stay at the checkpoint; no new checkpoint is saved.
    """
    cfg = dict(saved["config"])
    original_bath = cfg["dissipation_type"]
    if bath_type is not None:
        cfg["dissipation_type"] = bath_type
    model = CBIMTorus3D(
        shape=tuple(cfg["shape"]), velocities=cfg["velocities"],
        content_dim=cfg["content_dim"], collision_layers=cfg["collision_layers"],
        relative_address=True, readout_type=cfg["readout_type"],
        queries=cfg.get("readout_queries", 4), write_type=cfg["write_type"],
        micro_steps=cfg["K"], tau_0=cfg["tau_0"],
        dissipation_type=cfg["dissipation_type"],
        dissipation_rank=cfg["dissipation_rank"],
        continuous_velocities=cfg.get("continuous_velocities", False),
    ).cuda()
    replaced = original_bath != cfg["dissipation_type"]
    if replaced:
        weights = {key: value for key, value in saved["model"].items()
                   if not key.startswith("bath.")}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected or any(not key.startswith("bath.") for key in missing):
            raise RuntimeError(f"Unexpected benchmark load: {missing}, {unexpected}")
    else:
        model.load_state_dict(saved["model"], strict=True)
    return model, cfg, replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--bath", choices=("quadratic", "unified", "selective"),
                        help="Explicit computation-only bath replacement")
    parser.add_argument("--collision-kernel", choices=("auto", "native"),
                        default="auto", help="Matched compiled collision comparison")
    parser.add_argument("--collision-projection", choices=("auto", "dense"),
                        default="auto", help="Matched exact nullspace application")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model, cfg, bath_replaced = load_benchmark_model(saved, args.bath)
    if args.collision_projection == "dense":
        model.collision._structured_projection = False
    if args.collision_kernel == "native":
        from information_boltzmann.core.triton_givens import native_givens
        model.collision._triton_givens = native_givens
    if not args.no_compile:
        os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
        for module in (model.collision, model.bath, model.write_agent, model.readout):
            module.forward = torch.compile(module.forward, dynamic=False)
        model.transport.apply_multiplier = torch.compile(model.transport.apply_multiplier)
    print("Compiling/warming and capturing the production backward graph...", flush=True)
    runner = TruncatedBeliefGraphTrainer(
        model, tokens=cfg["tokens"], chunk_tokens=cfg["bptt_chunk_tokens"],
        lr=cfg["lr"], max_grad_norm=cfg["max_grad_norm"])
    with torch.no_grad():
        runner.field.copy_(saved["state"])
        runner.precision.copy_(saved["precision"])
    # Model weights, physical state and posterior precision start at the
    # recorded checkpoint. Warmup and measurement form one continuing stream.
    train = np.load(Path(cfg["data"]) / "train.npy", mmap_mode="r")
    offset = int(saved.get("step", 0)) * cfg["tokens"]
    del saved
    gpu_ms, wall_ms, losses = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.repeats):
        ids, targets = _batch(train, offset + index * cfg["tokens"], cfg["tokens"])
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        loss, _, _ = runner.step(ids, targets)
        end.record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - before) * 1000
        value = float(loss.detach())
        if index >= args.warmup:
            gpu_ms.append(start.elapsed_time(end))
            wall_ms.append(elapsed)
            losses.append(value)
        print(f"update {index + 1}: {elapsed:.1f} ms, joint loss {value:.6f}", flush=True)
    source = ROOT / "information_boltzmann/core/torus3d.py"
    kernel_source = ROOT / "information_boltzmann/core/triton_givens.py"
    projection_source = ROOT / "information_boltzmann/core/triton_projection.py"
    result = {
        "kind": "execution_benchmark_not_capability_training",
        "checkpoint": str(args.checkpoint), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "collision_source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
        "projection_source_sha256": hashlib.sha256(projection_source.read_bytes()).hexdigest(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "compiled": not args.no_compile, "K": cfg["K"], "dt": cfg["tau_0"],
        "shape": cfg["shape"], "channels": cfg["channels"], "bath": cfg["dissipation_type"],
        "bath_replaced_for_timing": bath_replaced,
        "collision_kernel": args.collision_kernel,
        "collision_projection": args.collision_projection,
        "tokens_per_update": cfg["tokens"], "bptt_chunk_tokens": cfg["bptt_chunk_tokens"],
        "start_data_offset": offset, "warmup_updates": args.warmup,
        "timed_updates": args.repeats, "wall_ms": wall_ms, "cuda_event_ms": gpu_ms,
        "median_wall_ms": statistics.median(wall_ms), "mean_wall_ms": statistics.mean(wall_ms),
        "tokens_per_second": cfg["tokens"] * 1000 / statistics.median(wall_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "joint_losses": losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "median_wall_ms", "tokens_per_second", "peak_allocated_mib", "peak_reserved_mib")}), flush=True)


if __name__ == "__main__":
    main()
