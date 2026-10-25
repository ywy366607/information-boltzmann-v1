"""Benchmark coefficient arithmetic on the current finite port kernel.

This is an operator microbenchmark, not a capability or training experiment.
The transition table mirrors the registered 24-state PortSeparation witness;
all coefficient modes apply the same 16 deterministic input kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


CELLS = ("vacuum", "search", "aSpent", "bSpent", "aReady", "bReady", "aHeld", "bHeld")
SIGNALS = ("none", "calm", "answerA", "answerB", "feed", "scarce", "miss", "probe")
NONCALM = {"answerA", "answerB", "feed", "scarce", "miss", "probe"}


def rl_incoming(signal: str, cell: str) -> int:
    if cell == "vacuum" or signal == "calm":
        return 0
    if signal in {"scarce", "miss"}:
        return 0
    if signal == "feed":
        return {"aSpent": 4, "bSpent": 4, "aHeld": 2, "bHeld": 2}.get(cell, 1)
    return 1


def rl_next(signal: str, cell: str) -> str:
    if cell == "vacuum":
        return "vacuum"
    if signal in {"calm", "probe"}:
        return cell
    if signal == "scarce":
        return "vacuum"
    if signal == "answerA":
        return "aSpent" if cell == "search" else cell
    if signal == "answerB":
        return "bSpent" if cell == "search" else cell
    if signal == "feed":
        return {"aSpent": "aReady", "aHeld": "aReady", "bSpent": "bReady", "bHeld": "bReady"}.get(cell, cell)
    if signal == "miss":
        return "search" if cell in {"aReady", "bReady"} else "vacuum"
    raise ValueError(signal)


def sensory_next(signal: str, cell: str, reserve: int) -> tuple[str, int]:
    if signal == "calm":
        return cell, reserve
    if cell == "vacuum":
        return "vacuum", 0
    if signal == "none":
        return (cell, reserve - 1) if reserve else ("vacuum", 0)
    new_cell = rl_next(signal, cell)
    if new_cell == "vacuum":
        return "vacuum", 0
    return new_cell, 2 if rl_incoming(signal, cell) > 0 else reserve


def port_next(signal: str, supplied: bool, cell: str, reserve: int) -> tuple[str, int]:
    if supplied:
        supplied_signal = "probe" if signal == "none" else signal
        return sensory_next(supplied_signal, cell, reserve)
    if cell == "vacuum" or reserve == 0:
        return "vacuum", 0
    next_cell = cell if signal in {"none", "feed"} else rl_next(signal, cell)
    return (next_cell, reserve - 1) if next_cell != "vacuum" else ("vacuum", 0)


def transition_kernels() -> torch.Tensor:
    n = len(CELLS) * 3
    kernels = torch.zeros((len(SIGNALS) * 2, n, n), dtype=torch.float64)
    for signal_index, signal in enumerate(SIGNALS):
        for supplied in (False, True):
            k = signal_index * 2 + int(supplied)
            for cell_index, cell in enumerate(CELLS):
                for reserve in range(3):
                    new_cell, new_reserve = port_next(signal, supplied, cell, reserve)
                    source = cell_index * 3 + reserve
                    target = CELLS.index(new_cell) * 3 + new_reserve
                    kernels[k, source, target] = 1.0
    return kernels


def propagate(kernels: torch.Tensor, x: torch.Tensor, steps: int, scale: int | None = None) -> torch.Tensor:
    for _ in range(steps):
        y = torch.einsum("kij,bkj->bki", kernels, x)
        x = y if scale is None else torch.div(y, scale, rounding_mode="floor")
    return x


def timed_run(
    kernels: torch.Tensor,
    x: torch.Tensor,
    steps: int,
    repeats: int,
    scale: int | None,
) -> tuple[float, torch.Tensor]:
    for _ in range(3):
        propagate(kernels, x, steps, scale)
    if kernels.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = x
    for _ in range(repeats):
        out = propagate(kernels, x, steps, scale)
    if kernels.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / repeats, out


@torch.no_grad()
def implicit_packed_run(args, device: torch.device) -> dict[str, object]:
    """One checked packed-coefficient transport pass; scales are storage only."""
    blocks, width, tile = args.blocks, args.width, args.tile
    if min(blocks, width, tile) < 1 or width % 2 or tile % 2:
        raise ValueError("blocks must be positive; width and tile must be positive even integers")
    kernel_path = Path(__file__).resolve().parents[1] / "information_boltzmann/core/cgb_language_kernel.py"
    spec = importlib.util.spec_from_file_location("cgb_packed_kernel", kernel_path)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    groups = (width + 63) // 64
    estimated = blocks * width * (width // 2 + 2 * groups + 8)
    cuda = device.type == "cuda"
    baseline = {"allocated": 0, "reserved": 0}
    free = total = None
    if cuda:
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        baseline = {"allocated": torch.cuda.memory_allocated(device),
                    "reserved": torch.cuda.memory_reserved(device)}
        if free < estimated + 512 * 2**20:
            raise RuntimeError(f"packed allocation refused: free={free}, required={estimated + 512 * 2**20}")
        torch.cuda.reset_peak_memory_stats(device)
    packed = torch.randint(0, 256, (blocks, width, width // 2), dtype=torch.uint8, device=device)
    scales = torch.ones((blocks, width, groups), dtype=torch.float16, device=device)
    states = torch.zeros((blocks, 2 * width), dtype=torch.int32, device=device)
    resident = {name: value.untyped_storage().nbytes()
                for name, value in (("packed_weights", packed), ("scales", scales), ("nodes", states))}
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    resident_allocated = torch.cuda.memory_allocated(device) if cuda else None
    tile_buffer = torch.empty((min(tile, width), min(tile, width)), dtype=torch.int8, device=device)
    graph = kernel.ImplicitBipartiteGraph(width, width)
    visited = 0
    start = time.perf_counter()
    for block in range(blocks):
        def producer(old_charge, rs, cs):
            nonlocal visited
            part = packed[block, rs, slice(cs.start // 2, cs.stop // 2)]
            flow = tile_buffer[:rs.stop - rs.start, :cs.stop - cs.start]
            flow[:, 0::2] = (part & 15).to(torch.int8) - 8
            flow[:, 1::2] = (part >> 4).to(torch.int8) - 8
            visited += flow.numel()
            return flow
        before = int(states[block].sum(dtype=torch.int64))
        states[block].copy_(kernel.charge_update(graph, states[block], producer, tile_shape=(tile, tile)))
        if int(states[block].sum(dtype=torch.int64)) != before:
            raise AssertionError("packed transport changed total charge")
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    assert visited == blocks * width * width and sum(resident.values()) == estimated
    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
    return {
        "scope": {"operator_only": True, "training": False, "ability": False,
                  "scales_used_for_arithmetic": False, "bptt": False},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "device": str(device), "torch_version": torch.__version__, "seed": args.seed,
        "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "blocks": blocks, "width": width, "tile": tile, "scale_group": 64,
        "P": visited, "E": visited, "nodes": blocks * 2 * width,
        "passes": 1, "checked_coefficients": visited, "checked_conservation_blocks": blocks,
        "single_pass_wall_seconds": seconds, "flow_range": [-8, 7],
        "bytes": {"resident": resident, "resident_total": estimated, "endpoint_indices": 0,
                  "cuda_free_before": free, "cuda_total": total, "cuda_baseline": baseline,
                  "cuda_resident_allocated": resident_allocated, "cuda_peak_allocated": peak_allocated,
                  "cuda_peak_reserved": peak_reserved,
                  "execution_extra_peak_allocated": peak_allocated - resident_allocated if cuda else None},
        "limits": "Integer transport test, not a neural activity or learning model; CPU peak is unmeasured.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--implicit-packed", action="store_true")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--tile", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if args.implicit_packed:
        results = implicit_packed_run(args, device)
        print(json.dumps(results, indent=2))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return
    kernels_float = transition_kernels().to(device)
    x_int = torch.randint(
        0, 4, (args.batch, len(SIGNALS) * 2, len(CELLS) * 3), dtype=torch.int64
    ).to(device)
    x_float = x_int.to(torch.float32)
    scale = 1 << 12
    kernels_fixed = (kernels_float * scale).to(torch.int64)
    x_fixed = x_int * scale

    modes = {
        "float32": (kernels_float.to(torch.float32), x_float, None),
        "float64": (kernels_float, x_int.to(torch.float64), None),
        "int64": (kernels_float.to(torch.int64), x_int, None),
        "fixed_q12_int64": (kernels_fixed, x_fixed, scale),
    }
    if device.type == "cuda":
        modes.update({
            "float16": (kernels_float.to(torch.float16), x_float.to(torch.float16), None),
            "bfloat16": (kernels_float.to(torch.bfloat16), x_float.to(torch.bfloat16), None),
        })
    results: dict[str, object] = {
        "scope": "finite PortSeparation kernel operator microbenchmark; no training or capability claim",
        "device": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "states": len(CELLS) * 3,
        "inputs": len(SIGNALS) * 2,
        "batch": args.batch,
        "steps": args.steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "modes": {},
    }
    outputs: dict[str, torch.Tensor] = {}
    for name, (kernels, x, mode_scale) in modes.items():
        try:
            seconds, out = timed_run(kernels, x, args.steps, args.repeats, mode_scale)
            outputs[name] = out
            reported = out.to(torch.float64) / (mode_scale or 1)
            results["modes"][name] = {
                "status": "ok",
                "seconds_per_run": seconds,
                "runs_per_second": 1.0 / seconds,
                "checksum": float(reported.sum().item()),
            }
        except (RuntimeError, NotImplementedError) as exc:
            results["modes"][name] = {
                "status": "unsupported",
                "error": str(exc).splitlines()[0],
            }

    reference = outputs["float64"]
    for name, out in outputs.items():
        mode_scale = modes[name][2]
        diff = (out.to(torch.float64) / (mode_scale or 1) - reference.to(torch.float64)).abs().max().item()
        results["modes"][name]["max_abs_error_vs_float64"] = float(diff)

    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
