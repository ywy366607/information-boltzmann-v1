"""CUDA execution/storage microbenchmark, with supplied gradients and no model.

This measures optimizer machinery only. No dataset, learning quality, convergence
or superiority claim is made. Full lifelong optimizer admission remains closed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.conservative_cycle_optimizer import ConservativeCycle


def measure(name, constructor, count, updates):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.manual_seed(491)
    parameter = torch.randn(count, device="cuda", dtype=torch.float32, requires_grad=True)
    parameter.grad = torch.randn_like(parameter)
    optimizer = constructor(parameter)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    torch.cuda.synchronize()
    initialization_peak = torch.cuda.max_memory_allocated()
    state_bytes = sum(value.numel() * value.element_size()
                      for state in optimizer.state.values() for value in state.values()
                      if isinstance(value, torch.Tensor))
    optimizer.step()
    torch.cuda.synchronize()
    steady = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(updates):
        optimizer.step()
    end_event.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_wall
    result = dict(
        name=name, parameters=count, parameter_dtype="float32", gradient_dtype="float32",
        supplied_gradient_updates=updates + 2, measured_updates=updates,
        parameter_and_gradient_bytes=before, persistent_tensor_state_bytes=state_bytes,
        steady_allocated_bytes=steady, initialization_peak_allocated_bytes=initialization_peak,
        measured_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        extra_workspace_peak_bytes=torch.cuda.max_memory_allocated() - steady,
        cuda_interval_ms_per_update=start_event.elapsed_time(end_event) / updates,
        wall_ms_per_update=1000 * elapsed / updates,
    )
    del optimizer, parameter
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, default=2**20)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this execution-only measurement")
    if args.parameters < 1 or args.updates < 1:
        raise ValueError("positive dimensions required")
    arms = [
        ("cycle_chunk65536", lambda p: ConservativeCycle([p], 0.001, chunk_cycles=65536)),
        ("cycle_chunk8192", lambda p: ConservativeCycle([p], 0.001, chunk_cycles=8192)),
        ("cycle_fused_cuda", lambda p: ConservativeCycle([p], 0.001, fused=True)),
        ("torch_adam_default", lambda p: torch.optim.Adam([p], lr=0.001)),
    ]
    receipt = dict(
        scope="execution/storage microbenchmark only; supplied fixed gradients; no model or data",
        full_optimizer_admitted=False, execution_enabled=False, review_status="self_reviewed",
        torch_version=torch.__version__, device=torch.cuda.get_device_name(),
        optimizer_source_sha256=hashlib.sha256(
            (ROOT / "information_boltzmann/conservative_cycle_optimizer.py").read_bytes()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(
            (ROOT / "information_boltzmann/conservative_cycle_cuda.py").read_bytes()).hexdigest(),
        arms=[measure(name, constructor, args.parameters, args.updates) for name, constructor in arms],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
