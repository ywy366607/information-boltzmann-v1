"""Record actual module intervals inside a captured forward/backward chunk.

External CUDA events become graph nodes. Backward intervals bracket each
compiled module's autograd node, rather than extrapolating eager timings.
Instrumentation adds event overhead; unassigned time is reported explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D
from scripts.ib.train_q8_port_agents import TruncatedBeliefGraphTrainer, _batch
from scripts.ib.benchmark_port_execution import load_benchmark_model


def tensors(output):
    if isinstance(output, torch.Tensor):
        yield output
    elif isinstance(output, dict):
        for value in output.values():
            yield from tensors(value)
    elif isinstance(output, (tuple, list)):
        for value in output:
            yield from tensors(value)


def instrument(function, name, intervals, node_names):
    def wrapped(*args, **kwargs):
        if not torch.cuda.is_current_stream_capturing():
            return function(*args, **kwargs)
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        begin.record()
        output = function(*args, **kwargs)
        end.record()
        intervals.append((name, "forward", begin, end))
        seen = set()
        for tensor in tensors(output):
            node = tensor.grad_fn
            if node is None or node in seen:
                continue
            seen.add(node)
            node_names.setdefault(name, set()).add(node.name())
            # Each differentiable return from a fullgraph AOT module shares
            # the same compiled backward node. Avoid double counting it.
            if "CompiledFunctionBackward" not in node.name():
                raise RuntimeError(f"Unattributable {name} backward node: {node.name()}")
            backward_begin = torch.cuda.Event(enable_timing=True, external=True)
            backward_end = torch.cuda.Event(enable_timing=True, external=True)
            intervals.append((name, "backward", backward_begin, backward_end))
            node.register_prehook(lambda grads, event=backward_begin: event.record())
            node.register_hook(lambda outputs, inputs, event=backward_end: event.record())
        return output
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--bath", choices=("quadratic", "unified", "selective"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model, cfg, bath_replaced = load_benchmark_model(saved, args.bath)
    intervals, node_names = [], {}
    for name, module in (("collision", model.collision), ("bath", model.bath),
                         ("write", model.write_agent), ("read", model.readout),
                         ("decoder", model.decoder)):
        module.forward = instrument(torch.compile(module.forward, fullgraph=True, dynamic=False),
                                    name, intervals, node_names)
    model.transport.apply_multiplier = instrument(
        torch.compile(model.transport.apply_multiplier, fullgraph=True),
        "transport", intervals, node_names)
    print("Capturing module forward/backward event pairs...", flush=True)
    runner = TruncatedBeliefGraphTrainer(
        model, tokens=cfg["tokens"], chunk_tokens=cfg["bptt_chunk_tokens"],
        lr=cfg["lr"], max_grad_norm=cfg["max_grad_norm"])
    with torch.no_grad():
        runner.field.copy_(saved["state"])
        runner.precision.copy_(saved["precision"])
    data = np.load(Path(cfg["data"]) / "train.npy", mmap_mode="r")
    offset = int(saved["step"]) * cfg["tokens"]
    del saved
    samples = []
    for repeat in range(args.repeats + 2):
        ids, targets = _batch(data, offset + repeat * cfg["bptt_chunk_tokens"], cfg["bptt_chunk_tokens"])
        runner.ids.copy_(ids)
        runner.targets.copy_(targets)
        runner.optimizer.zero_grad(set_to_none=False)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        runner.graph.replay()
        end.record()
        torch.cuda.synchronize()
        row = {"total_graph_ms": begin.elapsed_time(end)}
        for name, direction, start_event, end_event in intervals:
            key = f"{name}_{direction}_ms"
            row[key] = row.get(key, 0.0) + start_event.elapsed_time(end_event)
        with torch.no_grad():
            runner.field.copy_(runner.next_field)
            runner.precision.copy_(runner.next_precision)
        if repeat >= 2:
            samples.append(row)
        print(json.dumps(row), flush=True)
    medians = {key: statistics.median(row[key] for row in samples) for key in samples[0]}
    medians["unassigned_ms"] = medians["total_graph_ms"] - sum(
        value for key, value in medians.items() if key != "total_graph_ms")
    result = {"scope": "instrumented_production_chunk_forward_backward_excludes_optimizer",
              "checkpoint": str(args.checkpoint), "K": cfg["K"], "dt": cfg["tau_0"],
              "bath": cfg["dissipation_type"], "bath_replaced_for_timing": bath_replaced,
              "chunk_tokens": cfg["bptt_chunk_tokens"], "medians": medians,
              "backward_nodes": {key: sorted(value) for key, value in node_names.items()},
              "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
