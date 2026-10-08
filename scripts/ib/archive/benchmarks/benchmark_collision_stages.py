"""Captured forward/backward collision stages, including coarse-grid sizes.

Isolated timings locate execution overhead; they are not additive percentages
of a production update and do not measure model capability.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import LocalInvariantCollision3D
from information_boltzmann.core.triton_projection import (
    NullspaceProjection, NullspaceReconstruction,
)


def timing(function, value, parameters, repeats, *, autotune=False):
    options = {"max_autotune": True, "triton.cudagraphs": False} if autotune else None
    compiled = torch.compile(function, fullgraph=True, dynamic=False, options=options)
    direction = torch.randn_like(function(value))

    def run():
        value.grad = None
        for parameter in parameters:
            parameter.grad = None
        output = compiled(value)
        (output * direction).sum().backward()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    calls_per_graph = 16
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            run()
    # Compilation can leave this desktop GPU at its idle clock. Exercise the
    # captured kernels before sampling; especially important for 32-site grids.
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        replays = 20
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) / (replays * calls_per_graph))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(89)
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    rows = []
    for d, nodes in ((128, 256), (128, 32), (256, 256)):
        collision = LocalInvariantCollision3D((nodes, 1, 1), 8, d // 8).cuda()
        # Match the channel-major layout produced by the spectral transport.
        value = torch.randn(1, d, nodes, device="cuda").transpose(1, 2).requires_grad_()
        basis = collision.nullspace.float().contiguous()
        left = collision.projection_left.float().contiguous()
        right = collision.projection_right.float().contiguous()
        base = value.detach()
        coefficient = torch.randn(1, nodes, d - 4, device="cuda", requires_grad=True)
        stages = (
            ("project_dense", lambda x: x @ basis, value, ()),
            ("project_structured", lambda x: NullspaceProjection.apply(x, left, right), value, ()),
            ("reconstruct_dense", lambda x: base + x @ basis.T, coefficient, ()),
            ("reconstruct_structured", lambda x: NullspaceReconstruction.apply(base, x, left, right), coefficient, ()),
            ("angle_network", lambda x: collision.angle(collision.norm(x)), value,
             tuple(collision.angle.parameters()) + tuple(collision.norm.parameters())),
            ("full_collision_structured", lambda x: collision(x, 4.0)[0], value,
             tuple(collision.parameters())),
        )
        for name, function, inp, parameters in stages:
            print(f"Measuring d={d}, nodes={nodes}, {name}...", flush=True)
            row = {"d": d, "nodes": nodes, "stage": name,
                   **timing(function, inp, parameters, args.repeats, autotune=args.autotune)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        collision._structured_projection = False
        rows.append({"d": d, "nodes": nodes, "stage": "full_collision_dense",
                     **timing(lambda x: collision(x, 4.0)[0], value,
                              tuple(collision.parameters()), args.repeats,
                              autotune=args.autotune)})
    result = {"scope": "isolated_compiled_cuda_graph_forward_backward_not_additive",
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "autotune": args.autotune, "field_layout": "channel_major_fft",
              "calls_per_graph": 16, "warmup_graph_replays": 1000, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
