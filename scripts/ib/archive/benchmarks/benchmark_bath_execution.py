"""Captured bath cost on the same checkpoint field, without training claims."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.torus3d import QuadraticTorusBath, UnifiedTorusDissipation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    torch.manual_seed(11)
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = saved["config"]
    field_value = saved["state"].detach()
    probe = torch.randn_like(field_value)
    rows = []
    for name in ("unified", "quadratic"):
        if name == "unified":
            bath = UnifiedTorusDissipation(shape=tuple(cfg["shape"]), d=cfg["channels"],
                                           rank=cfg["dissipation_rank"]).cuda()
            bath.load_state_dict({key.removeprefix("bath."): value
                                  for key, value in saved["model"].items()
                                  if key.startswith("bath.")})
        else:
            # The same shape/dtype/state/dt; the standard local bath.
            # This comparison measures computation, not loss after a swap.
            bath = QuadraticTorusBath(shape=tuple(cfg["shape"]), d=cfg["channels"]).cuda()
        field = field_value.clone().requires_grad_()
        compiled = torch.compile(bath.forward, fullgraph=True)

        def forward_backward():
            output, _ = compiled(field, cfg["tau_0"])
            (output * probe).sum().backward()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                bath.zero_grad(set_to_none=True)
                field.grad = None
                forward_backward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            field.grad.zero_()
            for parameter in bath.parameters():
                parameter.grad.zero_()
            forward_backward()
        samples = []
        for _ in range(8):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 100)
        row = {"bath": name, "median_forward_backward_ms": statistics.median(samples),
               "samples_ms": samples}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graph, compiled, bath, field
    result = {"scope": "isolated_compiled_captured_bath_forward_backward_not_capability",
              "checkpoint": str(args.checkpoint), "shape": cfg["shape"],
              "channels": cfg["channels"], "dt": cfg["tau_0"],
              "gpu": torch.cuda.get_device_name(), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
