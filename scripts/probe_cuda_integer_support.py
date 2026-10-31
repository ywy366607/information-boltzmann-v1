"""Probe CUDA integer elementwise and matrix operator availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "dtypes": {},
    }
    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2))
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    result["device"] = props.name
    result["compute_capability"] = list(torch.cuda.get_device_capability(device))
    result["total_memory_bytes"] = props.total_memory
    result["matrix_shape"] = [256, 256]
    for dtype in (torch.int8, torch.int32, torch.int64):
        name = str(dtype).split(".")[-1]
        x = torch.arange(1, 1 + 256 * 256, device=device, dtype=dtype)
        y = torch.ones_like(x)
        a = torch.ones((256, 256), device=device, dtype=dtype)
        b = torch.ones((256, 256), device=device, dtype=dtype)
        ops = {
            "add": lambda: x + y,
            "mul": lambda: x * y,
            "sum": lambda: x.sum(),
            "where": lambda: torch.where(x > 0, x, y),
            "mm": lambda: torch.mm(a, b),
        }
        records: dict[str, object] = {"bytes_per_element": torch.empty((), dtype=dtype).element_size()}
        for op, fn in ops.items():
            try:
                out = fn()
                torch.cuda.synchronize()
                records[op] = {"status": "ok", "output_dtype": str(out.dtype)}
            except (RuntimeError, NotImplementedError) as exc:
                records[op] = {"status": "unsupported", "error": str(exc).splitlines()[0]}
        result["dtypes"][name] = records

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
