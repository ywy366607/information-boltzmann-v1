"""Benchmark hand-written Numba CUDA integer matmul on Torch CUDA tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from numba import cuda, int8, int32


TILE = 16


@cuda.jit
def int32_matmul_kernel(a, b, c):
    shared_a = cuda.shared.array((TILE, TILE), dtype=int32)
    shared_b = cuda.shared.array((TILE, TILE), dtype=int32)
    row = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    acc = int32(0)
    n = a.shape[0]
    for base in range(0, n, TILE):
        ak = base + cuda.threadIdx.x
        bk = base + cuda.threadIdx.y
        shared_a[cuda.threadIdx.y, cuda.threadIdx.x] = a[row, ak] if row < n and ak < n else 0
        shared_b[cuda.threadIdx.y, cuda.threadIdx.x] = b[bk, col] if bk < n and col < n else 0
        cuda.syncthreads()
        for j in range(TILE):
            acc += shared_a[cuda.threadIdx.y, j] * shared_b[j, cuda.threadIdx.x]
        cuda.syncthreads()
    if row < n and col < n:
        c[row, col] = acc


@cuda.jit
def int8_matmul_kernel(a, b, c):
    shared_a = cuda.shared.array((TILE, TILE), dtype=int8)
    shared_b = cuda.shared.array((TILE, TILE), dtype=int8)
    row = cuda.blockIdx.y * TILE + cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + cuda.threadIdx.x
    acc = int32(0)
    n = a.shape[0]
    for base in range(0, n, TILE):
        ak = base + cuda.threadIdx.x
        bk = base + cuda.threadIdx.y
        shared_a[cuda.threadIdx.y, cuda.threadIdx.x] = a[row, ak] if row < n and ak < n else 0
        shared_b[cuda.threadIdx.y, cuda.threadIdx.x] = b[bk, col] if bk < n and col < n else 0
        cuda.syncthreads()
        for j in range(TILE):
            acc += int32(shared_a[cuda.threadIdx.y, j]) * int32(shared_b[j, cuda.threadIdx.x])
        cuda.syncthreads()
    if row < n and col < n:
        c[row, col] = acc


def custom_i32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)
    blocks = ((a.shape[0] + TILE - 1) // TILE, (b.shape[1] + TILE - 1) // TILE)
    int32_matmul_kernel[blocks, (TILE, TILE)](a, b, c)
    return c


def custom_i8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)
    blocks = ((a.shape[0] + TILE - 1) // TILE, (b.shape[1] + TILE - 1) // TILE)
    int8_matmul_kernel[blocks, (TILE, TILE)](a, b, c)
    return c


def timed(fn, repeats: int) -> tuple[float, torch.Tensor]:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(repeats):
        out = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats / 1000.0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not cuda.is_available():
        raise RuntimeError("CUDA is required")

    n = args.size
    torch.manual_seed(20261025)
    a8 = torch.randint(0, 2, (n, n), device="cuda", dtype=torch.int8)
    b8 = torch.randint(0, 2, (n, n), device="cuda", dtype=torch.int8)
    a32, b32 = a8.to(torch.int32), b8.to(torch.int32)
    a_float, b_float = a8.to(torch.float32), b8.to(torch.float32)
    a_bf16, b_bf16 = a8.to(torch.bfloat16), b8.to(torch.bfloat16)
    reference = torch.mm(a_float, b_float)
    modes = {
        "custom_numba_int8_acc_int32": lambda: custom_i8(a8, b8),
        "custom_numba_int32_acc_int32": lambda: custom_i32(a32, b32),
        "torch_float32": lambda: torch.mm(a_float, b_float),
        "torch_bfloat16": lambda: torch.mm(a_bf16, b_bf16),
    }
    result: dict[str, object] = {
        "scope": "hand-written Numba CUDA integer matmul; no training or capability claim",
        "torch_version": torch.__version__,
        "numba_cuda": True,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "size": n,
        "repeats": args.repeats,
        "storage_bytes": {
            "int8_A_or_B": a8.numel() * a8.element_size(),
            "int32_A_or_B": a32.numel() * a32.element_size(),
            "float32_A_or_B": a_float.numel() * a_float.element_size(),
            "bfloat16_A_or_B": a_bf16.numel() * a_bf16.element_size(),
            "int32_output": n * n * 4,
        },
        "modes": {},
    }
    working_sets = {
        "custom_numba_int8_acc_int32": 2 * a8.numel() * a8.element_size() + n * n * 4,
        "custom_numba_int32_acc_int32": 2 * a32.numel() * a32.element_size() + n * n * 4,
        "torch_float32": 2 * a_float.numel() * a_float.element_size() + n * n * 4,
        "torch_bfloat16": 2 * a_bf16.numel() * a_bf16.element_size() + n * n * 2,
    }
    for name, fn in modes.items():
        seconds, out = timed(fn, args.repeats)
        error = (out.to(torch.float32) - reference).abs().max().item()
        result["modes"][name] = {
            "seconds_per_run": seconds,
            "runs_per_second": 1.0 / seconds,
            "output_dtype": str(out.dtype),
            "working_set_bytes": working_sets[name],
            "max_abs_error_vs_float32": float(error),
        }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
