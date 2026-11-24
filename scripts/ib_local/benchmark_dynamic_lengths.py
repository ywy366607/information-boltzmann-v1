"""Old length specialization versus shared kernels; runtime/numerical audit."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import types
from unittest.mock import patch

import numpy as np
import torch
import triton
import triton.language as tl

from scripts.ib_local import causal_workspace_scan as workspace
from scripts.ib_local import bfss_momentum_fused as momentum


def length_specialized(kernel):
    """Reconstruct only the prior T constexpr annotation; same kernel body."""
    fn = kernel.fn
    old = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    old.__annotations__ = dict(fn.__annotations__, T=tl.constexpr)
    old.__module__ = fn.__module__
    return triton.jit(old)


@contextmanager
def static_length_runtime():
    with patch.object(workspace, '_forward', length_specialized(workspace._forward)), \
         patch.object(workspace, '_backward', length_specialized(workspace._backward)), \
         patch.object(momentum, '_capture', length_specialized(momentum._capture)):
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(1618)
    lengths = (17, 31, 65, 89, 117, 128, 137, 480, 512)
    original = workspace._forward, workspace._backward, momentum._capture
    static = tuple(length_specialized(k) for k in original)
    errors, times, variants = {}, {}, {}
    for mode, kernels in (('static', static), ('dynamic', original)):
        values, cold, warm = [], [], []
        for t in lengths:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(t)
                aa = torch.rand(1, t, 8, 32, device='cuda')*.5+.2
                u = torch.randn_like(aa)*.1
                s = torch.randn(1, 8, 32, device='cuda')*.1
                dy = torch.randn_like(u)
                k = torch.nn.functional.normalize(torch.randn(1, t, 8, 32, device='cuda'), dim=-1)
                v = torch.randn_like(k)*.1
                alpha = torch.rand(1, t, 8, device='cuda')*.1+.8
                beta = torch.rand_like(alpha)
                sm = torch.randn(1, 8, 32, 32, device='cuda')*.1
                pm = torch.randn_like(sm)*.1
            def call():
                output = torch.empty_like(u)
                da, du, ds = torch.empty_like(aa), torch.empty_like(u), torch.empty_like(s)
                kernels[0][(1, 2)](aa, u, s, output, t, 256, 128)
                kernels[1][(1, 2)](aa, s, output, dy, da, du, ds, t, 256, 128)
                ms, mp = torch.empty_like(sm), torch.empty_like(pm)
                correction = torch.empty(8, device='cuda')
                kernels[2][(8,)](k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
                    alpha.transpose(1, 2).contiguous(), beta.transpose(1, 2).contiguous(),
                    sm, pm, ms, mp, correction, t, 32, 32., num_warps=4, enable_fp_fusion=False)
                return output, da, du, ds, ms, mp, correction
            torch.cuda.synchronize(); start = time.perf_counter()
            value = call()
            torch.cuda.synchronize(); cold.append(time.perf_counter()-start)
            values.append(tuple(v.cpu() for v in value))
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(10): call()
            torch.cuda.synchronize(); warm.append((time.perf_counter()-start)/10)
        times[mode] = dict(first_use_seconds=cold, warm_seconds=warm,
                           total_first_use=sum(cold), median_warm=float(np.median(warm)))
        variants[mode] = [sum(len(v) for v in kernel.cache.values()) for kernel in kernels]
        if mode == 'static':
            reference = values
        else:
            for i, (old, new) in enumerate(zip(reference, values)):
                errors[str(lengths[i])] = max(float((x-y).abs().max()) for x, y in zip(old, new))
                for x, y in zip(old, new): torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)
    if variants['dynamic'] != [1, 1, 1]:
        raise RuntimeError(f'shared T still specialized: {variants}')
    result = dict(schema=1, lengths=lengths, kernels=('workspace_forward', 'workspace_backward', 'momentum'),
                  compiled_variants=variants, timings=times, max_errors=errors,
                  optimizer_updates=0, world_steps=0, scope='kernel execution parity, not capability evidence')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
