"""Optional fused execution of the existing cycle law, with no global workspace.

Numba is imported only when requested. Each CUDA thread reads one implicit
four-coordinate block and one int16 state word. Torch owns all tensor storage;
the kernel uses Torch's current stream. No topology/index/history array is added.
"""
from functools import lru_cache
import math

import torch


@lru_cache(maxsize=2)
def _kernel(double_precision):
    from numba import cuda, float32, float64, int32
    from numba.cuda import libdevice

    scalar = float64 if double_precision else float32
    add = libdevice.dadd_rn if double_precision else libdevice.fadd_rn
    if double_precision:
        @cuda.jit(device=True, inline=True)
        def subtract(a, b):
            return libdevice.dadd_rn(a, -b)
        sub = subtract
    else:
        sub = libdevice.fsub_rn
    mul = libdevice.dmul_rn if double_precision else libdevice.fmul_rn
    div = libdevice.ddiv_rn if double_precision else libdevice.fdiv_rn

    @cuda.jit(fastmath=False)
    def run(parameter, gradient, codes, has_gradient, lr):
        block = cuda.grid(1)
        if block >= codes.size:
            return
        first = 4 * block
        count = parameter.size
        zero = scalar(0)
        g0, g1, g2, g3 = zero, zero, zero, zero
        if has_gradient:
            g0 = scalar(gradient[first])
            if first + 1 < count:
                g1 = scalar(gradient[first + 1])
            if first + 2 < count:
                g2 = scalar(gradient[first + 2])
            if first + 3 < count:
                g3 = scalar(gradient[first + 3])
        scale = max(abs(g0), abs(g1), abs(g2), abs(g3))
        cue = zero
        if scale > zero:
            z0, z1, z2, z3 = div(g0, scale), div(g1, scale), div(g2, scale), div(g3, scale)
            diagonal = add(mul(z0, z0), mul(z3, z3))
            cross = add(mul(z1, z1), mul(z2, z2))
            cue = div(mul(scalar(2), sub(diagonal, cross)), add(diagonal, cross))
        cue = min(scalar(2), max(scalar(-2), cue))
        cue_code = int32(math.floor(add(mul(scalar(32), cue), scalar(0.5))))
        phase = int32(codes[block] % 65) - int32(32)
        reserve = int32(codes[block] // 65) + int32(8)
        numerator = 2 * (4 + reserve) * phase + reserve * cue_code
        next_phase = min(32, max(-32, (numerator + 8) // 16))
        next_reserve = reserve + (19 - reserve) // 4 - reserve * abs(phase) // 128
        codes[block] = next_phase + 32 + 65 * (next_reserve - 8)
        diagonal_share = div(scalar(32 + next_phase), scalar(64))
        cross_share = div(scalar(32 - next_phase), scalar(64))
        rate = div(mul(lr, scalar(reserve)), scalar(16))
        # Explicit round-to-nearest operations avoid changing the reference
        # multiplication/subtraction order through fused multiply-add.
        parameter[first] = sub(parameter[first], mul(mul(g0, diagonal_share), rate))
        if first + 1 < count:
            parameter[first + 1] = sub(parameter[first + 1], mul(mul(g1, cross_share), rate))
        if first + 2 < count:
            parameter[first + 2] = sub(parameter[first + 2], mul(mul(g2, cross_share), rate))
        if first + 3 < count:
            parameter[first + 3] = sub(parameter[first + 3], mul(mul(g3, diagonal_share), rate))

    return run


def fused_step(parameter, gradient, codes, lr):
    """In-place FP32/64 path; BF16/FP16 use the chunked Torch reference."""
    import numpy as np
    from numba import cuda

    with torch.cuda.device(parameter.device):
        parameter_view = cuda.as_cuda_array(parameter.detach().view(-1), sync=False)
        gradient_view = (parameter_view if gradient is None else
                         cuda.as_cuda_array(gradient.detach().view(-1), sync=False))
        state_view = cuda.as_cuda_array(codes, sync=False)
        stream = cuda.external_stream(torch.cuda.current_stream(parameter.device).cuda_stream)
        double = parameter.dtype == torch.float64
        scalar_lr = np.float64(lr) if double else np.float32(lr)
        _kernel(double)[(codes.numel() + 255) // 256, 256, stream](
            parameter_view, gradient_view, state_view, gradient is not None, scalar_lr)
        # Raw CUDA writes must invalidate saved autograd views just like a
        # normal in-place optimizer update. The update law is unchanged.
        torch.autograd.graph.increment_version(parameter)
