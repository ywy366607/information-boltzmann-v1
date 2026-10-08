"""Exact rank-four representation of the collision's SVD nullspace basis.

For this four-invariant SVD completion, N = J + L R^T, where J selects
coordinates 4:D. L and R are fixed basis factors, not learned low-rank
approximations of the field. CUDA kernels fuse all four reductions per site.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _project(x, left, right, output, D: tl.constexpr, BLOCK: tl.constexpr,
             NODES: tl.constexpr, XB: tl.constexpr, XN: tl.constexpr,
             XD: tl.constexpr):
    row = tl.program_id(0)
    offset = (row // NODES) * XB + (row % NODES) * XN
    index = tl.arange(0, BLOCK)
    value = tl.load(x + offset + index * XD, index < D, 0.0)
    a0 = tl.sum(value * tl.load(left + index * 4, index < D, 0.0), 0)
    a1 = tl.sum(value * tl.load(left + index * 4 + 1, index < D, 0.0), 0)
    a2 = tl.sum(value * tl.load(left + index * 4 + 2, index < D, 0.0), 0)
    a3 = tl.sum(value * tl.load(left + index * 4 + 3, index < D, 0.0), 0)
    valid = index < D - 4
    tail = tl.load(x + offset + (index + 4) * XD, valid, 0.0)
    result = tail + a0 * tl.load(right + index * 4, valid, 0.0)
    result += a1 * tl.load(right + index * 4 + 1, valid, 0.0)
    result += a2 * tl.load(right + index * 4 + 2, valid, 0.0)
    result += a3 * tl.load(right + index * 4 + 3, valid, 0.0)
    tl.store(output + row * (D - 4) + index, result, valid)


@triton.jit
def _lift(x, left, right, base, output, D: tl.constexpr,
          BLOCK: tl.constexpr, ADD_BASE: tl.constexpr, NODES: tl.constexpr,
          BB: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
          OB: tl.constexpr, ON: tl.constexpr, OD: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < D - 4
    value = tl.load(x + row * (D - 4) + index, valid, 0.0)
    a0 = tl.sum(value * tl.load(right + index * 4, valid, 0.0), 0)
    a1 = tl.sum(value * tl.load(right + index * 4 + 1, valid, 0.0), 0)
    a2 = tl.sum(value * tl.load(right + index * 4 + 2, valid, 0.0), 0)
    a3 = tl.sum(value * tl.load(right + index * 4 + 3, valid, 0.0), 0)
    tail = tl.load(x + row * (D - 4) + index - 4,
                   (index >= 4) & (index < D), 0.0)
    result = tail + a0 * tl.load(left + index * 4, index < D, 0.0)
    result += a1 * tl.load(left + index * 4 + 1, index < D, 0.0)
    result += a2 * tl.load(left + index * 4 + 2, index < D, 0.0)
    result += a3 * tl.load(left + index * 4 + 3, index < D, 0.0)
    if ADD_BASE:
        base_offset = (row // NODES) * BB + (row % NODES) * BN
        result += tl.load(base + base_offset + index * BD, index < D, 0.0)
    out_offset = (row // NODES) * OB + (row % NODES) * ON
    tl.store(output + out_offset + index * OD, result, index < D)


def _project_cuda(x, left, right):
    d = left.shape[0]
    out = x.new_empty((*x.shape[:-1], d - 4))
    _project[(x.numel() // d,)](x, left, right, out, d,
                                triton.next_power_of_2(d), x.shape[1],
                                x.stride(0), x.stride(1), x.stride(2), num_warps=4,
                                enable_fp_fusion=False)
    return out


def _lift_cuda(x, left, right, base=None):
    d = left.shape[0]
    out = torch.empty_like(base) if base is not None else x.new_empty((*x.shape[:-1], d))
    base_strides = ((base.stride(0), base.stride(1), base.stride(2))
                    if base is not None else (0, 0, 0))
    _lift[(x.numel() // (d - 4),)](x, left, right, base, out, d,
                                   triton.next_power_of_2(d), base is not None,
                                   x.shape[1], *base_strides,
                                   out.stride(0), out.stride(1), out.stride(2),
                                   num_warps=4, enable_fp_fusion=False)
    return out


class NullspaceProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, left, right):
        ctx.save_for_backward(left, right)
        return _project_cuda(value, left, right)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        left, right = ctx.saved_tensors
        return _lift_cuda(gradient.contiguous(), left, right), None, None


class NullspaceReconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, value, left, right):
        ctx.save_for_backward(left, right)
        return _lift_cuda(value, left, right, base)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        left, right = ctx.saved_tensors
        return gradient, _project_cuda(gradient, left, right), None, None


def supports_projection(value: torch.Tensor) -> bool:
    return (value.is_cuda and value.dtype == torch.float32
            and value.ndim == 3 and 4 < value.shape[-1] <= 512)
