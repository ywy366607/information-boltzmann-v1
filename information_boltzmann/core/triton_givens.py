"""Local Givens scattering with fused state and angle derivatives.

One CUDA program owns one complete site, including both collision layers.
The schedule is exactly ``roll(arange(D), layer).reshape(-1, 2)``. CPU,
FP64 and noncontiguous inputs retain the native differentiable expression.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

# One warp covers up to 32 pair lanes. Two warps cover the d128 model's
# 62 pairs without the idle lanes of the old four-warp launch.
GIVENS_NUM_WARPS = 2


@triton.jit
def _first_layer(values, angles, row, index, valid, D: tl.constexpr):
    own = tl.load(values + row * D + index, valid, 0.0)
    other = tl.load(values + row * D + (index ^ 1), valid, 0.0)
    theta = tl.load(angles + row * D + index // 2, valid, 0.0)
    sign = tl.where(index % 2 == 0, -1.0, 1.0)
    return tl.cos(theta) * own + sign * tl.sin(theta) * other


@triton.jit
def _second_layer_adjoint(grad_output, angles, row, index, valid,
                          D: tl.constexpr):
    pair = ((index + 1) // 2) % (D // 2)
    partner = tl.where(index % 2 == 0, (index + D - 1) % D, (index + 1) % D)
    own = tl.load(grad_output + row * D + index, valid, 0.0)
    other = tl.load(grad_output + row * D + partner, valid, 0.0)
    theta = tl.load(angles + row * D + D // 2 + pair, valid, 0.0)
    sign = tl.where(index % 2 == 0, -1.0, 1.0)
    return tl.cos(theta) * own + sign * tl.sin(theta) * other


@triton.jit
def _givens_forward_kernel(values, angles, output, D: tl.constexpr,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    left, right = (2 * pair + D - 1) % D, 2 * pair
    u_left = _first_layer(values, angles, row, left, valid, D)
    u_right = _first_layer(values, angles, row, right, valid, D)
    theta = tl.load(angles + row * D + D // 2 + pair, valid, 0.0)
    cosine, sine = tl.cos(theta), tl.sin(theta)
    tl.store(output + row * D + left, cosine * u_left - sine * u_right, valid)
    tl.store(output + row * D + right, sine * u_left + cosine * u_right, valid)


@triton.jit
def _givens_backward_kernel(values, angles, grad_output, grad_values,
                            grad_angles, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    left, right = (2 * pair + D - 1) % D, 2 * pair
    u_left = _first_layer(values, angles, row, left, valid, D)
    u_right = _first_layer(values, angles, row, right, valid, D)
    theta1 = tl.load(angles + row * D + D // 2 + pair, valid, 0.0)
    cosine1, sine1 = tl.cos(theta1), tl.sin(theta1)
    y_left = cosine1 * u_left - sine1 * u_right
    y_right = sine1 * u_left + cosine1 * u_right
    g_left = tl.load(grad_output + row * D + left, valid, 0.0)
    g_right = tl.load(grad_output + row * D + right, valid, 0.0)
    tl.store(grad_angles + row * D + D // 2 + pair,
             g_right * y_left - g_left * y_right, valid)

    # Read the adjoint of layer 1 at layer-0 pairs. This recomputes local
    # intermediates in registers instead of storing full-field scratch.
    even, odd = 2 * pair, 2 * pair + 1
    g_even = _second_layer_adjoint(grad_output, angles, row, even, valid, D)
    g_odd = _second_layer_adjoint(grad_output, angles, row, odd, valid, D)
    x_even = tl.load(values + row * D + even, valid, 0.0)
    x_odd = tl.load(values + row * D + odd, valid, 0.0)
    theta0 = tl.load(angles + row * D + pair, valid, 0.0)
    cosine0, sine0 = tl.cos(theta0), tl.sin(theta0)
    u_even = cosine0 * x_even - sine0 * x_odd
    u_odd = sine0 * x_even + cosine0 * x_odd
    tl.store(grad_angles + row * D + pair,
             g_odd * u_even - g_even * u_odd, valid)
    tl.store(grad_values + row * D + even, cosine0 * g_even + sine0 * g_odd, valid)
    tl.store(grad_values + row * D + odd, -sine0 * g_even + cosine0 * g_odd, valid)


class TritonGivensFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, angles):
        d = values.shape[-1]
        output = torch.empty_like(values)
        _givens_forward_kernel[(values.numel() // d,)](
            values, angles, output, d, triton.next_power_of_2(d // 2),
            num_warps=GIVENS_NUM_WARPS, enable_fp_fusion=False)
        ctx.save_for_backward(values, angles)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, angles = ctx.saved_tensors
        d = values.shape[-1]
        grad_values, grad_angles = torch.empty_like(values), torch.empty_like(angles)
        _givens_backward_kernel[(values.numel() // d,)](
            values, angles, grad_output.contiguous(), grad_values, grad_angles,
            d, triton.next_power_of_2(d // 2), num_warps=GIVENS_NUM_WARPS,
            enable_fp_fusion=False)
        return grad_values, grad_angles


def native_givens(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reference scattering, including all state/angle and higher derivatives."""
    d = values.shape[-1]
    if d % 2 or angles.shape[:-2] != values.shape[:-1] or angles.shape[-1] != d // 2:
        raise ValueError("Expected values [...,D even] and angles [...,layers,D/2]")
    if angles.shape[-2] == 2:
        # Preserve the original compiled training reference for this common
        # schedule, including its slice layout and operation ordering.
        left, right = values[..., 0::2], values[..., 1::2]
        theta = angles[..., 0, :]
        value = values.clone()
        value[..., 0::2] = theta.cos() * left - theta.sin() * right
        value[..., 1::2] = theta.sin() * left + theta.cos() * right
        left = torch.cat((value[..., -1:], value[..., 1:-1:2]), -1)
        right = value[..., 0::2]
        theta = angles[..., 1, :]
        updated = value.clone()
        updated[..., -1] = theta[..., 0].cos() * left[..., 0] - theta[..., 0].sin() * right[..., 0]
        updated[..., 0] = theta[..., 0].sin() * left[..., 0] + theta[..., 0].cos() * right[..., 0]
        updated[..., 1:-1:2] = theta[..., 1:].cos() * left[..., 1:] - theta[..., 1:].sin() * right[..., 1:]
        updated[..., 2::2] = theta[..., 1:].sin() * left[..., 1:] + theta[..., 1:].cos() * right[..., 1:]
        return updated
    value = values
    for layer in range(angles.shape[-2]):
        pair = torch.roll(torch.arange(d, device=values.device), layer).reshape(-1, 2)
        left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
        theta = angles[..., layer, :]
        cosine, sine = theta.cos(), theta.sin()
        updated = value.clone()
        updated[..., pair[:, 0]] = cosine * left - sine * right
        updated[..., pair[:, 1]] = sine * left + cosine * right
        value = updated
    return value


def triton_givens(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Fused FP32 CUDA scattering; native fallback for other tensor layouts.

    Supports all batches and even nullspace widths for two layers, rather
    than hardcoding batch=1 and width=124. The CUDA backward includes angle
    gradients, so the generator continues to learn from the task objective.
    """
    d = values.shape[-1]
    if d % 2 or angles.shape[:-2] != values.shape[:-1] or angles.shape[-1] != d // 2:
        raise ValueError("Expected values [...,D even] and angles [...,layers,D/2]")
    if (values.is_cuda and angles.is_cuda and values.dtype == torch.float32
            and angles.dtype == torch.float32 and values.is_contiguous()
            and angles.is_contiguous() and 0 < d <= 512
            and angles.shape[-2] == 2):
        return TritonGivensFunction.apply(values, angles)
    return native_givens(values, angles)


def triton_givens_adjoint(grad_output: torch.Tensor,
                         angles: torch.Tensor) -> torch.Tensor:
    """State-only adjoint with reverse layer order, for legacy callers."""
    value = grad_output
    d = value.shape[-1]
    for layer in reversed(range(angles.shape[-2])):
        pair = torch.roll(torch.arange(d, device=value.device), layer).reshape(-1, 2)
        left, right = value[..., pair[:, 0]], value[..., pair[:, 1]]
        theta = angles[..., layer, :]
        cosine, sine = theta.cos(), theta.sin()
        updated = value.clone()
        updated[..., pair[:, 0]] = cosine * left + sine * right
        updated[..., pair[:, 1]] = -sine * left + cosine * right
        value = updated
    return value
