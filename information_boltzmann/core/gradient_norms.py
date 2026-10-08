"""Memory-bounded L2 gradient measurement and direction-preserving clipping.

FP32 gradients can have finite entries while their FP32 sum of squares
overflows. Accumulate chunk norms in FP64; never replace non-finite entries.
The chunk size is a workspace limit, not a learning or dynamics parameter.
"""

import math

import torch


def stable_grad_norm(parameters, *, chunk_elements=1 << 20):
    """Return a FP64 total L2 norm for gradients on one device.

FP32 chunks need at most 8 MiB of temporary FP64 storage at the default
workspace limit. Missing/empty gradients are ignored, as in PyTorch clipping.
"""
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    norms = []
    device = None
    with torch.no_grad():
        for parameter in parameters:
            grad = parameter.grad
            if grad is None or not grad.numel():
                continue
            if grad.is_sparse or grad.is_complex():
                raise ValueError("Stable clipping requires dense real gradients")
            if device is not None and grad.device != device:
                raise ValueError("Stable clipping requires one gradient device")
            device = grad.device
            for chunk in grad.detach().reshape(-1).split(chunk_elements):
                norms.append(torch.linalg.vector_norm(chunk, dtype=torch.float64))
        if not norms:
            return torch.tensor(0.0, dtype=torch.float64)
        return torch.linalg.vector_norm(torch.stack(norms))


def stable_clip_grad_norm_(parameters, max_norm, *, error_if_nonfinite=True,
                           chunk_elements=1 << 20):
    """Clip all gradients by one common factor, returning their original norm.

For exceptionally large finite FP32 gradients, divide by a representable
scale first. This avoids a clipping coefficient underflowing to zero before
it multiplies the gradient. The rule matches L2 clipping with PyTorch's
1e-6 denominator epsilon, up to arithmetic rounding.
"""
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    parameters = list(parameters)
    total = stable_grad_norm(parameters, chunk_elements=chunk_elements)
    finite = bool(torch.isfinite(total).item())
    if not finite:
        if error_if_nonfinite:
            raise FloatingPointError("Gradient entries or FP64 total norm are non-finite")
        # Preserve the explicit non-finite result; caller must decide what to do.
        return total
    norm_value = float(total)
    coefficient = min(1.0, max_norm / (norm_value + 1e-6))
    with torch.no_grad():
        for parameter in parameters:
            grad = parameter.grad
            if grad is None or not grad.numel() or coefficient == 1.0:
                continue
            if coefficient >= torch.finfo(grad.dtype).tiny or coefficient == 0.0:
                grad.mul_(coefficient)
            else:
                scale = min(norm_value, torch.finfo(grad.dtype).max)
                grad.div_(scale).mul_(scale * coefficient)
    return total
