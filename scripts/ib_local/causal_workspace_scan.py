"""Causal affine workspace scan, with a fused CUDA forward and pullback.

The workspace is small, but its time axis must not become a Python GPU loop.
CPU reference is retained solely for numerical and gradient checks.
T is deliberately a non-specialized runtime argument: variable utterance
length must not create another forward/backward binary at every new length.
"""
import torch


def reference_scan(a, u, initial):
    state = initial
    output = []
    for t in range(u.shape[1]):
        state = a[:, t] * state + u[:, t]
        output.append(state)
    return torch.stack(output, 1)


try:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['T'])
    def _forward(A, U, S, Y, T, N: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = j < N
        state = tl.load(S + b*N+j, mask, 0)
        for t in range(T):
            offset = (b*T+t)*N+j
            state = tl.load(A+offset, mask, 0)*state+tl.load(U+offset, mask, 0)
            tl.store(Y+offset, state, mask)

    @triton.jit(do_not_specialize=['T'])
    def _backward(A, S, Y, DY, DA, DU, DS, T, N: tl.constexpr,
                  BLOCK: tl.constexpr):
        b = tl.program_id(0)
        j = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
        mask = j < N
        adjoint = tl.full((BLOCK,), 0., tl.float32)
        for step in range(T):
            t = T-1-step
            offset = (b*T+t)*N+j
            adjoint += tl.load(DY+offset, mask, 0)
            if t == 0:
                previous = tl.load(S+b*N+j, mask, 0)
            else:
                previous = tl.load(Y+offset-N, mask, 0)
            tl.store(DA+offset, adjoint*previous, mask)
            tl.store(DU+offset, adjoint, mask)
            adjoint *= tl.load(A+offset, mask, 0)
        tl.store(DS+b*N+j, adjoint, mask)

except ImportError:
    triton = None


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, u, initial):
        a, u, initial = a.contiguous(), u.contiguous(), initial.contiguous()
        result = torch.empty_like(u)
        b, t = u.shape[:2]
        n = initial[0].numel()
        _forward[(b, triton.cdiv(n, 128))](a, u, initial, result, t, n, 128)
        ctx.save_for_backward(a, initial, result)
        return result

    @staticmethod
    def backward(ctx, derivative):
        a, initial, result = ctx.saved_tensors
        da, du, ds = torch.empty_like(a), torch.empty_like(result), torch.empty_like(initial)
        b, t = result.shape[:2]
        n = initial[0].numel()
        _backward[(b, triton.cdiv(n, 128))](
            a, initial, result, derivative.contiguous(), da, du, ds, t, n, 128)
        return da, du, ds


def causal_scan(a, u, initial):
    if a.shape != u.shape or initial.shape != (u.shape[0], *u.shape[2:]) or not u.shape[1]:
        raise ValueError('workspace scan needs matching [batch,time,slots,width] tensors')
    if u.is_cuda:
        if triton is None or u.dtype != torch.float32:
            raise RuntimeError('CUDA workspace requires the FP32 fused scan')
        return _Scan.apply(a, u, initial)
    return reference_scan(a, u, initial)
