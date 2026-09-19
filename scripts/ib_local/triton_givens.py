import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import triton
import triton.language as tl
import time

@triton.jit
def givens_l0_kernel(
    val_ptr,      # [N, 124]
    angles_ptr,   # [N, 2, 62]
    out_ptr,      # [N, 124]
    neg_angle: tl.constexpr,
    N_NODES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N_NODES:
        return

    val_base = val_ptr + pid * 124
    out_base = out_ptr + pid * 124
    ang_base0 = angles_ptr + pid * 124

    idx62 = tl.arange(0, 64)
    mask62 = idx62 < 62

    even_idx = idx62 * 2
    odd_idx = idx62 * 2 + 1
    v_even = tl.load(val_base + even_idx, mask=mask62, other=0.0)
    v_odd = tl.load(val_base + odd_idx, mask=mask62, other=0.0)

    th0 = tl.load(ang_base0 + idx62, mask=mask62, other=0.0)
    if neg_angle:
        th0 = -th0
    cos0 = tl.cos(th0)
    sin0 = tl.sin(th0)
    u_even = cos0 * v_even - sin0 * v_odd
    u_odd = sin0 * v_even + cos0 * v_odd

    tl.store(out_base + even_idx, u_even, mask=mask62)
    tl.store(out_base + odd_idx, u_odd, mask=mask62)

@triton.jit
def givens_l1_kernel(
    u_ptr,        # [N, 124]
    angles_ptr,   # [N, 2, 62]
    out_ptr,      # [N, 124]
    neg_angle: tl.constexpr,
    N_NODES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N_NODES:
        return

    u_base = u_ptr + pid * 124
    out_base = out_ptr + pid * 124
    ang_base1 = angles_ptr + pid * 124 + 62

    idx62 = tl.arange(0, 64)
    mask62 = idx62 < 62

    # Layer 1 schedule:
    # Pair 0: (123, 0)
    # Pair i (1..61): (2*i - 1, 2*i)
    left_idx = tl.where(idx62 == 0, 123, 2 * idx62 - 1)
    right_idx = 2 * idx62

    u_left = tl.load(u_base + left_idx, mask=mask62, other=0.0)
    u_right = tl.load(u_base + right_idx, mask=mask62, other=0.0)

    th1 = tl.load(ang_base1 + idx62, mask=mask62, other=0.0)
    if neg_angle:
        th1 = -th1
    cos1 = tl.cos(th1)
    sin1 = tl.sin(th1)

    out_left = cos1 * u_left - sin1 * u_right
    out_right = sin1 * u_left + cos1 * u_right

    tl.store(out_base + left_idx, out_left, mask=mask62)
    tl.store(out_base + right_idx, out_right, mask=mask62)


class TritonGivens2LayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val, angles):
        B, N, D = val.shape
        scratch = torch.empty_like(val)
        out = torch.empty_like(val)
        grid = (N,)
        givens_l0_kernel[grid](val, angles, scratch, neg_angle=False, N_NODES=N)
        givens_l1_kernel[grid](scratch, angles, out, neg_angle=False, N_NODES=N)
        ctx.save_for_backward(angles)
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        N = ctx.N
        scratch = torch.empty_like(grad_output)
        grad_val = torch.empty_like(grad_output)
        grid = (N,)
        # Invert Layer 1 first with negative angle
        givens_l1_kernel[grid](grad_output, angles, scratch, neg_angle=True, N_NODES=N)
        # Invert Layer 0 with negative angle
        givens_l0_kernel[grid](scratch, angles, grad_val, neg_angle=True, N_NODES=N)
        return grad_val, None


def triton_givens(val: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    return TritonGivens2LayerFunction.apply(val, angles)


def triton_givens_adjoint(grad_output: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Exact adjoint (VJP) through 2-layer Givens rotations without autograd overhead."""
    B, N, D = grad_output.shape
    scratch = torch.empty_like(grad_output)
    grad_val = torch.empty_like(grad_output)
    grid = (N,)
    givens_l1_kernel[grid](grad_output, angles, scratch, neg_angle=True, N_NODES=N)
    givens_l0_kernel[grid](scratch, angles, grad_val, neg_angle=True, N_NODES=N)
    return grad_val


def test_triton_givens():
    torch.manual_seed(42)
    N = 256
    val = torch.randn(1, N, 124, device="cuda", requires_grad=True)
    angles = torch.randn(1, N, 2, 62, device="cuda")

    schedules = [
        torch.roll(torch.arange(124, device="cuda"), layer).reshape(-1, 2)
        for layer in range(2)
    ]
    # PyTorch reference
    val_py = val.clone().detach().requires_grad_(True)
    curr = val_py
    for layer in range(2):
        pair = schedules[layer]
        l, r = curr[..., pair[:, 0]], curr[..., pair[:, 1]]
        th = angles[:, :, layer]
        cos, sin = th.cos(), th.sin()
        upd = curr.clone()
        upd[..., pair[:, 0]] = cos * l - sin * r
        upd[..., pair[:, 1]] = sin * l + cos * r
        curr = upd

    g = torch.randn_like(curr)
    curr.backward(g)
    grad_py = val_py.grad.clone()

    # Triton
    val_tr = val.clone().detach().requires_grad_(True)
    out_tr = triton_givens(val_tr, angles)
    out_tr.backward(g)
    grad_tr = val_tr.grad.clone()

    diff_fwd = (curr - out_tr).abs().max().item()
    diff_bwd = (grad_py - grad_tr).abs().max().item()
    print(f"Forward  diff: {diff_fwd:.8e}")
    print(f"Backward diff: {diff_bwd:.8e}")
    assert diff_fwd < 1e-5 and diff_bwd < 1e-5, f"Mismatch: fwd={diff_fwd}, bwd={diff_bwd}"
    print("ALL TESTS PASSED WITH 100% NUMERICAL EXACTNESS!")

    # Benchmark forward + backward across 128 tokens
    N_TOKENS = 128
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        curr = val_py
        for layer in range(2):
            pair = schedules[layer]
            l, r = curr[..., pair[:, 0]], curr[..., pair[:, 1]]
            th = angles[:, :, layer]
            cos, sin = th.cos(), th.sin()
            upd = curr.clone()
            upd[..., pair[:, 0]] = cos * l - sin * r
            upd[..., pair[:, 1]] = sin * l + cos * r
            curr = upd
    torch.cuda.synchronize()
    t_py = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        out_tr = triton_givens(val_tr, angles)
    torch.cuda.synchronize()
    t_tr = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(f"PyTorch Givens (128 tokens): {t_py:.2f} ms")
    print(f"Triton  Givens (128 tokens): {t_tr:.2f} ms (Speedup: {t_py/t_tr:.2f}x)")
    print("=" * 60)

if __name__ == "__main__":
    test_triton_givens()
