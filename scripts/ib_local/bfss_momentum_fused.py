"""Inference-only S/P observation replay, one CUDA program per head.

The reference in bfss_gdn_capture remains the CPU and audit implementation.
Only already observed input projections are consumed; no learning rule changes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _capture(kp, vp, ap, bp, sp, pp, so, po, co, T: tl.constexpr,
             D: tl.constexpr, BUDGET: tl.constexpr):
    bh = tl.program_id(0)
    z = tl.arange(0, D)
    ij = z[:, None] * D + z[None, :]
    s = tl.load(sp + bh * D * D + ij)
    p = tl.load(pp + bh * D * D + ij)
    correction = tl.full((), 0., tl.float32)
    for t in range(T):
        k = tl.load(kp + (bh * T + t) * D + z)
        v = tl.load(vp + (bh * T + t) * D + z)
        a = tl.load(ap + bh * T + t)
        b = tl.load(bp + bh * T + t)
        prior = a * s
        error = v - tl.sum(prior * k[:, None], axis=0)
        innovation = b * (k[:, None] * error[None, :])
        p = a * p + innovation
        proposed = prior + innovation
        s = proposed
        if BUDGET > 0:
            mag = tl.abs(proposed)
            row = tl.sum(mag, axis=1)
            col = tl.sum(mag, axis=0)
            den = tl.maximum(tl.maximum(row[:, None], col[None, :]), BUDGET)
            s = proposed * (BUDGET / den)
            diff = s - proposed
            correction += tl.sum(tl.sum(diff * diff, axis=1), axis=0)
    tl.store(so + bh * D * D + ij, s)
    tl.store(po + bh * D * D + ij, p)
    tl.store(co + bh, correction)


@torch.no_grad()
def capture(k, v, alpha, beta, s, p, budget):
    """Inputs [B,T,H,D], [B,T,H]; output matrices and total correction norm."""
    if s.dtype != torch.float32 or s.shape[-1] != s.shape[-2]:
        raise ValueError('fused observation replay requires square FP32 states')
    b, t, h, d = k.shape
    k, v = (a.transpose(1, 2).contiguous() for a in (k, v))
    alpha, beta = (a.transpose(1, 2).contiguous() for a in (alpha, beta))
    s, p = s.contiguous(), p.contiguous()
    out_s, out_p = torch.empty_like(s), torch.empty_like(p)
    corr = torch.empty(b*h, device=s.device, dtype=torch.float32)
    _capture[(b*h,)](k, v, alpha, beta, s, p, out_s, out_p, corr,
                      T=t, D=d, BUDGET=float(budget), num_warps=4,
                      enable_fp_fusion=False)
    return out_s, out_p, corr.sum().sqrt()
