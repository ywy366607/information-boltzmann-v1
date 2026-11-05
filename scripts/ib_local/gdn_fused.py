"""Fused Triton recurrence for `decay_then_delta` and `feedback_delta` (2026-11-03).

The reference layer in `gated_deltanet.py` runs the delta recurrence one event at a
time; inside a CUDA graph each event still launches dozens of tiny kernels per layer,
which bounds training at about 1.3 kB/s on a GTX 1650.  Here one program per
(batch, head) runs the whole window in registers: one forward and one backward kernel
per layer and window.

The arithmetic is the reference's, event by event:

    r = S^T k
    target = v + eta * (softsign(g r + v) - v)      (eta = 0: plain decay_then_delta)
    e = target - alpha r
    P = alpha S + beta k e^T
    S' = P * b / max(b, row_L1(P), col_L1(P))       (only with a content budget)
    o = S'^T q

The backward pass replays the window in reverse from the stored per-event states and
returns exact gradients for q, k, v, alpha, beta, eta, g and the initial state.  The
budget's max() follows PyTorch's tie rule only away from ties (measure zero).

`enable()` swaps `GatedDeltaNetLayer.forward` for a version that calls the fused kernels
for these two recurrences and keeps the reference path for every other configuration.
`tests/test_gdn_fused.py` checks outputs, states, diagnostics and all gradients against
the reference layer.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # the trainers import the package path; tests may import the flat module
    from scripts.ib_local import gated_deltanet
except ImportError:
    import gated_deltanet

FUSED = ("decay_then_delta", "feedback_delta")


@triton.jit
def _forward_kernel(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, eta_ptr, gain_ptr, s0_ptr,
                    o_ptr, states_ptr, esq_ptr, corr_ptr, T, H, budget,
                    D: tl.constexpr, FEEDBACK: tl.constexpr, BUDGET: tl.constexpr):
    bh = tl.program_id(0)
    h = bh % H
    offs = tl.arange(0, D)
    grid = offs[:, None] * D + offs[None, :]
    S = tl.load(s0_ptr + bh * D * D + grid)
    tl.store(states_ptr + bh * (T + 1) * D * D + grid, S)
    eta = 0.0
    gain = 0.0
    if FEEDBACK:
        eta = tl.load(eta_ptr + h)
        gain = tl.load(gain_ptr + h)
    corr = tl.sum(tl.sum(S * 0.0, axis=1), axis=0)
    for t in range(T):
        row = (bh * T + t) * D
        kt = tl.load(k_ptr + row + offs)
        vt = tl.load(v_ptr + row + offs)
        qt = tl.load(q_ptr + row + offs)
        at = tl.load(a_ptr + bh * T + t)
        bt = tl.load(b_ptr + bh * T + t)
        r = tl.sum(S * kt[:, None], axis=0)
        target = vt
        if FEEDBACK:
            z = gain * r + vt
            target = vt + eta * (z / (1.0 + tl.abs(z)) - vt)
        e = target - at * r
        P = at * S + bt * kt[:, None] * e[None, :]
        S = P
        if BUDGET:
            mag = tl.abs(P)
            R = tl.sum(mag, axis=1)
            C = tl.sum(mag, axis=0)
            den = tl.maximum(tl.maximum(R[:, None], C[None, :]), budget)
            S = P * (budget / den)
            diff = S - P
            corr = tl.sum(tl.sum(diff * diff, axis=1), axis=0)
        tl.store(esq_ptr + bh * T + t, tl.sum(e * e, axis=0))
        tl.store(o_ptr + row + offs, tl.sum(S * qt[:, None], axis=0))
        tl.store(states_ptr + (bh * (T + 1) + t + 1) * D * D + grid, S)
    tl.store(corr_ptr + bh, corr)


@triton.jit
def _backward_kernel(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, eta_ptr, gain_ptr, states_ptr,
                     do_ptr, dsT_ptr, dq_ptr, dk_ptr, dv_ptr, da_ptr, db_ptr,
                     deta_ptr, dgain_ptr, ds0_ptr, T, H, budget,
                     D: tl.constexpr, FEEDBACK: tl.constexpr, BUDGET: tl.constexpr):
    bh = tl.program_id(0)
    h = bh % H
    offs = tl.arange(0, D)
    grid = offs[:, None] * D + offs[None, :]
    dS = tl.load(dsT_ptr + bh * D * D + grid)
    eta = 0.0
    gain = 0.0
    if FEEDBACK:
        eta = tl.load(eta_ptr + h)
        gain = tl.load(gain_ptr + h)
    zero = tl.sum(tl.sum(dS * 0.0, axis=1), axis=0)
    deta = zero
    dgain = zero
    for i in range(T):
        t = T - 1 - i
        row = (bh * T + t) * D
        kt = tl.load(k_ptr + row + offs)
        vt = tl.load(v_ptr + row + offs)
        qt = tl.load(q_ptr + row + offs)
        dot = tl.load(do_ptr + row + offs)
        at = tl.load(a_ptr + bh * T + t)
        bt = tl.load(b_ptr + bh * T + t)
        S_new = tl.load(states_ptr + (bh * (T + 1) + t + 1) * D * D + grid)
        S_old = tl.load(states_ptr + (bh * (T + 1) + t) * D * D + grid)
        # o = S_new^T q
        dS += qt[:, None] * dot[None, :]
        tl.store(dq_ptr + row + offs, tl.sum(S_new * dot[None, :], axis=1))
        # Recompute the proposed matrix.
        r = tl.sum(S_old * kt[:, None], axis=0)
        target = vt
        z = vt
        soft = vt
        if FEEDBACK:
            z = gain * r + vt
            soft = z / (1.0 + tl.abs(z))
            target = vt + eta * (soft - vt)
        e = target - at * r
        P = at * S_old + bt * kt[:, None] * e[None, :]
        dP = dS
        if BUDGET:
            mag = tl.abs(P)
            R = tl.sum(mag, axis=1)
            C = tl.sum(mag, axis=0)
            Rm = R[:, None] + 0.0 * C[None, :]
            Cm = C[None, :] + 0.0 * R[:, None]
            den = tl.maximum(tl.maximum(Rm, Cm), budget)
            G = dS * P * (-budget / (den * den))
            row_sel = (Rm >= Cm) & (Rm > budget)
            col_sel = (Cm > Rm) & (Cm > budget)
            dR = tl.sum(tl.where(row_sel, G, 0.0), axis=1)
            dC = tl.sum(tl.where(col_sel, G, 0.0), axis=0)
            sign = tl.where(P > 0, 1.0, tl.where(P < 0, -1.0, 0.0))
            dP = dS * (budget / den) + sign * (dR[:, None] + dC[None, :])
        # P = a S_old + b k e^T
        da = tl.sum(tl.sum(dP * S_old, axis=1), axis=0)
        dPe = tl.sum(dP * e[None, :], axis=1)
        db = tl.sum(kt * dPe, axis=0)
        dk = bt * dPe
        de = bt * tl.sum(dP * kt[:, None], axis=0)
        dS_old = at * dP
        # e = target - a r
        da -= tl.sum(de * r, axis=0)
        dr = -at * de
        dv = de
        if FEEDBACK:
            slope = 1.0 / ((1.0 + tl.abs(z)) * (1.0 + tl.abs(z)))
            dz = eta * slope * de
            dv = (1.0 - eta) * de + dz
            dr += gain * dz
            deta += tl.sum(de * (soft - vt), axis=0)
            dgain += tl.sum(dz * r, axis=0)
        # r = S_old^T k
        dS_old += kt[:, None] * dr[None, :]
        dk += tl.sum(S_old * dr[None, :], axis=1)
        tl.store(dk_ptr + row + offs, dk)
        tl.store(dv_ptr + row + offs, dv)
        tl.store(da_ptr + bh * T + t, da)
        tl.store(db_ptr + bh * T + t, db)
        dS = dS_old
    tl.store(ds0_ptr + bh * D * D + grid, dS)
    tl.store(deta_ptr + bh, deta)
    tl.store(dgain_ptr + bh, dgain)


class FusedDelta(torch.autograd.Function):
    """Inputs [B,H,T,D] for q/k/v, [B,H,T] for alpha/beta, [H] for eta/gain, [B,H,D,D] state."""

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, eta, gain, state, budget, feedback):
        B, H, T, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, state = alpha.contiguous(), beta.contiguous(), state.contiguous()
        out = torch.empty_like(v)
        states = torch.empty(B, H, T + 1, D, D, device=q.device, dtype=q.dtype)
        esq = torch.empty(B, H, T, device=q.device, dtype=q.dtype)
        corr = torch.empty(B, H, device=q.device, dtype=q.dtype)
        _forward_kernel[(B * H,)](q, k, v, alpha, beta, eta, gain, state, out, states, esq, corr,
                                  T, H, float(budget), D=D, FEEDBACK=feedback,
                                  BUDGET=bool(budget), num_warps=4)
        ctx.save_for_backward(q, k, v, alpha, beta, eta, gain, states)
        ctx.budget, ctx.feedback = float(budget), feedback
        ctx.mark_non_differentiable(esq, corr)
        return out, states[:, :, -1].clone(), esq, corr

    @staticmethod
    def backward(ctx, d_out, d_final, _esq, _corr):
        q, k, v, alpha, beta, eta, gain, states = ctx.saved_tensors
        B, H, T, D = q.shape
        d_out = torch.zeros_like(q) if d_out is None else d_out.contiguous()
        d_final = (torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
                   if d_final is None else d_final.contiguous())
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        da, db = torch.empty_like(alpha), torch.empty_like(beta)
        deta = torch.empty(B * H, device=q.device, dtype=q.dtype)
        dgain = torch.empty_like(deta)
        ds0 = torch.empty(B, H, D, D, device=q.device, dtype=q.dtype)
        _backward_kernel[(B * H,)](q, k, v, alpha, beta, eta, gain, states, d_out, d_final,
                                   dq, dk, dv, da, db, deta, dgain, ds0, T, H, ctx.budget,
                                   D=D, FEEDBACK=ctx.feedback, BUDGET=bool(ctx.budget), num_warps=4)
        deta = deta.view(B, H).sum(0) if ctx.feedback else None
        dgain = dgain.view(B, H).sum(0) if ctx.feedback else None
        return dq, dk, dv, da, db, deta, dgain, ds0, None, None


_reference_forward = gated_deltanet.GatedDeltaNetLayer.forward


def fused_forward(self, x, state):
    """Same computation and outputs as GatedDeltaNetLayer.forward for the fused recurrences."""
    if self.recurrence not in FUSED or self.share_size or not x.is_cuda:
        return _reference_forward(self, x, state)
    B, T, d = x.shape
    if T == 0:
        raise ValueError("at least one input event is required")
    x_norm = self.norm(x)
    q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
    k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
    v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
    S = (state[:, :self.matrix_size].reshape(B, self.heads, self.dk, self.dv)
         if self.conv_kernel else state)
    if self.conv_kernel:
        history = state[:, self.matrix_size:].reshape(B, 3*d, self.conv_kernel-1)
        projected = torch.cat((q.reshape(B, T, d), k.reshape(B, T, d),
                               v.reshape(B, T, d)), dim=-1).transpose(1, 2)
        mixed, next_history = self._mix_qkv(projected, history)
        q, k, v = [part.reshape(B, T, self.heads, self.dk)
                   for part in mixed.transpose(1, 2).split(d, dim=-1)]
    g = self.g_proj(x_norm)
    alpha = self.retention(x_norm)
    beta = torch.sigmoid(self.beta_proj(x_norm))
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    feedback = self.recurrence == "feedback_delta"
    if feedback:
        strength = self.feedback_logit.sigmoid()
        feedback_gain = F.softplus(self.feedback_gain_raw)
    else:
        strength = feedback_gain = torch.zeros(self.heads, device=x.device, dtype=x.dtype)
    out, S, esq, corr = FusedDelta.apply(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        alpha.transpose(1, 2), beta.transpose(1, 2), strength, feedback_gain, S,
        self.content_budget, feedback)
    out = out.transpose(1, 2).reshape(B, T, d)
    result = x + self.o_proj(out * F.silu(g))
    if self.mlp is not None:
        normalized = self.mlp_norm(result)
        result = result + self.mlp["down"](
            F.silu(self.mlp["gate"](normalized)) * self.mlp["up"](normalized))
    diagnostics = {
        "alpha_mean": alpha.detach().mean(),
        "beta_mean": beta.detach().mean(),
        "error_norm": (esq.mean() / self.dv).sqrt(),
        "state_norm": S.detach().square().mean().sqrt(),
    }
    if self.content_budget:
        diagnostics.update(
            budget_correction_rms=(corr.sum() / (B * self.matrix_size)).sqrt(),
            content_mass_max=torch.maximum(S.detach().abs().sum(-1).max(),
                                           S.detach().abs().sum(-2).max()))
    if feedback:
        diagnostics.update(feedback_strength_mean=strength.detach().mean(),
                           feedback_gain_mean=feedback_gain.detach().mean())
    if self.conv_kernel:
        S = torch.cat([S.reshape(B, self.matrix_size),
                       next_history.reshape(B, self.history_size)], dim=-1)
    return result, S, diagnostics


def _layer_classes():
    for name in ("scripts.ib_local.gated_deltanet", "gated_deltanet"):
        module = sys.modules.get(name)
        if module is not None:
            yield module.GatedDeltaNetLayer


def enable() -> str:
    """Route the fused recurrences of every loaded GatedDeltaNetLayer through the kernels."""
    for cls in _layer_classes():
        cls.forward = fused_forward
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def disable() -> None:
    for cls in _layer_classes():
        cls.forward = _reference_forward
