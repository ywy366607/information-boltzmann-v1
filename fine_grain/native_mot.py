"""Native multimodal MoT stack (product-spec).

Per layer maintains:
  X_l ∈ R^{N×d_x}     full-res visual point field (mini: d_x=d recommended)
  S_l ∈ R^{M×d}        temporary visual slices — *layer-local* params (not shared)
  H_l ∈ R^{T×d}        language state

Block (spec):
  1) Modal-independent Q/K/V (+ separate RMSNorm)
  2) Shared attention *space*: K=[Kv;Kt], V=[Vv;Vt]; Av=softmax(Qv K^T)V, At=...
  3) Modal-independent Wo + FFN experts
  4) Field update (P0 residual cell):
       S = SliceRead(X)
       S', H' = MoTBlock(S, H)
       S+ = S + g_j · η · O(S'−S)
       H+ = H + g_G · (H'−H)
       X+ = X + π_X · Deslice(S+−S) + π_X · g_G · LocalResidual
       g_G = pool_LSE/top-k(U_j)  (not mean; 1px must not dilute)

Vision point field and language state advance in parallel; Slice is ephemeral.

Ablation knobs on SliceRead / Deslice (default all OFF for clean baseline):
  - use_ada_temp: Transolver++ adaptive temperature (else fixed T=1)
  - use_gumbel:   Transolver++ Gumbel noise on assignment logits (train only)
  - deslice_topk: repo sparse write (0 = full soft scatter)
  - use_stiefel:  repo Newton–Schulz Stiefel on slice directions after mass-norm
  - use_null_slice: softmax over M+1, last dim is ∅ (not explained). Content
    masses need not sum to 1. Deslice preserves leftover (no row-renorm).
  - use_yield_read: Yield-GDN analog on Read. After softmax, ReLU(w−τ_h)
    per head, no renormalize. Uniform 1/M sits in the dead zone (∅).
  - use_ticket_read: optical-flow tickets. S0 is pred_n = f(H, x_n, y_n),
    not a broadcast language vector. s_n = ||X_n − pred_n||²; s ≤ τ → ∅.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.bayesian_surprise import BayesianSurpriseGate, global_gate_from_surprise
from fine_grain.hard_admit import YieldGate
from fine_grain.write_yield import retain_and_write, yield_residual
from fine_grain.saccade import residual_pixel_mass, residual_rel
from fine_grain.frontends import FrontendOut
from fine_grain.mm_projector import MMProjector
from fine_grain.flow_match import AdaLNZero, TimeCondition, TimestepEmbedder
from fine_grain.forward_optim import ForwardStateOpt, normalize_kind
from fine_grain.active_gdn2 import ActiveInferenceGDN2, ActiveInferenceState
from fine_grain.models import (
    NS_A,
    NS_B,
    NS_C,
    NS_EPS,
    NS_STEPS_DEFAULT,
    coords,
    newton_schulz,
    sparse_deslice_weights,
)


# --------------------------------------------------------------------------- norms / FFN


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class SwiGLUFFN(nn.Module):
    """Modality-private FFN (SwiGLU)."""

    def __init__(self, dim: int, hidden_mult: float = 2.67):
        super().__init__()
        hidden = int(dim * hidden_mult)
        # round to multiple of 64
        hidden = max(64, (hidden + 63) // 64 * 64)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------- SliceRead / Deslice / LocalVisual


class SliceRead(nn.Module):
    """X [B,N,d_x] → S [B,M,d], assignment w [B,N,M] for deslice.

    Soft mass-normalized pool (Transolver-style); S is *temporary* layer state.

    Knobs (default OFF = clean / prereg primary spirit):
      use_ada_temp — Transolver++ proj_temperature; else fixed T=1
      use_gumbel   — Transolver++ Gumbel on logits in train
      use_stiefel  — repo NS Stiefel on mass-normalized slice directions
      use_null_slice — softmax over M content + 1 sink ∅. If point_pe is
        given, ℓ_∅ = τ − γ e_n (PE-modulated refuse). Else a Linear on the
        point. Content w need not sum to 1; S mass-norms over content.
      use_yield_read — per-head dead zone on assignment: ReLU(w−τ_h),
        τ_h=softplus(θ_h). Uniform 1/M does not enter any slice.
      point_u — lagged surprise tickets (BLT pack). Not a new entropy model.
    """

    def __init__(
        self,
        d_x: int,
        d: int,
        n_slices: int,
        n_heads: int = 4,
        use_ada_temp: bool = False,
        use_gumbel: bool = False,
        use_stiefel: bool = False,
        use_null_slice: bool = False,
        use_yield_read: bool = False,
    ):
        super().__init__()
        self.M = int(n_slices)
        self.d_x = d_x
        self.d = d
        self.h = n_heads
        self.use_ada_temp = bool(use_ada_temp)
        self.use_gumbel = bool(use_gumbel)
        self.use_stiefel = bool(use_stiefel)
        self.use_null_slice = bool(use_null_slice)
        self.use_yield_read = bool(use_yield_read)
        self.proj_in = nn.Linear(d_x, d)
        # assignment head dim
        self.dh = max(16, d // n_heads)
        use = n_heads * self.dh
        self.use = use
        self.to_head = nn.Linear(d, use) if use != d else nn.Identity()
        self.to_logits = nn.Linear(self.dh, self.M)
        nn.init.orthogonal_(self.to_logits.weight)
        # Extra logit is a function of the point (shared sink bias is not enough:
        # background must be allowed to score high on ∅ independently).
        if self.use_null_slice:
            self.to_null = nn.Linear(self.dh, 1)
            nn.init.zeros_(self.to_null.weight)
            nn.init.zeros_(self.to_null.bias)
            # ℓ_∅ = τ − γ e. τ>0 so r=0 (content logits ~0) prefers ∅.
            self.pe_null_tau = nn.Parameter(torch.tensor(2.0))
            self.pe_null_gamma_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.to_null = None
            self.pe_null_tau = None
            self.pe_null_gamma_raw = None
        # Per-head yield τ_h = softplus(θ_h). Init small so rows start open
        # (grads flow). Do not pin to 1/M: that sits on the ReLU kink of a
        # uniform softmax and freezes θ_h.
        if self.use_yield_read:
            self.yield_raw = nn.Parameter(torch.full((1, n_heads, 1, 1), -6.0))
        else:
            self.yield_raw = None
        # Ada-Temp head always constructed (params exist) but only used if flag on
        self.temp = nn.Sequential(
            nn.Linear(self.dh, self.M), nn.GELU(), nn.Linear(self.M, 1), nn.GELU(),
        )
        self.bias = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 0.5)
        self.out = nn.Linear(d, d)
        self.ns_steps = NS_STEPS_DEFAULT
        self.ns_coefficients = (NS_A, NS_B, NS_C)
        self.ns_eps = NS_EPS
        self.last_null = None
        self.last_pack_alpha = None
        self.last_admit_alpha = None

    def forward(
        self,
        x: torch.Tensor,
        w_override: Optional[torch.Tensor] = None,
        pixel_mass: Optional[torch.Tensor] = None,
        point_u: Optional[torch.Tensor] = None,
        point_admit: Optional[torch.Tensor] = None,
        admit_tau: Optional[torch.Tensor] = None,
        point_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        h, dh, M = self.h, self.dh, self.M
        xp = self.proj_in(x)  # [B,N,d]
        self.last_null = None
        self.last_pack_alpha = None
        self.last_admit_alpha = None
        if w_override is not None:
            w_pts = w_override.to(dtype=xp.dtype, device=xp.device)
            if w_pts.shape != (B, N, M):
                raise ValueError(f"w_override {tuple(w_pts.shape)} != {(B, N, M)}")
            if self.use_null_slice:
                # Leftover 1−∑w is ∅. Do not force a partition of unity.
                self.last_null = (1.0 - w_pts.sum(dim=-1)).clamp_min(0.0).detach()
            else:
                w_pts = w_pts / w_pts.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        else:
            xm = self.to_head(xp).reshape(B, N, h, dh).permute(0, 2, 1, 3)
            logits = self.to_logits(xm)  # B,h,N,M
            if self.use_null_slice:
                if point_pe is not None:
                    e = point_pe.reshape(B, 1, N, 1).to(dtype=logits.dtype, device=logits.device)
                    gamma = F.softplus(self.pe_null_gamma_raw)
                    ell_null = self.pe_null_tau.to(dtype=logits.dtype) - gamma * e
                    logits = torch.cat([logits, ell_null.expand(B, h, N, 1)], dim=-1)
                else:
                    logits = torch.cat([logits, self.to_null(xm)], dim=-1)
            if self.use_ada_temp:
                temp = torch.clamp(self.temp(xm) + self.bias, min=0.01)
            else:
                temp = torch.ones((), device=x.device, dtype=x.dtype)
            # Transolver++: Gumbel noise on assignment logits during training
            if self.training and self.use_gumbel:
                u = torch.rand_like(logits)
                logits = logits - torch.log(-torch.log(u + 1e-8) + 1e-8)
            w = F.softmax(logits / temp, dim=-1)
            if self.use_null_slice:
                self.last_null = w[..., -1].mean(dim=1).detach()  # B,N
                w = w[..., :-1]  # content; rows sum to 1 − p_∅
            if self.use_yield_read and self.yield_raw is not None:
                # Do not renormalize: leftover is ∅.
                w = F.relu(w - F.softplus(self.yield_raw))
            w_pts = w.mean(dim=1)  # B,N,M  (assignment for deslice)
            if self.use_yield_read:
                self.last_null = (1.0 - w_pts.sum(dim=-1)).clamp_min(0.0).detach()
        if point_u is not None:
            w_pts, alpha = surprise_pack_weights(w_pts, point_u)
            self.last_pack_alpha = alpha.detach()
        if point_admit is not None:
            tau = admit_tau if admit_tau is not None else point_admit.new_zeros(())
            w_pts, a2 = admit_read_weights(w_pts, point_admit, tau)
            self.last_admit_alpha = a2.detach()
            self.last_null = (1.0 - w_pts.sum(dim=-1)).clamp_min(0.0).detach()
        pool_w = w_pts
        if pixel_mass is not None:
            pm = pixel_mass.to(dtype=xp.dtype, device=xp.device).reshape(B, N, 1)
            pool_w = w_pts * pm.clamp_min(0.0)
        # mass-norm pool of full d features
        mass = pool_w.sum(1).clamp_min(1e-5).unsqueeze(-1)  # B,M,1
        S = torch.einsum("bnm,bnd->bmd", pool_w, xp) / mass
        # Stiefel on directions of S [B,M,d] → treat as [B,d,M] (columns = slices)
        if self.use_stiefel:
            # S: [B,M,d] → Xns [B,d,M]; NS then re-apply per-slice magnitude
            Xns = S.transpose(1, 2)  # [B,d,M]
            mag = Xns.norm(dim=1, keepdim=True).clamp_min(1e-6)  # [B,1,M]
            U = newton_schulz(
                Xns,
                steps=self.ns_steps,
                coefficients=self.ns_coefficients,
                eps=self.ns_eps,
            )
            S = (U * mag).transpose(1, 2)  # [B,M,d]
        S = self.out(S)
        return S, w_pts


def surprise_pack_weights(
    w_pts: torch.Tensor,
    point_u: torch.Tensor,
    min_cv: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BLT pack using existing surprise. w [B,N,M], u [B,N] (detached).

    Low u → slice 0 (one long patch, still in MoT). High u → slices 1..M-1.
    Spatially flat u (collapsed scatter) leaves w unchanged.
    Null leftover is kept: we only re-split the mass that already entered content.
    """
    B, N, M = w_pts.shape
    if M < 2:
        z = w_pts.new_zeros(B, N)
        return w_pts, z
    u = point_u.detach().reshape(B, N).to(device=w_pts.device, dtype=w_pts.dtype)
    u = u.clamp_min(0.0)
    mean = u.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    cv = u.std(dim=-1, keepdim=True) / mean
    g = u / mean
    alpha = g / (g + 1.0)
    row = w_pts.sum(dim=-1, keepdim=True).clamp_min(0.0)
    detail = w_pts[..., 1:]
    detail = detail / detail.sum(dim=-1, keepdim=True).clamp_min(1e-5)
    packed = w_pts.new_zeros(w_pts.shape)
    a = alpha.unsqueeze(-1)
    packed[..., :1] = (1.0 - a) * row
    packed[..., 1:] = a * row * detail
    use = cv > min_cv
    w_out = torch.where(use.unsqueeze(-1), packed, w_pts)
    alpha_out = torch.where(use, alpha, torch.zeros_like(alpha))
    return w_out, alpha_out


def field_rarity(x: torch.Tensor) -> torch.Tensor:
    """Per-point contrast vs the field mean. [B,N], mean ≈ 1.

    This is the BLT-style 'is this location like the rest?' score on the
    live point field (includes xy). Detach: tickets are not a gameable loss.
    """
    xd = x.detach()
    mu = xd.mean(dim=1, keepdim=True)
    r = (xd - mu).pow(2).mean(dim=-1)
    return r / r.mean(dim=-1, keepdim=True).clamp_min(1e-6)


def flow_residual(
    X: torch.Tensor,
    pred: torch.Tensor,
) -> torch.Tensor:
    """Optical-flow analog: Δ = X − pred, per-point MSE. [B,N]."""
    return (X.detach() - pred.detach()).pow(2).mean(dim=-1)


LANG_S0_FREQ = 4  # Fourier bands on (y,x); matches slice_four in models.py


def xy_features(xy: torch.Tensor, n_freq: int = LANG_S0_FREQ) -> torch.Tensor:
    """Raw (y,x) plus sin/cos(2^k π xy). [..., 2] → [..., 2+4n].

    Stem already cats raw xy. Fourier lets S0 paint 1px structure that a
    linear (y,x) map cannot.
    """
    if n_freq <= 0:
        return xy
    f = (2.0 ** torch.arange(n_freq, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy.unsqueeze(-1) * f
    return torch.cat([xy, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)


def admit_participation(s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """s ≤ τ → 0 (invisible). Optical-flow dead zone, not relative-to-mean."""
    score = s.detach().reshape(s.shape[0], -1).clamp_min(0.0)
    t = tau.reshape(()).to(device=score.device, dtype=score.dtype)
    return F.relu(score - t)


def admit_read_weights(
    w_pts: torch.Tensor,
    s: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero content mass where prediction error ≤ τ. No row-renorm."""
    gate = admit_participation(s, tau).to(device=w_pts.device, dtype=w_pts.dtype)
    return w_pts * gate.unsqueeze(-1), gate


def next_admit_score(
    X: torch.Tensor,
    U_x: Optional[torch.Tensor] = None,
    X_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Δ vs language-prior field if present, else field rarity (first look)."""
    if X_prior is not None:
        return flow_residual(X, X_prior)
    return field_rarity(X)


def next_read_tickets(
    pixel_mass: Optional[torch.Tensor],
    U_x: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Tickets for the next SliceRead: point residual × relative Bayes U.

    residual_pixel_mass is already pointwise (1px vs blend). Scatter(U, w) is
    flat while Read is collapsed; multiplying by a flat field is a no-op, so
    the first non-trivial packing comes from the residual we already compute.
    """
    if pixel_mass is None and U_x is None:
        return None
    if pixel_mass is None:
        return U_x.detach()
    pm = pixel_mass.detach()
    if U_x is None:
        return pm
    ux = U_x.detach().reshape_as(pm)
    rel = ux / ux.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return pm * rel


def slice_mass_loss_weights(w: torch.Tensor) -> torch.Tensor:
    """Supervision measure dual to SliceRead mass-norm. w [B,N,M] → π [B,N,1].

    Each slice casts one vote; points share it as w_nj / mass_j. A slice that
    locked onto a few points (thin structure, distant predator) makes those
    points loud in the loss. Uniform assignment ⇒ π = 1/N (same as area mean).
    No stroke / class mask — any content-adaptive assignment works.
    Detach w: the measure is the current partition, not a gameable weight.
    With null-slice Read, leftover mass is ∅ so π≈0 on unexplained points.
    """
    wd = w.detach()
    mass = wd.sum(dim=1, keepdim=True).clamp_min(1e-5)  # [B,1,M]
    hat = wd / mass
    return hat.mean(dim=-1, keepdim=True)


class DesliceWrite(nn.Module):
    r"""S [B,M,d] -> residual field delta [B,N,d_x] via assignment write.

    The shipped layer passes the increment (S+ - S). write_delta is
    weight-only so a zero increment writes exactly 0 (a proj bias would
    break S+=S => X+=X).

    deslice_topk=0: full soft scatter (clean). topk>0: repo sparse write.
    preserve_mass: keep leftover (null-slice). Do not renormalize a 1%
    content assignment up to 1 and paint the sink.
    """

    def __init__(
        self, d: int, d_x: int, deslice_topk: int = 0, beta: float = 1.0,
        preserve_mass: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(d, d_x)
        self.deslice_topk = int(deslice_topk)
        self.beta = float(beta)
        self.preserve_mass = bool(preserve_mass)

    def _write_w(self, w_pts: torch.Tensor) -> torch.Tensor:
        return sparse_deslice_weights(
            w_pts.unsqueeze(1), topk=self.deslice_topk,
            renorm=not self.preserve_mass,
        ).squeeze(1)

    def write_delta(self, S: torch.Tensor, w_pts: torch.Tensor) -> torch.Tensor:
        # S is an increment: 0 → 0. Bias is intentionally unused on the write.
        delta_s = F.linear(S, self.proj.weight, None)
        return self.beta * torch.einsum("bnm,bmd->bnd", self._write_w(w_pts), delta_s)

    def forward(
        self, S: torch.Tensor, w_pts: torch.Tensor, X: torch.Tensor,
    ) -> torch.Tensor:
        return X + self.write_delta(S, w_pts)

    def scatter_to_points(self, val: torch.Tensor, w_pts: torch.Tensor) -> torch.Tensor:
        """Deslice a per-slice scalar onto points with the *write* weights.

        val [B,M] or [B,M,1] → [B,N]. Same sparse top-k as ΔX. Slices stay
        ephemeral; uncertainty of the percept lives on the point field.
        """
        v = val.reshape(val.shape[0], -1)
        w_write = self._write_w(w_pts)
        return torch.einsum("bnm,bm->bn", w_write, v.to(dtype=w_write.dtype))


class LocalVisual(nn.Module):
    """Point-field local mix after Deslice.

    kind:
      "dw3"  — residual DWConv3×3 + 1×1 (default)
      "none" — identity (ablation: no local convolution)
      "cnx7" — ConvNeXt-ish residual: DWConv7 + LN + PW expand/contract
    """

    def __init__(self, d_x: int, res: int, kind: str = "dw3"):
        super().__init__()
        self.res = res
        self.kind = str(kind).lower()
        if self.kind in ("none", "off", "identity", "0", "false"):
            self.kind = "none"
            self.body = None
        elif self.kind in ("dw3", "default", "conv", "true", "1"):
            self.kind = "dw3"
            self.norm = RMSNorm(d_x)
            self.dw = nn.Conv2d(d_x, d_x, 3, padding=1, groups=d_x, bias=False)
            self.pw = nn.Conv2d(d_x, d_x, 1)
            self.body = "dw3"
        elif self.kind in ("cnx7", "convnext", "cnx"):
            self.kind = "cnx7"
            # channels-last LN via LayerNorm on last dim after reshape to BHWC
            self.dw = nn.Conv2d(d_x, d_x, 7, padding=3, groups=d_x, bias=False)
            self.norm = nn.LayerNorm(d_x)
            hidden = int(4 * d_x)
            self.pw1 = nn.Linear(d_x, hidden)
            self.pw2 = nn.Linear(hidden, d_x)
            self.body = "cnx7"
        else:
            raise ValueError(f"unknown LocalVisual kind={kind!r}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return X
        B, N, C = X.shape
        R = self.res
        assert N == R * R, (N, R)
        if self.kind == "dw3":
            x = self.norm(X)
            g = x.transpose(1, 2).reshape(B, C, R, R)
            g = self.pw(self.dw(g))
            return X + g.flatten(2).transpose(1, 2)
        # cnx7
        g = X.transpose(1, 2).reshape(B, C, R, R)
        g = self.dw(g)
        # B,C,H,W -> B,H,W,C for LN / MLP
        y = g.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pw2(F.gelu(self.pw1(y)))
        y = y.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        return X + y


# --------------------------------------------------------------------------- Patch embed / unpatch (on point field X)


class PointPatchEmbed(nn.Module):
    """Non-overlap patch embed on point field X [B,N,d_x] → P [B,N_p,d]."""

    def __init__(self, d_x: int, d: int, res: int, patch_size: int = 4):
        super().__init__()
        p = int(patch_size)
        assert res % p == 0, (res, p)
        self.res = int(res)
        self.p = p
        self.gh = self.res // p
        self.n_p = self.gh * self.gh
        self.proj = nn.Linear(d_x * p * p, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_p, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, C = X.shape
        R, p = self.res, self.p
        assert N == R * R, (N, R)
        x = X.transpose(1, 2).reshape(B, C, R, R)
        patches = F.unfold(x, kernel_size=p, stride=p)  # B, C*p*p, N_p
        patches = patches.transpose(1, 2)
        return self.proj(patches) + self.pos


class PointUnpatch(nn.Module):
    """Unpatch P [B,N_p,d] → field delta [B,N,d_x] (fold, same structure as deslice scatter)."""

    def __init__(self, d: int, d_x: int, res: int, patch_size: int = 4):
        super().__init__()
        p = int(patch_size)
        assert res % p == 0, (res, p)
        self.res = int(res)
        self.p = p
        self.proj = nn.Linear(d, d_x * p * p)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        B, Np, _ = P.shape
        R, p = self.res, self.p
        y = self.proj(P).transpose(1, 2)  # B, d_x*p*p, N_p
        x = F.fold(y, output_size=(R, R), kernel_size=p, stride=p)  # B,d_x,R,R
        return x.flatten(2).transpose(1, 2)


# --------------------------------------------------------------------------- MoT block (spec 1–3)


class NativeMoTBlock(nn.Module):
    """Cross-modal attention with *separate* vision/language experts.

    Shared: only the concatenated K/V attention space (not parameters).
    Vision sequence may be S only, or concat[P; S] for dual patch+slice.
    """

    def __init__(
        self,
        d: int = 512,
        n_heads: int = 8,
        ffn_mult: float = 2.67,
        residual_scale_init: float = 1.0,
    ):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.dh = d // n_heads
        scale = self.dh ** -0.5
        self.scale = scale

        # --- vision projections (private) ---
        self.rms_v = RMSNorm(d)
        self.Wq_v = nn.Linear(d, d, bias=False)
        self.Wk_v = nn.Linear(d, d, bias=False)
        self.Wv_v = nn.Linear(d, d, bias=False)
        self.Wo_v = nn.Linear(d, d, bias=False)
        self.rms_v_ffn = RMSNorm(d)
        self.ffn_v = SwiGLUFFN(d, ffn_mult)
        self.res_v = nn.Parameter(torch.tensor(float(residual_scale_init)))

        # --- language projections (private, NOT shared with vision) ---
        self.rms_t = RMSNorm(d)
        self.Wq_t = nn.Linear(d, d, bias=False)
        self.Wk_t = nn.Linear(d, d, bias=False)
        self.Wv_t = nn.Linear(d, d, bias=False)
        self.Wo_t = nn.Linear(d, d, bias=False)
        self.rms_t_ffn = RMSNorm(d)
        self.ffn_t = SwiGLUFFN(d, ffn_mult)
        self.res_t = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [B,L,d] -> [B,h,L,dh]
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B,h,L,dh] -> [B,L,d]
        B, h, L, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * dh)

    def forward(
        self,
        S: torch.Tensor,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        P: Optional[torch.Tensor] = None,
        visual_attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        S: [B,M,d], H: [B,T,d], optional P: [B,N_p,d]
        If P is set, visual sequence is concat[P; S] (shared vision expert).
        Returns (S_out, H_out, P_out|None).
        """
        B, M, d = S.shape
        T = H.shape[1]
        assert H.shape[0] == B and H.shape[2] == d

        if P is None:
            V = S
            n_p = 0
        else:
            assert P.shape[0] == B and P.shape[2] == d
            n_p = P.shape[1]
            V = torch.cat([P, S], dim=1)  # [B, N_p+M, d]
        L_v = V.shape[1]

        # 1) modal-independent QKV
        Vn, Hv = self.rms_v(V), self.rms_t(H)
        Qv = self._shape(self.Wq_v(Vn))
        Kv = self._shape(self.Wk_v(Vn))
        Vv = self._shape(self.Wv_v(Vn))
        Qt = self._shape(self.Wq_t(Hv))
        Kt = self._shape(self.Wk_t(Hv))
        Vt = self._shape(self.Wv_t(Hv))

        # 2) shared attention space: concat K/V  [visual; text]
        K = torch.cat([Kv, Kt], dim=2)  # [B,h,L_v+T,dh]
        Val = torch.cat([Vv, Vt], dim=2)

        if text_mask is None:
            text_mask = torch.ones(B, T, device=S.device, dtype=torch.bool)
        else:
            text_mask = text_mask.bool()
        if prompt_mask is None:
            prompt_mask = text_mask
        else:
            prompt_mask = prompt_mask.bool() & text_mask

        neg = torch.finfo(S.dtype).min
        # --- visual queries: all visual keys + prompt text keys only ---
        av = torch.matmul(Qv, K.transpose(-2, -1)) * self.scale
        if visual_attn_bias is not None:
            vb = visual_attn_bias.to(device=av.device, dtype=av.dtype)
            if vb.shape[:2] != (B, self.h):
                raise ValueError("visual_attn_bias must start with [B,n_heads]")
            if vb.shape[-2:] == (M, M):
                av[:, :, n_p:, n_p:L_v] = av[:, :, n_p:, n_p:L_v] + vb
            elif vb.shape[-2:] == (L_v, L_v):
                av[:, :, :, :L_v] = av[:, :, :, :L_v] + vb
            else:
                raise ValueError("visual_attn_bias has incompatible token axes")
        vis_key_ok = torch.cat([
            torch.ones(B, L_v, device=S.device, dtype=torch.bool),
            prompt_mask,
        ], dim=1)
        av = av.masked_fill(~vis_key_ok[:, None, None, :], neg)
        av = torch.softmax(av, dim=-1)
        Av = self._merge(torch.matmul(av, Val))
        # Per-visual-token mass on text keys. Slice-selective language uses this.
        self.last_text_mass = av[:, :, :, L_v:].sum(dim=-1).mean(dim=1)  # [B, L_v]
        # Mean visual-head mass on each text token. Diagnose language routing.
        self.last_text_token_mass = av[:, :, :, L_v:].mean(dim=(1, 2))  # [B, T]

        # --- language queries: all visual + causal text ---
        at = torch.matmul(Qt, K.transpose(-2, -1)) * self.scale
        txt_key_ok = torch.cat([
            torch.ones(B, L_v, device=S.device, dtype=torch.bool),
            text_mask,
        ], dim=1)
        at = at.masked_fill(~txt_key_ok[:, None, None, :], neg)
        i = torch.arange(T, device=S.device)[:, None]
        j = torch.arange(T, device=S.device)[None, :]
        causal_tt = j <= i
        allow = torch.ones(T, L_v + T, device=S.device, dtype=torch.bool)
        allow[:, L_v:] = causal_tt
        at = at.masked_fill(~allow[None, None, :, :], neg)
        at = torch.softmax(at, dim=-1)
        At = self._merge(torch.matmul(at, Val))

        # 3) modal-independent output experts
        V_bar = V + self.res_v * self.Wo_v(Av)
        H_bar = H + self.res_t * self.Wo_t(At)
        V_out = V_bar + self.res_v * self.ffn_v(self.rms_v_ffn(V_bar))
        H_out = H_bar + self.res_t * self.ffn_t(self.rms_t_ffn(H_bar))
        H_out = torch.where(text_mask.unsqueeze(-1), H_out, H)

        if n_p == 0:
            return V_out, H_out, None
        P_out, S_out = V_out[:, :n_p], V_out[:, n_p:]
        return S_out, H_out, P_out


# --------------------------------------------------------------------------- full layer + stack


def coerce_pi_x(pi_x, X: torch.Tensor) -> torch.Tensor:
    """Broadcast a write-permission flag to [B, 1, 1] on X's device/dtype.

    None / 1 → write the field. 0 → freeze X (read-only ports).
    A 1-d tensor is treated as a per-sample flag.
    """
    B = X.shape[0]
    if pi_x is None:
        return X.new_ones(B, 1, 1)
    if not torch.is_tensor(pi_x):
        return X.new_full((B, 1, 1), float(pi_x))
    t = pi_x.to(device=X.device, dtype=X.dtype)
    if t.ndim == 0:
        return t.view(1, 1, 1).expand(B, 1, 1)
    if t.ndim == 1:
        return t.reshape(-1, 1, 1)
    return t


def need_pix_to_pi_x(need_pix, ref: torch.Tensor) -> torch.Tensor:
    """Legacy output-to-write adapter for historical ablations.

    Canonical unified ports must not use this: output selection (``need_pix``)
    is distinct from latent-field evidence clamping (``pi_x``).
    """
    if torch.is_tensor(need_pix):
        vals = need_pix.detach().to(device=ref.device, dtype=ref.dtype).reshape(-1)
    else:
        vals = ref.new_tensor([1.0 if bool(x) else 0.0 for x in list(need_pix)])
    return vals.reshape(-1, 1, 1)


@dataclass
class NativeLayerTrace:
    layer: int
    x_delta: float
    h_delta: float
    M: int
    T: int
    surprise_u: float = 0.0
    surprise_gate: float = 1.0
    g_global: float = 1.0
    vfe_F: float = 0.0
    vfe_gap: float = 0.0
    rms_s: float = 0.0
    rms_delta: float = 0.0
    rms_ratio: float = 0.0
    kalman_k: float = 0.0
    lang_frac: float = 1.0


class NativeMoTLayer(nn.Module):
    """One full layer: SliceRead → MoTBlock → Deslice → LocalVisual.

    dual_patch: also PatchEmbed(X)→P into MoT with S; optional Unpatch(P') on X.
    Params are *per layer* (not shared across depth).
    """

    def __init__(
        self,
        d_x: int = 128,
        d: int = 512,
        n_slices: int = 64,
        n_heads: int = 8,
        res: int = 64,
        deslice_topk: int = 0,
        use_ada_temp: bool = False,
        use_gumbel: bool = False,
        use_stiefel: bool = False,
        local_kind: str = "dw3",
        dual_patch: bool = False,
        patch_size: int = 4,
        use_unpatch: bool = True,
        surprise_mode: str = "baseline",
        surprise_beta: float = 1.0,
        surprise_detach: bool = True,
        detach_pred_target: bool = True,
        s_update: str = "raw",
        interact_prenorm: bool = False,
        trust_rho: float = 0.1,
        eta_init: float = 1.0,
        evidence_decay: float = 0.0,
        sigma_r: float = 1.0,
        gate_on: str = "u",
        deslice_write: str = "absolute",
        gate_h_local: bool = False,
        saccade: bool = False,
        saccade_gain: float = 1.0,
        s_kalman_update: bool = False,
        s_lang_topk: int = 0,
        prior_write: float = 0.0,
        prior_write_by_t: bool = True,
        use_null_slice: bool = False,
        pack_by_surprise: bool = False,
        hard_admit: bool = False,
        use_yield_read: bool = False,
        use_ticket_read: bool = False,
        use_write_yield: bool = False,
        write_alpha: float = 1.0,
        use_residual_read: bool = False,
        use_action_rel_bias: bool = False,
        use_action_transport: bool = False,
        use_action_slice_transition: bool = False,
    ):
        super().__init__()
        self.dual_patch = bool(dual_patch)
        self.use_unpatch = bool(use_unpatch)
        self.patch_size = int(patch_size)
        self.surprise_mode = str(surprise_mode)
        self.surprise_beta = float(surprise_beta)
        self.surprise_detach = bool(surprise_detach)
        kind = normalize_kind(s_update)
        self.s_update = kind
        self.interact_prenorm = bool(interact_prenorm)
        self.trust_rho = float(trust_rho)
        self.fwd_opt = ForwardStateOpt(
            d, kind=kind, eta_init=eta_init, trust_rho=self.trust_rho,
            evidence_decay=float(evidence_decay),
        )
        self.eta = self.fwd_opt.eta
        self.s_in_norm = RMSNorm(d)
        self.h_in_norm = RMSNorm(d)
        self.use_residual_read = bool(use_residual_read)
        self.use_action_rel_bias = bool(use_action_rel_bias)
        self.use_action_transport = bool(use_action_transport)
        self.use_action_slice_transition = bool(use_action_slice_transition)
        self.res = int(res)
        self.read = SliceRead(
            d_x, d, n_slices, n_heads=n_heads,
            use_ada_temp=use_ada_temp,
            use_gumbel=use_gumbel,
            use_stiefel=use_stiefel,
            use_null_slice=use_null_slice or self.use_residual_read,
            use_yield_read=use_yield_read,
        )
        self.mot = NativeMoTBlock(d=d, n_heads=n_heads)
        if self.use_action_rel_bias:
            hidden = max(16, d // 2)
            self.action_rel_mlp = nn.Sequential(
                nn.Linear(6, hidden), nn.SiLU(),
                nn.Linear(hidden, n_heads, bias=False),
            )
            nn.init.normal_(self.action_rel_mlp[-1].weight, std=0.02)
        else:
            self.action_rel_mlp = None
        if self.use_action_transport:
            # Learned displacement of Slice assignments on the fixed Eulerian
            # grid. Zero initialization leaves every existing path unchanged.
            self.action_transport = nn.Linear(2, 2, bias=False)
            nn.init.zeros_(self.action_transport.weight)
        else:
            self.action_transport = None
        if self.use_action_slice_transition:
            hidden = max(16, d // 2)
            self.action_transition_mlp = nn.Sequential(
                nn.Linear(6, hidden), nn.SiLU(),
                nn.Linear(hidden, 1, bias=False),
            )
            nn.init.zeros_(self.action_transition_mlp[-1].weight)
            self.action_transition_rate_raw = nn.Parameter(torch.tensor(-2.1972246))
        else:
            self.action_transition_mlp = None
            self.action_transition_rate_raw = None
        self.surprise_gate = BayesianSurpriseGate(
            d_model=d,
            n_slices=n_slices,
            beta=surprise_beta,
            mode=surprise_mode,
            detach_gate=surprise_detach,
            detach_pred_target=detach_pred_target,
            n_heads=n_heads,
            sigma_r=sigma_r,
            gate_on=gate_on,
        )
        self.deslice = DesliceWrite(
            d, d_x, deslice_topk=deslice_topk, beta=1.0,
            preserve_mass=use_null_slice or use_yield_read or use_ticket_read
            or bool(use_residual_read),
        )
        # dual_patch: local default none (patch stream carries spatial local bias)
        if self.dual_patch and local_kind == "dw3":
            local_kind = "none"
        self.local = LocalVisual(d_x, res=res, kind=local_kind)
        self.M = n_slices
        self.local_kind = self.local.kind
        if self.dual_patch:
            self.patch_embed = PointPatchEmbed(d_x, d, res, patch_size=self.patch_size)
            self.unpatch = PointUnpatch(d, d_x, res, patch_size=self.patch_size)
        else:
            self.patch_embed = None
            self.unpatch = None
        # Runtime knobs. Recognition default = Champion B write A, H/Local ungated.
        # increment: D(S+ − S) time velocity (P0)
        # absolute:  D(S+) belief broadcast (A)
        # workspace: D(S+ − Read(W)) consistency residual (C)
        self.deslice_write = str(deslice_write)
        self.gate_h_local = bool(gate_h_local)
        self.gate_on = str(gate_on)
        self.sigma_r = float(sigma_r)
        self.saccade = bool(saccade)
        self.saccade_gain = float(saccade_gain)
        # Update-only Kalman: S ← μ* = μp + K(S−μp). No predict A.
        self.s_kalman_update = bool(s_kalman_update)
        # 0 = all slices take MoT Δ (product default). >0 = only top-k by text attn.
        self.s_lang_topk = int(s_lang_topk)
        # Action on the canvas: Deslice(μp − S). 0 = recognition (no overwrite).
        self.prior_write = float(prior_write)
        self.prior_write_by_t = bool(prior_write_by_t)
        self.use_null_slice = bool(use_null_slice)
        self.pack_by_surprise = bool(pack_by_surprise)
        self.hard_admit = bool(hard_admit)
        self.use_yield_read = bool(use_yield_read)
        self.use_ticket_read = bool(use_ticket_read)
        self.use_write_yield = bool(use_write_yield)
        self.write_alpha = float(write_alpha)
        self.admit = YieldGate(d_x) if self.hard_admit else None
        # Admission τ, not leak. Init open (softplus(-3)≈0.05); learns; not 1/M.
        self.write_yield_raw = (
            nn.Parameter(torch.full((1, 1, d_x), -3.0))
            if self.use_write_yield else None
        )
        # S0 = f(H, xy). Points query the language sequence (not mean-pool).
        self.lang_s0_freq = LANG_S0_FREQ
        if self.use_ticket_read or self.use_residual_read:
            feat_dim = 2 + 4 * self.lang_s0_freq
            self.s0_q = nn.Linear(feat_dim, d)
            self.s0_kv = nn.Linear(d, 2 * d)
            self.lang_s0 = nn.Sequential(
                nn.Linear(d + feat_dim, d_x),
                nn.SiLU(),
                nn.Linear(d_x, d_x),
            )
        else:
            self.s0_q = None
            self.s0_kv = None
            self.lang_s0 = None
        self.ticket_raw = nn.Parameter(torch.tensor(-4.0)) if self.use_ticket_read else None
        # Open-Sora split: adaLN carries diffusion t only. Language stays in MoT.
        self.ada_x = AdaLNZero(d_x)
        self.ada_s = AdaLNZero(d)
        self.ada_write = nn.Sequential(nn.SiLU(), nn.Linear(d_x, d_x))
        nn.init.zeros_(self.ada_write[-1].weight)
        nn.init.zeros_(self.ada_write[-1].bias)
        self.cond_to_s = nn.Identity() if d == d_x else nn.Linear(d_x, d)

    @property
    def lang_proto(self):
        """Back-compat alias for the spatial S0 net."""
        return self.lang_s0

    def _transport_write_weights(
        self,
        w: torch.Tensor,
        action_vector: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Learn p(write point | read assignment, action) at fixed addresses."""
        self.last_transport_shift_yx = None
        if self.action_transport is None or action_vector is None:
            return w
        B, N, M = w.shape
        R = self.res
        if N != R * R:
            raise ValueError("action transport requires a square full-resolution field")
        action_yx = action_vector[..., [1, 0]].to(device=w.device, dtype=w.dtype)
        shift_yx = torch.tanh(self.action_transport(action_yx)) * (0.5 * R)
        strength = action_yx.norm(dim=-1).clamp(0.0, 1.0)
        if action_mask is not None:
            strength = strength * action_mask.to(device=w.device, dtype=w.dtype)

        # Backward semi-Lagrangian sampling of the assignment field. A 3x3
        # tile implements the periodic micro-world without moving X addresses.
        field = w.transpose(1, 2).reshape(B, M, R, R)
        tiled = field.repeat(1, 1, 3, 3)
        axis = torch.arange(R, device=w.device, dtype=w.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        source_y = yy[None] - shift_yx[:, 0, None, None] + R
        source_x = xx[None] - shift_yx[:, 1, None, None] + R
        denom = float(3 * R - 1)
        grid = torch.stack(
            [2.0 * source_x / denom - 1.0, 2.0 * source_y / denom - 1.0],
            dim=-1,
        )
        moved = F.grid_sample(
            tiled, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).flatten(2).transpose(1, 2)
        moved = moved / moved.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.last_transport_shift_yx = shift_yx.detach()
        return w + strength[:, None, None] * (moved - w)

    def _action_transition_prior(
        self,
        S: torch.Tensor,
        w: torch.Tensor,
        action_vector: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
        enabled: bool,
    ) -> torch.Tensor:
        """Apply one normalized p(S_dst | S_src, action) prior transition."""
        self.last_action_transition = None
        if (
            not enabled
            or bool(getattr(self, "disable_action_transition", False))
            or self.action_transition_mlp is None
            or action_vector is None
        ):
            return S
        B, M, _ = S.shape
        grid = coords(self.res, S.device).to(dtype=S.dtype).expand(B, -1, -1)
        mass = w.sum(dim=1).clamp_min(1e-5).unsqueeze(-1)
        centers = torch.einsum("bnm,bnd->bmd", w, grid) / mass
        # Matrix axes are [destination, source].
        rel = centers[:, :, None, :] - centers[:, None, :, :]
        action_yx = action_vector[..., [1, 0]].to(device=S.device, dtype=S.dtype)
        act = action_yx[:, None, None, :].expand(-1, M, M, -1)
        feat = torch.cat([rel, act, rel - act], dim=-1)
        learned = self.action_transition_mlp(feat).squeeze(-1)
        identity = torch.eye(M, device=S.device, dtype=S.dtype)[None]
        proposal = torch.softmax(learned, dim=1)
        strength = action_yx.norm(dim=-1).clamp(0.0, 1.0)
        if action_mask is not None:
            strength = strength * action_mask.to(device=S.device, dtype=S.dtype)
        rate = torch.sigmoid(self.action_transition_rate_raw).to(dtype=S.dtype)
        transition = identity + strength[:, None, None] * rate * (proposal - identity)
        self.last_action_transition = transition.detach()
        self.last_action_transition_rate = float(rate.detach())
        return torch.einsum("bij,bjd->bid", transition, S)

    def _apply_s_update(
        self,
        S: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        opt_state: Optional[Dict] = None,
        S_ev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        S_new, st = self.fwd_opt.step(S, delta, gate, state=opt_state, S_ev=S_ev)
        self.last_opt_state = st
        return S_new

    def _mask_lang_delta(self, delta: torch.Tensor, n_slices: int) -> torch.Tensor:
        """Zero MoT Δ on slices outside the top-k by visual→text attention mass."""
        k = int(getattr(self, "s_lang_topk", 0))
        if k <= 0:
            self.last_lang_keep = None
            self.last_lang_frac = 1.0
            return delta
        mass = getattr(self.mot, "last_text_mass", None)
        if mass is None:
            self.last_lang_keep = None
            self.last_lang_frac = 1.0
            return delta
        if mass.shape[1] != n_slices:
            mass = mass[:, -n_slices:]
        k = min(k, n_slices)
        idx = mass.topk(k, dim=-1).indices
        keep = torch.zeros(mass.shape[0], n_slices, 1, device=delta.device, dtype=delta.dtype)
        keep.scatter_(1, idx.unsqueeze(-1), 1.0)
        self.last_lang_keep = keep.detach()
        self.last_lang_frac = float(keep.mean().item())
        return delta * keep

    def _pool_h(self, H: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if text_mask is None:
            return H.mean(dim=1)
        m = text_mask.unsqueeze(-1).to(dtype=H.dtype)
        return (H * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def language_prior(
        self,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """pred_n = f(H, x_n, y_n). [B,N,d_x].

        Each point queries the language *sequence* (not a pooled vector).
        Same graph for captions, edits, and digits — no class mask.
        """
        R = int(getattr(getattr(self, "local", None), "res", 0) or round(n_points ** 0.5))
        if R * R == n_points:
            xy = coords(R, device).to(dtype=dtype)
        else:
            t = torch.linspace(-1.0, 1.0, n_points, device=device, dtype=dtype)
            xy = torch.stack([t, t], dim=-1).unsqueeze(0)
        B = H.shape[0]
        feat = xy_features(xy, n_freq=self.lang_s0_freq).expand(B, n_points, -1)
        q = self.s0_q(feat)
        k, v = self.s0_kv(H).chunk(2, dim=-1)
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        none = None
        if text_mask is not None:
            keep = text_mask.bool()
            none = ~keep.any(dim=-1)
            attn = attn.masked_fill(~keep.unsqueeze(1), torch.finfo(attn.dtype).min)
            if bool(none.any()):
                attn = attn.masked_fill(none[:, None, None], 0.0)
        h_n = torch.matmul(torch.softmax(attn, dim=-1), v)
        if none is not None and bool(none.any()):
            h_n = h_n.masked_fill(none[:, None, None], 0.0)
        return self.lang_s0(torch.cat([h_n, feat], dim=-1))

    def flow_tickets(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        X_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Δ = X − S0. S0 is pred_n = f(H, x_n, y_n), not a broadcast vector.

        X_prior is an explicit override (tests / a real previous canvas). The
        stack does not feed last Deslice back here: that write is spatially
        constant while Read is collapsed, which is the deadlock we are
        breaking.
        """
        s0 = None
        if self.lang_s0 is not None:
            # Live S0: accuracy vs real o is a separate term. Tickets detach.
            s0 = self.language_prior(H, text_mask, X.shape[1], X.device, X.dtype)
        self.last_s0 = s0
        if X_prior is not None:
            pred = X_prior
        elif s0 is not None:
            pred = s0
        else:
            pred = torch.zeros_like(X)
        self.last_lang_pred = pred.detach()
        return flow_residual(X, pred)

    def _recon_from_slices(self, S: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bnm,bmd->bnd", w,
            F.linear(S, self.deslice.proj.weight, self.deslice.proj.bias),
        )

    def _surprise_on_points(self, s_meta: Dict, w: torch.Tensor) -> Optional[torch.Tensor]:
        """Same energy the Bayes gate uses (U / gap / F), Deslice-scattered to X.

        Not ||X − recon||². Flat or missing surprise → None (do not refuse).
        """
        on = str(getattr(self, "gate_on", "u")).lower()
        if on == "gap" and s_meta.get("gap") is not None:
            val = s_meta["gap"]
        elif on in ("f", "vfe") and s_meta.get("F") is not None:
            val = s_meta["F"]
        else:
            val = s_meta.get("surprise")
        if val is None:
            return None
        s = self.deslice.scatter_to_points(val.reshape(val.shape[0], -1).detach(), w.detach())
        if float(s.abs().sum()) < 1e-8:
            return None
        return s

    def forward(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        opt_state: Optional[Dict] = None,
        X_orig: Optional[torch.Tensor] = None,
        pi_x=1.0,
        force_gate: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        pixel_mass: Optional[torch.Tensor] = None,
        point_u: Optional[torch.Tensor] = None,
        point_admit: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        action_vector: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        apply_action_transition: bool = True,
        causal_delta_s: Optional[torch.Tensor] = None,
        causal_write_w: Optional[torch.Tensor] = None,
        causal_prior_gate: Optional[torch.Tensor] = None,
        causal_delta_x: Optional[torch.Tensor] = None,
        X_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeLayerTrace]:
        if cond is not None:
            X = self.ada_x(X, cond)
        X0, H0 = X, H
        admit_tau = None
        self.last_s0 = None
        point_pe = None
        x_read = X
        # Visual writes (S0 / tickets / language prior) may only read observed
        # prompt tokens. Answer tokens stay on the causal language path.
        vis_h_mask = prompt_mask if prompt_mask is not None else text_mask
        if self.use_residual_read and self.lang_s0 is not None:
            s0 = self.language_prior(H, vis_h_mask, X.shape[1], X.device, X.dtype)
            self.last_s0 = s0
            # Detach S0 in the residual: assignment must not game the prior.
            r = X - s0.detach()
            x_read = r
            point_pe = r.pow(2).mean(dim=-1)
        elif self.use_ticket_read and point_admit is None:
            point_admit = self.flow_tickets(X, H, vis_h_mask, X_prior)
            admit_tau = F.softplus(self.ticket_raw)
        S, w = self.read(
            x_read, pixel_mass=pixel_mass, point_u=point_u,
            point_admit=point_admit, admit_tau=admit_tau,
            point_pe=point_pe,
        )
        self.last_null = getattr(self.read, "last_null", None)
        self.last_pack_alpha = getattr(self.read, "last_pack_alpha", None)
        self.last_admit_alpha = getattr(self.read, "last_admit_alpha", None)
        visual_attn_bias = None
        if self.action_rel_mlp is not None and action_vector is not None:
            grid = coords(self.res, X.device).to(dtype=X.dtype).expand(X.shape[0], -1, -1)
            mass = w.sum(dim=1).clamp_min(1e-5).unsqueeze(-1)
            centers = torch.einsum("bnm,bnd->bmd", w, grid) / mass
            rel = centers[:, :, None, :] - centers[:, None, :, :]
            # External world actions use (dx, dy), while coords()/centroids use
            # (y, x). Close the chart before forming geometric residuals.
            action_yx = action_vector[..., [1, 0]]
            act = action_yx[:, None, None, :].to(dtype=X.dtype, device=X.device)
            act_full = act.expand(-1, rel.shape[1], rel.shape[2], -1)
            feat = torch.cat([rel, act_full, rel - act_full], dim=-1)
            visual_attn_bias = self.action_rel_mlp(feat).permute(0, 3, 1, 2)
            if action_mask is not None:
                visual_attn_bias = visual_attn_bias * action_mask.reshape(-1, 1, 1, 1).to(
                    device=X.device, dtype=X.dtype,
                )
        w_write = self._transport_write_weights(w, action_vector, action_mask)
        if cond is not None:
            S = self.ada_s(S, self.cond_to_s(cond))
        if self.interact_prenorm:
            S_in, H_in = self.s_in_norm(S), self.h_in_norm(H)
        else:
            S_in, H_in = S, H
        if self.dual_patch:
            P = self.patch_embed(X)
            if self.interact_prenorm:
                P = self.s_in_norm(P)
            S2n, H2n, P2 = self.mot(
                S_in, H_in, text_mask=text_mask, prompt_mask=prompt_mask, P=P,
                visual_attn_bias=visual_attn_bias,
            )
            delta = S2n - S_in
            v_H = H2n - H_in
            P2 = (P2 - P) if P2 is not None else None
        else:
            S2n, H2n, _ = self.mot(
                S_in, H_in, text_mask=text_mask, prompt_mask=prompt_mask,
                visual_attn_bias=visual_attn_bias,
            )
            delta = S2n - S_in
            v_H = H2n - H_in
            P2 = None
        delta = self._mask_lang_delta(delta, S.shape[1])
        gate, s_meta = self.surprise_gate(
            S, H, delta_S=delta, text_mask=vis_h_mask,
        )
        u_field = s_meta.get("surprise", gate.new_zeros(gate.shape))
        if force_gate is not None:
            gate = force_gate.to(device=gate.device, dtype=gate.dtype)
            if gate.shape != (S.shape[0], S.shape[1], 1):
                gate = gate.expand(S.shape[0], S.shape[1], 1)
            # Forced brake: pool the injected gate, do not let live U reopen H/Local.
            u_field = torch.zeros_like(gate)
        g_G = global_gate_from_surprise(
            u_field, slice_gate=gate, beta=self.surprise_beta, kind="lse",
        )
        g_hl = g_G if self.gate_h_local else g_G.new_ones(g_G.shape)
        H2 = H + g_hl * v_H
        S_ev = None
        if self.fwd_opt.evidence_decay > 0.0 and X_orig is not None:
            S_ev, _ = self.read(X_orig)
        mu_star = s_meta.get("mu_star")
        if self.s_kalman_update and mu_star is not None:
            # Closed-form posterior mean replaces MoT Δ on S. H still uses MoT.
            S_write = mu_star
        else:
            S_write = self._apply_s_update(S, delta, gate, opt_state=opt_state, S_ev=S_ev)
        mu_p = s_meta.get("mu_p")
        if mu_p is None:
            mu_p = s_meta.get("s_hat")
        self.last_mu_p = mu_p
        # Schrödinger reference W^H: language prior is the drift, not a
        # parallel write. Scale by t so we only paint when assignment w
        # has structure (near data). Recognition: prior_write=0.
        pw = float(getattr(self, "prior_write", 0.0))
        if pw != 0.0 and mu_p is not None:
            gain = S.new_full((), pw)
            if t is not None and getattr(self, "prior_write_by_t", True):
                gain = (pw * t.reshape(-1, 1, 1)).to(dtype=S.dtype)
            S_write = S_write + gain * (mu_p - S)
        S_write = self._action_transition_prior(
            S_write, w, action_vector, action_mask, enabled=apply_action_transition,
        )
        write = str(getattr(self, "deslice_write", "increment")).lower()
        S_W = None
        if write in ("absolute", "abs", "s", "state"):
            # Belief broadcast: scatter proj(S_write), including bias (Champion B).
            delta_x = self.deslice(S_write, w_write, X) - X
        elif write in ("workspace", "ws", "consistency", "consist"):
            # Idempotent absolute broadcast onto workspace W, not time increment.
            # R(W) ≠ R(E+W)=S, so this is not P0. Empty W ⇒ ≈ D(S) (first stamp).
            if W is None:
                W = torch.zeros_like(X)
            S_W, _ = self.read(W)
            delta_x = self.deslice.write_delta(S_write - S_W, w_write)
        else:
            # Velocity write: scatter proj(S_write − S), bias-free so 0 → 0.
            delta_x = self.deslice.write_delta(S_write - S, w_write)
        # Physical-time prior action stays in its persistent atlas all the way
        # to X. Reusing transient/content w here would discard the address
        # identity that the causal memory was introduced to provide. This is
        # still the same bias-free Deslice projection and the same X residual.
        self.last_causal_prior = None
        self.last_causal_gate = None
        if causal_delta_s is not None:
            if causal_write_w is None:
                raise ValueError("causal_delta_s requires persistent Deslice weights")
            if causal_delta_s.shape != S.shape:
                raise ValueError(
                    f"causal_delta_s {tuple(causal_delta_s.shape)} != {tuple(S.shape)}"
                )
            if causal_write_w.shape != w.shape:
                raise ValueError(
                    f"causal_write_w {tuple(causal_write_w.shape)} != {tuple(w.shape)}"
                )
            if causal_prior_gate is None:
                cg = S.new_ones(S.shape[0], 1, 1)
            else:
                cg = torch.as_tensor(
                    causal_prior_gate, device=S.device, dtype=S.dtype,
                ).reshape(S.shape[0], 1, 1).clamp(0.0, 1.0)
            delta_x = delta_x + cg * self.deslice.write_delta(
                causal_delta_s, causal_write_w,
            )
            self.last_causal_prior = causal_delta_s.detach()
            self.last_causal_gate = cg.detach()
        if causal_delta_x is not None:
            if causal_delta_x.shape != X.shape:
                raise ValueError(
                    f"causal_delta_x {tuple(causal_delta_x.shape)} != {tuple(X.shape)}"
                )
            if causal_prior_gate is None:
                cg = X.new_ones(X.shape[0], 1, 1)
            else:
                cg = torch.as_tensor(
                    causal_prior_gate, device=X.device, dtype=X.dtype,
                ).reshape(X.shape[0], 1, 1).clamp(0.0, 1.0)
            delta_x = delta_x + cg * causal_delta_x
            self.last_causal_prior = causal_delta_x.detach()
            self.last_causal_gate = cg.detach()
        if self.dual_patch and self.use_unpatch and P2 is not None:
            delta_x = delta_x + self.unpatch(P2)
        if cond is not None:
            # Bounded residual scale. Raw Linear*Δ exploded (loss 10^4).
            # 1+tanh: identity at zero-init, |gain| ≤ 2.
            delta_x = (1.0 + torch.tanh(self.ada_write(cond)).unsqueeze(1)) * delta_x
        recon = None
        if self.saccade or self.pack_by_surprise:
            recon = self._recon_from_slices(S, w)
        self.last_m = None
        self.last_a = None
        self.last_tau = None
        self.last_admit_frac = 1.0
        yield_scale = None
        if self.hard_admit and self.admit is not None:
            s_pts = self._surprise_on_points(s_meta, w)
            if s_pts is not None:
                scale, tau, _ex = self.admit(s_pts, X0)
                delta_x = scale * delta_x
                yield_scale = scale
                self.last_m = (scale > 0).to(dtype=scale.dtype).squeeze(-1).detach()
                self.last_a = scale.detach()
                self.last_tau = tau.detach()
                self.last_admit_frac = float((scale > 0).float().mean().item())
        if self.use_write_yield and self.write_yield_raw is not None:
            tau = F.softplus(self.write_yield_raw)
            delta_x = yield_residual(delta_x, tau)
            self.last_write_tau = tau.detach()
            self.last_write_admit = float((delta_x.abs() > 0).float().mean().item())
        else:
            self.last_write_tau = None
            self.last_write_admit = 1.0
        pi = coerce_pi_x(pi_x, X)
        X1 = retain_and_write(X, delta_x, self.write_alpha, pi)
        loc_out = self.local(X1)
        loc_res = loc_out - X1
        if yield_scale is not None:
            loc_res = yield_scale * loc_res
        X2 = X1 + pi * g_hl * loc_res
        # Language-only prior canvas. Workspace wrote μp on slices; the
        # prior that seeing compares against is this field on points.
        if mu_p is not None:
            self.last_X_prior = X0 + pi * self.deslice.write_delta(mu_p - S, w_write)
        else:
            self.last_X_prior = None
        rms_s = float(S.detach().pow(2).mean().sqrt())
        rms_d = float(delta.detach().pow(2).mean().sqrt())
        self.last_rms_s = rms_s
        self.last_rms_delta = rms_d
        self.last_rms_ratio = rms_d / max(rms_s, 1e-6)
        # Store intermediate diagnostics for visualization / probe
        self.last_w = w.detach()
        self.last_w_write = w_write.detach()
        self.last_u = s_meta.get("surprise", None)
        self.last_u_mu = s_meta.get("u_mu", None)
        self.last_u_sigma = s_meta.get("u_sigma", None)
        self.last_lv_q = s_meta.get("lv_q", None)
        self.last_gate = gate.detach()
        self.last_g_G = g_G.detach()
        self.last_S = S.detach()
        self.last_S_write = S_write.detach()
        self.last_S_W = None if S_W is None else S_W.detach()
        self.last_dW = (pi * delta_x).detach()
        self.last_C = (
            float((S_write - S_W).detach().pow(2).mean().sqrt())
            if S_W is not None
            else float((S_write - S).detach().pow(2).mean().sqrt())
        )
        self.last_pi_x = pi.detach()
        self.last_X = X2.detach()
        self.last_pred_loss = s_meta.get("pred_loss")
        self.last_vfe_train_loss = s_meta.get("vfe_train_loss")
        self.last_sigreg_loss = s_meta.get("sigreg_loss")
        self.last_H_ctx = s_meta.get("h_ctx")
        self.last_F = s_meta.get("F", None)
        self.last_gap = s_meta.get("gap", None)
        self.last_F_min = s_meta.get("F_min", None)
        self.last_acc_mean = s_meta.get("acc_mean", None)
        self.last_acc_tr = s_meta.get("acc_tr", None)
        k_t = s_meta.get("K", None)
        self.last_K = None if k_t is None else k_t.detach()
        self.last_mean_K = float(s_meta.get("mean_K", 0.0))
        # Point-field (pixel) uncertainty: same Deslice write as ΔX, not soft read.
        self.last_U_x = None
        self.last_gap_x = None
        self.last_usig_x = None
        self.last_sigq_x = None
        w_det = w_write.detach()
        if s_meta.get("surprise") is not None:
            self.last_U_x = self.deslice.scatter_to_points(s_meta["surprise"].detach(), w_det)
        if s_meta.get("gap") is not None:
            self.last_gap_x = self.deslice.scatter_to_points(s_meta["gap"].detach(), w_det)
        if s_meta.get("u_sigma") is not None:
            self.last_usig_x = self.deslice.scatter_to_points(s_meta["u_sigma"].detach(), w_det)
        lv_q = s_meta.get("lv_q")
        if lv_q is not None:
            sig_m = torch.exp(0.5 * lv_q.detach()).mean(dim=-1)
            self.last_sigq_x = self.deslice.scatter_to_points(sig_m, w_det)
        self.last_pixel_mass = None
        self.last_resid_rel = None
        if self.saccade or self.pack_by_surprise:
            recon_x = recon if recon is not None else self._recon_from_slices(S, w)
            gain = self.saccade_gain if self.saccade else 1.0
            self.last_pixel_mass = residual_pixel_mass(
                X0, recon_x, gain=gain,
            )
            self.last_resid_rel = residual_rel(X0, recon_x)
            # Contrast: is any pixel much more unexplained than the mean?
            # Absolute r/||X||² stays ~1 because Deslice is not an autoencoder.
            pm = self.last_pixel_mass
            self.last_resid_focus = (
                pm.max(dim=-1).values / pm.mean(dim=-1).clamp_min(1e-6)
            )

        tr = NativeLayerTrace(
            layer=layer_idx,
            x_delta=float((X2 - X0).detach().norm(dim=-1).mean()),
            h_delta=float((H2 - H0).detach().norm(dim=-1).mean()),
            M=S.shape[1],
            T=H.shape[1],
            surprise_u=float(s_meta.get("mean_u", 0.0)),
            surprise_gate=float(s_meta.get("mean_gate", 1.0)),
            g_global=float(g_G.detach().mean()),
            vfe_F=float(s_meta.get("mean_F", 0.0)),
            vfe_gap=float(s_meta.get("mean_gap", 0.0)),
            rms_s=rms_s,
            rms_delta=rms_d,
            rms_ratio=rms_d / max(rms_s, 1e-6),
            kalman_k=float(s_meta.get("mean_K", 0.0)),
            lang_frac=float(getattr(self, "last_lang_frac", 1.0)),
        )
        return X2, H2, tr


class LTIInjection(nn.Module):
    """Parcae / OpenMythos diagonal injection: ``h ← A⊙h + B⊙e``.

    ``A = exp(-exp(log_dt + log_A))`` is in (0, 1) per channel, so ρ(A) < 1
    by construction. Used only as a mix *before* the shared Φ; Φ itself stays
    anonymous (no loop index).
    """

    def __init__(self, dim: int, b_init: float = 0.1):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * float(b_init))

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20.0, 20.0)))

    def rho(self) -> torch.Tensor:
        return self.get_A().detach().max()

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.get_A() * h + self.B * e

    def assemble(self, h: torch.Tensor, e: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Parcae residual: ``Ā h + B̄ e + Δ``. Δ must not already contain h."""
        return self.forward(h, e) + delta


class NativeMoTStack(nn.Module):
    """L native layers + stem for RGB image → X, text emb → H, readout to d_llm.

    Default: each layer owns independent SliceRead / Deslice / MoT / LocalVisual.
    share_layers=True: one NativeMoTLayer Φ looped n_loops times. After each
    residual cell, assemble X ← ĀX + B̄ e + (X_Φ−X). No loop-index embedding.
    """

    def __init__(
        self,
        d_llm: int = 640,
        res: int = 64,
        d_x: int = 128,
        d: int = 512,
        n_slices: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        deslice_topk: int = 0,
        use_ada_temp: bool = False,
        use_gumbel: bool = False,
        use_stiefel: bool = False,
        local_kind: str = "dw3",
        dual_patch: bool = False,
        patch_size: int = 4,
        use_unpatch: bool = True,
        projector: str = "mlp",
        surprise_mode: str = "baseline",
        surprise_beta: float = 1.0,
        surprise_detach: bool = True,
        detach_pred_target: bool = True,
        s_update: str = "raw",
        interact_prenorm: bool = False,
        trust_rho: float = 0.1,
        evidence_decay: float = 0.0,
        sigma_r: float = 1.0,
        gate_on: str = "u",
        deslice_write: str = "absolute",
        gate_h_local: bool = False,
        share_layers: bool = False,
        n_loops: Optional[int] = None,
        lti_inject: bool = True,
        saccade: bool = False,
        saccade_gain: float = 1.0,
        saccade_inner: int = 2,
        saccade_halt: bool = False,
        saccade_halt_eps: float = 0.05,
        saccade_halt_train: bool = False,
        s_kalman_update: bool = False,
        s_lang_topk: int = 0,
        prior_write: float = 0.0,
        prior_write_by_t: bool = True,
        use_null_slice: bool = False,
        pack_by_surprise: bool = False,
        hard_admit: bool = False,
        use_yield_read: bool = False,
        use_ticket_read: bool = False,
        use_write_yield: bool = False,
        write_alpha: float = 1.0,
        use_residual_read: bool = False,
        use_modal_precision: bool = False,
        use_target_time: bool = False,
        use_target_time_adaln: bool = False,
        use_horizon_tokens: bool = False,
        gate_action_by_horizon: bool = False,
        history_size: int = 0,
        action_dim: int = 0,
        use_action_adaln: bool = False,
        use_action_tokens: bool = False,
        use_action_rel_bias: bool = False,
        use_action_transport: bool = False,
        use_action_slice_transition: bool = False,
        use_goal_adaln: bool = False,
        use_active_gdn2: bool = False,
        use_active_gdn2_history_transport: bool = False,
        active_gdn2_initial_trust: float = 0.0,
        terminal_token_atlas: bool = False,
    ):
        super().__init__()
        self.res = res
        self.d_x = d_x
        self.d = d
        self.d_llm = d_llm
        self.n_slices = n_slices
        self.n_layers = n_layers
        self.share_layers = bool(share_layers)
        self.n_loops = int(n_layers if n_loops is None else n_loops)
        self.lti_inject = bool(lti_inject)
        self.saccade = bool(saccade)
        self.saccade_gain = float(saccade_gain)
        self.saccade_inner = max(1, int(saccade_inner))
        self.saccade_halt = bool(saccade_halt)
        self.saccade_halt_eps = float(saccade_halt_eps)
        self.saccade_halt_train = bool(saccade_halt_train)
        self.s2a_head = None
        self.s2a_eval_halt = False
        self.s2a_halt_eps = 0.5
        self.deslice_topk = int(deslice_topk)
        self.use_ada_temp = bool(use_ada_temp)
        self.use_gumbel = bool(use_gumbel)
        self.use_stiefel = bool(use_stiefel)
        self.dual_patch = bool(dual_patch)
        self.patch_size = int(patch_size)
        self.use_unpatch = bool(use_unpatch)
        self.surprise_mode = str(surprise_mode)
        self.surprise_beta = float(surprise_beta)
        self.surprise_detach = bool(surprise_detach)
        self.detach_pred_target = bool(detach_pred_target)
        self.s_update = str(s_update)
        self.interact_prenorm = bool(interact_prenorm)
        self.trust_rho = float(trust_rho)
        self.evidence_decay = float(evidence_decay)
        self.sigma_r = float(sigma_r)
        self.gate_on = str(gate_on)
        self.deslice_write = str(deslice_write)
        self.gate_h_local = bool(gate_h_local)
        self.s_kalman_update = bool(s_kalman_update)
        self.s_lang_topk = int(s_lang_topk)
        self.prior_write = float(prior_write)
        self.prior_write_by_t = bool(prior_write_by_t)
        self.use_null_slice = bool(use_null_slice)
        self.pack_by_surprise = bool(pack_by_surprise)
        self.hard_admit = bool(hard_admit)
        self.use_yield_read = bool(use_yield_read)
        self.use_ticket_read = bool(use_ticket_read)
        self.use_write_yield = bool(use_write_yield)
        self.write_alpha = float(write_alpha)
        self.use_residual_read = bool(use_residual_read)
        self.use_modal_precision = bool(use_modal_precision)
        self.use_target_time = bool(use_target_time)
        self.use_target_time_adaln = bool(use_target_time_adaln)
        self.use_horizon_tokens = bool(use_horizon_tokens)
        self.gate_action_by_horizon = bool(gate_action_by_horizon)
        self.history_size = max(0, int(history_size))
        self.action_dim = max(0, int(action_dim))
        self.use_action_adaln = bool(use_action_adaln)
        self.use_action_tokens = bool(use_action_tokens)
        self.use_action_rel_bias = bool(use_action_rel_bias)
        self.use_action_transport = bool(use_action_transport)
        self.use_action_slice_transition = bool(use_action_slice_transition)
        self.use_goal_adaln = bool(use_goal_adaln)
        self.use_active_gdn2 = bool(use_active_gdn2)
        self.use_active_gdn2_history_transport = bool(
            use_active_gdn2_history_transport
        )
        self.active_gdn2_initial_trust = float(active_gdn2_initial_trust)
        self.terminal_token_atlas = bool(terminal_token_atlas)
        if self.terminal_token_atlas:
            side = int(round(self.n_slices ** 0.5))
            if side * side != self.n_slices:
                raise ValueError(
                    "terminal_token_atlas requires a square n_slices"
                )
        if self.use_target_time_adaln and not self.use_target_time:
            raise ValueError("use_target_time_adaln requires use_target_time")
        if self.use_horizon_tokens and not self.use_target_time:
            raise ValueError("use_horizon_tokens requires use_target_time")
        if self.use_horizon_tokens and self.use_target_time_adaln:
            raise ValueError("horizon tokens and horizon AdaLN are competing ablations")
        if self.gate_action_by_horizon and not self.use_target_time:
            raise ValueError("gate_action_by_horizon requires use_target_time")
        if self.use_action_adaln and self.action_dim <= 0:
            raise ValueError("use_action_adaln requires action_dim > 0")
        if self.use_action_tokens and self.action_dim <= 0:
            raise ValueError("use_action_tokens requires action_dim > 0")
        if self.use_action_tokens and self.use_action_adaln:
            raise ValueError("action tokens and action AdaLN are competing ablations")
        if self.use_action_rel_bias and self.action_dim != 2:
            raise ValueError("action-relative Slice bias currently requires action_dim=2")
        if self.use_action_transport and self.action_dim != 2:
            raise ValueError("action Slice transport currently requires action_dim=2")
        if self.use_action_slice_transition and self.action_dim != 2:
            raise ValueError("action Slice transition currently requires action_dim=2")
        if self.use_active_gdn2 and self.action_dim not in (0, 2):
            raise ValueError("active GDN-2 currently expects no action or a 2D action")
        if self.use_active_gdn2 and self.use_action_slice_transition:
            raise ValueError(
                "active GDN-2 replaces the transient action Slice transition"
            )
        # dual_patch: default no extra LocalVisual (patch stream is the local bias)
        if self.dual_patch and local_kind == "dw3":
            local_kind = "none"
        self.local_kind = str(local_kind)

        # stem: RGB+xy → d_x. t is a 6th point coordinate (not adaLN, not language).
        self.stem = nn.Linear(5, d_x)
        self.t_coord = nn.Linear(1, d_x, bias=False)
        nn.init.normal_(self.t_coord.weight, std=0.02)
        if self.use_modal_precision:
            # Boundary conditions, not task IDs: zero means the corresponding
            # modality is unobserved; one means observed evidence.
            self.image_precision_coord = nn.Linear(1, d_x, bias=False)
            self.text_precision_coord = nn.Linear(1, d, bias=False)
            nn.init.zeros_(self.image_precision_coord.weight)
            nn.init.zeros_(self.text_precision_coord.weight)
        else:
            self.image_precision_coord = None
            self.text_precision_coord = None
        if self.use_target_time and not self.use_horizon_tokens and not self.gate_action_by_horizon:
            # A physical horizon separates reconstruction (0) from prediction
            # (>0). It is not a port/readout switch.
            self.target_time_coord = nn.Linear(1, d_x, bias=False)
            nn.init.zeros_(self.target_time_coord.weight)
        else:
            self.target_time_coord = None
        if self.history_size > 0:
            # Transolver-style Eulerian history: ordered frame channels at the
            # same point enter the one full-resolution stem. No video backbone
            # or permanent temporal tokens are introduced.
            self.history_stem = nn.Linear(
                3 * self.history_size, d_x, bias=False,
            )
            nn.init.normal_(self.history_stem.weight, std=0.02)
        else:
            self.history_stem = None
        if self.action_dim > 0:
            # A sequence is encoded stepwise and pooled in chronological order.
            # Subtracting the zero-action embedding gives exact action=0
            # identity even though the affine layer has a bias.
            self.action_step = nn.Linear(self.action_dim, d_x, bias=True)
            self.action_pos = nn.Parameter(
                torch.zeros(1, max(1, self.history_size), d_x)
            )
            nn.init.normal_(self.action_pos, std=0.02)
            self.action_to_x = nn.Linear(d_x, d_x, bias=False)
            self.action_to_h = nn.Linear(d_x, d, bias=False)
            self.action_token_type = nn.Parameter(torch.zeros(1, 1, d))
            # The precision gate already makes pi_a=0 an exact identity.  A
            # live projection avoids the previous double-zero path
            # (zero action projection + zero-AdaLN), which let optimization
            # solve the toy dynamics from history while ignoring the action.
            nn.init.normal_(self.action_to_x.weight, std=0.02)
            nn.init.normal_(self.action_to_h.weight, std=0.02)
            nn.init.normal_(self.action_token_type, std=0.02)
        else:
            self.action_step = None
            self.action_pos = None
            self.action_to_x = None
            self.action_to_h = None
            self.action_token_type = None
        self.stem_local = nn.Sequential(
            nn.Conv2d(d_x, d_x, 3, padding=1, groups=d_x),
            nn.Conv2d(d_x, d_x, 1),
        )
        self.text_in = nn.Linear(d_llm, d)
        self.text_out = nn.Linear(d, d_llm)
        # Zero-init residual gates: H_llm = embedding + g_h ΔH, tok = g_v proj(S).
        # g=0 keeps the frozen LM on observed token embeddings (token interface).
        self.text_out_gate = nn.Parameter(torch.zeros(()))
        self.proj_gate = nn.Parameter(torch.zeros(()))
        if self.terminal_token_atlas:
            # Generic cross-Slice likelihood connector. A token-wise projector
            # cannot express 2-D topology while Pythia is frozen. Zero-init
            # residual keeps old checkpoints exact until token NLL trains it.
            width = self.n_slices * d
            self.terminal_atlas_mix = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.terminal_atlas_mix.weight)
            self.terminal_atlas_to_text = nn.Linear(width, d_llm, bias=True)
            nn.init.zeros_(self.terminal_atlas_to_text.weight)
            nn.init.zeros_(self.terminal_atlas_to_text.bias)
        else:
            self.terminal_atlas_mix = None
            self.terminal_atlas_to_text = None
        # Shared time condition (flow matching). Zero-init: absent t is identity.
        self.time_cond = TimeCondition(d_x)
        # Kept for old FM ckpts. Live path is t_coord on each point.
        self.t_embed = TimestepEmbedder(d_x)
        self.y_to_c = nn.Linear(d, d_x)
        if self.use_horizon_tokens:
            self.horizon_to_h = nn.Linear(d_x, d, bias=False)
            self.horizon_token_type = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.horizon_to_h.weight, std=0.02)
            nn.init.normal_(self.horizon_token_type, std=0.02)
        else:
            self.horizon_to_h = None
            self.horizon_token_type = None

        n_mods = 1 if self.share_layers else n_layers
        layer_kw = dict(
            d_x=d_x, d=d, n_slices=n_slices, n_heads=n_heads,
            res=res, deslice_topk=deslice_topk,
            use_ada_temp=use_ada_temp,
            use_gumbel=use_gumbel,
            use_stiefel=use_stiefel,
            local_kind=local_kind,
            dual_patch=self.dual_patch,
            patch_size=self.patch_size,
            use_unpatch=self.use_unpatch,
            surprise_mode=self.surprise_mode,
            surprise_beta=self.surprise_beta,
            surprise_detach=self.surprise_detach,
            detach_pred_target=self.detach_pred_target,
            s_update=self.s_update,
            interact_prenorm=self.interact_prenorm,
            trust_rho=self.trust_rho,
            evidence_decay=self.evidence_decay,
            sigma_r=self.sigma_r,
            gate_on=self.gate_on,
            deslice_write=self.deslice_write,
            gate_h_local=self.gate_h_local,
            saccade=self.saccade,
            saccade_gain=self.saccade_gain,
            s_kalman_update=self.s_kalman_update,
            s_lang_topk=self.s_lang_topk,
            prior_write=self.prior_write,
            prior_write_by_t=self.prior_write_by_t,
            use_null_slice=self.use_null_slice,
            pack_by_surprise=self.pack_by_surprise,
            hard_admit=self.hard_admit,
            use_yield_read=self.use_yield_read,
            use_ticket_read=self.use_ticket_read,
            use_write_yield=self.use_write_yield,
            write_alpha=self.write_alpha,
            use_residual_read=self.use_residual_read,
            use_action_rel_bias=self.use_action_rel_bias,
            use_action_transport=self.use_action_transport,
        )
        self.layers = nn.ModuleList([
            NativeMoTLayer(
                **layer_kw,
                use_action_slice_transition=(self.use_action_slice_transition and idx == 0),
            )
            for idx in range(n_mods)
        ])
        if self.use_active_gdn2:
            # Adding an experimental module must not consume the global RNG
            # stream and silently change downstream heads absent from an old
            # checkpoint. fork_rng restores the caller's CPU RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1702)
                self.active_gdn2 = ActiveInferenceGDN2(
                    d_model=d_x, n_slots=n_slices, res=res,
                    initial_prior_trust=self.active_gdn2_initial_trust,
                )
        else:
            self.active_gdn2 = None
        if self.use_target_time_adaln or self.use_action_adaln or self.use_goal_adaln:
            # tau=0 and pi_action=0 must remain exact identity boundaries.
            # A trainable affine bias would leak the future control operator
            # into reconstruction/generation even when the control is absent.
            for layer in self.layers:
                for module in (
                    layer.ada_x.net[-1],
                    layer.ada_s.net[-1],
                    layer.ada_write[-1],
                ):
                    nn.init.zeros_(module.bias)
                    module.bias.requires_grad_(False)
        if self.share_layers:
            self.lti_x = LTIInjection(d_x)
            self.lti_h = LTIInjection(d)
            self.e_x_norm = RMSNorm(d_x)
            self.e_h_norm = RMSNorm(d)
        else:
            self.lti_x = None
            self.lti_h = None
            self.e_x_norm = None
            self.e_h_norm = None
        # interface: mean pool final S via last SliceRead for LLM tokens
        # same knobs as layers (read path only; no deslice)
        self.readout = SliceRead(
            d_x, d, n_slices, n_heads=n_heads,
            use_ada_temp=use_ada_temp,
            use_gumbel=use_gumbel,
            use_stiefel=use_stiefel,
            use_null_slice=use_null_slice,
            use_yield_read=use_yield_read,
        )
        self.proj = MMProjector(d, d_llm, kind=projector)
        self._last_X = None
        self._last_X_prior = None
        self._last_H = None
        self._last_X_stem = None
        self._last_H_stem = None
        self._last_trace_text_mask = None
        self._last_X_steps: List[torch.Tensor] = []
        self.record_field_trace = False
        self._last_W = None
        self._last_step_H: List[torch.Tensor] = []
        self._last_rho: Dict[str, float] = {"rho_x": 0.0, "rho_h": 0.0}
        self._last_looks: Optional[torch.Tensor] = None
        self._last_traces: List[NativeLayerTrace] = []
        self._last_causal_state: Optional[ActiveInferenceState] = None
        self._last_causal_prior_mu: Optional[torch.Tensor] = None
        self._last_causal_prior_logvar: Optional[torch.Tensor] = None
        self._last_causal_posterior_mu: Optional[torch.Tensor] = None
        self._last_causal_posterior_logvar: Optional[torch.Tensor] = None
        self._last_causal_diagnostics: Dict[str, torch.Tensor] = {}

    def set_record_field_trace(self, enabled: bool = True) -> None:
        """Opt in to detached X_0..X_K snapshots for mechanism probes.

        Full-resolution traces are disabled by default so normal training does
        not retain another copy of every point field.
        """
        self.record_field_trace = bool(enabled)

    @torch.no_grad()
    def common_f2_anchor_energy(
        self,
        X: torch.Tensor,
        anchor_layer: int = 0,
    ) -> torch.Tensor:
        """Evaluate one field with the fixed prompt and F2 anchor coordinate."""
        if self._last_H_stem is None:
            raise RuntimeError(
                "no prompt anchor; call set_record_field_trace(True) before forward"
            )
        if not 0 <= int(anchor_layer) < len(self.layers):
            raise IndexError(f"anchor_layer={anchor_layer} outside layer range")
        layer = self.layers[int(anchor_layer)]
        gate = layer.surprise_gate
        if getattr(gate, "mode", None) != "v1_bayes":
            raise RuntimeError("common F2 anchor requires surprise_mode='v1_bayes'")
        S, _ = layer.read(X)
        pred = gate.prior_predictive(
            S,
            self._last_H_stem,
            text_mask=self._last_trace_text_mask,
        )
        return pred["F_min"].float().mean(dim=(1, 2))

    @torch.no_grad()
    def common_f2_anchor_trace(
        self,
        anchor_layer: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate one fixed F2 prior-predictive energy on every saved X_k.

        The selected layer contributes one trained SliceRead and one trained
        language-prior head. Those modules and the initial prompt state H_0
        are reused unchanged for all k. The method is diagnostic-only: it does
        not call MoT, Deslice, LocalVisual, or mutate the persistent field.
        """
        if not self._last_X_steps or self._last_H_stem is None:
            raise RuntimeError(
                "no field trace; call set_record_field_trace(True) before forward"
            )
        energies = [
            self.common_f2_anchor_energy(X_k, anchor_layer=anchor_layer)
            for X_k in self._last_X_steps
        ]
        energy = torch.stack(energies, dim=1)
        return {
            "energy": energy,
            "delta": energy[:, 1:] - energy[:, :-1],
            "terminal_delta": energy[:, -1] - energy[:, 0],
        }

    def set_write_knobs(self, deslice_write: str = "increment", gate_h_local: bool = True) -> None:
        """Eval-time switch: absolute S broadcast vs ΔS velocity; gate H/Local or not."""
        for layer in self.layers:
            layer.deslice_write = str(deslice_write)
            layer.gate_h_local = bool(gate_h_local)

    def set_q_infer(self, q_infer: str = "amortized", thresh: float = 0.05) -> None:
        """Eval-time q ← q* on all layers or residual layers (gap > thresh)."""
        for layer in self.layers:
            g = layer.surprise_gate
            if hasattr(g, "q_infer"):
                g.q_infer = str(q_infer)
                g.q_star_thresh = float(thresh)

    def _pool_h(self, H: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if text_mask is None:
            return H.mean(dim=1)
        m = text_mask.unsqueeze(-1).to(dtype=H.dtype)
        return (H * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def injection_rho(self) -> Dict[str, float]:
        if self.lti_x is None or self.lti_h is None:
            return {"rho_x": 0.0, "rho_h": 0.0}
        return {
            "rho_x": float(self.lti_x.rho().item()),
            "rho_h": float(self.lti_h.rho().item()),
        }

    @staticmethod
    def _point_condition(
        value,
        batch: int,
        points: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> torch.Tensor:
        """Broadcast a scalar/batch/map boundary value to [B,N,1]."""
        if value is None:
            return torch.full((batch, points, 1), default, device=device, dtype=dtype)
        v = torch.as_tensor(value, device=device, dtype=dtype)
        if v.ndim == 0:
            return v.reshape(1, 1, 1).expand(batch, points, 1)
        if v.ndim == 1:
            return v.reshape(batch, 1, 1).expand(batch, points, 1)
        if v.ndim == 2:
            if v.shape[1] == 1:
                return v.reshape(batch, 1, 1).expand(batch, points, 1)
            return v.reshape(batch, points, 1)
        if v.ndim == 3:
            return v.expand(batch, points, 1)
        if v.ndim == 4 and v.shape[1] == 1:
            return v.flatten(2).transpose(1, 2)
        raise ValueError(f"cannot broadcast boundary condition with shape {tuple(v.shape)}")

    def _transition_action_precision(
        self,
        action_precision,
        target_time,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Apply action only across a nonzero prediction horizon."""
        if not self.gate_action_by_horizon:
            return action_precision
        if target_time is None:
            tau = torch.zeros(batch, device=device, dtype=dtype)
        else:
            tau = torch.as_tensor(target_time, device=device, dtype=dtype).reshape(-1)
            if tau.numel() == 1:
                tau = tau.expand(batch)
            if tau.numel() != batch:
                raise ValueError("target_time must be scalar or [B]")
        gate = tau.clamp(0.0, 1.0)
        if action_precision is None:
            return gate
        pi = torch.as_tensor(action_precision, device=device, dtype=dtype)
        if pi.ndim == 0:
            return pi * gate
        if pi.ndim == 1:
            return pi * gate
        return pi * gate.reshape(batch, *([1] * (pi.ndim - 1)))

    def _history_features(
        self,
        history_images,
        history_precision,
        batch: int,
        res: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Encode ordered Eulerian frame values into [B,N,d_x]."""
        if self.history_stem is None or history_images is None:
            return None
        hist = torch.as_tensor(history_images, device=device, dtype=dtype)
        if hist.ndim != 5 or hist.shape[0] != batch or hist.shape[2] != 3:
            raise ValueError(
                "history_images must have shape [B,T,3,R,R], got "
                f"{tuple(hist.shape)}"
            )
        if hist.shape[-2:] != (res, res):
            raise ValueError(
                f"history resolution {tuple(hist.shape[-2:])} != {(res, res)}"
            )
        t_hist = hist.shape[1]
        if history_precision is None:
            pi = torch.ones(batch, t_hist, device=device, dtype=dtype)
        else:
            pi = torch.as_tensor(
                history_precision, device=device, dtype=dtype,
            )
            if pi.ndim == 0:
                pi = pi.reshape(1, 1).expand(batch, t_hist)
            elif pi.ndim == 1:
                if pi.numel() == batch:
                    pi = pi.reshape(batch, 1).expand(batch, t_hist)
                else:
                    pi = pi.reshape(1, -1).expand(batch, -1)
            elif pi.ndim != 2:
                raise ValueError("history_precision must be scalar, [B], or [B,T]")
            if pi.shape != (batch, t_hist):
                raise ValueError(
                    f"history_precision shape {tuple(pi.shape)} != {(batch, t_hist)}"
                )
        if t_hist > self.history_size:
            hist = hist[:, -self.history_size :]
            pi = pi[:, -self.history_size :]
        elif t_hist < self.history_size:
            pad = self.history_size - t_hist
            hist = torch.cat(
                [hist.new_zeros(batch, pad, 3, res, res), hist], dim=1,
            )
            pi = torch.cat([pi.new_zeros(batch, pad), pi], dim=1)
        hist = hist * pi[:, :, None, None, None]
        point_hist = hist.permute(0, 3, 4, 1, 2).reshape(
            batch, res * res, 3 * self.history_size,
        )
        return self.history_stem(point_hist)

    def _action_embedding(
        self,
        action,
        action_precision,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Encode a chronological action sequence with exact pi_a=0 identity."""
        if self.action_step is None:
            return None
        if action is None:
            return torch.zeros(batch, self.d_x, device=device, dtype=dtype)
        act = torch.as_tensor(action, device=device, dtype=dtype)
        if act.ndim == 1:
            if self.action_dim == 1 and act.numel() == batch:
                act = act.reshape(batch, 1, 1)
            else:
                act = act.reshape(1, 1, self.action_dim).expand(batch, -1, -1)
        elif act.ndim == 2:
            act = act.unsqueeze(1)
        if act.ndim != 3 or act.shape[0] != batch or act.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must have shape [B,A] or [B,T,A], got {tuple(act.shape)}"
            )
        t_act = act.shape[1]
        if action_precision is None:
            pi = torch.ones(batch, t_act, 1, device=device, dtype=dtype)
        else:
            pi = torch.as_tensor(action_precision, device=device, dtype=dtype)
            if pi.ndim == 0:
                pi = pi.reshape(1, 1, 1).expand(batch, t_act, 1)
            elif pi.ndim == 1:
                pi = pi.reshape(batch, 1, 1).expand(batch, t_act, 1)
            elif pi.ndim == 2:
                pi = pi.unsqueeze(-1)
            if pi.shape != (batch, t_act, 1):
                raise ValueError(
                    f"action_precision shape {tuple(pi.shape)} != {(batch, t_act, 1)}"
                )
        slots = self.action_pos.shape[1]
        if t_act > slots:
            act = act[:, -slots:]
            pi = pi[:, -slots:]
            t_act = slots
        pos = self.action_pos[:, :t_act].to(device=device, dtype=dtype)
        zero = torch.zeros_like(act)
        delta = self.action_step(act * pi) - self.action_step(zero)
        # Slot-specific multiplicative coordinates preserve action order;
        # adding position alone would vanish under mean pooling or leak when
        # the action is missing.
        delta = delta * (1.0 + torch.tanh(pos)) * pi
        return delta.mean(dim=1)

    def _action_token(
        self,
        action,
        action_precision,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One masked control token; pi_a=0 is invisible to every Slice query."""
        emb = self._action_embedding(
            action, action_precision, batch, device, dtype,
        )
        if emb is None or self.action_to_h is None:
            raise RuntimeError("action token requested without an action encoder")
        if action is None:
            present = torch.zeros(batch, device=device, dtype=torch.bool)
        elif action_precision is None:
            present = torch.ones(batch, device=device, dtype=torch.bool)
        else:
            pi = torch.as_tensor(action_precision, device=device, dtype=dtype)
            if pi.ndim == 0:
                present = (pi > 0).expand(batch)
            elif pi.ndim == 1:
                if pi.numel() != batch:
                    raise ValueError("1D action_precision must have one value per batch item")
                present = pi > 0
            else:
                if pi.shape[0] != batch:
                    raise ValueError("action_precision first dimension must equal batch")
                present = pi.reshape(batch, -1).amax(dim=1) > 0
        token = self.action_to_h(emb).unsqueeze(1)
        token = token + self.action_token_type.to(device=device, dtype=dtype)
        token = token * present[:, None, None].to(dtype=dtype)
        return token, present[:, None]

    def _action_vector(
        self,
        action,
        action_precision,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precision-weighted geometric action for relative Slice attention."""
        if action is None:
            return (
                torch.zeros(batch, 2, device=device, dtype=dtype),
                torch.zeros(batch, device=device, dtype=torch.bool),
            )
        act = torch.as_tensor(action, device=device, dtype=dtype)
        if act.ndim == 1:
            act = act.reshape(1, 1, 2).expand(batch, -1, -1)
        elif act.ndim == 2:
            act = act.unsqueeze(1)
        if act.ndim != 3 or act.shape[0] != batch or act.shape[-1] != 2:
            raise ValueError("geometric action must have shape [B,2] or [B,T,2]")
        if action_precision is None:
            pi = torch.ones(batch, act.shape[1], 1, device=device, dtype=dtype)
        else:
            pi = torch.as_tensor(action_precision, device=device, dtype=dtype)
            if pi.ndim == 0:
                pi = pi.reshape(1, 1, 1).expand(batch, act.shape[1], 1)
            elif pi.ndim == 1:
                pi = pi.reshape(batch, 1, 1).expand(batch, act.shape[1], 1)
            elif pi.ndim == 2:
                pi = pi.unsqueeze(-1)
            if pi.shape != (batch, act.shape[1], 1):
                raise ValueError("action_precision is incompatible with geometric action")
        present = pi.amax(dim=(1, 2)) > 0
        denom = pi.sum(dim=1).clamp_min(1.0)
        vector = (act * pi).sum(dim=1) / denom
        return vector, present

    def _horizon_token(
        self,
        target_time,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Temporary operator token; tau=0 is masked and never enters X."""
        if self.horizon_to_h is None:
            raise RuntimeError("horizon token requested without horizon encoder")
        if target_time is None:
            tau = torch.zeros(batch, device=device, dtype=dtype)
        else:
            tau = torch.as_tensor(target_time, device=device, dtype=dtype).reshape(-1)
            if tau.numel() == 1:
                tau = tau.expand(batch)
            if tau.numel() != batch:
                raise ValueError("target_time must be scalar or have one value per batch item")
        zero = torch.zeros_like(tau)
        delta = self.t_embed(tau) - self.t_embed(zero)
        present = tau.abs() > 0
        token = self.horizon_to_h(delta).unsqueeze(1)
        token = token + self.horizon_token_type.to(device=device, dtype=dtype)
        token = token * present[:, None, None].to(dtype=dtype)
        return token, present[:, None]

    def encode_X(
        self,
        img: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        image_precision=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
    ) -> torch.Tensor:
        """Point field. Recognition: [rgb, x, y]. FM: + t on every point.

        t is a coordinate, same as xy — SliceRead can pool it. Language
        still only meets vision in MoT. t=None is exact old stem (F2-safe).
        """
        B, _, R, _ = img.shape
        assert R == self.res, (R, self.res)
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        image_pi = None
        if self.use_modal_precision:
            image_pi = self._point_condition(
                image_precision, B, R * R, img.device, img.dtype, default=1.0,
            )
            # Missing RGB carries no accidental "black image" evidence.
            pts = pts * image_pi
        x = self.stem(torch.cat([pts, p], -1))
        if image_pi is not None:
            x = x + self.image_precision_coord(image_pi)
        if t is not None:
            tt = t.reshape(-1, 1, 1).to(dtype=x.dtype, device=x.device).expand(B, R * R, 1)
            x = x + self.t_coord(tt)
        if self.use_target_time and not self.use_horizon_tokens and not self.gate_action_by_horizon:
            horizon = self._point_condition(
                target_time, B, R * R, img.device, img.dtype, default=0.0,
            )
            x = x + self.target_time_coord(horizon)
        hist = self._history_features(
            history_images, history_precision, B, R, img.device, img.dtype,
        )
        if hist is not None:
            x = x + hist
        effective_action_precision = self._transition_action_precision(
            action_precision, target_time, B, img.device, img.dtype,
        )
        action_emb = self._action_embedding(
            action, effective_action_precision, B, img.device, img.dtype,
        )
        if action_emb is not None and not self.use_action_tokens:
            x = x + self.action_to_x(action_emb).unsqueeze(1)
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.stem_local(g).flatten(2).transpose(1, 2)
        return x

    @staticmethod
    def _batch_scalar(
        value,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), default, device=device, dtype=dtype)
        out = torch.as_tensor(value, device=device, dtype=dtype)
        if out.ndim == 0:
            return out.expand(batch)
        if out.shape[0] != batch:
            raise ValueError(f"boundary first dimension {out.shape[0]} != batch {batch}")
        if out.ndim == 1:
            return out
        return out.reshape(batch, -1).mean(dim=-1)

    def _active_memory_prior(
        self,
        image: torch.Tensor,
        X: torch.Tensor,
        H: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        image_precision,
        text_precision,
        target_time,
        history_images,
        history_precision,
        action,
        action_precision,
        causal_state: Optional[ActiveInferenceState],
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """Build/query the physical-time belief; never recur over depth."""
        memory = self.active_gdn2
        if memory is None:
            self._last_causal_state = None
            self._last_causal_prior_mu = None
            self._last_causal_prior_logvar = None
            self._last_causal_posterior_mu = None
            self._last_causal_posterior_logvar = None
            self._last_causal_diagnostics = {}
            return None, None, None
        batch = X.shape[0]
        device, dtype = X.device, X.dtype
        point_coords = coords(self.res, device).to(dtype=dtype).expand(batch, -1, -1)
        state = causal_state
        if state is None:
            state = memory.initial_state(batch, device, dtype)
            # Ordered history is consumed by one shared stem/read atlas.  This
            # is a causal scan over physical frames, not over network layers.
            if history_images is not None:
                hist = torch.as_tensor(history_images, device=device, dtype=dtype)
                if hist.ndim != 5 or hist.shape[0] != batch:
                    raise ValueError("history_images must have shape [B,T,3,R,R]")
                t_hist = hist.shape[1]
                if history_precision is None:
                    hpi = torch.ones(batch, t_hist, device=device, dtype=dtype)
                else:
                    hpi = torch.as_tensor(history_precision, device=device, dtype=dtype)
                    if hpi.ndim == 0:
                        hpi = hpi.expand(batch, t_hist)
                    elif hpi.ndim == 1:
                        hpi = hpi.reshape(batch, 1).expand(batch, t_hist)
                    if hpi.shape != (batch, t_hist):
                        raise ValueError(
                            f"history_precision {tuple(hpi.shape)} != {(batch, t_hist)}"
                        )
                zeros = torch.zeros(batch, device=device, dtype=dtype)
                for j in range(t_hist):
                    Xh = self.encode_X(
                        hist[:, j], image_precision=hpi[:, j], target_time=zeros,
                        history_images=None, history_precision=None,
                        action=None, action_precision=None,
                    )
                    Xh_origin = self.encode_X(
                        torch.zeros_like(hist[:, j]),
                        image_precision=hpi[:, j], target_time=zeros,
                        history_images=None, history_precision=None,
                        action=None, action_precision=None,
                    )
                    # Transport visual content relative to the fixed Eulerian
                    # coordinate basis; absolute xy features must not move.
                    Sh, ch = memory.pool_field(Xh - Xh_origin, point_coords)
                    state = memory.assimilate(
                        state, Sh, ch, hpi[:, j], update_motion=True,
                        transport_memory=self.use_active_gdn2_history_transport,
                    )

        # Observe the current frame without history/action conditioning. Those
        # are prior factors, not pixels in the likelihood. This prevents the
        # predictive control from leaking into q(o_t).
        zeros = torch.zeros(batch, device=device, dtype=dtype)
        X_observed = self.encode_X(
            image, image_precision=image_precision, target_time=zeros,
            history_images=None, history_precision=None,
            action=None, action_precision=None,
        )
        X_origin = self.encode_X(
            torch.zeros_like(image), image_precision=image_precision,
            target_time=zeros, history_images=None, history_precision=None,
            action=None, action_precision=None,
        )
        S, centers = memory.pool_field(X_observed - X_origin, point_coords)
        image_pi = self._batch_scalar(
            image_precision, batch, device, dtype, default=1.0,
        ).clamp(0.0, 1.0)
        # In the registered history convention the final history frame is the
        # current frame. Do not replace the inferred velocity by a duplicate
        # zero displacement, but do assimilate its observation precision.
        state = memory.assimilate(
            state, S, centers, image_pi, update_motion=history_images is None,
        )
        posterior_mu, posterior_lv, _ = memory.query_atlas(
            state, horizon=torch.zeros_like(image_pi),
            action=None, action_precision=None,
        )
        self._last_causal_posterior_mu = posterior_mu
        self._last_causal_posterior_logvar = posterior_lv

        tau = self._batch_scalar(
            target_time, batch, device, dtype, default=0.0,
        ).clamp_min(0.0)
        effective_action_precision = self._transition_action_precision(
            action_precision, target_time, batch, device, dtype,
        )
        state = memory.dynamic_prior(
            state, tau, action=action,
            action_precision=effective_action_precision,
        )
        prior_mu, prior_lv, diagnostics = memory.query_atlas(
            state, horizon=tau, action=action,
            action_precision=effective_action_precision,
        )
        # Semantic generation already minimizes the F2 language prior in the
        # shared MoT graph. GDN-2 is the physical-time dynamic prior only.
        boundary = tau.clamp(0.0, 1.0)
        causal_delta = memory.prior_residual(posterior_mu, prior_mu)
        causal_write_w = memory.point_to_atlas_weights(point_coords)
        causal_delta_x = torch.einsum(
            "bnk,bkd->bnd", causal_write_w, causal_delta,
        )
        causal_gate = boundary
        if bool(getattr(self, "disable_active_gdn2", False)):
            causal_gate = torch.zeros_like(causal_gate)
        self._last_causal_state = state
        self._last_causal_prior_mu = prior_mu
        self._last_causal_prior_logvar = prior_lv
        self._last_causal_diagnostics = diagnostics
        return causal_delta_x, None, causal_gate

    def forward_native(
        self,
        img: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        pi_x=1.0,
        force_gate: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
        image_precision=None,
        text_precision=None,
        target_time=None,
        history_images=None,
        history_precision=None,
        action=None,
        action_precision=None,
        causal_state: Optional[ActiveInferenceState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[NativeLayerTrace]]:
        """
        text_emb: [B,T,d_llm]
        pi_x: write permission for the point field (0 = read-only I2T / text).
        n_loops: override recurrent depth when share_layers=True (eval extrapolation).
        Returns: X, H_llm [B,T,d_llm], interface_tokens [B,M,d_llm], traces
        """
        X = self.encode_X(
            img,
            t=t,
            image_precision=image_precision,
            target_time=target_time,
            history_images=history_images,
            history_precision=history_precision,
            action=action,
            action_precision=action_precision,
        )
        self._last_X_stem = X.detach()
        self._last_X_prior = None
        if self.use_modal_precision:
            text_pi = self._point_condition(
                text_precision,
                text_emb.shape[0],
                text_emb.shape[1],
                text_emb.device,
                text_emb.dtype,
                default=1.0,
            )
            # pi_h=0 removes lexical evidence while retaining a valid null
            # token that can be updated from vision by MoT.
            text_evidence = text_emb * text_pi
            H = self.text_in(text_evidence)
            H = H + self.text_precision_coord(text_pi)
        else:
            text_evidence = text_emb
            H = self.text_in(text_emb)
        text_token_count = H.shape[1]
        self._last_text_evidence = text_evidence
        record_trace = bool(getattr(self, "record_field_trace", False))
        field_steps: List[torch.Tensor] = [X.detach()] if record_trace else []
        self._last_H_stem = H.detach() if record_trace else None
        self._last_trace_text_mask = (
            None
            if not record_trace or text_mask is None
            else text_mask.detach()
        )
        # Text content stays in MoT. Optional AdaLN is a temporal control plane
        # only. The live sinusoidal embedding lets zero-init AdaLN weights learn;
        # subtracting the tau=0 embedding keeps reconstruction/generation exact.
        cond = None
        effective_action_precision = self._transition_action_precision(
            action_precision, target_time, X.shape[0], X.device, X.dtype,
        )
        if self.use_goal_adaln:
            # H already participates in every MoT interaction. Its pooled
            # semantic goal also supplies a global control plane so all layers
            # know whether the requested update is reconstruction, editing,
            # generation, or prediction. This is still one graph/head set.
            goal_cond = self.y_to_c(self._pool_h(H, text_mask))
            if self.use_modal_precision:
                goal_cond = goal_cond * text_pi.mean(dim=1)
            cond = goal_cond
        if self.use_target_time_adaln:
            if target_time is None:
                tau = X.new_zeros(X.shape[0])
            else:
                tau = torch.as_tensor(
                    target_time, device=X.device, dtype=X.dtype,
                ).reshape(-1)
            zero = torch.zeros_like(tau)
            tau_cond = self.t_embed(tau) - self.t_embed(zero)
            cond = tau_cond if cond is None else cond + tau_cond
        if self.use_action_adaln:
            action_cond = self._action_embedding(
                action, effective_action_precision, X.shape[0], X.device, X.dtype,
            )
            cond = action_cond if cond is None else cond + action_cond
        if self.use_action_tokens or self.use_horizon_tokens:
            base_text_mask = (
                torch.ones(
                    X.shape[0], text_token_count, device=X.device,
                    dtype=torch.bool,
                )
                if text_mask is None else text_mask.to(device=X.device).bool()
            )
            base_prompt_mask = (
                base_text_mask
                if prompt_mask is None
                else prompt_mask.to(device=X.device).bool() & base_text_mask
            )
            control_tokens = []
            control_masks = []
            if self.use_horizon_tokens:
                horizon_token, horizon_mask = self._horizon_token(
                    target_time, X.shape[0], X.device, X.dtype,
                )
                control_tokens.append(horizon_token)
                control_masks.append(horizon_mask)
            if self.use_action_tokens:
                action_token, action_mask = self._action_token(
                    action, effective_action_precision, X.shape[0], X.device, X.dtype,
                )
                control_tokens.append(action_token)
                control_masks.append(action_mask)
            controls = torch.cat(control_tokens, dim=1)
            controls_mask = torch.cat(control_masks, dim=1)
            H = torch.cat([H, controls], dim=1)
            text_mask = torch.cat([base_text_mask, controls_mask], dim=1)
            # Slice queries read controls directly. Text output tokens remain
            # causal and are stripped back to their original length.
            prompt_mask = torch.cat([base_prompt_mask, controls_mask], dim=1)
        action_vector = action_rel_mask = None
        if (
            self.use_action_rel_bias
            or self.use_action_transport
            or self.use_action_slice_transition
        ):
            action_vector, action_rel_mask = self._action_vector(
                action, effective_action_precision, X.shape[0], X.device, X.dtype,
            )
        causal_delta_x, causal_write_w, causal_prior_gate = self._active_memory_prior(
            img, X, H, text_mask,
            image_precision=image_precision,
            text_precision=text_precision,
            target_time=target_time,
            history_images=history_images,
            history_precision=history_precision,
            action=action,
            action_precision=action_precision,
            causal_state=causal_state,
        )
        traces = []
        pred_terms = []
        vfe_terms = []
        sig_terms = []
        step_H: List[torch.Tensor] = []
        X_orig = X
        opt_state = None
        W = torch.zeros_like(X)
        share = bool(self.share_layers)
        use_lti = share and bool(self.lti_inject) and self.lti_x is not None
        if share:
            n_steps = int(self.n_loops if n_loops is None else n_loops)
            e_x = self.e_x_norm(X) if use_lti else None
            e_h = self.e_h_norm(H) if use_lti else None
        else:
            n_steps = len(self.layers)
            e_x = e_h = None
        inner = self.saccade_inner if (self.saccade and not share) else 1
        halt_on = bool(self.saccade_halt) and (
            bool(self.saccade_halt_train) or not self.training
        )
        s2a_on = (
            self.s2a_head is not None
            and bool(getattr(self, "s2a_eval_halt", False))
            and not self.training
        )
        looks = X.new_zeros(X.shape[0])
        pixel_mass = None
        pack_u = None
        X_prior = None
        prev_rel = None
        prev_g = None
        for i in range(n_steps):
            layer = self.layers[0] if share else self.layers[i]
            # New unique layer → own time. Do not carry the previous layer's mass.
            if not share:
                pixel_mass = None
                prev_rel = None
                prev_g = None
            n_inner = inner if not share else 1
            if share and self.saccade:
                n_inner = 1
            for look in range(n_inner):
                if look > 0 and s2a_on and prev_g is not None and not bool((prev_g > self.s2a_halt_eps).any()):
                    break
                if halt_on and look > 0 and prev_rel is not None and not bool((prev_rel > self.saccade_halt_eps).any()):
                    break
                X_in, H_in = X, H
                point_u = pack_u if self.pack_by_surprise else None
                X_phi, H_phi, tr = layer(
                    X_in, H_in, text_mask=text_mask, prompt_mask=prompt_mask,
                    layer_idx=i if n_inner == 1 else i * n_inner + look,
                    opt_state=opt_state, X_orig=X_orig,
                    pi_x=pi_x, force_gate=force_gate, W=W, pixel_mass=pixel_mass,
                    point_u=point_u, t=t, cond=cond, X_prior=X_prior,
                    action_vector=action_vector, action_mask=action_rel_mask,
                    apply_action_transition=(i == 0 and look == 0),
                    causal_delta_s=None,
                    causal_write_w=(
                        causal_write_w if i == 0 and look == 0 else None
                    ),
                    causal_prior_gate=(
                        causal_prior_gate if i == 0 and look == 0 else None
                    ),
                    causal_delta_x=(
                        causal_delta_x if i == 0 and look == 0 else None
                    ),
                )
                if use_lti:
                    X_phi = self.lti_x.assemble(X_in, e_x, X_phi - X_in)
                    H_phi = self.lti_h.assemble(H_in, e_h, H_phi - H_in)
                rel = getattr(layer, "last_resid_focus", None)
                if rel is None:
                    rel = getattr(layer, "last_resid_rel", None)
                if look == 0 and s2a_on:
                    prev_g = torch.sigmoid(self.s2a_head(self._pool_h(H_phi, text_mask)))
                if s2a_on and look > 0 and prev_g is not None:
                    more = prev_g > self.s2a_halt_eps
                    take = more.view(-1, 1, 1)
                    X = torch.where(take, X_phi, X)
                    H = torch.where(take, H_phi, H)
                    looks = looks + more.to(dtype=looks.dtype)
                elif halt_on and look > 0 and prev_rel is not None:
                    more = prev_rel > self.saccade_halt_eps
                    take = more.view(-1, 1, 1)
                    X = torch.where(take, X_phi, X)
                    H = torch.where(take, H_phi, H)
                    looks = looks + more.to(dtype=looks.dtype)
                else:
                    X, H = X_phi, H_phi
                    looks = looks + 1.0
                if record_trace:
                    field_steps.append(X.detach())
                prev_rel = rel
                pixel_mass = getattr(layer, "last_pixel_mass", None) if self.saccade else None
                if self.pack_by_surprise:
                    pack_u = next_read_tickets(
                        getattr(layer, "last_pixel_mass", None),
                        getattr(layer, "last_U_x", None),
                    )
                # Tickets stay pred_n=f(H,xy) every layer. Do not replace S0
                # with Deslice(μp): that field is flat until Read specializes.
                dW = getattr(layer, "last_dW", None)
                if dW is not None and str(getattr(layer, "deslice_write", "")) in (
                    "workspace", "ws", "consistency", "consist",
                ):
                    W = W + dW
                opt_state = getattr(layer, "last_opt_state", None)
                traces.append(tr)
                step_H.append(H)
                pl = getattr(layer, "last_pred_loss", None)
                if pl is not None:
                    pred_terms.append(pl)
                if getattr(layer, "last_X_prior", None) is not None:
                    self._last_X_prior = layer.last_X_prior
                vl = getattr(layer, "last_vfe_train_loss", None)
                if vl is not None:
                    vfe_terms.append(vl)
                sl = getattr(layer, "last_sigreg_loss", None)
                if sl is not None:
                    sig_terms.append(sl)
        self._last_looks = looks
        self._last_W = W.detach()
        self._last_step_H = step_H
        self._last_rho = self.injection_rho()
        terminal_text_message = None
        if self.terminal_token_atlas:
            # A fixed Eulerian atlas preserves 2-D topology for the terminal
            # token likelihood. It is a read-only Slice(X), not another image
            # encoder, and it never participates in Deslice/write dynamics.
            side = int(round(self.n_slices ** 0.5))
            field = X.transpose(1, 2).reshape(
                X.shape[0], self.d_x, self.res, self.res,
            )
            S_out = F.adaptive_avg_pool2d(
                field, (side, side),
            ).flatten(2).transpose(1, 2)
            flat = S_out.flatten(1)
            S_out = S_out + self.terminal_atlas_mix(flat).reshape_as(S_out)
            terminal_text_message = self.terminal_atlas_to_text(
                S_out.flatten(1)
            )
        else:
            S_out, _ = self.readout(
                X,
                point_u=pack_u if self.pack_by_surprise else None,
            )
        self._last_terminal_token_slices = S_out.detach()
        tok = self.proj_gate * self.proj(S_out)
        H_text = H[:, :text_token_count]
        H_llm = text_evidence + self.text_out_gate * self.text_out(H_text)
        if terminal_text_message is not None:
            # Same cross-modal message at every causal language position. It
            # is independent of answer tokens, so suffix leakage remains
            # impossible while the frozen decoder can read the live X field.
            H_llm = H_llm + terminal_text_message.unsqueeze(1)
        self._last_X = X.detach()
        self._last_H = H_text.detach()
        self._last_X_steps = field_steps
        self._last_traces = traces
        if pred_terms:
            self._last_pred_loss = torch.stack([p.reshape(()) for p in pred_terms]).mean()
        else:
            self._last_pred_loss = X.new_zeros(())
        if vfe_terms:
            self._last_vfe_train_loss = torch.stack([v.reshape(()) for v in vfe_terms]).mean()
        else:
            self._last_vfe_train_loss = X.new_zeros(())
        if sig_terms:
            self._last_sigreg_loss = torch.stack([s.reshape(()) for s in sig_terms]).mean()
        else:
            self._last_sigreg_loss = X.new_zeros(())
        return X, H_llm, tok, traces

    def forward(
        self,
        img: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        external_h: Optional[torch.Tensor] = None,
    ) -> FrontendOut:
        B = img.shape[0]
        if text_emb is None:
            text_emb = torch.zeros(B, 1, self.d_llm, device=img.device, dtype=img.dtype)
            if external_h is not None:
                text_emb = text_emb + external_h.unsqueeze(1)
            text_mask = torch.ones(B, 1, device=img.device)
        X, H_llm, tok, traces = self.forward_native(img, text_emb, text_mask)
        return FrontendOut(
            tokens=tok,
            T=tok.shape[1],
            meta={
                "kind": "native_mot",
                "spec": "modal_private_qkv_ffn_shared_kv_space",
                "d_x": self.d_x,
                "d": self.d,
                "N": int(X.shape[1]),
                "M": self.n_slices,
                "n_layers": self.n_layers,
                "n_loops": self.n_loops if self.share_layers else 1,
                "point_field": True,
                "slice_ephemeral": True,
                "layer_param_tying": bool(self.share_layers),
                "use_ada_temp": self.use_ada_temp,
                "use_gumbel": self.use_gumbel,
                "deslice_topk": self.deslice_topk,
                "use_stiefel": self.use_stiefel,
                "use_null_slice": self.use_null_slice,
                "pack_by_surprise": self.pack_by_surprise,
                "hard_admit": self.hard_admit,
                "use_yield_read": self.use_yield_read,
                "use_ticket_read": self.use_ticket_read,
                "use_write_yield": self.use_write_yield,
                "use_residual_read": self.use_residual_read,
                "local_kind": self.local_kind,
                "dual_patch": self.dual_patch,
                "patch_size": self.patch_size,
                "use_unpatch": self.use_unpatch,
                "layer_traces": [
                    {
                        "layer": t.layer,
                        "x_delta": t.x_delta,
                        "h_delta": t.h_delta,
                        "M": t.M,
                        "T": t.T,
                    }
                    for t in traces
                ],
            },
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def estimate_pretrain_tokens_needed(n_params: int, min_ratio: float = 1.0) -> int:
    """Validation eligibility: data tokens ≥ min_ratio × parameter count."""
    return int(n_params * min_ratio)
