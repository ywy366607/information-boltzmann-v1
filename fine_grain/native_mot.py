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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.bayesian_surprise import BayesianSurpriseGate, global_gate_from_surprise
from fine_grain.hard_admit import YieldGate
from fine_grain.saccade import residual_pixel_mass, residual_rel
from fine_grain.frontends import FrontendOut
from fine_grain.mm_projector import MMProjector
from fine_grain.flow_match import AdaLNZero, TimeCondition, TimestepEmbedder
from fine_grain.forward_optim import ForwardStateOpt, normalize_kind
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
      use_null_slice — softmax over M content + 1 sink ∅. A point may spend
        mass on ∅ and not enter any content slice. Returned w is content-only
        and need not sum to 1. S still mass-norms over the M content slices.
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
        else:
            self.to_null = None
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

    def forward(
        self,
        x: torch.Tensor,
        w_override: Optional[torch.Tensor] = None,
        pixel_mass: Optional[torch.Tensor] = None,
        point_u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        h, dh, M = self.h, self.dh, self.M
        xp = self.proj_in(x)  # [B,N,d]
        self.last_null = None
        self.last_pack_alpha = None
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
        vis_key_ok = torch.cat([
            torch.ones(B, L_v, device=S.device, dtype=torch.bool),
            prompt_mask,
        ], dim=1)
        av = av.masked_fill(~vis_key_ok[:, None, None, :], neg)
        av = torch.softmax(av, dim=-1)
        Av = self._merge(torch.matmul(av, Val))
        # Per-visual-token mass on text keys. Slice-selective language uses this.
        self.last_text_mass = av[:, :, :, L_v:].sum(dim=-1).mean(dim=1)  # [B, L_v]

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
    """π_X from omni ``need_pix`` (True = write port)."""
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
        self.read = SliceRead(
            d_x, d, n_slices, n_heads=n_heads,
            use_ada_temp=use_ada_temp,
            use_gumbel=use_gumbel,
            use_stiefel=use_stiefel,
            use_null_slice=use_null_slice,
            use_yield_read=use_yield_read,
        )
        self.mot = NativeMoTBlock(d=d, n_heads=n_heads)
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
            preserve_mass=use_null_slice or use_yield_read,
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
        self.admit = YieldGate(d_x) if self.hard_admit else None
        # Open-Sora split: adaLN carries diffusion t only. Language stays in MoT.
        self.ada_x = AdaLNZero(d_x)
        self.ada_s = AdaLNZero(d)
        self.ada_write = nn.Sequential(nn.SiLU(), nn.Linear(d_x, d_x))
        nn.init.zeros_(self.ada_write[-1].weight)
        nn.init.zeros_(self.ada_write[-1].bias)
        self.cond_to_s = nn.Identity() if d == d_x else nn.Linear(d_x, d)

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
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, NativeLayerTrace]:
        if cond is not None:
            X = self.ada_x(X, cond)
        X0, H0 = X, H
        S, w = self.read(X, pixel_mass=pixel_mass, point_u=point_u)
        self.last_null = getattr(self.read, "last_null", None)
        self.last_pack_alpha = getattr(self.read, "last_pack_alpha", None)
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
            )
            delta = S2n - S_in
            v_H = H2n - H_in
            P2 = (P2 - P) if P2 is not None else None
        else:
            S2n, H2n, _ = self.mot(
                S_in, H_in, text_mask=text_mask, prompt_mask=prompt_mask,
            )
            delta = S2n - S_in
            v_H = H2n - H_in
            P2 = None
        delta = self._mask_lang_delta(delta, S.shape[1])
        gate, s_meta = self.surprise_gate(
            S, H, delta_S=delta, text_mask=text_mask,
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
        write = str(getattr(self, "deslice_write", "increment")).lower()
        S_W = None
        if write in ("absolute", "abs", "s", "state"):
            # Belief broadcast: scatter proj(S_write), including bias (Champion B).
            delta_x = self.deslice(S_write, w, X) - X
        elif write in ("workspace", "ws", "consistency", "consist"):
            # Idempotent absolute broadcast onto workspace W, not time increment.
            # R(W) ≠ R(E+W)=S, so this is not P0. Empty W ⇒ ≈ D(S) (first stamp).
            if W is None:
                W = torch.zeros_like(X)
            S_W, _ = self.read(W)
            delta_x = self.deslice.write_delta(S_write - S_W, w)
        else:
            # Velocity write: scatter proj(S_write − S), bias-free so 0 → 0.
            delta_x = self.deslice.write_delta(S_write - S, w)
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
        pi = coerce_pi_x(pi_x, X)
        X1 = X + pi * delta_x
        loc_out = self.local(X1)
        loc_res = loc_out - X1
        if yield_scale is not None:
            loc_res = yield_scale * loc_res
        X2 = X1 + pi * g_hl * loc_res
        # Language-only prior canvas. Workspace wrote μp on slices; the
        # prior that seeing compares against is this field on points.
        if mu_p is not None:
            self.last_X_prior = X0 + pi * self.deslice.write_delta(mu_p - S, w)
        else:
            self.last_X_prior = None
        rms_s = float(S.detach().pow(2).mean().sqrt())
        rms_d = float(delta.detach().pow(2).mean().sqrt())
        self.last_rms_s = rms_s
        self.last_rms_delta = rms_d
        self.last_rms_ratio = rms_d / max(rms_s, 1e-6)
        # Store intermediate diagnostics for visualization / probe
        self.last_w = w.detach()
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
        w_det = w.detach()
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
        # dual_patch: default no extra LocalVisual (patch stream is the local bias)
        if self.dual_patch and local_kind == "dw3":
            local_kind = "none"
        self.local_kind = str(local_kind)

        # stem: RGB+xy → d_x. t is a 6th point coordinate (not adaLN, not language).
        self.stem = nn.Linear(5, d_x)
        self.t_coord = nn.Linear(1, d_x, bias=False)
        nn.init.normal_(self.t_coord.weight, std=0.02)
        self.stem_local = nn.Sequential(
            nn.Conv2d(d_x, d_x, 3, padding=1, groups=d_x),
            nn.Conv2d(d_x, d_x, 1),
        )
        self.text_in = nn.Linear(d_llm, d)
        self.text_out = nn.Linear(d, d_llm)
        # Shared time condition (flow matching). Zero-init: absent t is identity.
        self.time_cond = TimeCondition(d_x)
        # Kept for old FM ckpts. Live path is t_coord on each point.
        self.t_embed = TimestepEmbedder(d_x)
        self.y_to_c = nn.Linear(d, d_x)

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
        )
        self.layers = nn.ModuleList([NativeMoTLayer(**layer_kw) for _ in range(n_mods)])
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
        self._last_W = None
        self._last_step_H: List[torch.Tensor] = []
        self._last_rho: Dict[str, float] = {"rho_x": 0.0, "rho_h": 0.0}
        self._last_looks: Optional[torch.Tensor] = None
        self._last_traces: List[NativeLayerTrace] = []

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

    def encode_X(self, img: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Point field. Recognition: [rgb, x, y]. FM: + t on every point.

        t is a coordinate, same as xy — SliceRead can pool it. Language
        still only meets vision in MoT. t=None is exact old stem (F2-safe).
        """
        B, _, R, _ = img.shape
        assert R == self.res, (R, self.res)
        pts = img.reshape(B, 3, R * R).transpose(1, 2)
        p = coords(R, img.device).expand(B, -1, -1)
        x = self.stem(torch.cat([pts, p], -1))
        if t is not None:
            tt = t.reshape(-1, 1, 1).to(dtype=x.dtype, device=x.device).expand(B, R * R, 1)
            x = x + self.t_coord(tt)
        g = x.transpose(1, 2).reshape(B, -1, R, R)
        x = x + self.stem_local(g).flatten(2).transpose(1, 2)
        return x

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[NativeLayerTrace]]:
        """
        text_emb: [B,T,d_llm]
        pi_x: write permission for the point field (0 = read-only I2T / text).
        n_loops: override recurrent depth when share_layers=True (eval extrapolation).
        Returns: X, H_llm [B,T,d_llm], interface_tokens [B,M,d_llm], traces
        """
        X = self.encode_X(img, t=t)
        self._last_X_stem = X.detach()
        self._last_X_prior = None
        H = self.text_in(text_emb)
        # t lives on the point. Do not FiLM language into X. cond stays None.
        cond = None
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
                    point_u=point_u, t=t, cond=cond,
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
                prev_rel = rel
                pixel_mass = getattr(layer, "last_pixel_mass", None) if self.saccade else None
                if self.pack_by_surprise:
                    pack_u = next_read_tickets(
                        getattr(layer, "last_pixel_mass", None),
                        getattr(layer, "last_U_x", None),
                    )
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
        S_out, _ = self.readout(X, point_u=pack_u if self.pack_by_surprise else None)
        tok = self.proj(S_out)
        H_llm = self.text_out(H) + text_emb  # residual in LLM space
        self._last_X = X.detach()
        self._last_H = H.detach()
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
