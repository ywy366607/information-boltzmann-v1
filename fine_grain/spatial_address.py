"""Prompt-Conditioned Spatial Address Prior and Multi-Frequency Coordinate Basis.

Addresses the foundational T2I bottleneck:
  On a blank canvas (pi_x = 0), visual pixels are all zero.
  Without prompt conditioning before SliceRead, layer-0 slice assignment W
  is 100% prompt-blind: W(digit 7) == W(digit 0).
  Furthermore, with only linear (y, x) coordinates, slice boundaries are strictly
  affine half-planes, incapable of representing 1px sharp curved strokes.

Three-tier solution:
  Tier 1: Continuous multi-frequency Fourier coordinate basis phi(x,y).
  Tier 2: Language-conditioned spatial address prior P = CrossAttn(phi, H),
          producing prompt-aware slice routing logits w^p = softmax(g([X, P, phi])).
  Tier 3: Active inference address KL loss D_KL(q(W|o,H) || p(W|H)) and
          address mutual-information diversity regularization.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.models import coords


_coord_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}


def multi_freq_coords(
    R: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_freq: int = 4,
) -> torch.Tensor:
    """Continuous multi-frequency coordinate basis phi(x, y).

    Cached across calls for identical (R, n_freq, device, dtype) to eliminate
    redundant meshgrid and trigonometric allocations.
    Returns:
      phi: [1, N, 2 + 4 * n_freq] where N = R * R, with coordinates in [-1, 1].
           [y, x, sin(2^0 pi y), cos(2^0 pi y), sin(2^0 pi x), cos(2^0 pi x), ...]
    """
    key = (R, int(n_freq), str(device), str(dtype))
    cached = _coord_cache.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached

    xy = coords(R, device).to(dtype=dtype)  # [1, N, 2]
    if n_freq <= 0:
        _coord_cache[key] = xy
        return xy
    f = (2.0 ** torch.arange(n_freq, device=device, dtype=dtype)) * math.pi
    a = xy.unsqueeze(-1) * f  # [1, N, 2, n_freq]
    sin_a = torch.sin(a).flatten(-2)  # [1, N, 2 * n_freq]
    cos_a = torch.cos(a).flatten(-2)  # [1, N, 2 * n_freq]
    res = torch.cat([xy, sin_a, cos_a], dim=-1)
    _coord_cache[key] = res
    return res


class LanguageAddressPrior(nn.Module):
    """Computes prompt-conditioned spatial address routing prior W^p = softmax(ell).

    Allows language prompt H to determine where slices attend on the canvas
    BEFORE visual aggregation, enabling different prompts (e.g. digit '7' vs '0')
    to activate distinct, high-frequency spatial slice footprints on a blank canvas.
    """

    def __init__(
        self,
        d_model: int,
        n_slices: int = 32,
        n_freq: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.d = int(d_model)
        self.M = int(n_slices)
        self.n_freq = int(n_freq)
        self.coord_dim = 2 + 4 * self.n_freq if self.n_freq > 0 else 2
        self.hidden_dim = int(hidden_dim)

        # Coordinate query projection
        self.q_proj = nn.Linear(self.coord_dim, self.hidden_dim)
        # Language key / value projection
        self.k_proj = nn.Linear(self.d, self.hidden_dim)
        self.v_proj = nn.Linear(self.d, self.hidden_dim)

        # Output MLP predicting slice routing logits: [hidden_dim + coord_dim] -> M
        self.to_logits = nn.Sequential(
            nn.Linear(self.hidden_dim + self.coord_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.M),
        )

        # Scale factor
        self.scale = self.hidden_dim ** -0.5

    def forward(
        self,
        H: torch.Tensor,
        R: int,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """H: [B, T, d] language hidden states

        R: canvas grid resolution (N = R * R)
        text_mask: optional [B, T] bool mask (1=keep)
        Returns:
          logits: [B, N, M] slice routing logits
          w_prior: [B, N, M] soft spatial assignment probabilities
        """
        B, T, _ = H.shape
        device = H.device
        dtype = H.dtype
        N = R * R

        # 1. Multi-frequency coordinate basis (cached)
        phi_1 = multi_freq_coords(R, device, dtype, n_freq=self.n_freq)  # [1, N, coord_dim]

        # 2. Hoisted query projection: compute on [1, N] once, expand to [B, N]
        q = self.q_proj(phi_1).expand(B, -1, -1)   # [B, N, hidden_dim]
        k = self.k_proj(H)                         # [B, T, hidden_dim]
        v = self.v_proj(H)                         # [B, T, hidden_dim]

        # 3. Fused C++ SDPA kernel (memory-efficient, no [B, N, T] tensor allocated)
        attn_mask = text_mask.unsqueeze(1).unsqueeze(2) if text_mask is not None else None
        p_lang = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            attn_mask=attn_mask,
        ).squeeze(1)  # [B, N, hidden_dim]

        # 4. Router Logits from language prior + spatial harmonics
        phi_B = phi_1.expand(B, -1, -1)
        feat = torch.cat([p_lang, phi_B], dim=-1)    # [B, N, hidden_dim + coord_dim]
        logits = self.to_logits(feat)                # [B, N, M]
        w_prior = torch.softmax(logits, dim=-1)

        return logits, w_prior


def compute_address_kl_loss(
    q_w: torch.Tensor,
    p_w: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Analytical KL divergence between posterior address q(W|o,H) and prior address p(W|H).

    Both q_w and p_w: [B, N, M], probabilities summing to 1 over M at each pixel.
    D_KL(q || p) = sum_m q_m * log( (q_m + eps) / (p_m + eps) )
    """
    q_w = torch.clamp(q_w, min=eps)
    p_w = torch.clamp(p_w, min=eps)
    kl = q_w * (torch.log(q_w) - torch.log(p_w))  # [B, N, M]
    return kl.sum(dim=-1).mean()                  # Mean over B, N


def compute_address_diversity_loss(
    w: torch.Tensor,
    eps: float = 1e-7,
    balance_weight: float = 1.0,
    sharpness_weight: float = 0.5,
    decollinear_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Information-Theoretic and Geometric Regularization on Address Matrix W.

    Enforces:
      1. Slot Mass Balance: Maximize entropy of mean slot mass H(w_bar) across canvas,
         preventing all points from collapsing into a single slice.
      2. Point Sharpness: Minimize conditional entropy H(w_i) at each individual point,
         encouraging decisive, non-uniform spatial assignment.
         (Maximizes Mutual Information I(X; M) = H(w_bar) - H(w|X)).
      3. De-collinearity: Penalize off-diagonal correlations of normalized W^T @ W.
    """
    B, N, M = w.shape

    # 1. Slot Mass Balance
    w_bar = w.mean(dim=1).clamp_min(eps)  # [B, M], average mass across all pixels
    slot_entropy = -(w_bar * torch.log(w_bar)).sum(dim=-1).mean()
    # Loss is negative entropy (maximize entropy -> uniform slot utilization)
    loss_balance = -slot_entropy

    # 2. Point Sharpness (conditional entropy)
    w_safe = w.clamp_min(eps)
    point_entropy = -(w_safe * torch.log(w_safe)).sum(dim=-1).mean()  # Mean over B, N
    loss_sharpness = point_entropy  # minimize point entropy -> sharp assignment

    # 3. De-collinearity across slices
    w_centered = w - w.mean(dim=1, keepdim=True)
    gram = torch.bmm(w_centered.transpose(1, 2), w_centered) / float(N)  # [B, M, M]
    norms = torch.sqrt(torch.diagonal(gram, dim1=1, dim2=2).clamp_min(eps)).unsqueeze(-1)  # [B, M, 1]
    gram_norm = gram / (torch.bmm(norms, norms.transpose(1, 2)) + eps)  # cosine correlation [B, M, M]

    eye = torch.eye(M, device=w.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    offdiag_corr = gram_norm[~eye].pow(2).mean()

    total_diversity_loss = (
        balance_weight * loss_balance
        + sharpness_weight * loss_sharpness
        + decollinear_weight * offdiag_corr
    )

    return {
        "address_diversity_loss": total_diversity_loss,
        "slot_entropy": slot_entropy.detach(),
        "point_entropy": point_entropy.detach(),
        "mutual_information": (slot_entropy - point_entropy).detach(),
        "offdiag_corr": offdiag_corr.detach(),
    }


def compute_address_compactness_loss(
    w: torch.Tensor,
    *,
    balance_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Make Slice addresses spatially compact without adding a new visual path.

    ``w`` is the existing point-to-Slice assignment [B, N, M].  A row-wise
    softmax alone can divide *every* point among global slots; its entropy is
    not a locality measure.  This computes each slot's second moment around
    its own soft centroid, plus a light slot-mass balance term so all slots do
    not collapse onto one small region.  It intentionally says nothing about
    target identity: target-dependent posterior alignment supplies that signal.
    """
    if w.ndim != 3:
        raise ValueError("w must be [B, N, M]")
    batch, points, _ = w.shape
    side = int(math.isqrt(points))
    if side * side != points:
        raise ValueError("address compactness requires a square point field")
    axis = torch.linspace(-1.0, 1.0, side, device=w.device, dtype=w.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)  # [N,2]
    mass = w.sum(dim=1).clamp_min(1e-8)                         # [B,M]
    centres = torch.einsum("bnm,nc->bmc", w, xy) / mass.unsqueeze(-1)
    delta2 = (xy[None, :, None, :] - centres[:, None, :, :]).square().sum(-1)
    radius2 = (w * delta2).sum(dim=1) / mass
    compactness = radius2.mean()
    slot_mass = mass / float(points)
    slot_entropy = -(slot_mass.clamp_min(1e-8) * slot_mass.clamp_min(1e-8).log()).sum(-1).mean()
    balance = -slot_entropy
    total = compactness + float(balance_weight) * balance
    return {
        "address_compactness_loss": total,
        "address_radius2": radius2.mean().detach(),
        "address_slot_entropy": slot_entropy.detach(),
    }
