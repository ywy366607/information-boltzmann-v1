"""Sketched Isotropic Gaussian Regularization (SIGReg) from LeJEPA / LeWorldModel (LeWM).

References:
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026), "LeWorldModel" (arXiv:2603.19312)
- Balestriero & LeCun (2025), "LeJEPA: Provably Stable Joint-Embedding Predictive Architecture"

Mathematical Definition:
SIGReg enforces that embeddings Z in R^{B x D} match an isotropic standard Gaussian N(0, I).
By the Cramer-Wold theorem, a d-dimensional distribution is isotropic Gaussian iff all its
1D projections are univariate standard normal N(0, 1).

Algorithm:
1. Slicing: Sample M random unit-norm projection directions U in R^{D x M} on the hypersphere.
   Compute 1D projections H = Z @ U in R^{B x M}.
2. Epps-Pulley Statistic: For each 1D slice h in R^B, measure the L2-weighted distance
   between the empirical characteristic function phi_B(t) and the standard normal CF phi_0(t) = exp(-t^2 / 2):
     phi_B(t) = (1/B) sum_{j=1}^B exp(i * t * h_j) = C(t) + i * S(t)
     where C(t) = mean(cos(t * h)), S(t) = mean(sin(t * h))
     |phi_B(t) - phi_0(t)|^2 = (C(t) - exp(-t^2/2))^2 + S(t)^2
3. Numerical Quadrature:
   Compute integral over t in [t_min, t_max] with Gaussian weight w(t) = exp(-t^2 / (2 * lam^2)):
     T(h) = int w(t) * |phi_B(t) - phi_0(t)|^2 dt
   using K trapezoidal knots.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (SIGReg) without heuristics."""

    def __init__(
        self,
        dim: int,
        num_slices: int = 64,
        num_knots: int = 17,
        t_min: float = 0.2,
        t_max: float = 4.0,
        lam: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_slices = num_slices
        self.num_knots = num_knots
        self.lam = lam

        # Setup trapezoidal quadrature knots t_k in [t_min, t_max]
        t_knots = torch.linspace(t_min, t_max, num_knots)
        # Trapezoid integration step
        dt = (t_max - t_min) / (num_knots - 1)
        # Trapezoid rule weights: [dt/2, dt, dt, ..., dt/2]
        trapz_w = torch.full((num_knots,), dt)
        trapz_w[0] *= 0.5
        trapz_w[-1] *= 0.5

        # Weight function w(t) = exp(-t^2 / (2 * lam^2))
        gaussian_weight = torch.exp(-0.5 * (t_knots / lam) ** 2)
        quad_weights = trapz_w * gaussian_weight

        # Target standard normal characteristic function: phi_0(t) = exp(-t^2 / 2)
        phi_0 = torch.exp(-0.5 * (t_knots ** 2))

        # Register non-trainable buffers
        self.register_buffer("t_knots", t_knots)
        self.register_buffer("quad_weights", quad_weights)
        self.register_buffer("phi_0", phi_0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Computes SIGReg anti-collapse loss on embeddings z.

        Args:
            z: [B, D] batch of latent embeddings.

        Returns:
            Scalar regularizer loss >= 0. Exactly 0 iff z is isotropic N(0, I).
        """
        B, D = z.shape
        if B < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)

        # 1. Sample M random unit vectors on hypersphere S^{D-1}
        # Draw from standard normal, then normalize
        raw_proj = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
        U = raw_proj / torch.linalg.norm(raw_proj, dim=0, keepdim=True).clamp_min(1e-8)

        # 2. Slicing: H in [B, M]
        H = z @ U

        # 3. Compute empirical characteristic function on 1D projections
        # Broadcast over knots: H has [B, M], t_knots has [K]
        # Angles shape: [K, B, M]
        angles = self.t_knots[:, None, None] * H[None, :, :]

        # Real part C(t) and Imaginary part S(t): shape [K, M]
        C = torch.cos(angles).mean(dim=1)
        S = torch.sin(angles).mean(dim=1)

        # 4. Squared discrepancy against target Gaussian phi_0(t)
        # phi_0 shape: [K, 1]
        diff_real = C - self.phi_0[:, None]
        diff_imag = S
        discrepancy = diff_real.square() + diff_imag.square() # [K, M]

        # 5. Integrate over t via quadrature weights: [M]
        # quad_weights has shape [K]
        stat_per_slice = (self.quad_weights[:, None] * discrepancy).sum(dim=0)

        # Average over all M random slices
        return stat_per_slice.mean()
