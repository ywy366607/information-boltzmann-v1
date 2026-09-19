"""Unified adaptive force and multi-timescale dissipation for Information Boltzmann.

Protocol:
1. Retains full continuous phase-space dynamical skeleton: persistent particles, transport, local elastic collisions.
   - Birth operator B_theta: responsible only for initial distribution at t=0.
   - Force operator F_theta: responsible for token-by-token input drive.
   - Transport: responsible for position evolution dx/dt = v.
   - Collisions: responsible for local elastic state exchange, preserving mass, momentum, and kinetic energy.
   - Gamma and thermal bath: responsible for dissipation and thermalization.
2. Local environment descriptor e_i = [rho_i, v_bar_i, vartheta_i, h_theta(x_i)]:
   - rho_i: local particle density from compact support kernel K_h(x_i - x_j).
   - v_bar_i: local bulk velocity.
   - vartheta_i: local velocity fluctuation variance (microscopic kinetic temperature of the gas, distinct from bath T).
   - h_theta(x_i): learnable spatial coordinate features.
   - Explicit fallback: isolated particles have v_bar_i = v_i, vartheta_i = 0.
   - Zero cross-sample mixing across batches.
3. Shared condition encoder:
   c_{t, i} = Encoder[u_t, x_i, v_i, e_i]
   Outputs two independent heads:
   F_{t, i}^{input} = W_F c_{t, i} + b_F
   gamma_{t, i} = softplus(b_{gamma, i} + W_gamma c_{t, i})
4. GDN-2 inspired multi-timescale gamma initialization:
   tau_i in [tau_min, tau_max] tokens (default: [32, 1024]).
   gamma_{0, i} = 1 / (tau_i * Delta t_token).
   b_{gamma, i} = softplus^{-1}(gamma_{0, i}).
   W_gamma is zero-initialized so initial damping strictly matches the multi-timescale schedule.
5. Integration into ODE sub-steps:
   Computed once per token, held constant across the 4 sub-steps of that token.
   Substep decay: alpha_{i, sub} = exp(-gamma_{t, i} * Delta t_sub).
   Replaces old static token drive, no duplicate input, no duplicate damping.
"""
import math
import os
import sys

# Prevent local scripts/ib_local from shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch import nn
from torch.nn import functional as F


class LocalEnvironmentField(nn.Module):
    """Computes local fluid/plasma environmental descriptor e_i = [rho_i, v_bar_i, vartheta_i, h_theta(x_i)]."""
    def __init__(self, dim: int = 4, d_space: int = 8, width: float = 1.0):
        super().__init__()
        self.dim = dim
        self.d_space = d_space
        self.width = width

        # Learnable spatial coordinate feature mapping
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, d_space),
            nn.SiLU(),
            nn.Linear(d_space, d_space),
        )

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute local environment without N^2 global attention, strictly preserving sample isolation.

        x: [N, d] or [B, N, d]
        v: [N, d] or [B, N, d]
        Returns: [N, d_env] or [B, N, d_env] where d_env = 1 + d + 1 + d_space (14 for d=4, d_space=8)
        """
        has_batch = (x.ndim == 3)
        if not has_batch:
            x_b = x.unsqueeze(0)
            v_b = v.unsqueeze(0)
        else:
            x_b = x
            v_b = v

        B, N, d = x_b.shape

        # Pairwise spatial displacements: [B, N, N, d]
        deltas = (x_b.unsqueeze(2) - x_b.unsqueeze(1)).abs()
        # Compact support tent kernel: [B, N, N]
        factors = torch.clamp(1.0 - deltas / self.width, min=0.0)
        # Unroll multiplication over phase dimension to avoid torch.prod CUDA graph capture incompatibility
        if d == 4:
            W = factors[..., 0] * factors[..., 1] * factors[..., 2] * factors[..., 3]
        elif d == 2:
            W = factors[..., 0] * factors[..., 1]
        else:
            W = factors.prod(dim=-1)  # [B, N, N], diagonal W_ii = 1.0

        # 1. Local density rho: [B, N, 1]
        rho = W.sum(dim=-1, keepdim=True)
        denom = rho.clamp_min(1e-6)

        # 2. Local bulk velocity v_bar: [B, N, d]
        v_bar = torch.bmm(W, v_b) / denom

        # 3. Local kinetic temperature vartheta (velocity fluctuation variance around local bulk): [B, N, 1]
        v_sq = (v_b ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        v_sq_mean = torch.bmm(W, v_sq) / denom        # [B, N, 1]
        v_bar_sq = (v_bar ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        vartheta = torch.clamp((v_sq_mean - v_bar_sq) / d, min=0.0)

        # Fallback check for isolated particles (W_ij == 0 for all j != i):
        # When rho_i == 1.0 (self only), W @ v == v_i, so v_bar == v_i, and vartheta == 0.0 naturally.
        isolated = (rho <= 1.0 + 1e-5)
        v_bar = torch.where(isolated, v_b, v_bar)
        vartheta = torch.where(isolated, torch.zeros_like(vartheta), vartheta)

        # 4. Spatial coordinate feature h_theta(x_i): [B, N, d_space]
        h_x = self.coord_mlp(x_b)

        # Concatenate: [B, N, 1 + d + 1 + d_space]
        e_t = torch.cat([rho, v_bar, vartheta, h_x], dim=-1)

        if not has_batch:
            e_t = e_t.squeeze(0)
        return e_t


class UnifiedAdaptiveForce(nn.Module):
    """Unified token-driven external force and multi-timescale local damping generator."""
    def __init__(
        self,
        vocab: int = 50257,
        dim: int = 4,
        hidden_dim: int = 128,
        n_particles: int = 256,
        d_space: int = 8,
        width: float = 1.0,
        tau_min: float = 32.0,
        tau_max: float = 1024.0,
        dt_token: float = 1.0,
        amplitude: float = 1.0,
    ):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.d_space = d_space
        self.width = width
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt_token = dt_token
        self.amplitude = amplitude

        # 1. Local environment aggregator
        self.environment = LocalEnvironmentField(dim=dim, d_space=d_space, width=width)
        d_env = 1 + dim + 1 + d_space  # 14 for dim=4, d_space=8

        # 2. GDN-2 inspired multi-timescale gamma initialization
        # tau_i log-uniformly spaced across particles in [tau_min, tau_max]
        taus = tau_min * (tau_max / tau_min) ** (torch.linspace(0, 1, n_particles))
        gamma_0 = 1.0 / (taus * dt_token)
        biases = torch.log(torch.expm1(gamma_0))
        self.gamma_bias = nn.Parameter(biases.unsqueeze(-1))  # [N, 1]

        # 3. Shared condition encoder: c_{t, i} = Encoder[u_t, x_i, v_i, e_i]
        in_dim = hidden_dim + 2 * dim + d_env
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 4. Output Heads
        # Force head: W_F c_{t, i} + b_F (zero-initialized for exact baseline preservation at init)
        self.force_head = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.force_head.weight)
        nn.init.zeros_(self.force_head.bias)

        # Gamma head: W_gamma c_{t, i} (zero-initialized so gamma_i = gamma_{0, i} at start)
        self.gamma_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        token_embedding: torch.Tensor,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Compute input external force F_{t, i} and adaptive damping gamma_{t, i} per token.

        x: [N, d]
        v: [N, d]
        token_embedding: [H] (from force.embedding(y_{t-1}))
        Returns:
            F_input: [N, d]
            gamma: [N, 1]
            diag: dict | None
        """
        N, d = x.shape
        z = torch.cat([x, v], dim=-1)           # [N, 2d]
        e_i = self.environment(x, v)            # [N, d_env]
        tok_exp = token_embedding.unsqueeze(0).expand(N, -1)  # [N, H]

        # Shared conditional encoding
        enc_in = torch.cat([tok_exp, z, e_i], dim=-1)
        c = self.encoder(enc_in)  # [N, H]

        # 1. Bounded force output (smooth tanh scaling, not squashing state)
        raw_F = self.force_head(c)  # [N, d]
        F_input = (self.amplitude / math.sqrt(d)) * torch.tanh(raw_F)

        # 2. Adaptive local gamma (initializes exactly to multi-timescale gamma_0)
        gamma = F.softplus(self.gamma_bias + self.gamma_head(c))  # [N, 1]

        diag = None
        if diagnostics:
            tau_eff = 1.0 / (gamma.mean() * self.dt_token)
            diag = {
                'gamma_mean': float(gamma.mean().item()),
                'gamma_min': float(gamma.min().item()),
                'gamma_max': float(gamma.max().item()),
                'effective_tau_mean': float(tau_eff.item()),
                'force_norm_mean': float(F_input.norm(dim=-1).mean().item()),
            }

        return F_input, gamma, diag
