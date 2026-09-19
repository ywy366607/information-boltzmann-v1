"""M03: Adaptive local memory with environment-aware writing and multi-timescale dissipation.

Equations and principles:
1. Retains full continuous phase-space dynamical skeleton: persistent particles, transport, local elastic collisions.
2. GDN-2 inspired multi-timescale gamma initialization:
     alpha_i = exp(-gamma_i * Delta t_token)
     Initial biases b_i distributed across timescales tau in [tau_min, tau_max] tokens:
     b_i = softplus^{-1}(Delta_i), Delta_i = 2.0 / tau_i.
3. Local environment field e_t(x_i) = [rho_t(x_i), u_t(x_i), T_{kin, t}(x_i), h_theta(x_i)]:
     - rho: local particle density from compact support kernel K_h.
     - u: local mean bulk velocity.
     - T_{kin}: local kinetic temperature (velocity fluctuation variance around local bulk).
     - h_theta(x): learnable spatial coordinate features.
4. Shared conditional encoder:
     c_i = Encoder(u_t, z_i, e_t(x_i))
     gamma_i = softplus(b_i + g_theta(c_i))
     w_i = sigma(q_theta(c_i))
     Delta z_i = beta_max * W_theta(c_i) / sqrt(1 + ||W_theta(c_i)||^2)
     Zero-initialized final projections guarantee exact identity mapping at initialization.
5. Direct sub-step integration:
     alpha_{i, sub} = exp(-gamma_i * Delta t_sub)
     gamma_i acts directly on ODE damping without duplicate outer decay.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LocalEnvironmentField(nn.Module):
    """Computes local fluid/plasma environmental descriptor e_t(x_i) = [rho, u, T_kin, h_theta(x)]."""
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
        """Compute local environment without N^2 global attention, preserving sample isolation.

        x: [N, d] or [B, N, d]
        v: [N, d] or [B, N, d]
        Returns: [N, 1 + d + 1 + d_space] or [B, N, 1 + d + 1 + d_space]
        """
        has_batch = (x.ndim == 3)
        if not has_batch:
            x_b = x.unsqueeze(0)
            v_b = v.unsqueeze(0)
        else:
            x_b = x
            v_b = v

        B, N, d = x_b.shape

        # Pairwise spatial distances: [B, N, N, d]
        deltas = (x_b.unsqueeze(2) - x_b.unsqueeze(1)).abs()
        # Compact support tent kernel: [B, N, N]
        factors = torch.clamp(1.0 - deltas / self.width, min=0.0)
        W = factors.prod(dim=-1)  # [B, N, N]

        # 1. Local density rho: [B, N, 1]
        rho = W.sum(dim=-1, keepdim=True)
        denom = rho.clamp_min(1e-6)

        # 2. Local bulk velocity u: [B, N, d]
        u = torch.bmm(W, v_b) / denom

        # 3. Local kinetic temperature (velocity fluctuation variance around bulk): [B, N, 1]
        v_sq = (v_b ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        v_sq_mean = torch.bmm(W, v_sq) / denom        # [B, N, 1]
        u_sq = (u ** 2).sum(dim=-1, keepdim=True)     # [B, N, 1]
        t_kin = torch.clamp((v_sq_mean - u_sq) / d, min=0.0)

        # 4. Spatial coordinate feature: [B, N, d_space]
        h_x = self.coord_mlp(x_b)

        # Concatenate: [B, N, 1 + d + 1 + d_space]
        e_t = torch.cat([rho, u, t_kin, h_x], dim=-1)

        if not has_batch:
            e_t = e_t.squeeze(0)
        return e_t


class AdaptiveMemoryController(nn.Module):
    """M03: Local adaptive forgetting + state- and environment-aware writing controller."""
    def __init__(
        self,
        dim: int = 4,
        hidden_dim: int = 128,
        n_particles: int = 256,
        d_space: int = 8,
        width: float = 1.0,
        tau_min: float = 2.0,
        tau_max: float = 50.0,
        beta_max: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.d_space = d_space
        self.width = width
        self.beta_max = beta_max

        # Local environment aggregator
        self.environment_field = LocalEnvironmentField(dim=dim, d_space=d_space, width=width)
        d_env = 1 + dim + 1 + d_space

        # GDN-2 inspired multi-timescale initial gamma bias
        # tau is log-spaced from tau_min to tau_max tokens
        taus = tau_min * (tau_max / tau_min) ** (torch.linspace(0, 1, n_particles))
        deltas = 2.0 / taus  # Target initial decay rate: e^{-gamma * tau / 2} = e^{-1}
        biases = torch.log(torch.expm1(deltas))
        self.gamma_bias = nn.Parameter(biases.unsqueeze(-1))  # [N, 1]

        # Shared conditional encoder: c_i = Encoder([u_t, z_i, e_t(x_i)])
        in_dim = hidden_dim + 2 * dim + d_env
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output Heads:
        # 1. Adaptive gamma modulation: zero-initialized to preserve multi-timescale bias at start
        self.gamma_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # 2. Write gate w_i in (0, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 0.0)

        # 3. Write content Delta z_i: zero-initialized to guarantee exact identity at start
        self.write_head = nn.Linear(hidden_dim, 2 * dim)
        nn.init.zeros_(self.write_head.weight)
        nn.init.zeros_(self.write_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        token_embedding: torch.Tensor,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        """Execute state-conditioned write and adaptive gamma computation.

        x: [N, d]
        v: [N, d]
        token_embedding: [H]
        Returns:
            x_new: [N, d]
            v_new: [N, d]
            gamma: [N, 1] (per-particle damping rate for subsequent substep ODE kicks)
            diag: dict | None
        """
        N, d = x.shape
        z = torch.cat([x, v], dim=-1)           # [N, 2d]
        e_t = self.environment_field(x, v)      # [N, d_env]
        tok_exp = token_embedding.unsqueeze(0).expand(N, -1)  # [N, H]

        enc_in = torch.cat([tok_exp, z, e_t], dim=-1)
        c = self.encoder(enc_in)  # [N, H]

        # Multi-timescale adaptive gamma
        gamma = F.softplus(self.gamma_bias + self.gamma_head(c))  # [N, 1]

        # Write gate w_i in (0, 1)
        w = torch.sigmoid(self.gate_head(c))  # [N, 1]

        # Bounded write content Delta z_i
        raw_dz = self.write_head(c)  # [N, 2d]
        norm_dz = raw_dz.norm(dim=-1, keepdim=True)
        bounded_dz = self.beta_max * raw_dz / torch.sqrt(1.0 + norm_dz.square())

        # State update (identity at init since write_head is zero-initialized)
        z_new = z + w * bounded_dz
        x_new = z_new[:, :d]
        v_new = z_new[:, d:]

        diag = None
        if diagnostics:
            diag = {
                'gamma_mean': float(gamma.mean().item()),
                'gamma_min': float(gamma.min().item()),
                'gamma_max': float(gamma.max().item()),
                'write_gate_mean': float(w.mean().item()),
                'write_norm_mean': float(bounded_dz.norm(dim=-1).mean().item()),
            }

        return x_new, v_new, gamma, diag
