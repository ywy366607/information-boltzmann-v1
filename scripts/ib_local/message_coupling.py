"""M03: Input particle message stream + persistent particle field coupling.

Protocol:
1. Two particle roles, one persistent system:
   - Persistent particles: S_t = {(x_i, v_i)}_{i=1}^N carry working memory across tokens.
   - Input particle messages: Q_t = B_theta(X_t) = {(x_m^{(q)}, h_m^{(q)})}_{m=1}^M
     are ephemeral messages generated per token, discarded after local coupling.
2. Local coupling operator:
   S_t^+ = I_theta(S_t, Q_t; e_t)
   - Message generator B_theta produces M discrete message particles with spatial coordinates.
   - Messages interact locally with persistent particles within compact support K_h(|x_i - x_m^{(q)}|).
   - Injected momentum Delta v_i and multi-timescale gamma_i are output from a shared encoder:
     c_i = Encoder[x_i, v_i, e_t(x_i), local_msg_i, rho_i^{(q)}]
   - Zero-initialized output heads ensure exact identity at step 0.
3. Subsequent continuous dynamics:
   Persistent particles S_t^+ undergo ODE sub-steps:
   - Damping by gamma_i and thermal bath Langevin fluctuations.
   - Harmonic confining trap.
   - Kinematic transport dx/dt = v.
   - Conservative binary elastic collisions C_theta.
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

from scripts.ib_local.adaptive_force import LocalEnvironmentField


class ParticleMessageCoupling(nn.Module):
    """Local coupling operator between incoming particle messages Q_t and persistent field S_t."""
    def __init__(
        self,
        dim: int = 4,
        hidden_dim: int = 128,
        n_particles: int = 256,
        n_messages: int = 16,
        d_msg: int = 16,
        d_space: int = 8,
        width: float = 1.0,
        tau_min: float = 32.0,
        tau_max: float = 1024.0,
        dt_token: float = 1.0,
        beta_v: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_particles = n_particles
        self.n_messages = n_messages
        self.d_msg = d_msg
        self.width = width
        self.dt_token = dt_token
        self.beta_v = beta_v

        # 1. Ephemeral Message Generator Q_t = B_theta(u_t)
        self.msg_pos_proj = nn.Linear(hidden_dim, n_messages * dim)
        self.msg_feat_proj = nn.Linear(hidden_dim, n_messages * d_msg)

        # 2. Local environment field of persistent medium
        self.environment = LocalEnvironmentField(dim=dim, d_space=d_space, width=width)
        d_env = 1 + dim + 1 + d_space

        # 3. GDN-2 inspired multi-timescale gamma initialization
        taus = tau_min * (tau_max / tau_min) ** (torch.linspace(0, 1, n_particles))
        gamma_0 = 1.0 / (taus * dt_token)
        biases = torch.log(torch.expm1(gamma_0))
        self.gamma_bias = nn.Parameter(biases.unsqueeze(-1))  # [N, 1]

        # 4. Shared coupling encoder: c_i = Encoder[x_i, v_i, e_i, local_msg_i, rho_i^q]
        in_dim = 2 * dim + d_env + d_msg + 1
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 5. Output heads: Delta v and gamma
        # Injected velocity head (zero-initialized for exact identity at init)
        self.v_head = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

        # Gamma modulation head (zero-initialized so gamma matches multi-timescale bias at start)
        self.gamma_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        token_embedding: torch.Tensor,
        particle_bias: torch.Tensor | None = None,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        """Couple ephemeral message particles into persistent particles locally.

        x: [N, d]
        v: [N, d]
        token_embedding: [H]
        particle_bias: [N, 1] optional intrinsic per-particle gamma bias (permuted with particles)
        Returns:
            x_new: [N, d] (positions unchanged by message impulse)
            v_new: [N, d] (velocities updated by localized message transfer)
            gamma: [N, 1] (per-particle damping rates)
            diag: dict | None
        """
        N, d = x.shape
        M = self.n_messages

        # Step 1: Generate ephemeral message particles Q_t = B_theta(u_t)
        # Bounded positions in [-1.5, 1.5]
        msg_x = 1.5 * torch.tanh(self.msg_pos_proj(token_embedding).view(M, d))
        msg_feat = self.msg_feat_proj(token_embedding).view(M, self.d_msg)

        # Step 2: Compute local environment of persistent particles
        e_i = self.environment(x, v)  # [N, d_env]

        # Step 3: Local coupling between message particles and persistent particles
        deltas_q = (x.unsqueeze(1) - msg_x.unsqueeze(0)).abs()  # [N, M, d]
        factors_q = torch.clamp(1.0 - deltas_q / self.width, min=0.0)
        if d == 4:
            A = factors_q[..., 0] * factors_q[..., 1] * factors_q[..., 2] * factors_q[..., 3]  # [N, M]
        elif d == 2:
            A = factors_q[..., 0] * factors_q[..., 1]
        else:
            A = factors_q.prod(dim=-1)

        rho_q = A.sum(dim=-1, keepdim=True)  # [N, 1]
        denom_q = rho_q.clamp_min(1e-6)
        local_msg = (A @ msg_feat) / denom_q  # [N, d_msg]

        # Step 4: Shared condition vector
        z_i = torch.cat([x, v], dim=-1)  # [N, 2d]
        c_in = torch.cat([z_i, e_i, local_msg, rho_q], dim=-1)
        c = self.encoder(c_in)  # [N, H]

        # Step 5: Output velocity injection Delta v_i and adaptive damping gamma_i
        raw_dv = self.v_head(c)
        norm_dv = raw_dv.norm(dim=-1, keepdim=True)

        # Scale by presence of message to guarantee zero impulse far away from messages
        presence = rho_q / (rho_q + 1.0)
        delta_v = presence * self.beta_v * raw_dv / torch.sqrt(1.0 + norm_dv.square())

        v_new = v + delta_v
        x_new = x
        bias = particle_bias if particle_bias is not None else self.gamma_bias
        gamma = F.softplus(bias + self.gamma_head(c))

        diag = None
        if diagnostics:
            diag = {
                'gamma_mean': float(gamma.mean().item()),
                'gamma_min': float(gamma.min().item()),
                'gamma_max': float(gamma.max().item()),
                'delta_v_norm_mean': float(delta_v.norm(dim=-1).mean().item()),
                'message_overlap_particles_count': int((rho_q > 0.01).sum().item()),
            }

        return x_new, v_new, gamma, diag
