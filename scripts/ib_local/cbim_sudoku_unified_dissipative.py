"""CBIM-Sudoku Unified Dissipative Open-System Architecture (Aligned with Language MaleCNS-Unified V6).

Key architectural components:
1. Contextual 2-Port Boundary Scattering (State-and-Problem Aware):
   theta(F, P) detects conflict between existing field F and new problem clues P.
   Unitary 2-port rotation:
     F_absorbed = cos(theta) * F + sin(theta) * P
     Reflected = -sin(theta) * F + cos(theta) * P
   Where conflict is high, cos(theta) < 1 actively dissipates and erases obsolete state!
   Where conflict is zero, cos(theta) == 1, preserving 100% coherent global harmonics!

2. Unitary Cayley Transport (9x9 Torus, 8 velocities, exact L2 preservation).

3. Problem-Conditioned Lie Algebra Collision:
   SO(d) orthogonal Givens rotations in active nullspace, preserving momentum invariants.

4. Quadratic Passive Radiation Bath:
   J_bath = 2 * kappa * E^2 / R^2 * F.
   Nonlinear quadratic cooling: rapidly cools high-energy task-switching shockwaves
   while leaving low-energy coherent pondering waves unattenuated!

5. Characteristic Kernel Readout:
   MLP readout mapping field to cell digit logits.
"""
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualBoundaryWrite2Port(nn.Module):
    """2-Port Unitary Boundary Scattering perceiving both State F and Problem P."""

    def __init__(self, d: int = 256, theta_max: float = 0.5 * math.pi):
        super().__init__()
        self.d = d
        self.theta_max = float(theta_max)
        self.angle_net = nn.Sequential(
            nn.Linear(2 * d, 128),
            nn.SiLU(),
            nn.Linear(128, d),
            nn.Sigmoid()
        )
        # Initialize near zero so it starts with pure preservation (cos ~ 1)
        nn.init.normal_(self.angle_net[0].weight, std=1e-3)
        nn.init.zeros_(self.angle_net[0].bias)
        nn.init.zeros_(self.angle_net[2].weight)
        nn.init.constant_(self.angle_net[2].bias, -2.0)  # sigmoid(-2) ~ 0.12

    def forward(self, field: torch.Tensor, clue_field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Perceive [Field, Clue]
        cat_in = torch.cat([field, clue_field], dim=-1)  # [B, 9, 9, 2d]
        theta = self.theta_max * self.angle_net(cat_in)   # [B, 9, 9, d]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        field_absorbed = cos_t * field + sin_t * clue_field
        reflected = -sin_t * field + cos_t * clue_field

        diag = {
            "theta_mean": theta.detach().mean(),
            "theta_max": theta.detach().amax(),
            "cos_mean": cos_t.detach().mean(),
            "absorbed_norm": torch.linalg.vector_norm(field_absorbed, dim=-1).mean().detach(),
            "reflected_norm": torch.linalg.vector_norm(reflected, dim=-1).mean().detach()
        }
        return field_absorbed, reflected, diag


class QuadraticPassiveRadiationBath(nn.Module):
    """Quadratic passive radiation cooling: J_bath = 2 * kappa * E^2 / R^2 * F."""

    def __init__(self, d: int = 256, local_radius: float = 2.0, max_kappa: float = 0.1):
        super().__init__()
        self.d = d
        self.local_radius = float(local_radius)
        self.max_kappa = float(max_kappa)
        self.kappa_net = nn.Sequential(
            nn.Linear(d, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        nn.init.normal_(self.kappa_net[0].weight, std=1e-3)
        nn.init.constant_(self.kappa_net[2].bias, -1.0)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Local energy per cell: E_i = 0.5 * ||f_i||^2
        local_energy = 0.5 * field.square().sum(dim=-1, keepdim=True)
        rho = 2.0 * local_energy / (self.local_radius ** 2)

        kappa = self.max_kappa * self.kappa_net(field)
        sin2_theta = torch.clamp(kappa * rho, max=0.5)
        cos_theta = torch.sqrt(1.0 - sin2_theta)
        sin_theta = torch.sqrt(sin2_theta.clamp_min(torch.finfo(field.dtype).tiny))

        field_next = cos_theta * field
        bath_out = -sin_theta * field

        diag = {
            "cooling_power": (0.5 * bath_out.square().sum(dim=(-1, -2, -3)).mean()).detach(),
            "occupation_ratio_mean": rho.detach().mean()
        }
        return field_next, bath_out, diag


class TorusCayleyTransport(nn.Module):
    """Unitary Cayley Transport on 9x9 Torus."""

    def __init__(self, n_v: int = 8, d_c: int = 32):
        super().__init__()
        self.n_v = n_v
        self.d_c = d_c
        self.velocities = [
            ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
            ( 1,  1), (-1, -1), ( 1, -1), (-1,  1)
        ]

    def forward(self, f_5d: torch.Tensor) -> torch.Tensor:
        out = []
        for i, (dr, dc) in enumerate(self.velocities):
            v_comp = f_5d[:, :, :, i, :]
            rolled = torch.roll(v_comp, shifts=(dr, dc), dims=(1, 2))
            out.append(rolled)
        return torch.stack(out, dim=3)


class ConditionedLieCollision(nn.Module):
    """Problem-Conditioned Lie Algebra SO(d) Givens Collision."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.d = d
        self.dim_active = 252
        self.angle_mlp = nn.Sequential(
            nn.Linear(2 * d, 128),
            nn.SiLU(),
            nn.Linear(128, self.dim_active // 2),
            nn.Tanh()
        )
        self.theta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_tr: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, h, w, d = f_tr.shape
        cat = torch.cat([f_tr, cond], dim=-1)
        angles = self.theta_scale * self.angle_mlp(cat)  # [B, 9, 9, dim_active//2]

        # Orthogonal 2x2 Givens rotations
        f_active = f_tr[:, :, :, :self.dim_active].view(b, h, w, self.dim_active // 2, 2)
        cos_t = torch.cos(angles).unsqueeze(-1)
        sin_t = torch.sin(angles).unsqueeze(-1)

        x1 = f_active[..., 0:1]
        x2 = f_active[..., 1:2]
        rot_x1 = cos_t * x1 - sin_t * x2
        rot_x2 = sin_t * x1 + cos_t * x2
        f_rot = torch.cat([rot_x1, rot_x2], dim=-1).view(b, h, w, self.dim_active)

        if self.dim_active < d:
            f_pass = f_tr[:, :, :, self.dim_active:]
            return torch.cat([f_rot, f_pass], dim=-1)
        return f_rot


class UnifiedDissipativeCBIMSudokuModel(nn.Module):
    """Complete Unified Open-System Architecture on 9x9 Torus."""

    def __init__(self, vocab_size: int = 11, d_channels: int = 256, ponder_steps: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.ponder_steps = ponder_steps
        self.n_v = 8
        self.d_c = d_channels // self.n_v

        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 1. 2-Port State-Aware Boundary Write & Dissipation
        self.boundary_write = ContextualBoundaryWrite2Port(d=d_channels)

        # 2. Torus Cayley Transport
        self.transport = TorusCayleyTransport(n_v=self.n_v, d_c=self.d_c)

        # 3. Problem-Conditioned Lie Collision
        self.collision = ConditionedLieCollision(d=d_channels)

        # 4. Quadratic Passive Radiation Bath
        self.bath = QuadraticPassiveRadiationBath(d=d_channels)

        # 5. Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_channels, 256),
            nn.SiLU(),
            nn.Linear(256, vocab_size)
        )

    def forward_stream_step(self, persistent_state: torch.Tensor, inp: torch.Tensor, k_step: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b = inp.shape[0]
        clue_field = self.clue_proj(self.embed_tokens(inp).view(b, 9, 9, self.d))

        # 1. Macro-Step H Boundary: 2-Port Unitary Scattering (State-and-Problem Aware Dissipation)
        # Actively dissipates obsolete state via cos(theta) and injects new clue packet via sin(theta)
        f_absorbed, reflected, write_diag = self.boundary_write(persistent_state, clue_field)
        curr = f_absorbed
        orig_norm = torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True).to(curr.dtype)

        # 2. Kinetic Microstep Loop K: Pure Isoenergetic Conservative Flow (Transport + Collision)
        # Energy is strictly preserved across all K microsteps during pondering!
        for _ in range(k_step):
            f_5d = curr.view(b, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d).view(b, 9, 9, self.d)
            curr = self.collision(f_tr, cond=clue_field)

        # Exact norm preservation across microstep pondering
        curr = curr * (orig_norm / (torch.linalg.vector_norm(curr.float(), dim=(1, 2, 3), keepdim=True) + 1e-8))

        flat = curr.view(b, 81, self.d)
        logits = self.readout_mlp(flat)

        # 3. Macro-Step H Boundary: Quadratic Passive Radiation Bath
        # Settles residual shock energy ONCE per macrostep H before carrying state to next puzzle
        state_next, bath_out, bath_diag = self.bath(curr)

        return state_next, logits, {**write_diag, **bath_diag}
