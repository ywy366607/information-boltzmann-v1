"""CBIM Master Baseline: Explicit Branch Axis Multi-Path Architecture with Unbounded 1/K Horizon Support.

Consolidates all verified breakthroughs into a single canonical module:
1. Multi-Branch Parallel Hypotheses: F in R^{M x 9 x 9 x d}.
   - M in {1, 2, 4, 8} branches initialized with Walsh-Hadamard orthogonal carrier seeds Z_h.
   - Vectorized shared-parameter nonlinear physical evolution.
   - Self-Scoring Confidence Head + Decision-level softmax mixture aggregation.
2. Exact Complex Exponential Spectral Advection on 9x9 Torus: M(dt) = exp(-i * dt * omega).
3. Continuous Time-Step Scaling: dt = 1.0 / K.
4. Transolver++ Mass-Normalized Soft Integral Readout: tok = sum(w*V) / sum(w).
5. 2-Port State-and-Problem Aware Boundary Scattering theta(F, P).
6. Quadratic Passive Radiation Bath for inter-macrostep stability.
7. Strictly Never-Reset single-stream execution.
"""
import math
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard

from scripts.ib_local.cbim_sudoku_unified_dissipative import (
    ContextualBoundaryWrite2Port,
    QuadraticPassiveRadiationBath
)
from scripts.ib_local.train_stage1_dt_scaling_3000 import ContinuousConditionedLieCollision
from scripts.ib_local.train_stage2_exact_exponential_3000 import ExactExponentialTorusTransport
from scripts.ib_local.train_stage3_mass_norm_readout_3000 import MassNormalizedSoftIntegralReadout


def get_orthogonal_hadamard_codes(M: int, d: int = 256, device: str = "cuda") -> torch.Tensor:
    """Generates M orthogonal Hadamard code vectors in {-1, +1}^d."""
    H = hadamard(d)[:M]  # [M, d]
    tensor_codes = torch.as_tensor(H, dtype=torch.float32, device=device)
    return tensor_codes.view(M, 1, 1, d)  # [M, 1, 1, d]


class CBIMMasterBaselineModel(nn.Module):
    """Canonical Master Baseline Architecture for CBIM."""

    def __init__(self, vocab_size: int = 11, d_channels: int = 256, alpha_max: float = 0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_channels
        self.n_v = 8
        self.d_c = d_channels // self.n_v
        self.alpha_max = float(alpha_max)

        self.embed_tokens = nn.Embedding(vocab_size, d_channels)
        self.clue_proj = nn.Sequential(
            nn.Linear(d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )

        # 1. H-Boundary 2-Port State-Aware Scattering theta(F, P)
        self.boundary_write = ContextualBoundaryWrite2Port(d=d_channels)

        # 2. Exact Complex Exponential Transport on 9x9 Torus
        self.transport = ExactExponentialTorusTransport(n_v=self.n_v, d_c=self.d_c)

        # 3. Continuous Lie Collision (Synchronized with dt!)
        self.collision = ContinuousConditionedLieCollision(d=d_channels)

        # 4. Continuous Arm C Corrective Flow (Synchronized with dt!)
        self.corrective_net = nn.Sequential(
            nn.Linear(2 * d_channels, d_channels),
            nn.SiLU(),
            nn.Linear(d_channels, d_channels)
        )
        self.angle_gate = nn.Sequential(
            nn.Linear(2 * d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.corrective_net[-1].weight)
        nn.init.zeros_(self.corrective_net[-1].bias)
        nn.init.zeros_(self.angle_gate[-1].weight)
        nn.init.constant_(self.angle_gate[-1].bias, -2.0)

        # 5. H-Boundary Quadratic Passive Bath
        self.bath = QuadraticPassiveRadiationBath(d=d_channels)

        # 6. Transolver++ Mass-Normalized Soft Integral Readout
        self.readout = MassNormalizedSoftIntegralReadout(d_channels=d_channels, vocab_size=vocab_size)

        # 7. Self-Scoring Branch Confidence Head
        self.branch_scorer = nn.Sequential(
            nn.Linear(d_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward_stream_step(self, persistent_state: torch.Tensor, inp: torch.Tensor,
                            M: int, k_step: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device = inp.device
        clue_field = self.clue_proj(self.embed_tokens(inp).view(1, 9, 9, self.d))  # [1, 9, 9, d]

        # 1. Macro-Step H Boundary: 2-Port Unitary Scattering theta(F, P)
        f_base, reflected, write_diag = self.boundary_write(persistent_state, clue_field)

        # 2. Orthogonal Exploration Seeds (Walsh-Hadamard Codes)
        if M == 1:
            branches = f_base  # [1, 9, 9, d]
        else:
            codes = get_orthogonal_hadamard_codes(M, d=self.d, device=device)  # [M, 1, 1, d]
            branch_seeds = (clue_field * codes) / math.sqrt(M)
            branches = f_base.expand(M, -1, -1, -1) + 0.15 * branch_seeds  # [M, 9, 9, d]

        orig_norm = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True).to(branches.dtype)
        dt = 1.0 / max(1, k_step)
        clue_expanded = clue_field.expand(M, -1, -1, -1)

        # 3. Independent Nonlinear Dynamics for all M branches
        for _ in range(k_step):
            f_5d = branches.view(M, 9, 9, self.n_v, self.d_c)
            f_tr = self.transport(f_5d, dt=dt).view(M, 9, 9, self.d)
            f_star = self.collision(f_tr, cond=clue_expanded, dt=dt)

            cat_in = torch.cat([f_star, clue_expanded], dim=-1)
            v_k = self.corrective_net(cat_in)
            dot_p = torch.sum(f_star.float() * v_k.float(), dim=(1, 2, 3), keepdim=True)
            norm_sq = torch.sum(f_star.float() ** 2, dim=(1, 2, 3), keepdim=True) + 1e-8
            u_k = v_k.float() - (dot_p / norm_sq) * f_star.float()
            u_hat = u_k / (torch.linalg.vector_norm(u_k, dim=(1, 2, 3), keepdim=True) + 1e-8)

            gate_val = torch.sigmoid(torch.mean(self.angle_gate(cat_in), dim=(1, 2, 3), keepdim=True))
            alpha_k = (dt * self.alpha_max) * gate_val.to(branches.dtype)

            f_norm_step = torch.linalg.vector_norm(f_star.float(), dim=(1, 2, 3), keepdim=True)
            branches = torch.cos(alpha_k) * f_star + f_norm_step * torch.sin(alpha_k) * u_hat.to(branches.dtype)

        # Norm preservation per branch
        branch_norms = torch.linalg.vector_norm(branches.float(), dim=(1, 2, 3), keepdim=True) + 1e-8
        branches = branches * (orig_norm / branch_norms)

        # 4. Independent Readout & Decision-Level Mixture
        logits_branches = self.readout(branches)  # [M, 81, 11]
        probs_branches = F.softmax(logits_branches, dim=-1)

        if M == 1:
            final_probs = probs_branches
            best_branch = branches
        else:
            # Self-Scoring Confidence
            pooled_feats = branches.mean(dim=(1, 2))
            scores = self.branch_scorer(pooled_feats).squeeze(-1)
            pi = F.softmax(scores, dim=0).view(M, 1, 1)

            final_probs = (probs_branches * pi).sum(dim=0, keepdim=True)
            best_idx = torch.argmax(scores)
            best_branch = branches[best_idx:best_idx+1]

        final_log_probs = torch.log(final_probs.clamp_min(1e-12))

        # 5. Macro-Step H Exit: Quadratic Passive Bath
        state_next, bath_out, bath_diag = self.bath(best_branch)

        return state_next, final_log_probs, {**write_diag, **bath_diag}
