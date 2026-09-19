"""Characteristic Kernel and Dynamic Linear Readout Probes for CBIM Physical Field.

Theoretical Principle:
"State is the world; Readout is the measuring instrument."
The recurrent Boltzmann physical field F_t in R^{256 x 128} is an immutable, all-order continuous
dynamical medium. Readout does not modify or compress F_t; it acts as a dynamic measurement probe.

Probes:
1. DynamicLinearReadout (Arm B):
   - Token-conditioned dynamic queries Q(x_t).
   - Standard linear key/value projections over z_i = [f_i, p_i] in R^134.
   - Scaled dot-product attention + token residual.

2. CharacteristicKernelReadout (Arms C & D):
   - Characteristic Gaussian kernel K_{hi} = exp(-||z_i - c_h||^2 / (2 \sigma_h^2)) in R^134.
     Embeds the empirical distribution injectively into RKHS (infinite-order moment representation).
   - Three physical readings per probe:
     a. Mean field reading: r_h = \sum_i \alpha_{hi} f_i (local expectation)
     b. Kernel response / partition function: s_h = log(\sum_i K_{hi} + \epsilon) (match evidence)
     c. Local fluctuation / uncertainty: e_h = \sum_i \alpha_{hi} ||f_i - r_h||^2 (phase coherence)
   - Recurrent measurement controller over R rounds (R=1 for Arm C, R=2 for Arm D):
     u_0 = RMSNorm(x_t)
     u_{r+1} = u_r + SwiGLU(W_m M_r + W_u u_r)
     c_{r+1}, \sigma_{r+1} = Q(u_{r+1})
   - Diagnostic instrumentation: probe attention entropy, logZ, spatial overlap, round query delta.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.cbim_torus3d import torus_grid, torus_features


class DynamicLinearReadout(nn.Module):
    """Arm B: Dynamic query Q(x_t) + linear multi-head attention + token residual."""

    def __init__(
        self,
        shape: Tuple[int, int, int] = (8, 8, 4),
        d: int = 128,
        heads: int = 8,
    ):
        super().__init__()
        self.shape = shape
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.nodes = math.prod(shape)

        coords = torus_grid(shape)
        pos_feat = torus_features(coords).reshape(self.nodes, 6)
        self.register_buffer("pos_features", pos_feat, persistent=False)

        self.z_dim = d + 6  # 134

        # Token-conditioned dynamic queries
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(self.z_dim, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)

        self.out_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        self.norm_u = nn.RMSNorm(d)

    def forward(
        self, field: torch.Tensor, token_embed: torch.Tensor, return_diag: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        B = field.shape[0]
        flat_field = field.reshape(B, self.nodes, self.d)
        pos = self.pos_features[None].expand(B, -1, -1)
        z = torch.cat([flat_field, pos], dim=-1)  # [B, N, 134]

        # Multi-head dynamic Q, K, V
        q = self.q_proj(token_embed).view(B, 1, self.heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d_h]
        k = self.k_proj(z).view(B, self.nodes, self.heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_h]
        v = self.v_proj(flat_field).view(B, self.nodes, self.heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_h]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, 1, N]
        alpha = torch.softmax(scores, dim=-1)  # [B, H, 1, N]
        read = torch.matmul(alpha, v).transpose(1, 2).reshape(B, self.d)  # [B, d]

        # Normalized token scale matches CBIM decoder operating point (~1.12 norm)
        tok_scaled = F.rms_norm(token_embed, (self.d,)) * 0.1
        h_t = tok_scaled + (self.out_proj(self.norm_u(read)) * 0.1)

        alpha_squeezed = alpha.squeeze(2)  # [B, H, N]
        self.last_entropy = (
            -(alpha_squeezed * torch.log(alpha_squeezed + 1e-12))
            .sum(dim=-1)
            .mean()
        ).detach()

        diag = None
        if return_diag:
            alpha_squeezed = alpha.squeeze(2)  # [B, H, N]
            entropy = (
                -(alpha_squeezed * torch.log(alpha_squeezed + 1e-12))
                .sum(dim=-1)
                .mean()
                .item()
            )
            # Spatial overlap between heads
            alpha_norm = F.normalize(alpha_squeezed, dim=-1)
            overlap_mat = torch.matmul(alpha_norm, alpha_norm.transpose(-1, -2))
            eye = torch.eye(self.heads, device=field.device)[None]
            overlap = (
                ((overlap_mat * (1.0 - eye)).sum(dim=(-1, -2)) / (self.heads * (self.heads - 1)))
                .mean()
                .item()
            )
            diag = {
                "probe_attention_entropy": entropy,
                "probe_spatial_overlap": overlap,
                "kernel_response_logZ": 0.0,
                "round_query_delta": 0.0,
            }

        return h_t, diag


class CharacteristicKernelReadout(nn.Module):
    """16-Channel Decoupled Key/Value Characteristic Readout with Multi-Scale Hierarchy (3 Pillars).

    Key principles:
    1. Key & Value Strict Decoupling (W_K != W_V):
       - self.k_proj: solely generates Key representations for spatial & content attention routing (alpha).
       - self.v_proj: solely generates Value representations for linear reading (r) and wave variance (e).
       Eliminates all quadratic feedback loops while translating physical velocity states to semantic spaces.
    2. 1024-Dimensional RKHS Characteristic Moment Representations:
       - 16 channels x 32 dims first-order mean field R (512 dims)
       - 16 channels x 32 dims second-order channel-wise wave variance E (512 dims)
       - Total m = [R, E] in R^1024, fully preserving the multi-body fluctuation spectrum.
    3. Multi-Scale Spatial Temperature Hierarchy:
       - 16 stationary learnable probes on a 2x2x4 lattice covering the (8, 8, 4) torus.
       - Head 0: beta=14.0 (sharp needle detectors: ~13 nodes)
       - Head 1: beta=8.0 (fine regional detectors: ~30 nodes)
       - Head 2: beta=4.0 (medium regional detectors: ~97 nodes)
       - Head 3: beta=2.0 (broad global background coverage: ~188 nodes, 0% blind spots from Step 0!)
    4. Arm-A-Style Post-Readout Mixing Block: Dense 2-layer MLP (merge 1024 -> 128 -> SiLU -> 128).
    """

    def __init__(
        self,
        shape: Tuple[int, int, int] = (8, 8, 4),
        d: int = 128,
        heads: int = 4,
        queries: int = 4,
        num_probes: Optional[int] = None,
        rounds: int = 1,
    ):
        super().__init__()
        self.shape = shape
        self.d = d
        if num_probes is not None and num_probes != heads * queries:
            if num_probes in (4, 8) and queries == 1:
                heads = num_probes
                queries = 1
            elif num_probes == 16:
                heads = 4
                queries = 4
        self.heads = heads
        self.queries = queries
        self.total_channels = heads * queries
        if d % heads != 0:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads})")
        self.d_h = d // heads  # 128 // 4 = 32
        self.rounds = rounds
        self.nodes = math.prod(shape)

        coords = torus_grid(shape)
        pos_feat = torus_features(coords).reshape(self.nodes, 6)
        self.register_buffer("pos_features", pos_feat, persistent=False)

        # 1024-dimensional RKHS moment representation: R (512) + E (512)
        self.m_total = self.total_channels * self.d_h * 2  # 16 * 32 * 2 = 1024

        # Stationary Learnable Spatial Probe Coordinates on T^3 [heads, queries, 3]
        init_pts = []
        for z in [0.125, 0.375, 0.625, 0.875]:
            for x in [0.25, 0.75]:
                for y in [0.25, 0.75]:
                    init_pts.append(torch.tensor([x, y, z], dtype=torch.float32))
        init_coords = torch.stack(init_pts).reshape(heads, queries, 3)
        self.probe_coords = nn.Parameter(init_coords)

        # Multi-scale temperature hierarchy across the 4 heads:
        # Head 0: beta=14.0 (sharp), Head 1: beta=8.0 (fine), Head 2: beta=4.0 (medium), Head 3: beta=2.0 (broad background)
        init_betas = torch.tensor([
            [math.log(14.0)],
            [math.log(8.0)],
            [math.log(4.0)],
            [math.log(2.0)],
        ]).expand(heads, queries).unsqueeze(-1)
        self.pos_log_scale = nn.Parameter(init_betas.clone())

        # Dynamic channel query (semantic listening in each 32-dim subspace)
        self.w_q = nn.Linear(d, heads * queries * self.d_h)
        nn.init.normal_(self.w_q.weight, std=1e-3)
        nn.init.zeros_(self.w_q.bias)

        self.field_norm = nn.RMSNorm(d)

        # Strictly decoupled Key and Value projections
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)

        # Arm-A-Style Post-Readout Mixing Block: merge (1024 -> 128) -> SiLU -> output (128 -> 128)
        self.merge = nn.Linear(self.m_total, d)
        self.output = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d)
        )
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def compute_spatial_score(self) -> torch.Tensor:
        """Stationary spatial alignment across 16 probes and 256 nodes."""
        probe_pos = torus_features(torch.remainder(self.probe_coords, 1.0))  # [H, Q, 6]
        probe_pos_hat = F.normalize(probe_pos, dim=-1)  # [H, Q, 6]
        pos_hat = F.normalize(self.pos_features, dim=-1)  # [N, 6]
        return torch.einsum("hqk,nk->hqn", probe_pos_hat, pos_hat)  # [H, Q, N]

    def measure(
        self, key_h: torch.Tensor, val_h: torch.Tensor, q_field: torch.Tensor,
        spatial_score: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Measure higher-order characteristic stats with decoupled K/V across all 16 channels."""
        B, H, N, d_h = key_h.shape
        Q = self.queries
        # 1. Stationary spatial alignment (passed in or computed once)
        if spatial_score is None:
            spatial_score = self.compute_spatial_score()
        spatial_score = spatial_score.to(device=key_h.device, dtype=key_h.dtype)  # [H, Q, N]

        # 2. Dynamic channel alignment with decoupled normalized Key & normalized Query (QK-Norm)
        k_hat = F.normalize(key_h, dim=-1)  # [B, H, N, d_h]
        q_hat = F.normalize(q_field, dim=-1)  # [B, H, Q, d_h]
        content_score = torch.einsum("bhqd,bhnd->bhqn", q_hat, k_hat)  # [B, H, Q, N] in [-1, 1]

        # 3. Total score = multi-scale temperature * (spatial anchor + content listening)
        scores = self.pos_log_scale.exp().unsqueeze(0) * (spatial_score.unsqueeze(0) + content_score)  # [B, H, Q, N]
        alpha = torch.softmax(scores, dim=-1)  # [B, H, Q, N]

        # A. Expected field reading: sum_i alpha_{hqi} val_{hi} (1st moment, 512 dims)
        r = torch.einsum("bhqn,bhnd->bhqd", alpha, val_h)  # [B, H, Q, d_h]

        # B. Second moment: channel-wise vector variance via Var(X) = E[X^2] - (E[X])^2
        e = (torch.einsum("bhqn,bhnd->bhqd", alpha, val_h.square()) - r.square()).clamp_min(0.0)  # [B, H, Q, d_h]

        R = r.reshape(B, self.heads * self.queries * self.d_h)  # [B, 512]
        E = e.reshape(B, self.heads * self.queries * self.d_h)  # [B, 512]
        m = torch.cat([R, E], dim=-1)  # [B, 1024]
        return m, alpha, scores.amax(dim=-1, keepdim=True)

    def forward(
        self, field: torch.Tensor, token_embed: torch.Tensor, return_diag: bool = False,
        spatial_score: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        B = field.shape[0]
        flat_field = field.reshape(B, self.nodes, self.d)
        normed_field = self.field_norm(flat_field)

        # Key: solely for attention routing
        keys = self.k_proj(normed_field)
        key_h = keys.reshape(B, self.nodes, self.heads, self.d_h).transpose(1, 2)

        # Value: solely for read values and wave variance extraction
        values = self.v_proj(normed_field)
        val_h = values.reshape(B, self.nodes, self.heads, self.d_h).transpose(1, 2)

        u = F.rms_norm(token_embed, (self.d,))
        q_field = self.w_q(u).view(B, self.heads, self.queries, self.d_h)
        m, alpha, s = self.measure(key_h, val_h, q_field, spatial_score=spatial_score)

        # Arm-A-Style Cross-Channel Dense Mixing (merge 1024 -> 128 -> SiLU -> 128)
        h_t = self.output(self.merge(m))

        diag = None
        if return_diag:
            flat_alpha = alpha.reshape(B, self.total_channels, self.nodes)
            self.last_entropy = (
                -(flat_alpha * torch.log(flat_alpha + 1e-12))
                .sum(dim=-1)
                .mean()
            ).detach()
            entropy = float(self.last_entropy.item())
            alpha_norm = F.normalize(flat_alpha, dim=-1)
            overlap_mat = torch.matmul(alpha_norm, alpha_norm.transpose(-1, -2))
            eye = torch.eye(self.total_channels, device=field.device)[None]
            overlap = (
                (
                    (overlap_mat * (1.0 - eye)).sum(dim=(-1, -2))
                    / max(self.total_channels * (self.total_channels - 1), 1)
                )
                .mean()
                .item()
            )
            diag = {
                "read_attention_entropy": entropy,
                "read_spatial_overlap": overlap,
                "read_score_max": s.mean().item(),
                "read_r_norm": r.norm(dim=-1).mean().item(),
                "read_e_norm": e.norm(dim=-1).mean().item(),
            }

        return h_t, diag
