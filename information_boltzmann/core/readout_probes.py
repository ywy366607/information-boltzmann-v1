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

from .torus3d import torus_grid, torus_features


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


class PredictivePhysicalReadAgent(nn.Module):
    """Field-only read policy over a complete local kinetic coordinate system.

    The action is a distribution over a finite Fourier--Galerkin atlas of
    continuous torus apertures.  It is selected from the posterior field and
    its channel precision, never from the current token or the reflected
    boundary wave.  At every spatial cell the measured value is expressed in
    an orthogonal decomposition::

        f = C m + N g,

    where ``C`` spans the collision invariants and ``N`` is the collision
    nullspace.  Thus the value coordinates ``[m, g]`` are an invertible
    reparameterization of the local field: conserved content and collision
    changes are both observable before learned semantic mixing.

    The atlas bandwidth follows its physical cell volume rather than a
    hand-selected attention radius.  Softmax mixtures keep every aperture
    strictly positive, so a finite read policy has no exact spatial blind
    spot at initialization.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        d: int,
        nullspace: torch.Tensor,
        *,
        heads: int = 4,
        queries: int = 4,
        atlas_shape: Tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        if d % heads:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads})")
        if nullspace.ndim != 2 or nullspace.shape[0] != d:
            raise ValueError("nullspace must be [d, collision_nullity]")
        self.shape, self.d = tuple(shape), int(d)
        self.heads, self.queries = int(heads), int(queries)
        self.head_dim = d // heads
        self.nodes = math.prod(shape)
        self.atlas_shape = tuple(atlas_shape)
        self.atlas_size = math.prod(self.atlas_shape)
        self.nullity = int(nullspace.shape[1])

        # ``nullspace`` comes from the collision SVD and is orthonormal.  A
        # complete QR supplies its orthogonal invariant complement C.
        q_complete, _ = torch.linalg.qr(nullspace.detach(), mode="complete")
        invariant_basis = q_complete[:, self.nullity:]
        if invariant_basis.shape[1] + self.nullity != d:
            raise RuntimeError("Kinetic coordinate decomposition is incomplete")
        self.register_buffer("nullspace", nullspace.detach(), persistent=False)
        self.register_buffer("invariant_basis", invariant_basis, persistent=False)

        coordinates = torus_grid(self.shape).reshape(self.nodes, 3)
        anchors = torus_grid(self.atlas_shape).reshape(self.atlas_size, 3)
        delta = torch.remainder(
            coordinates[None] - anchors[:, None] + 0.5, 1.0) - 0.5
        distance_sq = delta.square().sum(-1)
        # A chart cell has volume 1/A; its isotropic physical length is the
        # cubic root.  The normalized kernel produces a partition of unity
        # over the continuous torus independently of the sampled field grid.
        cell_length = float(self.atlas_size) ** (-1.0 / 3.0)
        log_kernel = -0.5 * distance_sq / (cell_length * cell_length)
        log_kernel = log_kernel - torch.logsumexp(log_kernel, dim=0, keepdim=True)
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("anchor_log_kernel", log_kernel, persistent=False)

        # The prior is built from invariant macrostate.  The posterior policy
        # may additionally use collision coordinates and persistent precision.
        invariant_dim = d - self.nullity
        self.prior_features = nn.Sequential(
            nn.Linear(2 * invariant_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.posterior_features = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_prior = nn.Linear(d, heads * queries * self.atlas_size)
        self.action_posterior = nn.Linear(d, heads * queries * self.atlas_size)
        self.q_proj = nn.Linear(d, heads * queries * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)

        # Cosine QK attention: exp(0)=1 is the ordinary unit-temperature
        # cosine model.  Temperature is entirely learned per head.
        self.head_log_scale = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.merge = nn.Linear(2 * queries * d, d, bias=False)
        self.correction = nn.Sequential(
            nn.RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

        for module in (self.prior_features, self.posterior_features):
            nn.init.normal_(module[-1].weight, std=1e-3)
            nn.init.zeros_(module[-1].bias)
        nn.init.zeros_(self.action_prior.weight)
        nn.init.zeros_(self.action_prior.bias)
        nn.init.normal_(self.action_posterior.weight, std=1e-3)
        nn.init.zeros_(self.action_posterior.bias)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        with torch.no_grad():
            self.merge.weight.zero_()
            # At initialization the first aperture is an exact physical mean
            # reader for every channel; later queries and variance are learned.
            self.merge.weight[:, :d] = torch.eye(d)
            self.correction[-1].weight.zero_()
            self.correction[-1].bias.zero_()

    def physical_coordinates(self, flat_field: torch.Tensor) -> torch.Tensor:
        """Return the exact [invariant, collision] local coordinates of f."""
        c = self.invariant_basis.to(dtype=flat_field.dtype)
        n = self.nullspace.to(dtype=flat_field.dtype)
        invariant = torch.einsum("dk,bnd->bnk", c, flat_field)
        collision = torch.einsum("dk,bnd->bnk", n, flat_field)
        return torch.cat((invariant, collision), dim=-1)

    def reconstruct_physical_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`physical_coordinates` exactly up to numerical error."""
        invariant_dim = self.invariant_basis.shape[1]
        c = self.invariant_basis.to(dtype=coordinates.dtype)
        n = self.nullspace.to(dtype=coordinates.dtype)
        return (
            torch.einsum("dk,bnk->bnd", c, coordinates[..., :invariant_dim])
            + torch.einsum("dk,bnk->bnd", n, coordinates[..., invariant_dim:]))

    @staticmethod
    def _categorical_kl(posterior: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(posterior.dtype).eps
        return (posterior * (
            posterior.clamp_min(eps).log() - prior.clamp_min(eps).log()
        )).sum(dim=-1).mean()

    def forward(
        self,
        field: torch.Tensor,
        precision: torch.Tensor,
        return_diag: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if precision.ndim != 2 or precision.shape != (field.shape[0], self.d):
            raise ValueError("precision must be [batch, d]")
        batch = field.shape[0]
        flat = field.reshape(batch, self.nodes, self.d)
        kinetic = self.physical_coordinates(flat)
        invariant_dim = self.invariant_basis.shape[1]
        invariant = kinetic[..., :invariant_dim]

        invariant_mean = invariant.mean(dim=1)
        invariant_energy = invariant.square().mean(dim=1).sqrt()
        prior_state = self.prior_features(torch.cat((
            F.rms_norm(invariant_mean, (invariant_dim,)), invariant_energy
        ), dim=-1))
        kinetic_mean = kinetic.mean(dim=1)
        kinetic_energy = kinetic.square().mean(dim=1).sqrt()
        posterior_state = self.posterior_features(torch.cat((
            F.rms_norm(kinetic_mean, (self.d,)),
            kinetic_energy * precision.clamp_min(torch.finfo(field.dtype).eps).sqrt(),
        ), dim=-1))

        prior_logits = self.action_prior(prior_state).reshape(
            batch, self.heads, self.queries, self.atlas_size)
        # q(a|F,Lambda) is a posterior correction of the invariant prior,
        # so both the predictive prior and evidence correction receive the
        # next-token likelihood gradient without adding an arbitrary KL weight.
        posterior_logits = prior_logits + self.action_posterior(posterior_state).reshape(
            batch, self.heads, self.queries, self.atlas_size)
        prior_action = torch.softmax(prior_logits, dim=-1)
        posterior_action = torch.softmax(posterior_logits, dim=-1)

        # Marginalize the categorical aperture exactly.  This is the expected
        # physical measurement under q(a_read | F, Lambda), not a sampled
        # token-conditioned selector.
        log_aperture = torch.logsumexp(
            posterior_logits[..., :, None] + self.anchor_log_kernel.to(field)[None, None, None],
            dim=-2,
        )
        keys = self.k_proj(F.rms_norm(kinetic, (self.d,))).reshape(
            batch, self.nodes, self.heads, self.head_dim).transpose(1, 2)
        query = self.q_proj(posterior_state).reshape(
            batch, self.heads, self.queries, self.head_dim)
        semantic = torch.einsum(
            "bhqd,bhnd->bhqn", F.normalize(query, dim=-1), F.normalize(keys, dim=-1))
        scores = semantic * self.head_log_scale.exp() + log_aperture
        attention = torch.softmax(scores, dim=-1)

        values = kinetic.reshape(batch, self.nodes, self.heads, self.head_dim).transpose(1, 2)
        mean = torch.einsum("bhqn,bhnd->bhqd", attention, values)
        variance = (
            torch.einsum("bhqn,bhnd->bhqd", attention, values.square())
            - mean.square()).clamp_min(0.0)
        measurement = torch.cat((
            mean.transpose(1, 2).reshape(batch, self.queries * self.d),
            variance.transpose(1, 2).reshape(batch, self.queries * self.d),
        ), dim=-1)
        base = self.merge(measurement)
        feature = base + self.correction(base)

        if not return_diag:
            return feature, None
        entropy = -(attention * attention.clamp_min(torch.finfo(field.dtype).eps).log()).sum(-1).mean()
        action_entropy = -(posterior_action * posterior_action.clamp_min(torch.finfo(field.dtype).eps).log()).sum(-1).mean()
        diag: Dict[str, torch.Tensor] = {
            "read_attention_entropy": entropy.detach(),
            "read_action_entropy": action_entropy.detach(),
            "read_action_kl": self._categorical_kl(posterior_action, prior_action).detach(),
            "read_temperature_mean": self.head_log_scale.exp().detach().mean(),
            "read_invariant_norm": invariant.detach().square().mean().sqrt(),
            "read_collision_norm": kinetic[..., invariant_dim:].detach().square().mean().sqrt(),
            "read_aperture_coverage": attention.detach().amin(dim=-1).mean(),
            # Retained inside the differentiable execution graph for a later
            # expected-free-energy action objective.  It is deliberately not
            # silently added to token likelihood here.
            "_read_action_complexity": self._categorical_kl(posterior_action, prior_action),
        }
        return feature, diag


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
