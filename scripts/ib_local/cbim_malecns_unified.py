"""CBIM-MaleCNS Unified Dynamic Equilibrium Architecture (v6).

Key architectural components:
1. Contextual Boundary Write:
   Full-rank state & neighborhood conditioned wavepacket with local capacity R_i
   and 2-port unitary boundary scattering with exact energy accounting.
2. Dual-Scale Directional Transport:
   Symmetric Strang splitting between local edge transport (D3Q8 velocity-aligned)
   and global low-rank Laplacian modal transport, preserving exact L2 norm.
3. Woodbury Non-Equilibrium Collision:
   State-dependent low-rank skew-symmetric generator in the 60-dim D3Q8 invariant
   nullspace, solved analytically via Sherman-Morrison-Woodbury identity.
4. Quadratic Passive Radiation Bath:
   Threshold-free quadratic dissipation J_bath = 2 * kappa * E^2 / R^2,
   guaranteeing global boundedness while keeping low-energy states coherent.
5. QK-Norm Readout:
   Multi-query spatial attention with L2 QK-norm and per-head learnable scales
   initialized to g_0 = log2(N^2 - N) approx 15.994.
"""
from __future__ import annotations

import os
import sys

# Prevent scripts/ib_local/types.py from shadowing the standard library types.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import math
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.geometric_transport import cube_velocities


class ContextualBoundaryWrite(nn.Module):
    """Full-rank state- and neighborhood-conditioned boundary scattering with local capacity."""

    def __init__(
        self,
        graph: Dict[str, torch.Tensor],
        vocab_size: int = 50257,
        d: int = 64,
        local_radius: float = 1.25,
        theta_open: float = 0.25,
    ) -> None:
        super().__init__()
        self.d = d
        self.local_radius = float(local_radius)
        self.theta_open = float(theta_open)

        coordinates = graph["coordinates"].float()
        node_features = graph["node_features"].float()
        self.nodes = coordinates.shape[0]

        self.embedding = nn.Embedding(vocab_size, d)
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)

        # Pairwise distance for safe width initialization
        pair_distance = torch.cdist(coordinates, coordinates)
        pair_distance.fill_diagonal_(float("inf"))
        reference_width = pair_distance.amin(-1).median().item()
        inv_softplus = math.log(math.expm1(reference_width))
        nn.init.zeros_(self.width.weight)
        nn.init.constant_(self.width.bias, inv_softplus)

        self.token_content = nn.Sequential(
            nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.state_content = nn.Linear(d, d, bias=False)
        self.neighbor_content = nn.Linear(d, d, bias=False)
        self.angle_net = nn.Sequential(
            nn.Linear(3 * d + node_features.shape[-1], d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        nn.init.normal_(self.token_content[-1].weight, std=1e-3)
        nn.init.zeros_(self.token_content[-1].bias)
        nn.init.normal_(self.state_content.weight, std=1e-3)
        nn.init.normal_(self.neighbor_content.weight, std=1e-3)
        nn.init.normal_(self.angle_net[-1].weight, std=1e-3)
        nn.init.constant_(self.angle_net[-1].bias, 0.0)

        self.register_buffer("coordinates", coordinates[None], persistent=False)
        self.register_buffer("node_features", node_features[None], persistent=False)

        if "neighbor_indices" in graph:
            neighbor_indices = graph["neighbor_indices"].long()
            weights = graph["neighbor_weights"].float()
        else:
            adjacency = graph["adjacency"].float()
            count = min(8, adjacency.shape[-1] - 1)
            weights, neighbor_indices = adjacency.topk(count, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("neighbor_indices", neighbor_indices, persistent=False)
        self.register_buffer("neighbor_weights", weights, persistent=False)

    @staticmethod
    def _energy(x: torch.Tensor) -> torch.Tensor:
        # Total field/port energy per sample.  Sum spatial ports so this is
        # directly comparable with the earlier MaleCNS implementations.
        return 0.5 * x.square().sum(dim=(-1, -2)).mean()

    def forward(self, field: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        batch, nodes, d = field.shape
        token = self.embedding(token_ids)  # [B, d]

        # 3D spatial Gaussian envelope
        center = torch.sigmoid(self.address(token))[:, None]  # [B, 1, 3]
        width = F.softplus(self.width(token))[:, None].clamp_min(1e-4)  # [B, 1, 3]
        dist = (self.coordinates - center) / width
        tail_limit = math.sqrt(-2.0 * math.log(torch.finfo(dist.dtype).tiny))
        safe_dist = dist.clamp(min=-tail_limit, max=tail_limit)
        envelope = torch.exp(-0.5 * safe_dist.square().sum(-1))  # [B, N] in [0, 1]

        # Context: neighbor aggregation + local state
        neighbors = field[:, self.neighbor_indices]  # [B, N, K, d]
        neighborhood = (neighbors * self.neighbor_weights[None, ..., None]).sum(2)  # [B, N, d]

        # Local content direction c_i
        c_i = (
            self.token_content(token)[:, None]
            + self.state_content(field)
            + self.neighbor_content(neighborhood)
        )
        c_i_norm = F.normalize(c_i, dim=-1)

        # Fixed total incident budget independent of address width or whether
        # a learned center falls between graph nodes.  The envelope determines
        # the spatial shape; its L2 normalization prevents accidental signal
        # starvation or amplification as that shape changes.
        spatial_norm = envelope.square().sum(-1, keepdim=True).clamp_min(1e-16).sqrt()
        spatial = envelope / spatial_norm
        packet = self.local_radius * spatial[..., None] * c_i_norm

        # Local impedance angle
        node_feats = self.node_features.expand(batch, -1, -1)
        angle_ctx = torch.cat(
            (token[:, None].expand(-1, nodes, -1), field, neighborhood, node_feats),
            dim=-1,
        )
        channel_angle = self.theta_open * torch.sigmoid(self.angle_net(angle_ctx))
        theta = envelope[..., None] * channel_angle  # Localized rotation

        cosine, sine = theta.cos(), theta.sin()
        field_next = cosine * field + sine * packet
        reflected = -sine * field + cosine * packet

        # Actual absorbed work
        e_before = self._energy(field)
        e_after = self._energy(field_next)
        j_absorb = e_after - e_before
        incident_e = self._energy(packet)
        reflected_e = self._energy(reflected)
        residual = (e_after + reflected_e - e_before - incident_e).abs()

        diagnostics = {
            "incident_energy": incident_e.detach(),
            "reflected_energy": reflected_e.detach(),
            "absorbed_power": j_absorb.detach(),
            "write_angle_abs_mean": theta.detach().abs().mean(),
            "write_balance_residual": residual.detach(),
            "envelope_mean": envelope.detach().mean(),
        }
        return field_next, reflected, diagnostics


class DualScaleGeometricTransport(nn.Module):
    """Symmetric Strang splitting between local directional edge transport and low-rank spectral modes."""

    def __init__(
        self,
        graph: Dict[str, torch.Tensor],
        velocities: int = 8,
        content_dim: int = 8,
        modes: int = 32,
    ) -> None:
        super().__init__()
        self.velocities = int(velocities)
        self.content_dim = int(content_dim)
        self.d = velocities * content_dim

        coordinates = graph["coordinates"].float()
        scale = graph.get("coordinate_scale", torch.ones(3)).float()
        nodes = coordinates.shape[0]
        self.nodes = nodes

        # 1. Local edge pairs geometry
        pairs = graph["collision_pairs"].long()
        edge_features = graph["collision_features"].float()
        physical = coordinates * scale
        displacement = physical[pairs[..., 1]] - physical[pairs[..., 0]]
        direction = displacement / displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        velocity = cube_velocities()
        projection = torch.einsum("lpk,qk->lpq", direction, velocity)
        geometry = torch.cat(
            (edge_features[..., None, :].expand(-1, -1, velocities, -1),
             projection[..., None]),
            dim=-1,
        )
        self.edge_layers = pairs.shape[0]
        self.register_buffer("pairs", pairs, persistent=False)
        self.register_buffer("geometry", geometry, persistent=False)
        # Geometry fixes the dimensionless generator; learning controls its
        # rate.  log_rate=0 means one normalized generator-time unit rather
        # than an almost-identity random angle.
        self.edge_log_rate = nn.Parameter(torch.zeros(velocities, content_dim))

        # 2. Global spectral basis
        basis = graph["laplacian_basis"][:, :modes].double()
        basis = torch.linalg.qr(basis, mode="reduced").Q
        self.modes = basis.shape[1]
        self.register_buffer("basis", basis, persistent=False)
        if "adjacency" in graph:
            adjacency = graph["adjacency"].double()
        else:
            adjacency = torch.zeros(nodes, nodes, dtype=torch.float64)
            flat_pairs = pairs.reshape(-1, 2)
            adjacency[flat_pairs[:, 0], flat_pairs[:, 1]] = 1.0
            adjacency[flat_pairs[:, 1], flat_pairs[:, 0]] = 1.0
        adjacency = .5 * (adjacency + adjacency.T)
        delta = physical.double()[None] - physical.double()[:, None]
        edge_direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        reduced_generators = []
        for vector in velocity.double():
            generator = adjacency * torch.einsum("ijk,k->ij", edge_direction, vector)
            generator = .5 * (generator - generator.T)
            reduced = basis.T @ generator @ basis
            reduced = reduced / torch.linalg.matrix_norm(reduced, ord=2).clamp_min(1e-8)
            reduced_generators.append(reduced)
        reduced_generators = torch.stack(reduced_generators)
        eigval, eigvec = torch.linalg.eigh(
            1j * reduced_generators.double().to(torch.complex128))
        self.register_buffer("spectral_eigenvalues", eigval, persistent=False)
        self.register_buffer("spectral_eigenvectors", eigvec, persistent=False)
        self.spectral_log_rate = nn.Parameter(torch.zeros(velocities, content_dim))

    def _spectral_step(self, field: torch.Tensor, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cayley rotation on paired graph Laplacian modes (modes x velocities x content)."""
        batch, nodes, v, a = field.shape
        basis = self.basis.to(dtype=field.dtype)
        coeffs = torch.einsum("mr,bmqa->bqra", basis, field)
        rate = scale * self.spectral_log_rate.exp()
        mu = self.spectral_eigenvalues.to(dtype=field.dtype)
        phase = ((1.0 - .5j * rate[:, None, :] * mu[:, :, None]) /
                 (1.0 + .5j * rate[:, None, :] * mu[:, :, None]))
        complex_dtype = (torch.complex128 if field.dtype == torch.float64
                         else torch.complex64)
        vectors = self.spectral_eigenvectors.to(dtype=complex_dtype)
        modal = torch.einsum(
            "qrs,bqsa->bqra", vectors.conj().transpose(-2, -1),
            coeffs.to(complex_dtype))
        rotated = torch.einsum(
            "qrs,bqsa->bqra", vectors, phase[None] * modal).real
        delta = torch.einsum("mr,bqra->bmqa", basis, rotated - coeffs)
        return field + delta, rate.detach().mean()

    def _edge_step(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Givens rotation on disjoint graph edges along velocity projection."""
        result = field
        angles = []
        for layer in range(self.edge_layers):
            left_idx, right_idx = self.pairs[layer].unbind(-1)
            left = result[:, left_idx]   # [B, P, v, a]
            right = result[:, right_idx]
            projection = self.geometry[layer, ..., -1:]
            theta = (projection * self.edge_log_rate.exp()[None])[None]
            cosine, sine = theta.cos(), theta.sin()
            left_new = cosine * left - sine * right
            right_new = sine * left + cosine * right
            updated = result.clone()
            updated[:, left_idx] = left_new
            updated[:, right_idx] = right_new
            result = updated
            angles.append(theta.detach().abs().mean())
        return result, torch.stack(angles).mean()

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Symmetric Strang splitting: Spectral(1/2) -> Edge(1) -> Spectral(1/2)."""
        batch, nodes, d = field.shape
        shaped = field.reshape(batch, nodes, self.velocities, self.content_dim)
        before_norm = shaped.square().sum()

        # Step 1: Half-step global spectral Cayley
        s1, theta_s1 = self._spectral_step(shaped, scale=0.5)
        # Step 2: Full-step local edge transport
        s2, theta_edge = self._edge_step(s1)
        # Step 3: Half-step global spectral Cayley
        s3, theta_s2 = self._spectral_step(s2, scale=0.5)

        after_norm = s3.square().sum()
        residual = (after_norm - before_norm).abs()

        diagnostics = {
            "transport_edge_angle_mean": theta_edge.detach(),
            "transport_spectral_angle_mean": 0.5 * (theta_s1.detach() + theta_s2.detach()),
            "transport_norm_residual": residual.detach(),
        }
        return s3.reshape(batch, nodes, d), diagnostics


def _inv_2x2(M: torch.Tensor) -> torch.Tensor:
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    inv_det = 1.0 / det.clamp_min(1e-12)
    row0 = torch.stack([M[..., 1, 1], -M[..., 0, 1]], dim=-1)
    row1 = torch.stack([-M[..., 1, 0], M[..., 0, 0]], dim=-1)
    return torch.stack([row0, row1], dim=-2) * inv_det[..., None, None]


def _inv_4x4(M: torch.Tensor) -> torch.Tensor:
    A = M[..., :2, :2]
    B = M[..., :2, 2:]
    C = M[..., 2:, :2]
    D = M[..., 2:, 2:]
    inv_A = _inv_2x2(A)
    S = D - C @ inv_A @ B
    inv_S = _inv_2x2(S)
    inv_A_B_inv_S = inv_A @ B @ inv_S
    inv_S_C_inv_A = inv_S @ C @ inv_A
    TL = inv_A + inv_A_B_inv_S @ C @ inv_A
    TR = -inv_A_B_inv_S
    BL = -inv_S_C_inv_A
    BR = inv_S
    top = torch.cat([TL, TR], dim=-1)
    bot = torch.cat([BL, BR], dim=-1)
    return torch.cat([top, bot], dim=-2)


class WoodburyKineticCollision(nn.Module):
    """Local D3Q8 collision in the invariant nullspace via rank-2r Woodbury Cayley transform."""

    def __init__(
        self,
        node_features: torch.Tensor,
        velocities: int = 8,
        content_dim: int = 8,
        rank: int = 2,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.velocities = int(velocities)
        self.content_dim = int(content_dim)
        self.d = velocities * content_dim
        self.rank = int(rank)

        # 1. Nullspace of mass and 3D momentum constraints
        velocity = cube_velocities(dtype=torch.float64)
        mass = torch.ones(1, velocities, content_dim, dtype=torch.float64)
        momentum = velocity.T[:, :, None].expand(-1, -1, content_dim)
        constraints = torch.cat((mass, momentum), 0).reshape(4, self.d)
        _, singular, right = torch.linalg.svd(constraints, full_matrices=True)
        rank_c = int((singular > 1e-10).sum())
        nullspace = right[rank_c:].T  # [d, K]
        self.nullity = nullspace.shape[1]
        self.register_buffer("constraints", constraints.float(), persistent=False)
        self.register_buffer("nullspace", nullspace.float(), persistent=False)

        # 2. Generator prediction: outputs U and V for A_C = U V^T - V U^T
        self.norm = nn.LayerNorm(self.d)
        node_dim = node_features.shape[-1]
        self.u_net = nn.Sequential(
            nn.Linear(self.d + node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.nullity * self.rank),
        )
        self.v_net = nn.Sequential(
            nn.Linear(self.d + node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.nullity * self.rank),
        )
        # If both factors start near zero, A=UV^T-VU^T and its gradient are
        # second-order tiny.  A zero U gives an identity initial collision,
        # while a non-degenerate V makes the first derivative with respect to
        # U finite so CE can wake the collision immediately.
        nn.init.zeros_(self.u_net[-1].weight)
        nn.init.zeros_(self.u_net[-1].bias)
        nn.init.xavier_uniform_(self.v_net[-1].weight)
        nn.init.zeros_(self.v_net[-1].bias)
        self.register_buffer("node_features", node_features.float()[None], persistent=False)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch, nodes, d = field.shape
        # Decompose into conserved subspace and invariant nullspace
        nullspace = self.nullspace.to(dtype=field.dtype)  # [d, K]
        z = torch.einsum("dk,bnd->bnk", nullspace, field)  # [B, N, K]
        conserved = field - torch.einsum("dk,bnk->bnd", nullspace, z)

        # Predict rank-2r factor matrices: U and V in [B, N, K, r]
        feats = self.node_features.expand(batch, -1, -1)
        generator_input = torch.cat((self.norm(field), feats), dim=-1)
        U = self.u_net(generator_input).reshape(
            batch, nodes, self.nullity, self.rank)
        V = self.v_net(generator_input).reshape(
            batch, nodes, self.nullity, self.rank)

        # Woodbury Sherman-Morrison-Woodbury formulation:
        # A_C = P Q^T with P = [U, -V] (K x 2r), Q = [V, U] (K x 2r)
        # (I - 1/2 P Q^T)^{-1} (I + 1/2 P Q^T) z = z + P (I_{2r} - 1/2 Q^T P)^{-1} Q^T z
        P = torch.cat((U, -V), dim=-1)  # [B, N, K, 2r]
        Q = torch.cat((V, U), dim=-1)   # [B, N, K, 2r]

        # Inner matrix C = I_{2r} - 0.5 Q^T P in [B, N, 2r, 2r]
        Q_T_P = torch.einsum("bnkp,bnkq->bnpq", Q, P)  # [B, N, 2r, 2r]
        eye = torch.eye(2 * self.rank, device=field.device, dtype=field.dtype)[None, None]
        C = eye - 0.5 * Q_T_P

        # Invert the tiny 2r x 2r matrix (rank=2 -> 4x4) analytically for CUDA Graph
        inv_C = _inv_4x4(C) if self.rank == 2 else torch.linalg.inv(C)

        # Compute Woodbury update: delta_z = P @ inv_C @ (Q^T @ z)
        Q_T_z = torch.einsum("bnkp,bnk->bnp", Q, z)  # [B, N, 2r]
        sol = torch.einsum("bnpq,bnq->bnp", inv_C, Q_T_z)  # [B, N, 2r]
        z_next = z + torch.einsum("bnkp,bnp->bnk", P, sol)  # [B, N, K]

        # Reconstruct full field
        output = conserved + torch.einsum("dk,bnk->bnd", nullspace, z_next)

        # Check exact norm & constraint invariants
        norm_diff = (z_next.square().sum(-1) - z.square().sum(-1)).abs().amax()
        gram_u = torch.einsum("bnkr,bnks->bnrs", U, U)
        gram_v = torch.einsum("bnkr,bnks->bnrs", V, V)
        cross = torch.einsum("bnkr,bnks->bnrs", U, V)
        generator_norm_sq = (2.0 * (gram_u * gram_v).sum((-1, -2))
                             - 2.0 * (cross * cross.transpose(-2, -1)).sum((-1, -2)))
        generator_norm = generator_norm_sq.clamp_min(0).sqrt().mean()
        relative_change = ((z_next - z).norm(dim=-1) /
                           z.norm(dim=-1).clamp_min(1e-8)).mean()
        diag = {
            "collision_generator_norm": generator_norm.detach(),
            "collision_relative_change": relative_change.detach(),
            "collision_norm_residual": norm_diff.detach(),
        }
        return output, diag


class QuadraticPassiveBath(nn.Module):
    """Threshold-free quadratic passive radiation bath J_bath = 2 * kappa * E^2 / R^2."""

    def __init__(
        self,
        node_features: torch.Tensor,
        d: int = 64,
        local_radius: float = 1.25,
        max_kappa: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = d
        self.local_radius = float(local_radius)
        self.max_kappa = float(max_kappa)
        node_dim = node_features.shape[-1]
        self.kappa_net = nn.Sequential(
            nn.Linear(node_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        nn.init.normal_(self.kappa_net[-1].weight, std=1e-3)
        nn.init.constant_(self.kappa_net[-1].bias, 0.0)
        self.register_buffer("node_features", node_features.float()[None], persistent=False)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Dimensionless local occupation ratio: rho_i = 2 * E_i / R_i^2 = ||f_i||^2 / R_i^2
        local_energy = 0.5 * field.square().sum(dim=-1, keepdim=True)
        rho = 2.0 * local_energy / (self.local_radius ** 2)

        kappa = self.max_kappa * torch.sigmoid(self.kappa_net(self.node_features))
        sin2_theta = torch.clamp(kappa * rho, max=0.5)
        cos_theta = torch.sqrt(1.0 - sin2_theta)
        # sqrt has an infinite derivative at exactly zero.  Empty parcels are
        # common at initialization, so clamp before sqrt rather than after it;
        # the outgoing wave is diagnostic and remains numerically zero after
        # multiplication by a zero field.
        sin_theta = torch.sqrt(sin2_theta.clamp_min(torch.finfo(field.dtype).tiny))

        field_next = cos_theta * field
        bath_out = -sin_theta * field
        j_bath = 0.5 * bath_out.square().sum(dim=(-1, -2)).mean()

        diag = {
            "bath_power": j_bath.detach(),
            "occupation_ratio_mean": rho.detach().mean(),
            "occupation_ratio_max": rho.detach().amax(),
            "cooling_sin2_mean": sin2_theta.detach().mean(),
        }
        return field_next, bath_out, diag


class QKNormReadout(nn.Module):
    """State-only multi-query attention readout with per-head learnable scale and QK-norm."""

    def __init__(
        self,
        node_features: torch.Tensor,
        d: int = 64,
        queries: int = 4,
        heads: int = 4,
        nodes: int = 256,
    ) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d = d
        self.queries = queries
        self.heads = heads
        self.head_dim = d // heads

        # Theory-derived initial logit scale: g0 = log2(N^2 - N)
        g0 = math.log2(nodes * nodes - nodes)
        self.head_log_scale = nn.Parameter(torch.full((1, heads, 1, 1), math.log(g0)))

        self.query = nn.Parameter(torch.randn(1, queries, d) * 0.02)
        self.position = nn.Sequential(
            nn.Linear(node_features.shape[-1], d), nn.SiLU(), nn.Linear(d, d)
        )
        self.norm = nn.LayerNorm(d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("node_features", node_features.float()[None], persistent=False)

    def position_encoding(self) -> torch.Tensor:
        return self.position(self.node_features)

    def forward(self, field: torch.Tensor, position_encoding: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, d = field.shape
        state = self.norm(field)
        pos = self.position_encoding() if position_encoding is None else position_encoding

        q = self.query.expand(batch, -1, -1)
        k = self.key(state + pos)
        v = self.value(state)

        split = lambda x: x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q_h, k_h, v_h = split(q), split(k), split(v)

        # Cosine QK-Norm
        q_norm = F.normalize(q_h, dim=-1)
        k_norm = F.normalize(k_h, dim=-1)

        scale = torch.exp(self.head_log_scale)
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        read = torch.matmul(attn, v_h).transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read)), attn


class CBIMMaleCNSUnifiedV6(nn.Module):
    """Complete Unified Open-System Boltzmann Kinetic Model on the MaleCNS Connectome."""

    architecture = "CBIM-MaleCNS-unified-kinetic-v6"

    def __init__(
        self,
        graph_path: str | Path,
        vocab_size: int = 50257,
        velocities: int = 8,
        content_dim: int = 8,
        queries: int = 4,
        heads: int = 4,
        modes: int = 32,
        collision_rank: int = 2,
        checkpoint_tokens: int = 1,
    ) -> None:
        super().__init__()
        graph = load_malecns_graph(graph_path)
        self.graph_path = str(graph_path)
        self.velocities = int(velocities)
        # Two state-dependent passes are the smallest composition that can
        # produce a nontrivial noncommuting scattering sequence.  More passes
        # exceeded the 1.5 s/update engineering gate on the target GPU.
        self.collision_passes = 2
        self.content_dim = int(content_dim)
        self.d = self.velocities * self.content_dim
        self.L = graph["coordinates"].shape[0]
        self.state_shape = (self.L, self.d)
        self.checkpoint_tokens = int(checkpoint_tokens)

        self.boundary_write = ContextualBoundaryWrite(graph, vocab_size=vocab_size, d=self.d)
        self.transport = DualScaleGeometricTransport(
            graph, velocities=velocities, content_dim=content_dim, modes=modes
        )
        self.collision = WoodburyKineticCollision(
            graph["node_features"], velocities=velocities, content_dim=content_dim, rank=collision_rank
        )
        self.bath = QuadraticPassiveBath(graph["node_features"], d=self.d)
        self.readout = QKNormReadout(graph["node_features"], d=self.d, queries=queries, heads=heads, nodes=self.L)
        self.decoder = nn.Linear(self.d, vocab_size)
        self.decoder.weight = self.boundary_write.embedding.weight

    def initial_state(self, batch_size: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        p = next(self.parameters())
        dev = p.device if device is None else device
        dt = p.dtype if dtype is None else dtype
        return torch.zeros(batch_size, self.L, self.d, device=dev, dtype=dt)

    def evolve(
        self,
        state: torch.Tensor,
        token_id: torch.Tensor,
        disable_transport: bool = False,
        disable_collision: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # 1. Full-rank boundary exchange
        state_after_w, reflected, write_diag = self.boundary_write(state, token_id)

        # 2. Dual-scale Strang-split transport
        if disable_transport:
            state_after_t = state_after_w
            trans_diag = {"transport_norm_residual": state.new_zeros(())}
        else:
            state_after_t, trans_diag = self.transport(state_after_w)

        # 3. Woodbury zero-space non-equilibrium collision
        if disable_collision:
            state_after_c = state_after_t
            col_diag = {"collision_norm_residual": state.new_zeros(())}
        else:
            state_after_c = state_after_t
            generator_norm = state.new_zeros(())
            relative_change = state.new_zeros(())
            norm_residual = state.new_zeros(())
            for _ in range(self.collision_passes):
                state_after_c, pass_diag = self.collision(state_after_c)
                generator_norm = generator_norm + pass_diag["collision_generator_norm"]
                relative_change = relative_change + pass_diag["collision_relative_change"]
                norm_residual = torch.maximum(
                    norm_residual, pass_diag["collision_norm_residual"])
            col_diag = {
                "collision_generator_norm": generator_norm / self.collision_passes,
                "collision_relative_change": relative_change / self.collision_passes,
                "collision_norm_residual": norm_residual,
            }

        # 4. Quadratic passive radiation cooling
        state_next, bath_out, bath_diag = self.bath(state_after_c)

        diagnostics = {
            **write_diag,
            **trans_diag,
            **col_diag,
            **bath_diag,
            "field_energy": 0.5 * state_next.square().sum(dim=(-1, -2)).mean().detach(),
        }
        return state_next, reflected, diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        state: torch.Tensor | None = None,
        disable_transport: bool = False,
        disable_collision: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        batch, tokens = input_ids.shape
        current = self.initial_state(batch, input_ids.device) if state is None else state
        pos = self.readout.position_encoding()

        diagnostic_names = (
            "incident_energy", "reflected_energy", "absorbed_power",
            "write_angle_abs_mean", "write_balance_residual", "envelope_mean",
            "transport_edge_angle_mean", "transport_spectral_angle_mean",
            "transport_norm_residual", "collision_generator_norm",
            "collision_norm_residual", "bath_power", "occupation_ratio_mean",
            "occupation_ratio_max", "cooling_sin2_mean", "field_energy",
        )

        def segment(segment_state, ids, position):
            segment_features, segment_diagnostics = [], []
            for index in range(ids.shape[1]):
                segment_state, _, diag = self.evolve(
                    segment_state, ids[:, index],
                    disable_transport=disable_transport,
                    disable_collision=disable_collision,
                )
                feature, _ = self.readout(segment_state, position)
                segment_features.append(feature)
                segment_diagnostics.append(torch.stack([
                    diag.get(name, segment_state.new_zeros(())).detach()
                    for name in diagnostic_names
                ]))
            return (segment_state, torch.stack(segment_features, 1),
                    torch.stack(segment_diagnostics).sum(0))

        features, diagnostic_sums = [], []
        for start in range(0, tokens, self.checkpoint_tokens):
            ids = input_ids[:, start:start + self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                current, feature, diagnostic = checkpoint(
                    segment, current, ids, pos, use_reentrant=False,
                    preserve_rng_state=False)
            else:
                current, feature, diagnostic = segment(current, ids, pos)
            features.append(feature)
            diagnostic_sums.append(diagnostic)

        logits = self.decoder(torch.cat(features, dim=1))
        loss = F.cross_entropy(logits.reshape(-1, self.decoder.out_features), targets.reshape(-1))
        mean_diagnostic = torch.stack(diagnostic_sums).sum(0) / tokens
        avg_diag = dict(zip(diagnostic_names, mean_diagnostic.unbind()))
        avg_diag["final_energy"] = 0.5 * current.square().sum(-1).mean().detach()
        return loss, current, avg_diag
