"""Directional finite-volume transport on an embedded graph.

The velocity channel is a physical direction.  It is not merely a conditioning
label for a scalar graph spectrum.  The transition is built once from anatomy
and executed as one batched dense matrix multiplication; at the current
MaleCNS size, the eight 256x256 transition matrices occupy about 2 MiB in FP32.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def cube_velocities(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the normalized D3Q8 corner velocities."""
    velocity = torch.tensor(
        [(x, y, z) for x in (-1.0, 1.0)
         for y in (-1.0, 1.0) for z in (-1.0, 1.0)], dtype=dtype)
    return velocity / math.sqrt(3.0)


class DirectionalFiniteVolumeTransport(nn.Module):
    """Conservative positive advection for ``[batch,node,velocity,content]``.

    ``transition[q, i, j]`` is the fraction transported from source node ``i``
    to destination node ``j`` in velocity channel ``q``.  A shared global CFL
    scale retains the reciprocity
    ``transition[q,i,j] == transition[q_bar,j,i]`` on symmetric graph support.

    For nonnegative occupancy the step preserves positivity and mass.  The same
    transition can transport the corresponding carried moment ``p * a``.  A
    general signed feature tensor is accepted numerically, but then positivity
    and a classical Boltzmann H functional have no physical interpretation.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        adjacency: torch.Tensor,
        coordinate_scale: torch.Tensor | None = None,
        stream_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        coordinates = torch.as_tensor(coordinates)
        if not coordinates.is_floating_point():
            coordinates = coordinates.float()
        dtype = coordinates.dtype
        adjacency = torch.as_tensor(adjacency, dtype=dtype)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [nodes, 3]")
        nodes = coordinates.shape[0]
        if adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency must have shape [nodes, nodes]")
        if not 0.0 <= stream_fraction <= 1.0:
            raise ValueError("stream_fraction must lie in [0, 1]")

        if coordinate_scale is not None:
            scale = torch.as_tensor(coordinate_scale, dtype=dtype)
            if scale.shape != (3,) or torch.any(scale <= 0):
                raise ValueError("coordinate_scale must contain three positives")
            coordinates = coordinates * scale

        # Synaptic direction and kinetic velocity are separate concepts.  The
        # support is symmetrized; c_q dot r_ij supplies the transport direction.
        support = adjacency.clamp_min(0)
        support = support + support.T
        support.fill_diagonal_(0)
        degree = support.sum(-1)
        conductance = support / torch.sqrt(
            (degree[:, None] * degree[None, :]).clamp_min(1e-12))

        displacement = coordinates[None, :, :] - coordinates[:, None, :]
        distance = displacement.norm(dim=-1, keepdim=True)
        direction = displacement / distance.clamp_min(1e-12)
        direction = torch.where(distance > 0, direction, torch.zeros_like(direction))
        velocity = cube_velocities(dtype=dtype)
        projection = torch.einsum("qk,ijk->qij", velocity, direction)
        raw_flux = conductance[None] * projection.clamp_min(0)

        # A single scale for q and its opposite preserves edge reciprocity.
        max_outflow = raw_flux.sum(-1).amax().clamp_min(1e-12)
        off_diagonal = float(stream_fraction) * raw_flux / max_outflow
        stay = 1.0 - off_diagonal.sum(-1)
        transition = off_diagonal + torch.diag_embed(stay)

        opposite = torch.arange(7, -1, -1)
        reciprocal = transition[:, :, :].clone()
        diagonal = torch.eye(nodes, dtype=torch.bool)[None]
        reciprocal_error = torch.where(
            diagonal, torch.zeros_like(reciprocal),
            reciprocal - transition[opposite].transpose(-1, -2)).abs().amax()
        moved = off_diagonal.sum()
        alignment = ((off_diagonal * projection.clamp_min(0)).sum()
                     / moved.clamp_min(1e-12))

        self.nodes = nodes
        self.velocities = 8
        self.register_buffer("transition", transition, persistent=True)
        self.register_buffer("velocity_vectors", velocity, persistent=False)
        self.register_buffer("directional_alignment", alignment, persistent=False)
        self.register_buffer("reciprocity_error", reciprocal_error, persistent=False)

    def forward(self, field: torch.Tensor):
        if field.ndim != 4:
            raise ValueError("field must have shape [batch,node,velocity,content]")
        if field.shape[1] != self.nodes or field.shape[2] != self.velocities:
            raise ValueError("field node/velocity dimensions do not match transport")
        values = field.permute(0, 2, 3, 1)
        transported = torch.matmul(values, self.transition.to(field.dtype))
        transported = transported.permute(0, 3, 1, 2).contiguous()
        before = field.sum(dim=1)
        after = transported.sum(dim=1)
        return transported, {
            "transport_mass_residual": (after - before).abs().amax().detach(),
            "transport_directional_alignment": self.directional_alignment.detach(),
            "transport_reciprocity_error": self.reciprocity_error.detach(),
        }


class DirectionalCayleyTransport(nn.Module):
    """Energy-preserving directional transport for a signed wave field.

    The local generator is assembled directly from anatomical edges,
    ``K_q[i,j] propto w_ij c_q dot (r_j-r_i)``. It is skew-symmetric, so its
    Cayley step is exactly orthogonal. Opposite D3Q8 channels use inverse
    propagators. This operator is for signed amplitudes and makes no positivity
    claim; the finite-volume class above remains the nonnegative-density path.
    """

    def __init__(self, coordinates: torch.Tensor, adjacency: torch.Tensor,
                 coordinate_scale: torch.Tensor | None = None,
                 stream_fraction: float = 0.5) -> None:
        super().__init__()
        coordinates = torch.as_tensor(coordinates)
        if not coordinates.is_floating_point():
            coordinates = coordinates.float()
        dtype = coordinates.dtype
        adjacency = torch.as_tensor(adjacency, dtype=dtype)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [nodes, 3]")
        nodes = coordinates.shape[0]
        if adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency must have shape [nodes, nodes]")
        if not 0.0 <= stream_fraction <= 1.0:
            raise ValueError("stream_fraction must lie in [0, 1]")
        if coordinate_scale is not None:
            scale = torch.as_tensor(coordinate_scale, dtype=dtype)
            if scale.shape != (3,) or torch.any(scale <= 0):
                raise ValueError("coordinate_scale must contain three positives")
            coordinates = coordinates * scale

        support = adjacency.clamp_min(0)
        support = support + support.T
        support.fill_diagonal_(0)
        degree = support.sum(-1)
        conductance = support / torch.sqrt(
            (degree[:, None] * degree[None, :]).clamp_min(1e-12))
        displacement = coordinates[None, :, :] - coordinates[:, None, :]
        distance = displacement.norm(dim=-1, keepdim=True)
        direction = displacement / distance.clamp_min(1e-12)
        direction = torch.where(distance > 0, direction, torch.zeros_like(direction))
        velocity = cube_velocities(dtype=dtype)
        projection = torch.einsum("qk,ijk->qij", velocity, direction)

        generator = conductance[None] * projection
        generator = .5 * (generator - generator.transpose(-1, -2))
        generator = generator / generator.abs().sum(-1).amax().clamp_min(1e-12)
        identity = torch.eye(nodes, dtype=dtype)
        half_step = .5 * float(stream_fraction)
        opposite = torch.arange(7, -1, -1)
        propagators = [None] * 8
        for channel in range(4):
            forward = torch.linalg.solve(
                identity - half_step * generator[channel],
                identity + half_step * generator[channel])
            propagators[channel] = forward
            propagators[7 - channel] = forward.T
        propagator = torch.stack(propagators)
        batched_identity = identity[None]
        orthogonal_error = (
            propagator.transpose(-1, -2) @ propagator - batched_identity
        ).abs().amax()
        reciprocity_error = (
            propagator - propagator[opposite].transpose(-1, -2)
        ).abs().amax()
        directional_weight = generator.clamp_min(0)
        alignment = ((directional_weight * projection.clamp_min(0)).sum()
                     / directional_weight.sum().clamp_min(1e-12))

        self.nodes = nodes
        self.velocities = 8
        self.register_buffer("propagator", propagator, persistent=True)
        self.register_buffer("velocity_vectors", velocity, persistent=False)
        self.register_buffer("directional_alignment", alignment, persistent=False)
        self.register_buffer("orthogonal_error", orthogonal_error, persistent=False)
        self.register_buffer("reciprocity_error", reciprocity_error, persistent=False)

    def forward(self, field: torch.Tensor):
        if field.ndim != 4:
            raise ValueError("field must have shape [batch,node,velocity,content]")
        if field.shape[1] != self.nodes or field.shape[2] != self.velocities:
            raise ValueError("field node/velocity dimensions do not match transport")
        values = field.permute(0, 2, 3, 1)
        transported = torch.matmul(values, self.propagator.to(field.dtype))
        transported = transported.permute(0, 3, 1, 2).contiguous()
        before = field.square().sum(dim=(1, 2, 3))
        after = transported.square().sum(dim=(1, 2, 3))
        return transported, {
            "transport_norm_residual": (after - before).abs().amax().detach(),
            "transport_directional_alignment": self.directional_alignment.detach(),
            "transport_reciprocity_error": self.reciprocity_error.detach(),
            "transport_orthogonal_error": self.orthogonal_error.detach(),
        }
