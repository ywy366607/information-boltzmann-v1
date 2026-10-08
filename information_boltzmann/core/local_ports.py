"""Persistent movable ports with compact C1 footprints on the unit torus."""
from __future__ import annotations

import math
import torch
from torch import nn


def balanced_grid(count: int) -> tuple[int, int, int]:
    grid, factor = [1, 1, 1], 2
    while count > 1:
        while count % factor == 0:
            grid[min(range(3), key=grid.__getitem__)] *= factor
            count //= factor
        factor += 1
    return tuple(grid)


class CompactTorusPorts(nn.Module):
    """Locations persist across events; content changes gates, not locations.

    Radius is an explicit communication budget in physical coordinates. The
    default uses the port-cell interior with a half-sampling-cell guard gap,
    and at least one sampling-cell radius for movable interpolation. Refinement should
    pass the saved physical radius to preserve the same aperture.
    An optional total aperture budget bounds count*prod(2*radius), the sum of
    continuous support-box volumes. Overlap is charged repeatedly and positions
    remain freely learnable; no separation or diversity objective is imposed.
    """

    def __init__(self, shape, count: int, radius=None, *, aperture_budget: float | None = None):
        super().__init__()
        if count < 1 or min(shape) < 2:
            raise ValueError('Positive port count and spatial axes >= 2 required')
        self.shape, self.count = tuple(shape), count
        grid = balanced_grid(count)
        axes = [(torch.arange(n) + 0.5) / n for n in grid]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(count, 3)
        # Mid-sample initialization prevents an exactly single-cell footprint
        # from starting at a location with zero coordinate derivative.
        self.centers = nn.Parameter((centers + 0.5 / torch.tensor(shape)).remainder(1))
        axes = [torch.arange(n) / n for n in shape]
        self.register_buffer('coordinates', torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3), persistent=False)
        radius = tuple(min(0.5, max(1.0 / n, 0.5 / p - 0.5 / n))
                       for p, n in zip(grid, shape)) if radius is None else tuple(radius)
        if len(radius) != 3 or any(not math.isfinite(r) or r <= 0.5 / n or r > 0.5 for r, n in zip(radius, shape)):
            raise ValueError('Radius must exceed half a sampling cell and be <= half the period')
        if aperture_budget is not None:
            if not math.isfinite(aperture_budget) or aperture_budget <= 0:
                raise ValueError('Aperture budget must be finite and positive')
            volume = count * math.prod(2 * r for r in radius)
            scale = min(1., (aperture_budget / volume) ** (1. / 3.))
            radius = tuple(r * scale for r in radius)
            if any(r <= 0.5 / n for r, n in zip(radius, shape)):
                raise ValueError('Aperture budget is too small for movable support at this grid/aspect ratio')
            # Round toward the feasible side if default-dtype construction
            # would put the continuous support sum just above its ceiling.
            represented = torch.tensor(radius)
            if count * math.prod(2 * float(r) for r in represented) > aperture_budget:
                represented = torch.nextafter(represented, torch.zeros_like(represented))
            radius = tuple(float(r) for r in represented)
            if any(r <= 0.5 / n for r, n in zip(radius, shape)):
                raise ValueError('Aperture budget leaves no representable movable support')
        self.physical_radius = radius
        self.aperture_budget = aperture_budget
        self.aperture_volume = count * math.prod(2 * r for r in radius)
        self.register_buffer('radius', torch.tensor(radius), persistent=False)

    def footprint(self):
        return compact_footprint(self.coordinates, self.centers, self.radius)

    def weights(self):
        footprint = self.footprint()
        return footprint / footprint.sum(-1, keepdim=True).clamp_min(torch.finfo(footprint.dtype).tiny)

    def observe(self, flat):
        return torch.einsum('pn,bnd->bpd', self.weights().to(flat), flat)


def compact_footprint(coordinates, centers, radius):
    delta = (coordinates[None] - centers.reshape(-1, 3)[:, None] + 0.5).remainder(1.0) - 0.5
    factors = (1.0 - (delta / radius).square()).clamp_min(0.0).square()
    # Explicit three-axis product avoids the reduction-product backward's
    # prefix-scan allocation path, which cannot be captured on this CUDA build.
    return factors[..., 0] * factors[..., 1] * factors[..., 2]


def port_overlap_diagnostics(write, read, field):
    """Report soft overlaps and support coverage; impose no diversity loss."""
    w, r = write.weights(), read.weights()
    wn, rn = torch.nn.functional.normalize(w, dim=-1), torch.nn.functional.normalize(r, dim=-1)
    def pair_overlap(values):
        n = len(values)
        gram = values @ values.T
        return (gram.sum() - gram.diagonal().sum()) / max(1, n * (n - 1))
    cross = wn @ rn.T
    write_support, read_support = w > 0, r > 0
    read_centers = read.probe_coords.reshape(-1, 3)
    delta = (write.centers[:, None] - read_centers[None] + 0.5).remainder(1.0) - 0.5
    spatial = field - field.mean((1, 2, 3), keepdim=True)
    eps = torch.finfo(field.dtype).eps
    return {
        'port_write_read_overlap_mean': cross.mean().detach(),
        'port_write_read_nearest_overlap': cross.max(-1).values.mean().detach(),
        'port_write_pair_overlap': pair_overlap(wn).detach(),
        'port_read_pair_overlap': pair_overlap(rn).detach(),
        'port_nearest_read_distance': delta.norm(dim=-1).min(-1).values.mean().detach(),
        'port_write_coverage': write_support.any(0).float().mean().detach(),
        'port_read_coverage': read_support.any(0).float().mean().detach(),
        'port_shared_coverage': (write_support.any(0) & read_support.any(0)).float().mean().detach(),
        'port_write_aperture_volume': field.new_tensor(write.aperture_volume),
        'port_read_aperture_volume': field.new_tensor(read.aperture_volume),
        'port_total_aperture_volume': field.new_tensor(write.aperture_volume + read.aperture_volume),
        'port_field_spatial_fraction': (spatial.square().sum() / field.square().sum().clamp_min(eps)).detach(),
        'write_port_coords': write.centers.detach().remainder(1.0),
        'read_port_coords': read_centers.detach().remainder(1.0),
        'write_port_weights': w.detach(), 'read_port_weights': r.detach(),
    }
