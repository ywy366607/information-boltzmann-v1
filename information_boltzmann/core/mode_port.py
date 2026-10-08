"""One energy-closed boundary mode on a spatial/content field.

The coupling vector is a complete spatial/content packet, not a factorization
into one spatial envelope and one content vector. Inner products use cell-volume
weights on the unit torus. The rational form avoids normalizing a vanishing
coupling vector and preserves its orthogonal complement exactly.
"""
from __future__ import annotations

import math

import torch


def scatter_contact_mode(field: torch.Tensor, coupling: torch.Tensor,
                         incident: torch.Tensor):
    """Rotate the contacted field mode and scalar environmental amplitude.

    Let r=||w||, psi=w/r, theta=atan(r). This implements
    a'=cos(theta)*a+sin(theta)*e, e'=-sin(theta)*a+cos(theta)*e
    with a=<psi,f>, leaving f-psi*a unchanged. No division by r is needed.
    ``incident`` has shape[B]; both field tensors have shape[B,X,Y,Z,D].
    Storage is .5*(mean_xyz sum_D f**2 + e**2).
    """
    if field.shape != coupling.shape or field.ndim != 5:
        raise ValueError('Matching [B,X,Y,Z,D] field and coupling required')
    if incident.shape != (field.shape[0],):
        raise ValueError('One scalar incident amplitude per batch item required')
    axes = (1, 2, 3, 4)
    volume = 1.0 / math.prod(field.shape[1:4])
    radius_squared = coupling.square().sum(axes) * volume
    contact = (coupling * field).sum(axes) * volume
    denominator = (1.0 + radius_squared).sqrt()
    correction = -contact / (denominator * (denominator + 1.0))
    injection = incident / denominator
    next_field = field + coupling * (correction + injection)[:, None, None, None, None]
    outgoing = (incident - contact) / denominator
    return next_field, outgoing, radius_squared
