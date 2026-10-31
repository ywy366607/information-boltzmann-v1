"""Shared local constitutive operators for generated native mechanisms.

These laws do not choose receptor identity or share physical states. Adapters
bind actual contact/node arrays and retain their existing kinetic parameters.
The operators introduce no clock, random draw, reward or retained history.
"""
import torch


def ligand_binding(occupancy, concentration, affinity, off_rate, *, clamp=False):
    """Exact constant-ligand step of a two-state binding reaction per tick.

    affinity and off_rate are positive native parameters in the caller's units.
    No global mean or cell pooling is performed. Broadcasting is explicit in
    the bound parameter arrays supplied by each generated component.
    """
    equilibrium=concentration/(concentration+affinity)
    survival=torch.exp(-off_rate*(1.+concentration/affinity))
    value=equilibrium+(occupancy-equilibrium)*survival
    return value.clamp(0.,1.) if clamp else value


def resource_recovery(available, released, retained_fraction):
    """Release consumes available resources before independent replenishment.

    Callers retain their distinct release-probability and facilitation laws;
    this common balance introduces no population or site aggregation.
    """
    return 1.+(available-released-1.)*retained_fraction
