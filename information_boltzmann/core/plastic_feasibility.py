"""Analytic generators and constructive feasibility witnesses, without training.

The supply below is a boundary-action witness, not the default W4 policy. It
establishes an admissible bounded-power phase-locking solution using existing
medium primitives; it does not certify random initialization or policy learning.
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch.nn import functional as F

from .plastic_medium import MediumState, PlasticMedium3D


def continuous_rhs(model: PlasticMedium3D, state: MediumState, *,
                   collision: bool = True, bath: bool = True) -> MediumState:
    """Exact infinitesimal generator of the implemented operator splitting."""
    material = model.material_field()
    speeds = model.speed_reference * model.edge_log_speeds(state, material).exp()
    stp = (None if model.short_term_plasticity is None else
           model.short_term_plasticity.coefficients(material))
    if stp is not None:
        speeds = speeds * model.short_term_plasticity.transmission_gain(state.transmission, stp)
    df = torch.zeros_like(state.field)
    dj = []
    for axis in range(3):
        weight = model.shape[axis] * speeds[..., axis, None]
        flow = weight * state.flux[axis]
        df = df + torch.roll(flow, 1, axis + 1) - flow
        dj.append(weight * (state.field - torch.roll(state.field, -1, axis + 1)))
    rhs = model._pack(MediumState(df, tuple(dj), state.elapsed))
    value = model._pack(state)
    features = model._features(value, material, state.receptors)
    if collision:
        transformed = model._reflect(value)
        free = transformed[..., 1:].flatten(-2)
        rates = model.collision_rate(features).reshape(
            *state.field.shape[:-1], model.collision_layers, model.free_width // 2)
        derivative = torch.zeros_like(free)
        for layer in range(model.collision_layers):
            lanes = torch.roll(free, -layer, -1).reshape(*free.shape[:-1], -1, 2)
            angle_rate = rates[..., layer, :]
            changes = torch.stack((-angle_rate * lanes[..., 1],
                                   angle_rate * lanes[..., 0]), -1)
            derivative = derivative + torch.roll(changes.flatten(-2), layer, -1)
        in_basis = torch.cat((torch.zeros_like(transformed[..., :1]),
                             derivative.reshape(*value.shape[:-1], model.channels - 1)), -1)
        rhs = rhs + model._reflect(in_basis)
    if bath:
        if model.conductance_response is not None:
            df_response, dj_response, ds = model.conductance_response.rhs(
                state.field, state.flux, state.receptors,
                model.conductance_response.coefficients(material))
            rhs = rhs + model._pack(MediumState(df_response, dj_response, state.elapsed))
        else:
            rate = F.softplus(model.bath_rate(features) + model.bath_bias).reshape_as(value)
            occupancy = value.square().mean((-2, -1), keepdim=True)
            rhs = rhs - rate * occupancy * value
    output = model._unpack(rhs, state)
    if state.receptors is not None:
        output = replace(output, receptors=ds if bath else torch.zeros_like(state.receptors))
    if model.conduction_plasticity is not None:
        gain, rate, metric = model.conduction_plasticity.coefficients(material)
        evidence = model.conduction_plasticity.adapted_evidence(
            state.field, state.flux, metric, gain,
            model.conductance_response.inhibition_history(state.receptors)
            if model.activity_adaptation else None)
        output = output.with_conduction(rate[None] * (gain[None] * evidence - state.conduction))
    if stp is not None:
        output = replace(output, transmission=model.short_term_plasticity.rhs(
            state.transmission, state.field, state.flux, stp))
    return output


def collision_pair_edges(width: int, layers: int):
    """Edges of the free-coordinate rotation graph, not spatial graph edges."""
    return [(int((2 * pair + layer) % width), int((2 * pair + layer + 1) % width))
            for layer in range(layers) for pair in range(width // 2)]


def configure_locking_witness(model: PlasticMedium3D, angular_rate: float,
                             bath_rate: float = 1.0) -> torch.Tensor:
    """Exact constant-generator member of the model's parameter family.

    A two-dimensional zero-mean content plane rotates equally in all four groups.
    The chosen angular rate is a free model parameter in this existence proof,
    not an imposed biological rhythm. No periodic signal is supplied.
    """
    if model.bath_rate is None:
        raise ValueError('This locking witness is specific to the quadratic bath; use the conductance certificate')
    if model.channels < 3 or model.collision_layers < 2 or bath_rate <= 0:
        raise ValueError("Witness requires D >= 3, two layers and a positive bath")
    with torch.no_grad():
        model.material.coefficients.zero_()
        for parameter in model.collision_rate.parameters():
            parameter.zero_()
        rates = model.collision_rate[-1].bias.reshape(model.collision_layers, -1)
        for group in range(4):
            offset = group * (model.channels - 1)
            layer = offset % 2
            pair = ((offset - layer) % model.free_width) // 2
            rates[layer, pair] = angular_rate
        for parameter in model.bath_rate.parameters():
            parameter.zero_()
        model.bath_rate[-1].bias.fill_(
            math.log(math.expm1(bath_rate)) - float(model.bath_bias))
    w = model.mean_reflector
    householder = torch.eye(model.channels, device=w.device, dtype=w.dtype) - 2 * w[:, None] * w[None, :]
    return householder[:, 1:3]


def bounded_boundary_supply(field: torch.Tensor, plane: torch.Tensor,
                            power: float, radius_squared: float) -> torch.Tensor:
    """Content-dependent admissible source u=P*Pf/(R²+|Pf|²).

    u·f <= P at every cell for all states. All constants are declared supply
    budgets/units. This controller is used only for a mathematical certificate.
    """
    if power <= 0 or radius_squared <= 0:
        raise ValueError("Positive power and supply radius required")
    projection = (field @ plane) @ plane.T
    occupancy = projection.square().sum(-1, keepdim=True)
    return power * projection / (radius_squared + occupancy)


def rotation_generator(value: torch.Tensor, plane: torch.Tensor,
                       angular_rate: float) -> torch.Tensor:
    coordinates = value @ plane
    return angular_rate * torch.stack((-coordinates[..., 1], coordinates[..., 0]), -1) @ plane.T


def rotate_back(value: torch.Tensor, plane: torch.Tensor, angle: float) -> torch.Tensor:
    coordinates = value @ plane
    x, y = coordinates[..., 0], coordinates[..., 1]
    rotated = torch.stack((math.cos(angle) * x + math.sin(angle) * y,
                           -math.sin(angle) * x + math.cos(angle) * y), -1)
    return value + (rotated - coordinates) @ plane.T


def graph_incidence(model: PlasticMedium3D) -> torch.Tensor:
    """Weighted incidence for the actual positive-axis propagation operator."""
    shape = model.shape
    sites = math.prod(shape)
    reference = model.material.coefficients
    ids = torch.arange(sites, device=reference.device).reshape(shape)
    weight = model.speed_reference * model.log_speed(model.material_field()).exp()
    result = reference.new_zeros(3 * sites, sites)
    for axis in range(3):
        rows = torch.arange(sites, device=reference.device) + axis * sites
        left = ids.flatten()
        right = torch.roll(ids, -1, axis).flatten()
        edge_weight = shape[axis] * weight[..., axis].flatten()
        result[rows, left] = edge_weight
        result[rows, right] = -edge_weight
    return result


def realize_edge_speeds(model: PlasticMedium3D, speeds: torch.Tensor) -> float:
    """Exact inverse synthesis on the full reference grid, without fitting.

    This audit operation solves V*coefficients*W.T = log(c/c_ref). It is not
    called during forward or used to prescribe trained spatial organization.
    """
    target = speeds.to(model.material.coefficients)
    if target.shape != (*model.shape, 3) or not bool(torch.isfinite(target).all() and (target > 0).all()):
        raise ValueError("Finite positive edge speeds [X,Y,Z,3] required")
    basis = model.material.basis(model.coordinates.to(target)).reshape(math.prod(model.shape), -1)
    if basis.shape[0] != basis.shape[1] or int(torch.linalg.matrix_rank(model.log_speed.weight)) != 3:
        raise ValueError("Square complete reference chart and full-row-rank speed map required")
    with torch.no_grad():
        log_target = (target / model.speed_reference).log().reshape(-1, 3)
        material = log_target @ torch.linalg.pinv(model.log_speed.weight.T)
        model.material.coefficients.copy_(torch.linalg.solve(basis, material))
        actual = model.speed_reference * model.log_speed(model.material_field()).exp()
        return float((actual - target).abs().max())


def spatial_heterogeneity_certificate() -> dict:
    """Construct compartment and localized wave modes in the actual parameter family."""
    model = PlasticMedium3D().double()
    shape = model.shape
    speeds = torch.ones(*shape, 3, dtype=torch.float64)
    slab = torch.zeros(shape, dtype=torch.bool)
    slab[:shape[0]//2] = True
    # Two weak cross-sections on a periodic domain, all interior edges unchanged.
    cross = slab != torch.roll(slab, -1, 0)
    weak_speed = 0.01
    speeds[..., 0][cross] = weak_speed
    slab_error = realize_edge_speeds(model, speeds)
    incidence = graph_incidence(model).detach()
    laplacian = incidence.T @ incidence
    spectrum = torch.linalg.eigvalsh(laplacian)
    contrast = (2 * slab.to(torch.float64) - 1).flatten()
    quotient = float(contrast @ laplacian @ contrast / contrast.square().sum())
    homogeneous_gap = min(4*n*n*math.sin(math.pi/n)**2 for n in shape)

    # A two-cell resonance with strong internal coupling and weak external edges.
    region = torch.zeros(shape, dtype=torch.bool)
    region[0, 0, 0] = region[1, 0, 0] = True
    speeds.fill_(1)
    for axis in range(3):
        outside = region != torch.roll(region, -1, axis)
        speeds[..., axis][outside] = weak_speed
    speeds[0, 0, 0, 0] = 4.0
    resonance_error = realize_edge_speeds(model, speeds)
    incidence = graph_incidence(model).detach()
    values, vectors = torch.linalg.eigh(incidence.T @ incidence)
    mode = vectors[:, -1]
    fraction = float(mode[region.flatten()].square().sum())
    ipr = float(mode.pow(4).sum())
    return {
        "shape": list(shape), "sites": math.prod(shape),
        "scope": "operator representation and spectrum; no learned anatomical claim",
        "weak_boundary_speed": weak_speed,
        "compartment": {
            "speed_synthesis_max_error": slab_error,
            "contrast_rayleigh_quotient": quotient,
            "first_nonzero_eigenvalue": float(spectrum[1]),
            "homogeneous_first_nonzero_eigenvalue": homogeneous_gap,
            "analytic_rayleigh_quotient": 8 * shape[0] * weak_speed**2,
        },
        "localized_resonance": {
            "speed_synthesis_max_error": resonance_error,
            "strong_internal_speed": 4.0,
            "top_squared_wave_frequency": float(values[-1]),
            "two_cell_mode_energy_fraction": fraction,
            "mode_ipr": ipr,
            "plane_wave_ipr": 1 / math.prod(shape),
        },
    }


def discrete_orbit_radius_squared(channels: int, duration: float, power: float,
                                  radius_squared: float, bath_rate: float) -> float:
    """Unique positive root of the implemented source/rotation/radial-bath map."""
    lower, upper = 0.0, 4 * channels / (2 * duration * bath_rate)
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        gain = 1 + duration * power / (radius_squared + middle)
        residual = gain * gain * (1 - 2 * duration * bath_rate * middle / (4 * channels)) - 1
        if residual > 0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)
