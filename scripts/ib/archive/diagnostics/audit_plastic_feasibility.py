"""Training-free representation and constructive phase-locking certificate.

The proof applies to a declared boundary-action witness, not the default W4
innovation policy. The small-grid Jacobian corroborates the all-grid analytic
argument. This is a numerical math audit, not a toy capability experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from information_boltzmann.core.plastic_medium import MediumState, PlasticMedium3D
from information_boltzmann.core.plastic_feasibility import (
    bounded_boundary_supply, configure_locking_witness, continuous_rhs,
    discrete_orbit_radius_squared, graph_incidence, rotate_back, rotation_generator,
    spatial_heterogeneity_certificate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(417)
    reference = PlasticMedium3D().double()
    basis = reference.material.basis(reference.coordinates).reshape(256, -1)
    singular = torch.linalg.svdvals(basis)
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8).double()
    omega, power, radius2, beta, dt = 1.3, 1.0, 1.0, 1.0, 0.005
    plane = configure_locking_witness(model, omega, beta)
    q = 4 * model.channels
    orbit2 = (-radius2 + math.sqrt(radius2**2 + 4*q*power/beta)) / 2
    empty = model.initial_state()
    state = empty.with_field((math.sqrt(orbit2) * plane[:, 0]).expand_as(empty.field))

    def rotating_rhs(packed):
        candidate = model._unpack(packed, state)
        rhs = model._pack(continuous_rhs(model, candidate))
        supply = bounded_boundary_supply(candidate.field, plane, power, radius2)
        return rhs + torch.stack((supply, *[torch.zeros_like(supply) for _ in range(3)]), -2) - rotation_generator(packed, plane, omega)

    packed = model._pack(state)
    jacobian = torch.autograd.functional.jacobian(rotating_rhs, packed).reshape(packed.numel(), -1)
    eigen = torch.linalg.eigvals(jacobian)
    neutral = eigen.abs() < 1e-9
    continuous_gap = float(-eigen.real[~neutral].max())
    continuous_residual = float(rotating_rhs(packed).abs().max().detach())
    incidence = graph_incidence(model)
    spectrum = torch.linalg.eigvalsh(incidence.T @ incidence)

    orbit_discrete2 = discrete_orbit_radius_squared(3, dt, power, radius2, beta)
    state = empty.with_field((math.sqrt(orbit_discrete2) * plane[:, 0]).expand_as(empty.field))

    def rotating_map(packed):
        candidate = model._unpack(packed, state)
        source = bounded_boundary_supply(candidate.field, plane, power, radius2)
        output, _ = model(candidate.with_field(candidate.field + dt*source), dt)
        return rotate_back(model._pack(output), plane, omega*dt)

    packed = model._pack(state)
    map_jacobian = torch.autograd.functional.jacobian(rotating_map, packed).reshape(packed.numel(), -1)
    multipliers = torch.linalg.eigvals(map_jacobian)
    neutral_map = (multipliers - 1).abs() < 1e-9
    transverse_radius = float(multipliers.abs()[~neutral_map].max())
    passed = (int(neutral.sum()) == 1 and continuous_gap > 0
              and int(neutral_map.sum()) == 1 and transverse_radius < 1)
    report = {
        "purpose": "mathematical feasibility; no training or capability verdict",
        "reference_shape": [8, 8, 4], "channels": 128,
        "material_basis_columns": basis.shape[1],
        "material_basis_rank": int(torch.linalg.matrix_rank(basis)),
        "material_basis_min_singular_value": float(singular.min()),
        "medium_parameters": sum(p.numel() for p in reference.parameters()),
        "spatial_heterogeneity": spatial_heterogeneity_certificate(),
        "witness": {
            "shape": [2, 2, 2], "channels": 3,
            "angular_rate_in_model_units": omega, "power_per_cell": power,
            "supply_radius_squared": radius2, "bath_rate": beta,
            "source": "u=P*projection(f)/(R²+|projection(f)|²)",
            "scope": "admissible orthogonal-boundary actions; not the default W4 actor",
            "external_periodic_driver": False,
            "orbit_radius_squared": orbit2,
            "rotating_rhs_residual": continuous_residual,
            "neutral_phase_directions": int(neutral.sum()),
            "continuous_transverse_gap": continuous_gap,
            "incidence_rank": int(torch.linalg.matrix_rank(incidence)),
            "laplacian_gap": float(spectrum[1].detach()),
            "integration_duration": dt,
            "discrete_orbit_radius_squared": orbit_discrete2,
            "rotating_map_residual": float((rotating_map(packed)-packed).abs().max().detach()),
            "neutral_map_directions": int(neutral_map.sum()),
            "discrete_transverse_spectral_radius": transverse_radius,
            "passed": passed,
        },
        "remaining_closure": [
            "The W4 policy must be shown to include a compatible energy-supplying feedback. Its innovation-only action can vanish exactly on perfectly predicted input.",
            "A two-plane witness is a proof configuration, not a prescribed allocation of production brain regions.",
            "Connectivity proves rotational control feasibility under independent controls; it does not prove the finite shared MLP can realize every spatial/time policy.",
            "The stability certificate is local to an open neighborhood of the driven orbit, not a claim for every arbitrary input stream or trained parameter setting.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Locking witness failed its stability certificate")


if __name__ == '__main__':
    main()
