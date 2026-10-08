"""Constructive runtime-plasticity and extended phase-locking certificate.

No optimizer, GPU, or corpus training. Declared forcing patterns certify rule
response, not semantic discovery. The locking source is an admissible-action
witness, not a certificate for the default innovation-only W4 actor.
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
    bounded_boundary_supply, configure_locking_witness, continuous_rhs, rotation_generator,
)


def phase_locking_certificate() -> dict:
    model = PlasticMedium3D((2, 2, 2), channels=3, hidden=8,
                            adaptive_conduction=True).double()
    omega = 1.3
    plane = configure_locking_witness(model, omega)
    q = 4 * model.channels
    orbit2 = (-1 + math.sqrt(1 + 4 * q)) / 2
    state = model.initial_state()
    state = state.with_field((math.sqrt(orbit2) * plane[:, 0]).expand_as(state.field))
    gain, rate, metric = model.conduction_plasticity.coefficients(model.material_field())
    equilibrium = gain[None] * model.conduction_plasticity.evidence(state.field, state.flux, metric)
    state = state.with_conduction(equilibrium)
    width = model._pack(state).numel()

    def flatten(value):
        return torch.cat((model._pack(value).flatten(), value.conduction.flatten()))

    def unpack(vector):
        packed = vector[:width].reshape_as(model._pack(state))
        return model._unpack(packed, state).with_conduction(vector[width:].reshape_as(state.conduction))

    def rotating_rhs(vector):
        candidate = unpack(vector)
        rhs = continuous_rhs(model, candidate)
        source = bounded_boundary_supply(candidate.field, plane, 1.0, 1.0)
        fast_rhs = model._pack(rhs)
        fast_rhs = fast_rhs + torch.stack((source, *[torch.zeros_like(source) for _ in range(3)]), -2)
        fast_rhs = fast_rhs - rotation_generator(model._pack(candidate), plane, omega)
        return torch.cat((fast_rhs.flatten(), rhs.conduction.flatten()))

    vector = flatten(state)
    jacobian = torch.autograd.functional.jacobian(rotating_rhs, vector)
    eigen = torch.linalg.eigvals(jacobian)
    neutral = eigen.abs() < 1e-9
    gap = float(-eigen.real[~neutral].max())
    residual = float(rotating_rhs(vector).abs().max().detach())
    # At the uniform orbit both endpoint difference and flux vanish, so varying
    # the conductance cannot drive first-order wave perturbations.
    wave_from_structure = float(jacobian[:width, width:].abs().max())
    structural_block = jacobian[width:, width:]
    expected = -torch.diag(rate[None].expand_as(state.conduction).flatten())
    block_residual = float((structural_block - expected).abs().max().detach())
    passed = (int(neutral.sum()) == 1 and gap > 0 and residual < 1e-12
              and wave_from_structure < 1e-12 and block_residual < 1e-12)
    return {
        "purpose": "conditional coupled-system existence certificate, not the W4 actor",
        "shape": list(model.shape), "channels": model.channels,
        "total_state_dimension": vector.numel(), "structural_state_dimension": state.conduction.numel(),
        "angular_rate_model_units": omega, "external_periodic_forcing": False,
        "orbit_radius_squared": orbit2, "neutral_phase_directions": int(neutral.sum()),
        "continuous_transverse_gap": gap, "rotating_rhs_residual": residual,
        "wave_from_structure_block_max": wave_from_structure,
        "structural_block_residual": block_residual, "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(439)
    model = PlasticMedium3D((4, 4, 4), channels=3,
                            adaptive_conduction=True).double()
    with torch.no_grad():
        model.conduction_plasticity.gain.bias.fill_(math.log(math.expm1(4)))
    empty = model.initial_state()
    field = torch.zeros_like(empty.field)
    field[:, :, 0, 0, 0] = 1
    first = model.adapt_conduction(empty.with_field(field), 10.0)
    first_speeds = model.edge_log_speeds(first).exp()
    new_field = torch.zeros_like(field)
    new_field[:, :, 2, 0, 0] = 1
    moved = model.adapt_conduction(first.with_field(new_field), 10.0)
    new_speeds = model.edge_log_speeds(moved).exp()
    random = MediumState(torch.randn_like(empty.field),
                         tuple(torch.randn_like(x) for x in empty.flux),
                         empty.elapsed, empty.conduction)
    output, info = model.advance(random, 0.03, substeps=4)
    production = PlasticMedium3D(adaptive_conduction=True)
    production_state = production.initial_state()
    certificate = phase_locking_certificate()
    report = {
        "purpose": "train-free math/interface audit; no semantic or speed superiority claim",
        "law": "tau*dp/dt = gain*s(field,flux,metric) - p; |s| <= 1",
        "construction": {"gain": 4.0, "duration_per_activity_pattern": 10.0,
                         "source": "prescribed coherent site pattern, no optimizer",
                         "initial_path_min_speed": float(first_speeds[0, :, 0, 0, 0].min().detach()),
                         "initial_transverse_max_speed": float(first_speeds[0, :, 0, 0, 1:].max().detach()),
                         "new_path_min_speed": float(new_speeds[0, :, 2, 0, 0].min().detach()),
                         "old_path_max_speed_after_change": float(new_speeds[0, :, 0, 0, 0].max().detach())},
        "adaptive_random_energy_residual": float((model.energy(output) + info['bath_out_energy']
                                                   - model.energy(random)).abs().max().detach()),
        "production_shape": list(production.shape), "production_channels": production.channels,
        "added_parameters": sum(p.numel() for p in production.conduction_plasticity.parameters()),
        "added_fp32_state_bytes_per_individual": production_state.conduction.numel() * 4,
        "phase_locking": certificate,
        "remaining": ["Task likelihood must learn semantic preferences; activity alone is not usefulness.",
                      "Full W4 boundary energy/action-support closure remains required before capability training.",
                      "Finite-step stability preserves energy, but high-speed propagation accuracy needs refinement.",
                      "The rule is bio-inspired, not a model of literal axon growth or biological myelination."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not certificate['passed']:
        raise RuntimeError('Extended phase-locking witness failed')


if __name__ == '__main__':
    main()
