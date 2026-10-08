"""CPU-only constitutive audit; no task training or capability measurements."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.structural_resource import (
    TransportCapacityBudget, transport_capacity)


def audit():
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    net = PlasticMedium3D((4, 4, 2), channels=8, hidden=8,
                         anisotropic_transport=True).double()
    material = net.material_field()
    speed = torch.ones(1, 4, 4, 2, 3, dtype=torch.float64)
    birth = net.transport_factor(material, speed).detach()
    budget_value = transport_capacity(birth).item()
    budget = TransportCapacityBudget(budget_value)
    state = net.initial_state()
    state = replace(state, field=torch.randn_like(state.field),
                    flux=tuple(torch.randn_like(x) for x in state.flux))
    energy = net.energy(state)
    with torch.no_grad():
        net.transport_shear.bias.copy_(torch.tensor([.4, -.3, .5]))
    raw = net.transport_factor(material, 2 * speed).detach()
    effective = budget(raw)
    output = net._tensor_transport(state, net._duration(.02, state.field), effective)
    capacity_error = abs(transport_capacity(effective).item() - budget_value)
    energy_error = abs((net.energy(output) - energy).item())
    eig = torch.linalg.eigvalsh(effective @ effective.transpose(-1, -2))

    width = state.field.numel()
    def local_response(vector, factor):
        blocks = vector.split(width)
        s = replace(state, field=blocks[0].reshape_as(state.field),
                    flux=tuple(x.reshape_as(state.field) for x in blocks[1:]))
        return net._tensor_transport(s, net._duration(.02, s.field), factor).field[0, 0, 0, 0, 0]
    origin = torch.zeros(4 * width, dtype=torch.float64, requires_grad=True)
    r0 = torch.autograd.functional.jacobian(lambda v: local_response(v, birth), origin)
    r1 = torch.autograd.functional.jacobian(lambda v: local_response(v, effective), origin)
    witness = r1 - (r1 @ r0) / (r0 @ r0) * r0
    witness = witness / witness.norm()
    return {
        'protocol': 'structural_resource_numerical_audit_v1',
        'device': 'cpu', 'dtype': 'float64', 'optimizer_updates': 0,
        'scope': 'Constitutive constraints and finite-time local observability; not language or memory capability',
        'budget_origin': 'mean trace(A) of the initial unit factor, not a tuned penalty',
        'birth_budget': budget_value,
        'raw_capacity': transport_capacity(raw).item(),
        'executed_capacity': transport_capacity(effective).item(),
        'capacity_residual': capacity_error,
        'transport_energy_residual': energy_error,
        'minimum_tensor_eigenvalue': eig.min().item(),
        'old_local_response_of_unit_witness': (r0 @ witness).item(),
        'new_local_response_of_same_witness': (r1 @ witness).item(),
        'readout_scope': 'one fixed site/channel after .02 time; linear conservative transport only',
        'interpretation': 'A fixed coordinate state can remain unchanged while structure changes its observable distinctions. This does not certify old semantic retrieval.',
        'cost_status': 'quadratic propagation-capacity proxy; not biochemical maintenance or growth metabolism',
        'production_opt_in': False,
        'review': 'Independent review advised fixed-budget prototype and separate memory/observation claims; incorporated. Capability and true metabolic law remain unreviewed/unvalidated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
