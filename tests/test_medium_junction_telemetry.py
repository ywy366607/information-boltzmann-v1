"""Read-only physical junction telemetry, not a capability benchmark."""
from dataclasses import replace
import json

import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.medium_health import spatial_medium_snapshot
from information_boltzmann.runtime.training import belief_tensors


def candidate(enabled):
    torch.manual_seed(739)
    return PlasticMediumPorts3D(vocab_size=13, shape=(2, 2, 2), channels=8,
        material_width=3, hidden=4, heads=1, queries=1, anisotropic_transport=True,
        bath_type='conductance', short_term_plasticity=True,
        activity_adaptation=True, hopf_recomposition=enabled,
        structure_options=dict(resource_density=4., speed_reference=4.,
            structure_time=1., prior_std=.2, initial_std=.2,
            maintenance_supply=2., initial_dual=.1)).double()


def test_disabled_and_zero_junction_have_actual_baseline_current():
    legacy, enabled = candidate(False), candidate(True)
    legacy_report = spatial_medium_snapshot(legacy, legacy.initial_belief())
    enabled_report = spatial_medium_snapshot(enabled, enabled.initial_belief())
    assert legacy_report['hopf_branch'] is None
    junction = enabled_report['hopf_branch']
    assert junction['protocol'] == 'capacity_paid_persistent_flux_junction_v2'
    assert junction['added_state_elements'] == 0
    # Reference material cells are fixed at 8x8x4; this observation grid is 2^3.
    assert junction['physical_length'] == [.125, .125, .25]
    assert junction['resource_owner_rows'] == [0, 1, 2]
    assert all(row == [0., 0., 0.] for row in junction['rates'])
    assert all(row == [0., 0., 0.] for row in junction['signed_fraction'])
    assert junction['capacity_metric'] == 'squared_coupling_norm'
    assert junction['capacity_balance_error'] == 0
    torch.testing.assert_close(torch.tensor(legacy_report['effective_transport_factor']),
                               torch.tensor(enabled_report['effective_transport_factor']))
    assert enabled_report['transport_energy_current'] == legacy_report['transport_energy_current']
    json.dumps(enabled_report, allow_nan=False)


def test_snapshot_reports_paid_physical_allocation_and_keeps_all_state():
    net = candidate(True)
    belief = net.initial_belief()
    values = torch.arange(belief.medium.field.numel(), dtype=torch.float64)
    values = values.reshape_as(belief.medium.field) / 10
    belief = replace(belief, medium=replace(belief.medium, field=values,
        flux=(values + .1, values + .2, values + .4)))
    with torch.no_grad():
        net.hopf_pathway.gate.bias.copy_(torch.tensor([.4, -.2, .1], dtype=torch.float64))
    before = [x.clone() for x in belief_tensors(belief)]
    rng = torch.get_rng_state().clone()
    result = spatial_medium_snapshot(net, belief)
    j = result['hopf_branch']
    assert j['capacity_balance_error'] < 1e-12
    assert 'not the frozen allocation tape' in j['scope']
    assert len(j['forest']) == 3
    actual = torch.tensor(result['effective_transport_factor'], dtype=torch.float64)
    paid = torch.tensor(j['full_row_capacity'], dtype=torch.float64)
    transport = torch.tensor(j['transport_row_capacity'], dtype=torch.float64)
    junction = torch.tensor(j['junction_row_capacity'], dtype=torch.float64)
    torch.testing.assert_close(actual.norm(dim=-1), transport)
    torch.testing.assert_close(transport.square() + junction.square(), paid.square())
    fractions = torch.tensor(j['signed_fraction'], dtype=torch.float64)
    torch.testing.assert_close(junction.square() / paid.square(), fractions.square())
    assert 'not electric consumption or power' in j['capacity_scope']
    torch.testing.assert_close(torch.tensor(j['flux_energy_shares']).sum(-1), torch.ones(8))
    expected_current = net.medium.transport_energy_current(
        belief.medium, actual.reshape(1, 2, 2, 2, 3, 3)).reshape(8, 3)
    torch.testing.assert_close(torch.tensor(result['transport_energy_current'], dtype=torch.float64),
                               expected_current)
    assert torch.equal(rng, torch.get_rng_state())
    for original, current in zip(before, belief_tensors(belief)):
        torch.testing.assert_close(original, current, atol=0, rtol=0)
    assert all(p.grad is None for p in net.parameters())
    json.dumps(result, allow_nan=False)
