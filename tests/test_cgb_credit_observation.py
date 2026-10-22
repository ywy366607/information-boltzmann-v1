"""Native stoichiometry and observational roles, not a learning benchmark."""
import math

import numpy as np
import pytest

from information_boltzmann.cgb.credit_observation import (
    contact_roles, resource_bandwidth, support_return_paths)
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.mobile_exchange import MobileExchange, MobileExchangeParams, MaterialEvent
from information_boltzmann.cgb.learning_observation import parameter_quench
from scripts.ib.generative.compare_support_transport import same


def material():
    return MobileExchange([[1., 1., 1.], [1.8, 1., 1.], [2.6, 1., 1.]],
                          material=2, free=0, charged=[2, 1, 0], depleted=[1, 0, 1],
                          pre=[0, 1], post=[1, 2], trace=[1, 0], bound=[0, 1],
                          params=MobileExchangeParams(capture_coupling=2., bound_energy=-2.))


def test_observers_preserve_all_state_rng_and_distinguish_structure_from_tag():
    m = material()
    original = m.state_dict()
    row = contact_roles(m, 0)
    assert row['installed'] and row['activity_configuration'] == 1
    assert row['capture_rate'] == 0 and row['dismantle_rate'] == 0
    assert row['actual_conductance'] > row['basal_installed_conductance']
    assert support_return_paths(m)
    assert resource_bandwidth(m)['nodes'][0]['use_rate'] > 0
    same(original, m.state_dict())


def test_actual_downstream_transport_and_release_deliver_capture_resource():
    m = material()
    path = support_return_paths(m)[0]
    assert path['upstream_edge'] == 0 and path['downstream_edge'] == 1
    m.apply_event(MaterialEvent('transport', 1, 2, 1))
    assert m.carried[1] == 1 and m.bound[1] == 0 and m.charged[1] == 0
    m.apply_event(MaterialEvent('release', 1))
    row = contact_roles(m, 0)
    expected = m.params.binding_rate/(1+math.exp(m.params.bound_energy-m.params.capture_coupling))
    assert row['capture_rate'] == pytest.approx(expected)
    m.apply_event(MaterialEvent('binding', 0, 1, 0))
    assert m.bound[0] == 1 and m.free[1] == 0
    # Clearing the activity configuration leaves the paid, supported carrier.
    m.apply_event(MaterialEvent('thermal', 0, 1, 0, -1))
    cleared = contact_roles(m, 0)
    assert cleared['installed'] and cleared['bound_support'] == 1
    assert cleared['actual_conductance'] == cleared['basal_installed_conductance']
    assert cleared['dismantle_rate'] == 0
    for value in m.balances().values():
        np.testing.assert_allclose(value, 0., atol=1e-12)


def test_marked_and_unmarked_preparations_match_charge_and_drive():
    original = material()
    m = fork_material(original)
    n = fork_material(original)
    m.apply_event(MaterialEvent('write', 0, 1, 0, -1))
    n.apply_event(MaterialEvent('active', 0, 1, 0, -1))
    for name in ('charge', 'charged', 'depleted', 'material', 'free', 'bound'):
        np.testing.assert_array_equal(getattr(m, name), getattr(n, name))
    assert m.trace[0] == 0 and n.trace[0] == 1
    assert contact_roles(m, 0)['basal_installed_conductance'] == contact_roles(n, 0)['basal_installed_conductance']


def test_native_usage_is_limited_by_available_drive_and_actual_recovery():
    m = material()
    start = int(m.charged.sum())
    use = recovery = 0
    for _ in range(2):
        e = MaterialEvent('active', 0, 1, 0)
        assert e in m.events()
        m.apply_event(e)
        use += 1
    assert MaterialEvent('active', 0, 1, 0) not in m.events()
    assert resource_bandwidth(m)['nodes'][0]['use_rate'] == 0
    m.apply_event(MaterialEvent('active', 0, 1, 0, -1))
    recovery += 1
    assert MaterialEvent('active', 0, 1, 0) in m.events()
    assert int(m.charged.sum()) == start+recovery-use
    assert use <= start+recovery


def test_flat_response_control_preserves_capture_and_all_physical_compositions():
    m = material()
    m.apply_event(MaterialEvent('transport', 1, 2, 1))
    m.apply_event(MaterialEvent('release', 1))
    old = fork_material(m)
    before = contact_roles(m, 0)
    receipt = parameter_quench(m, gate_high=m.params.gate_low)
    after = contact_roles(m, 0)
    assert receipt['external_work'] == 0 and receipt['free_energy_change'] == 0
    assert after['actual_conductance'] == after['basal_installed_conductance']
    assert before['capture_rate'] == after['capture_rate']
    for name in m._arrays:
        np.testing.assert_array_equal(getattr(m, name), getattr(old, name))
    for name in ('rng_reaction', 'rng_motion', 'rng_field'):
        assert getattr(m, name).bit_generator.state == getattr(old, name).bit_generator.state
