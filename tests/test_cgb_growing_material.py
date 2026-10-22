"""Conservation, reversible phase measure, real indexing and full continuation."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.special import gammaln

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from scripts.ib.generative.compare_support_transport import same
from test_cgb_boundary_material import material


def growing(enabled=True):
    m = material(1, 1, 1)
    m.field_scheme = 'symmetric_ou'
    return GrowingIdentityMobileExchange.from_boundary(m, domain_reactions_enabled=enabled)


@pytest.mark.parametrize('slot', [0, 1, 2])
def test_material_domain_birth_and_inverse_include_all_energy_composition_and_labels(slot):
    m = growing()
    old = {name:getattr(m, name).copy() for name in m._node_arrays}
    before = m.energy(), m.free_energy()
    bridge = m.pre.copy(), m.post.copy(), m.port_nodes.copy()
    # Explicit allowed-transition control, not a spontaneous-growth claim.
    receipt = m.birth_trial(0, [1., 1.8, 1.], .3, [0, 1, 1, 1, 2], 0., insertion_index=slot)
    assert receipt['accepted'] and m.n == 3
    assert abs(m.energy()-before[0]-receipt['delta_u']) < 1e-12
    assert abs(m.free_energy()-before[1]-receipt['delta_g']) < 1e-12
    assert abs(receipt['delta_g']-receipt['delta_g_physical']-math.log(3)) < 1e-13
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-12
    parent = receipt['parent_after']
    assert m.port_nodes[0] == parent
    assert m.cell_type[slot] == m.cell_type[parent]
    assert m.capacitance[slot] == m.capacitance[parent]
    # The actual bridge is immediately usable by the original local generator.
    assert MaterialEvent('write', parent, slot, len(m.pre)-1) in m.events()
    reverse = m.death_trial(parent, slot, len(m.pre)-1, 0.)
    assert reverse['accepted'] and m.n == 2
    assert abs(reverse['delta_u']+receipt['delta_u']) < 1e-12
    assert abs(reverse['delta_g']+receipt['delta_g']) < 1e-12
    assert abs(reverse['log_birth_acceptance_ratio']-receipt['log_birth_acceptance_ratio']) < 1e-12
    for name in m._node_arrays:
        np.testing.assert_allclose(getattr(m, name), old[name], atol=1e-15, rtol=0)
    for actual, expected in zip((m.pre, m.post, m.port_nodes), bridge):
        np.testing.assert_array_equal(actual, expected)
    m._validate()
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-12


def test_birth_reverse_density_contains_partition_charge_and_no_empty_slot_factor():
    m = growing()
    receipt = m.birth_trial(0, [1., 1.8, 1.], .3, [0, 1, 1, 1, 2], 0., insertion_index=0)
    log_f = sum(receipt[key] for key in ('log_position_density', 'log_charge_density',
                                        'log_partition_probability'))
    log_r = receipt['log_birth_acceptance_ratio']
    # Forward actual density includes uniform insertion 1/(N+1). Reverse
    # density is one leaf transition. Their ratio is exp(-full delta G).
    log_forward = log_f-math.log(3)+min(0., log_r)
    log_reverse = min(0., -log_r)
    assert abs(log_forward-log_reverse+receipt['delta_g']) < 1e-12
    old_degree = m.degrees()[receipt['parent_after']]-1
    assert m._domain_attempt_rate(old_degree) == m._domain_attempt_rate(1)
    available = np.array([1, 3, 2, 3, 5])
    child = np.array([0, 1, 1, 1, 2])
    direct = np.sum(gammaln(available+1)-gammaln(child+1)
                    -gammaln(available-child+1)-available*math.log(2))
    assert abs(direct-receipt['log_partition_probability']) < 1e-12


def test_domain_configuration_retains_paid_support_and_trace_before_recovery():
    m = growing()
    receipt = m.birth_trial(0, [1., 1.8, 1.], .3, [0, 1, 1, 1, 2], 0., insertion_index=2)
    parent, child, edge = receipt['parent_after'], receipt['child_after'], len(m.pre)-1
    m.apply_event(MaterialEvent('write', parent, child, edge))
    assert (parent, child, edge) not in m.death_channels()
    with pytest.raises(ValueError, match='Unavailable'):
        m.death_trial(parent, child, edge, 0.)
    m.apply_event(MaterialEvent('write', parent, child, edge, -1))
    m.apply_event(MaterialEvent('binding', parent, child, edge))
    assert (parent, child, edge) not in m.death_channels()
    m.apply_event(MaterialEvent('binding', parent, child, edge, -1))
    assert (parent, child, edge) in m.death_channels()
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-11


def test_outside_geometry_is_a_null_attempt_with_no_unpaid_change():
    m = growing()
    before = m.state_dict()
    receipt = m.birth_trial(0, [-.1, 1., 1.], .3, [0, 1, 1, 1, 2], 0.)
    assert not receipt['accepted']
    after = m.state_dict()
    after['domain_attempts'] = before['domain_attempts']
    same(before, after)


def test_closed_growth_barrier_reproduces_parent_full_future():
    parent = material(1, 1, 1)
    parent.field_scheme = 'symmetric_ou'
    m = GrowingIdentityMobileExchange.from_boundary(parent, domain_reactions_enabled=False)
    for _ in range(12):
        parent.step(2., [2.]); m.step(2., [2.])
    for name in parent._arrays:
        if name != 'initial_material':
            np.testing.assert_array_equal(getattr(m, name), getattr(parent, name))
    np.testing.assert_array_equal(m.pre, parent.pre)
    np.testing.assert_array_equal(m.post, parent.post)
    for name in ('rng_reaction', 'rng_motion', 'rng_field'):
        assert getattr(m, name).bit_generator.state == getattr(parent, name).bit_generator.state
    assert m.t == parent.t and m.work_in == parent.work_in and m.heat_out == parent.heat_out
    assert m.domain_attempts == 0
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10


def test_full_domain_archive_repeats_actual_future_reactions_without_rebirth(tmp_path):
    m = growing()
    m.params = replace(m.params, assembly_rate=.3, material_diffusion=.1,
                       support_diffusion=.1, drive_diffusion=.1)
    receipt = m.birth_trial(0, [1., 1.8, 1.], .3, [0, 1, 1, 1, 2], 0., insertion_index=0)
    assert receipt['accepted']
    path = tmp_path/'grown.npz'
    save_exchange(path, m, {'actual_time_ms':m.t})
    restored, meta = load_exchange(path)
    same(m.state_dict(), restored.state_dict())
    assert restored.program.fingerprint() == m.program.fingerprint()
    assert not restored.cell_type.flags.writeable
    assert meta['actual_time_ms'] == m.t
    for _ in range(20):
        m.step(.5, [2.]); restored.step(.5, [2.])
    same(m.state_dict(), restored.state_dict())
    assert m.domain_attempts > 0
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10


def test_new_domain_uses_nonzero_inherited_identity_and_native_transmission():
    from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
    from test_cgb_identity_material import material as identity_material
    parent = BoundaryIdentityMobileExchange.from_identity(identity_material(alpha=.4))
    m = GrowingIdentityMobileExchange.from_boundary(parent)
    before = m.energy(), m.free_energy()
    assert abs(float(m.recognition_free_energy(0, 0))) > 1e-3
    receipt = m.birth_trial(0, [1., .2, 1.], .15, [0, 1, 1, 1, 1], 0., insertion_index=0)
    assert receipt['accepted']
    assert abs(m.energy()-before[0]-receipt['delta_u']) < 1e-12
    assert abs(m.free_energy()-before[1]-receipt['delta_g']) < 1e-12
    i, j, edge = receipt['parent_after'], receipt['child_after'], len(m.pre)-1
    expected = m.unit_conductance*m.params.gate_low*float(m._transport_factor(i, j))
    assert abs(m.bridge_conductance()[edge]-expected) < 1e-15
    source_support = m.free[j]
    m.apply_event(MaterialEvent('binding', i, j, edge))
    assert m.free[j] == source_support-1 and m.bound[edge] == 1
    m.apply_event(MaterialEvent('write', i, j, edge))
    assert m.trace[edge] == 1
    assert m.bridge_conductance()[edge] > expected
    m.step(.5, [.4, -.3])
    m._validate()
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10
