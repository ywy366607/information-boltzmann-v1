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


@pytest.mark.parametrize('alpha', [0., .4, 1.])
def test_exact_birth_ratio_eliminates_auxiliary_charge_and_resource_marks(alpha):
    from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
    from test_cgb_identity_material import material as identity_material
    m = GrowingIdentityMobileExchange.from_boundary(
        BoundaryIdentityMobileExchange.from_identity(identity_material(alpha=alpha)))
    old = np.array([8, 3, 2, 2, 6])
    available = old+[-2, 0, 0, -1, 1]
    rng = np.random.default_rng(608)
    # Conditional numerical states, not unpaid interventions in an individual.
    for _ in range(32):
        length = float(rng.uniform(.3, 1.5))
        xyz = m.coordinates[0]+[length, 0., 0.]
        q = float(rng.normal()*2)
        child_counts = rng.binomial(available, .5)
        full = m._birth_metrics(0, xyz, q, child_counts, old_counts=old,
                                old_charge=1.2, old_degree=0)
        marginal = m.birth_log_ratio(0, xyz, old_counts=old, old_charge=1.2, old_degree=0)
        assert abs(full['log_birth_acceptance_ratio']-marginal) < 3e-13


def test_marginal_ratio_is_the_normalized_joint_conditional_integral():
    import itertools
    from numpy.polynomial.hermite import hermgauss
    m = growing()
    old = np.array([2, 1, 0, 1, 0])
    available = old+[-2, 0, 0, -1, 1]
    xyz = m.coordinates[0]+[.8, 0., 0.]
    log_ratio = m.birth_log_ratio(0, xyz, old_counts=old, old_charge=.4, old_degree=0)
    nodes, weights = hermgauss(8)
    acceptance = normalization = 0.
    for child in itertools.product(*(range(int(n)+1) for n in available)):
        child = np.asarray(child)
        probability = np.exp(np.sum(gammaln(available+1)-gammaln(child+1)
                                    -gammaln(available-child+1)-available*math.log(2)))
        for node, weight in zip(nodes, weights):
            q = .2+node*math.sqrt(m.capacitance[0])
            full = m._birth_metrics(0, xyz, q, child, old_counts=old, old_charge=.4, old_degree=0)
            w = probability*weight/math.sqrt(math.pi)
            normalization += w
            acceptance += w*math.exp(min(0., full['log_birth_acceptance_ratio']))
    assert abs(normalization-1) < 1e-14
    assert abs(acceptance-math.exp(min(0., log_ratio))) < 1e-14


def test_thinned_recovery_is_the_original_attempt_times_exact_acceptance():
    m = growing()
    receipt = m.birth_trial(0, [1., 1.8, 1.], .3, [0, 1, 1, 1, 2], 0., insertion_index=0)
    i, j = receipt['parent_after'], receipt['child_after']
    log_ratio = m.death_log_ratio(i, j)
    assert abs(log_ratio-receipt['log_birth_acceptance_ratio']) < 1e-12
    rate = m._domain_attempt_rate(m.degrees()[i]-1)*math.exp(min(0., -log_ratio))
    rows = m.domain_generator_rates()['inverse_channels']
    assert any(row['parent']==i and row['child']==j and row['accepted_rate']==rate for row in rows)
    assert rate <= m._domain_attempt_rate(m.degrees()[i]-1)
    m.domain_reactions_enabled = False
    rates = m.domain_generator_rates()
    assert rates['birth_attempt_rate_upper_bound'] == rates['recovery_accepted_rate'] == 0


def test_marginal_sampling_has_explicit_schema_and_complete_future(tmp_path):
    m = growing()
    assert m.state_dict()['schema'] == 1 and 'domain_sampling' not in m.state_dict()
    m.domain_sampling = 'marginal'
    m._validate()
    assert m.state_dict()['schema'] == 2
    m.params = replace(m.params, assembly_rate=.3, material_diffusion=.1,
                       support_diffusion=.1, drive_diffusion=.1)
    for _ in range(8):
        m.step(.5, [2.])
    path = tmp_path/'marginal.npz'
    save_exchange(path, m)
    restored, _ = load_exchange(path)
    assert restored.domain_sampling == 'marginal'
    for _ in range(16):
        m.step(.5, [2.]); restored.step(.5, [2.])
    same(m.state_dict(), restored.state_dict())
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10
    corrupt = m.state_dict()
    corrupt['schema'] = 1
    with pytest.raises(ValueError, match='schema'):
        GrowingIdentityMobileExchange.from_state_dict(corrupt)


def test_growing_learning_drift_separates_native_exact_terms_and_domain_bounds():
    from information_boltzmann.cgb.learning_observation import native_learning_drift
    m = growing()
    drift = native_learning_drift(m)
    assert 'domains' in drift and 'spatial mark' in drift['scope']
    bounds = drift['domains']
    assert bounds['birth_attempt_rate_upper_bound'] >= 0
    assert bounds['domain_count_drift_bounds'][0] <= bounds['domain_count_drift_bounds'][1]


def test_material_fork_shares_only_immutable_genes_and_repeats_full_physics():
    from information_boltzmann.cgb.material_fork import fork_material
    m = growing();m.domain_sampling = 'marginal'
    original = m.state_dict()
    fork = fork_material(m)
    assert fork.program is m.program
    for name in m.program.array_names:
        assert getattr(fork, 'gene_'+name) is getattr(m, 'gene_'+name)
    for name in ('_normalized', '_occ_left', '_strength_left'):
        assert getattr(fork.program, name) is getattr(m.program, name)
        assert not getattr(fork.program, name).flags.writeable
    for name in m._node_arrays:
        assert not np.shares_memory(getattr(fork, name), getattr(m, name))
    fork.step(2., [2.])
    same(m.state_dict(), original)
    m.step(2., [2.])
    same(m.state_dict(), fork.state_dict())
    m.material[0] += 1
    assert m.material[0] == fork.material[0]+1


def test_original_capture_transport_and_release_change_same_physical_phase_ratio():
    m = growing()
    xyz0 = m.coordinates[0]+[0., .8, 0.]
    xyz1 = m.coordinates[1]+[0., .8, 0.]
    ratio0, ratio1 = m.birth_log_ratio(0, xyz0), m.birth_log_ratio(1, xyz1)
    m.apply_event(MaterialEvent('binding', 0, 1, 0))
    assert abs(m.birth_log_ratio(1, xyz1)-ratio1+math.log(2)) < 1e-13
    assert m.birth_log_ratio(0, xyz0) == ratio0
    fuel, depleted = int(m.charged[0]), int(m.depleted[0])
    m.apply_event(MaterialEvent('transport', 0, 1, 0))
    expected = math.log((fuel-1)*(depleted+1)/(fuel*(depleted+2)))+math.log(2)
    assert abs(m.birth_log_ratio(0, xyz0)-ratio0-expected) < 1e-13
    ratio = m.birth_log_ratio(0, xyz0)
    m.apply_event(MaterialEvent('release', 0))
    assert abs(m.birth_log_ratio(0, xyz0)-ratio) < 1e-13
    assert max(abs(float(v)) for v in m.balances().values()) < 1e-10
