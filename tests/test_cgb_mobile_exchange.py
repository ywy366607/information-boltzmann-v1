"""Finite material kinetics/geometry checks, not synthetic learning tasks."""
from dataclasses import replace
import math

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent, MobileExchange, MobileExchangeParams


def material(c=0, r=0, seed=17, **param_changes):
    params = replace(MobileExchangeParams(), **param_changes)
    return MobileExchange(np.array([[1., 1., 1.], [1.8, 1., 1.], [1.4, 1.7, 1.]]),
        np.full(3, 3), np.full(3, 2), np.full(3, 3), np.full(3, 2), carried=2,
        end_capacity=4, pre=[0, 1], post=[1, 2], bound=c, trace=r,
        charge=[.1, -.2, .3], params=params, seed=seed, port_nodes=[0, 2])


def opposite(event, old_edges):
    edge = old_edges if event.kind == 'assembly' and event.sign > 0 else event.edge
    if event.kind == 'assembly' and event.sign < 0:
        edge = -1
    return replace(event, sign=-event.sign, edge=edge)


@pytest.mark.parametrize('c,r', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_every_reaction_has_exact_joint_energy_and_detailed_balance(c, r):
    m = material(c, r)
    events = m.events()
    assert {'assembly', 'binding', 'thermal', 'write', 'transport', 'release',
            'conversion', 'active', 'exchange'} <= {e.kind for e in events}
    for event in events:
        clone = MobileExchange.from_state_dict(m.state_dict())
        before_u, before_g = clone.energy(), clone.free_energy()
        du, dg = clone.event_delta(event)
        forward = clone.event_rate(event)
        clone.apply_event(event)
        assert abs(clone.energy()-before_u-du) < 3e-14
        assert abs(clone.free_energy()-before_g-dg) < 8e-14
        reverse = opposite(event, len(m.pre))
        assert reverse in clone.events()
        backward = clone.event_rate(reverse)
        assert abs(math.log(forward/backward)+dg) < 2e-14
        clone.apply_event(reverse)
        assert abs(clone.energy()-before_u) < 3e-14
        assert abs(clone.free_energy()-before_g) < 8e-14
        for value in clone.balances().values():
            assert abs(float(value)) < 1e-12


def test_elastic_force_is_the_same_energy_derivative_and_preserves_rigid_symmetry():
    m = material()
    force = m.forces()
    np.testing.assert_allclose(force.sum(axis=0), np.zeros(3), atol=1e-15)
    h = 1e-5
    for i in range(m.n):
        for axis in range(3):
            old = m.coordinates[i, axis]
            m.coordinates[i, axis] = old+h
            plus = m.energy()
            m.coordinates[i, axis] = old-h
            minus = m.energy()
            m.coordinates[i, axis] = old
            assert abs((plus-minus)/(2*h)+force[i, axis]) < 2e-9
    before = m.energy()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    m.coordinates = m.coordinates@rotation.T+np.array([3., 0., 0.])
    assert abs(m.energy()-before) < 1e-13
    np.testing.assert_allclose(m.forces(), force@rotation.T, atol=1e-15)


def test_transport_creates_a_new_encounter_and_paid_storage_without_a_candidate_table():
    m = MobileExchange([[.6, .6, .6], [2.4, .6, .6]], [2, 2], [1, 1], [3, 3], [1, 1])
    assert len(m.encounters()) == 0 and len(m.pre) == 0 and len(m.trace) == 0
    assert m.move_trial(0, [1.1, .6, .6], .4)
    assert len(m.encounters()) == 1
    event = MaterialEvent('assembly', 0, 1)
    assert event in m.events()
    before_b, before_p, before_d = m.material.copy(), m.charged.copy(), m.depleted.copy()
    m.apply_event(event)
    assert len(m.pre) == len(m.trace) == len(m.bound) == 1
    assert m.trace[0] == m.bound[0] == 0
    assert m.material[1] == before_b[1]-1 and m.charged[1] == before_p[1]-1
    assert m.depleted[1] == before_d[1]+1
    np.testing.assert_array_equal(m.remaining_ends(), [2, 2])
    for value in m.balances().values():
        assert abs(float(value)) < 1e-12


def test_unreleased_configuration_keeps_its_carrier_until_actual_recovery():
    m = material(c=1, r=1)
    assert not any(e.kind == 'assembly' and e.sign < 0 for e in m.events())
    m.apply_event(MaterialEvent('binding', 0, 1, 0, -1))
    assert not any(e.kind == 'assembly' and e.sign < 0 and e.edge == 0 for e in m.events())
    m.apply_event(MaterialEvent('thermal', 0, 1, 0, -1))
    removal = MaterialEvent('assembly', 0, 1, 0, -1)
    assert removal in m.events()
    before = m.material.sum()
    m.apply_event(removal)
    assert len(m.pre) == 1 and m.material.sum() == before+1
    assert len(m.trace) == len(m.pre)
    for value in m.balances().values():
        assert abs(float(value)) < 1e-12


def test_motion_acceptance_pays_thermal_energy_and_rejects_invalid_geometry():
    m = material()
    before = m.energy()
    assert m.move_trial(0, [.95, 1., 1.], 0.)
    delta = m.energy()-before
    assert delta > 0 and abs(m.motion_heat+delta) < 1e-13
    saved = m.coordinates.copy()
    assert not m.move_trial(0, m.coordinates[1].copy(), 0.)
    assert not m.move_trial(0, [-1., 1., 1.], 0.)
    assert not m.move_trial(0, [4.9, 4.9, 4.9], 0.)
    np.testing.assert_array_equal(m.coordinates, saved)
    assert abs(float(m.balances()['energy'])) < 1e-12


def test_local_transport_recovers_force_drift_in_small_step_limit():
    m = MobileExchange([[1., 1., 1.], [2.1, 1., 1.]], [2, 2], [1, 1], [3, 3], [1, 1],
        pre=[0], post=[1], params=MobileExchangeParams(mobility=.05))
    nodes, weights = hermgauss(15)
    offsets = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 3)
    w = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing='ij'), axis=-1), axis=-1).ravel()/np.pi**1.5
    target = m.params.mobility*m.forces()[0, 0]
    errors = []
    old_energy = .5*m.params.elastic_stiffness*(1.1-m.params.rest_length)**2
    for dt in (.01, .0025, .000625):
        dx = offsets*np.sqrt(4*m.params.mobility*dt)
        separation = m.coordinates[0]+dx-m.coordinates[1]
        du = .5*m.params.elastic_stiffness*(np.linalg.norm(separation, axis=1)-m.params.rest_length)**2-old_energy
        accepted = np.minimum(1., np.exp(-du))
        drift = np.dot(w, dx[:, 0]*accepted)/dt
        errors.append(abs(drift-target))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < .003


def test_full_material_world_and_three_rng_streams_restore_exact_future(tmp_path):
    a = material(seed=34, assembly_rate=.1, bulk_attempt=1., surface_attempt=1.,
        write_rate=.1, recovery_rate=.1, binding_rate=.1, transport_rate=.1)
    for _ in range(5):
        a.step(.5, np.array([3., -3.]))
    save_exchange(tmp_path/'mobile.npz', a, {'clock': 'same physical steps'})
    b, meta = load_exchange(tmp_path/'mobile.npz')
    assert meta['clock'] == 'same physical steps'
    assert isinstance(b, MobileExchange)
    for _ in range(10):
        a.step(.5, np.array([-3., 3.]))
        b.step(.5, np.array([-3., 3.]))
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.pre, b.pre)
    np.testing.assert_array_equal(a.post, b.post)
    for name in a._ledgers:
        assert getattr(a, name) == getattr(b, name)
    assert a.reaction_counts == b.reaction_counts
    assert a.rng_motion.bit_generator.state == b.rng_motion.bit_generator.state
    assert a.rng_reaction.bit_generator.state == b.rng_reaction.bit_generator.state
    assert a.rng_field.bit_generator.state == b.rng_field.bit_generator.state
    for value in a.balances().values():
        assert abs(float(value)) < 1e-10


def test_geometry_follows_finite_ends_with_no_branch_scheduler():
    m = MobileExchange([[1., 1., 1.], [1.8, 1., 1.], [1., 1.8, 1.], [1., 1., 1.8]],
        np.full(4, 4), np.ones(4), np.full(4, 5), np.ones(4), end_capacity=3)
    for j in (1, 2, 3):
        event = MaterialEvent('assembly', 0, j)
        assert event in m.events()
        m.apply_event(event)
    assert m.degrees()[0] == 3 and m.remaining_ends()[0] == 0
    assert not any(e.kind == 'assembly' and e.sign > 0 and 0 in (e.i, e.j) for e in m.events())
    assert len(m.trace) == 3


def test_bad_composition_geometry_and_corrupt_archives_rejected():
    with pytest.raises(ValueError): MobileExchange([[1., 1., 1.]], [1.2], [1], [1], [1])
    with pytest.raises(ValueError): MobileExchange([[1., 1., 1.], [1.1, 1., 1.]], [1, 1], [1, 1], [1, 1], [1, 1])
    with pytest.raises(ValueError): MobileExchangeParams(contact_radius=.1)
    state = material().state_dict()
    state['arrays']['material'][0] += 1
    with pytest.raises(ValueError, match='accounting'):
        MobileExchange.from_state_dict(state)
    m = material(c=1, r=1)
    with pytest.raises(ValueError, match='Unavailable'):
        m.apply_event(MaterialEvent('assembly', 0, 1, 0, -1))
    state = m.state_dict()
    state['capacity'][0] = .5
    with pytest.raises(ValueError, match='Corrupt'):
        MobileExchange.from_state_dict(state)


def test_pair_field_exchange_obeys_exact_fluctuation_dissipation_moments():
    m = MobileExchange([[1., 1., 1.], [1.8, 1., 1.]], [2, 2], [1, 1], [3, 3], [1, 1],
        pre=[0], post=[1], charge=[1., -1.], capacitance=[2., 3.],
        unit_conductance=.3, trace=0)
    # Supply fixed standard-normal quadrature nodes directly to the exact OU
    # update; no sampling tolerance masks a missing thermal covariance.
    nodes, weights = hermgauss(9)
    samples, heats = [], []
    before = m.energy()
    old = m.charge.copy()
    class FixedNormal:
        def __init__(self, value): self.value = value
        def normal(self): return self.value
    for z in nodes*np.sqrt(2):
        m.charge[:] = old
        m.rng_field = FixedNormal(z)
        m._field(.7, None)
        samples.append(old[0]-m.charge[0])
        heats.append(before-m.energy())
    w = weights/np.sqrt(np.pi)
    mean = np.dot(w, samples)
    variance = np.dot(w, (np.array(samples)-mean)**2)
    inv = 1/2+1/3
    expected_mean = (1/2+1/3)*(-np.expm1(-.7*.3*inv))/inv
    expected_variance = -np.expm1(-2*.7*.3*inv)/inv
    assert abs(mean-expected_mean) < 1e-15
    assert abs(variance-expected_variance) < 1e-15
    assert min(heats) < 0.  # The same bath can supply energy; no clipping.
    assert abs(m.charge.sum()-old.sum()) < 1e-15
