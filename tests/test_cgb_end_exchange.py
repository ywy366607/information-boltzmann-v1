"""Physical endpoint composition and port-response checks, not capability tasks."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from information_boltzmann.cgb.end_exchange import EndExchange
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.spatial_exchange import SpatialExchangeParams


def cross(params=None, bridge=.2, scale=1.):
    xyz = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                    [0., -1., 0.], [0., 0., 0.], [0., 0., 0.]])
    return EndExchange(np.array([0, 4, 2, 5, 4]), np.array([4, 1, 5, 3, 5]),
        np.r_[np.full(4, .3), bridge], np.full(6, .2), np.full(5, .05),
        np.full(6, .1), np.full(6, .8), np.full(6, .2), np.ones(5),
        np.full(6, .8), np.linspace(-.3, .3, 6), unit_conductance=1.,
        params=params, coordinates=xyz, end_capacity=np.r_[np.ones(4), 1.5, 1.5],
        reaction_scale=scale, trace=.2)


def test_joint_endpoint_chemical_potential_sets_forward_backward_ratio():
    m = cross()
    plus, minus = m.fluxes()['assembly']
    for e in range(5):
        g, h = m.free_energy(), 1e-7
        post = m.post[e]
        m.capacity[e] += h
        m.material[post] -= h
        m.charged[post] -= h
        m.depleted[post] += h
        derivative = (m.free_energy()-g)/h
        m.capacity[e] -= h
        m.material[post] += h
        m.charged[post] += h
        m.depleted[post] -= h
        assert abs(derivative+np.log(plus[e]/minus[e])) < 4e-6


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_endpoint_reactions_keep_all_compositions_charge_and_energy_closed(dt):
    m = cross()
    for _ in range(8):
        change = m.step(dt)
        assert change['free_energy_drop'] >= -1e-10
        assert np.all(m.remaining_ends() >= 0)
        assert np.all(m.capacity >= m.bound)
        for name, value in m.balances().items():
            assert np.max(np.abs(value), initial=0.) < 2e-10, name


def test_shared_finite_ends_limit_competing_bridges_without_clipping():
    p = SpatialExchangeParams(assembly_rate=10., binding_rate=0., transport_rate=0.,
        release_rate=0., conversion_rate=0., active_rate=0., write_rate=0.,
        recovery_rate=0., material_diffusion=0., support_diffusion=0., drive_diffusion=0.,
        bulk_attempt=1., surface_attempt=0., cohesion_energy=0.)
    m = cross(p, scale=1.)
    for _ in range(4):
        m.step(1000.)
        assert np.all(m.remaining_ends() >= 0.)
        assert m.capacity[0]+m.capacity[1]+m.capacity[4] <= 1.5
        assert m.capacity[2]+m.capacity[3]+m.capacity[4] <= 1.5


def electrical_panel(bridge):
    p = SpatialExchangeParams(assembly_rate=0., binding_rate=0., transport_rate=0.,
        release_rate=0., conversion_rate=0., active_rate=0., write_rate=0.,
        recovery_rate=0., material_diffusion=0., support_diffusion=0., drive_diffusion=0.,
        gate_low=1., gate_high=1.)
    m = cross(p)
    # Fresh physical preparation, never an intervention into a saved organism.
    m.capacity[:] = np.r_[np.ones(4), bridge]
    m.bound[:] = 0.
    m.end_capacity[:] = np.r_[np.ones(4), 3., 3.]
    m.initial_end_inventory = m.end_capacity.copy()
    m.charge[:] = np.r_[1., np.zeros(5)]
    m.initial_charge = m.charge.copy()
    m.initial_material = m.material+m._sum(m.post, m.capacity)
    m.initial_energy, m.initial_free_energy = m.energy(), m.free_energy()
    return m


def test_colocated_crossing_has_no_cross_current_before_a_bridge_exists():
    x, y = electrical_panel(0.), electrical_panel(1.)
    assert np.array_equal(x.coordinates[4], x.coordinates[5])
    assert x.charge[4] == x.charge[5]
    x.step(.5)
    y.step(.5)
    np.testing.assert_array_equal(x.charge[[2, 3, 5]], np.zeros(3))
    assert np.all(y.charge[[2, 3, 5]] > 0.)
    assert abs(x.charge.sum()-y.charge.sum()) < 1e-14


def test_independent_boundary_response_matches_cross_and_joined_limits():
    # Fix the four external voltages. Solve the two retained internal domain
    # potentials independently of the implementation's time integrator.
    for bridge in (0., .1, 1.):
        m = electrical_panel(bridge)
        g = m.conductance.copy()
        if bridge:
            g[-1] *= 10.  # A measured material conductance, not a domain merge.
        lap = np.zeros((6, 6))
        for a, b, conductance in zip(m.pre, m.post, g):
            lap[a, a] += conductance
            lap[b, b] += conductance
            lap[a, b] -= conductance
            lap[b, a] -= conductance
        v = np.r_[1., np.zeros(5)]
        v[4:] = np.linalg.solve(lap[4:, 4:], -lap[4:, :4]@v[:4])
        k = g[-1]
        np.testing.assert_allclose(v[4:], [(k+2)/(4*k+4), k/(4*k+4)], atol=1e-15)
        if not bridge:
            assert v[5] == 0.


def test_zero_attempt_scale_freezes_only_selected_contact_reaction():
    m = cross(scale=np.r_[np.ones(4), 0.])
    old = m.capacity.copy()
    m.step(1.)
    assert m.capacity[-1] == old[-1]
    assert not np.array_equal(m.capacity[:4], old[:4])
    assert np.any(m.trace_written != 0.)


def test_trace_capture_and_endpoint_constraints_share_one_free_energy():
    # Resolve enough physical turnover to measure the capture->retention
    # path; the default slow assembly gives only ~4e-8 change over 5 ms.
    a = cross(SpatialExchangeParams(assembly_rate=.05))
    b = cross(replace(a.params, capture_coupling=0.))
    for _ in range(10):
        a.step(.5)
        b.step(.5)
    assert np.max(np.abs(a.bound-b.bound)) > 1e-5
    assert np.max(np.abs(a.capacity-b.capacity)) > 1e-7
    assert np.all(a.remaining_ends() >= 0.)


def test_colocated_contact_does_not_merge_material_diffusion_reservoirs():
    p = SpatialExchangeParams(assembly_rate=0., binding_rate=0., transport_rate=0.,
        release_rate=0., conversion_rate=0., active_rate=0., write_rate=0.,
        recovery_rate=0., material_diffusion=.5, support_diffusion=0., drive_diffusion=0.)
    m = cross(p)
    m.material[:] = 0.
    m.material[0] = 1.
    m.initial_material = m.material+m._sum(m.post, m.capacity)
    m.initial_energy, m.initial_free_energy = m.energy(), m.free_energy()
    for _ in range(3):
        m.step(1.)
    assert m.material[4] > 0.
    np.testing.assert_array_equal(m.material[[2, 3, 5]], np.zeros(3))
    assert abs(m.material.sum()-1.) < 1e-14


def test_full_archive_restores_end_pairs_rates_and_future_bitwise(tmp_path):
    a = cross(scale=np.r_[np.ones(4), .7])
    for _ in range(3):
        a.step(.5)
    save_exchange(tmp_path/'ends.npz', a, {'scope': 'resolved material ends'})
    b, meta = load_exchange(tmp_path/'ends.npz')
    assert isinstance(b, EndExchange) and meta['scope'] == 'resolved material ends'
    for _ in range(4):
        a.step(.5)
        b.step(.5)
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in a._ledgers:
        assert getattr(a, name) == getattr(b, name)


def test_end_competition_converges_to_independent_nonlinear_ode():
    p = SpatialExchangeParams(assembly_rate=.3, binding_rate=0., transport_rate=0.,
        release_rate=0., conversion_rate=0., active_rate=0., write_rate=0., recovery_rate=0.,
        material_diffusion=0., support_diffusion=0., drive_diffusion=0.,
        bulk_attempt=.2, surface_attempt=.7, cohesion_energy=.4)
    source = cross(p)
    source.unit_conductance[:] = 0.
    start = np.r_[source.capacity, source.material, source.charged, source.depleted]
    pre, post = source.pre, source.post
    def rhs(_, y):
        n, b, fuel, waste = y[:5], y[5:11], y[11:17], y[17:23]
        incident = np.bincount(pre, weights=n, minlength=6)+np.bincount(post, weights=n, minlength=6)
        ends = source.end_capacity-incident
        neighbors = incident[pre]+incident[post]-2*n
        rate = .3*(.2+.7*neighbors)
        j = rate*(b[post]*fuel[post]*(1.-n)*ends[pre]*ends[post]
            -np.exp(p.structure_energy-.4*neighbors-p.charged_energy)*(n-.05)*waste[post])
        debit = np.bincount(post, weights=j, minlength=6)
        return np.r_[j, -debit, -debit, debit]
    exact = solve_ivp(rhs, [0., 2.], start, rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for dt in (.1, .05, .025):
        m = EndExchange.from_state_dict(source.state_dict())
        for _ in range(round(2/dt)):
            m.step(dt)
        errors.append(np.linalg.norm(np.r_[m.capacity, m.material, m.charged, m.depleted]-exact))
    assert 1.6 < errors[0]/errors[1] < 2.4
    assert 1.6 < errors[1]/errors[2] < 2.4


def test_corrupt_or_overcommitted_end_state_is_rejected():
    a = cross()
    state = a.state_dict()
    state['arrays']['end_capacity'][4] = .1
    with pytest.raises(ValueError, match='endpoint inventory'):
        EndExchange.from_state_dict(state)
    state = a.state_dict()
    state['arrays']['initial_end_inventory'][4] += .1
    with pytest.raises(ValueError, match='accounting'):
        EndExchange.from_state_dict(state)
