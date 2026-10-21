"""Physical/numerical contracts; not synthetic memory or learning studies."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.finite_exchange import ExchangeParams, FiniteExchange


def panel(params=None):
    return FiniteExchange(np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3),
                          np.array([1., .8, .9]), np.array([.2, .3, .4]),
                          np.array([.2, .3, .4]), np.array([.8, .7, .6]),
                          np.array([.2, .3, .4]), params)


def test_flux_ratios_obey_declared_free_energy_affinities():
    m = panel()
    p = m.params
    mu_f = np.log(m.free)
    mu_c = p.bound_energy+np.log(m.bound/(m.capacity-m.bound))
    mu_h = p.carried_energy+np.log(m.carried)
    mu_p, mu_d = p.charged_energy+np.log(m.charged), np.log(m.depleted)
    expected = dict(binding=mu_f[m.post]-mu_c,
                    transport=mu_c+mu_p[m.pre]-mu_h[m.pre]-mu_d[m.pre],
                    release=mu_h-mu_f)
    for name, (forward, backward) in m.fluxes().items():
        np.testing.assert_allclose(np.log(forward/backward), expected[name], atol=1e-14)
    assert m.entropy_rate() >= 0


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_closed_composition_energy_and_entropy_balances(dt):
    m = panel()
    for _ in range(40):
        result = m.step(dt)
        assert result['free_energy_drop'] >= -1e-12
        for residual in m.balances().values():
            assert np.max(np.abs(residual), initial=0.) < 2e-11
    assert m.charged.sum()+m.depleted.sum() == pytest.approx(3.)


def test_existing_contact_can_bind_without_an_enhancement_signal():
    # No tag/activity carrier occurs in this finite-contact physical model.
    m = FiniteExchange(np.array([0]), np.array([1]), [1.], [0., 1.], [0.],
                       [0., 0.], [.5, .5], [.5, .5],
                       ExchangeParams(transport_rate=0., release_rate=0.))
    m.step(10.)
    assert m.bound[0] > 0
    assert m.free[1] < 1.
    assert m.free_energy() < m.initial_free_energy


def test_transport_draws_on_finite_local_drive_composition():
    m = FiniteExchange(np.array([0]), np.array([0]), [1.], [0.], [.8],
                       [0.], [.01], [.99],
                       ExchangeParams(binding_rate=0., release_rate=0.))
    before = m.charged[0]
    m.step(1000.)
    assert 0 <= m.carried[0] <= before
    assert m.depleted[0] > .99
    np.testing.assert_allclose(m.balances()['drive'], 0., atol=1e-15)


def test_zero_support_inventory_cannot_create_support():
    m = FiniteExchange(np.array([0]), np.array([0]), [1.], [0.], [0.],
                       [0.], [.5], [.5])
    m.step(1e5)
    assert m.free.sum()+m.bound.sum()+m.carried.sum() == 0.


def test_zero_capacity_and_extreme_rates_preserve_feasible_pools():
    m = FiniteExchange(np.array([0, 0]), np.array([1, 1]), [0., 1.],
                       [1., 1.], [0., 0.], [1., 1.], [.5, .5], [.5, .5],
                       ExchangeParams(binding_rate=1e5, transport_rate=1e5, release_rate=1e5))
    m.step(1000.)
    assert m.bound[0] == 0.
    assert np.all(m.bound >= 0) and np.all(m.bound <= m.capacity)
    for value in (m.free, m.carried, m.charged, m.depleted):
        assert np.all(value >= 0)


def test_edge_order_does_not_grant_priority():
    a = panel()
    order = np.array([2, 0, 1])
    b = FiniteExchange(a.pre[order], a.post[order], a.capacity[order], a.free,
                       a.bound[order], a.carried, a.charged, a.depleted, a.params)
    for _ in range(40):
        a.step(.5)
        b.step(.5)
    for name in ('free', 'carried', 'charged', 'depleted'):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-14)
    np.testing.assert_allclose(a.bound, b.bound[np.argsort(order)], atol=1e-14)


def test_closed_one_location_equilibrium_independent_of_barrier_speed():
    initial = dict(pre=np.array([0]), post=np.array([0]), capacity=[1.],
                   free=[1.], bound=[.2], carried=[.3], charged=[.8], depleted=[.2])
    params = ExchangeParams(binding_rate=.5, transport_rate=.5, release_rate=.5,
                            bound_energy=-.5, carried_energy=.2, charged_energy=1.)
    expected_f = brentq(lambda f:f*(1+np.exp(-params.carried_energy))
                       +f/(np.exp(params.bound_energy)+f)-1.5, .001, 1.5)
    expected = [expected_f, expected_f/(np.exp(params.bound_energy)+expected_f),
                expected_f*np.exp(-params.carried_energy), 1/(1+np.exp(params.charged_energy))]
    for scale in (.5, 2.):
        m = FiniteExchange(**initial, params=replace(params,
            binding_rate=params.binding_rate*scale,
            transport_rate=params.transport_rate*scale,
            release_rate=params.release_rate*scale))
        for _ in range(600):
            m.step(1.)
        np.testing.assert_allclose([m.free[0], m.bound[0], m.carried[0], m.charged[0]],
                                   expected, atol=2e-7)


def test_step_refinement_matches_independent_continuous_equations():
    params = ExchangeParams(binding_rate=.7, transport_rate=.4, release_rate=.5,
                            bound_energy=-.5, carried_energy=.2, charged_energy=1.)
    start = np.array([1., .2, .4, .8, .2])
    def rhs(_, y):
        f, c, h, p, d = y
        binding = .7*(f*(1-c)-np.exp(-.5)*c)
        transit = .4*(c*p-np.exp(.2+.5-1.)*h*d*(1-c))
        release = .5*(h-np.exp(-.2)*f)
        return [-binding+release, binding-transit, transit-release, -transit, transit]
    exact = solve_ivp(rhs, [0, 2.], start, rtol=1e-11, atol=1e-13).y[:, -1]
    errors = []
    for dt in (.1, .05, .025):
        m = FiniteExchange(np.array([0]), np.array([0]), [1.], *([x] for x in start), params)
        for _ in range(round(2/dt)):
            m.step(dt)
        errors.append(np.linalg.norm(np.array([m.free[0],m.bound[0],m.carried[0],m.charged[0],m.depleted[0]])-exact))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_atomic_archive_restores_all_physical_state_and_future(tmp_path):
    m = panel()
    for _ in range(20):
        m.step(.25)
    path = tmp_path/'exchange.npz'
    save_exchange(path, m, {'scope':'numerical subsystem'})
    fork, provenance = load_exchange(path)
    assert provenance == {'scope':'numerical subsystem'}
    for _ in range(20):
        m.step(.25)
        fork.step(.25)
    for name in m._arrays:
        np.testing.assert_array_equal(getattr(m, name), getattr(fork, name))
    assert m.free_energy() == fork.free_energy() and m.t == fork.t


@pytest.mark.parametrize('change', [dict(binding_rate=-1.), dict(bound_energy=np.nan),
                                  dict(charged_energy=-1.), dict(bound_energy=1000.)])
def test_invalid_physical_or_numeric_parameters_rejected(change):
    with pytest.raises(ValueError):
        ExchangeParams(**change)


def test_tampered_composition_or_energy_archive_is_rejected():
    m = panel()
    state = m.state_dict()
    state['arrays']['free'][0] += 1.
    with pytest.raises(ValueError, match='accounting'):
        FiniteExchange.from_state_dict(state)
