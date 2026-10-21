"""Numerical/interface contracts, not a synthetic learning capability study."""
import numpy as np
import pytest
from scipy.linalg import expm

from information_boltzmann.cgb.checkpoint import load_medium, save_medium
from information_boltzmann.cgb.medium import CellClass, Medium, SynapseParams, TrophicParams
from information_boltzmann.cgb.support_transport import SupportTransport, SupportTransportParams


def trophic(**kwargs):
    values = dict(rule='constitutive', capture='tag', q0=0., kappa=.7,
                  tau_N=np.inf, tau_T=np.inf, phi=0.)
    values.update(kwargs)
    return TrophicParams(**values)


def test_closed_cycle_conserves_every_owner_and_module_without_minting():
    pre, post = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 2])
    free, bound = np.array([1., 2., 3.]), np.array([.1, .2, .3, .4])
    chain = SupportTransport(3, pre, post, free, bound, 0.,
                             SupportTransportParams(.4, .6, 0.))
    mass = free.sum() + bound.sum()
    for _ in range(300):
        chain.advance(free, bound, np.array([1., 3., .5, 2.]), np.ones(4),
                      np.ones(4, bool), .1, trophic())
    assert free.min() >= 0 and bound.min() >= 0 and chain.cargo.min() >= 0
    assert free.sum() + bound.sum() + chain.cargo.sum() == pytest.approx(mass, abs=2e-12)
    np.testing.assert_allclose(chain.balance(free, bound), 0., atol=2e-12)
    inventory = free + chain.cargo + np.bincount(pre, weights=bound, minlength=3)
    for module in ([0], [1, 2], [0, 2], [0, 1, 2]):
        member = np.isin(np.arange(3), module)
        incoming = chain.captured[~member[post] & member[pre]].sum()
        outgoing = chain.captured[member[post] & ~member[pre]].sum()
        assert inventory[member].sum() == pytest.approx(
            chain.initial[member].sum() + incoming - outgoing, abs=2e-12)


def test_open_sources_clearance_and_maintenance_are_paid_and_audited():
    free, bound = np.array([0., .1]), np.array([.02])
    chain = SupportTransport(2, [0], [1], free, bound, [0., .3],
                             SupportTransportParams(.2, .5, .1))
    p = trophic(tau_N=3., tau_T=4., phi=1.)
    for _ in range(200):
        chain.advance(free, bound, np.array([2.]), np.array([1.]),
                      np.ones(1, bool), .1, p)
    assert chain.cleared.sum() > 0 and chain.spent.sum() > 0
    assert chain.unpaid.sum() > 0
    np.testing.assert_allclose(chain.balance(free, bound), 0., atol=2e-12)
    assert free.sum() + bound.sum() + chain.cargo.sum() == pytest.approx(
        chain.initial.sum() + chain.imported.sum() - chain.cleared.sum()
        - chain.spent.sum(), abs=2e-12)


def test_huge_capture_competition_is_positive_and_rate_proportional():
    free, bound = np.array([1.]), np.zeros(3)
    chain = SupportTransport(1, [0, 0, 0], [0, 0, 0], free, bound, 0.)
    chain.advance(free, bound, np.array([1e8, 2e8, 7e8]), np.zeros(3),
                  np.ones(3, bool), 1e4, trophic())
    np.testing.assert_allclose(bound, [.1, .2, .7], atol=1e-15)
    assert free[0] >= 0
    np.testing.assert_allclose(chain.balance(free, bound), 0., atol=1e-15)


def test_zero_inventory_no_source_cannot_produce_support():
    free, bound = np.zeros(2), np.zeros(1)
    chain = SupportTransport(2, [0], [1], free, bound, 0.)
    for _ in range(20):
        chain.advance(free, bound, np.ones(1), np.ones(1), np.ones(1, bool),
                      100., trophic(phi=1.))
    assert free.sum() + bound.sum() + chain.cargo.sum() == 0.
    assert chain.unpaid.sum() > 0


def test_reverse_chain_requires_transport_and_release_without_same_step_cascade():
    # Neural direction 0 -> 1 -> 2; support enters only at boundary 2.
    results = []
    for rate in (0., .5):
        free, bound = np.array([0., 0., 1.]), np.zeros(2)
        chain = SupportTransport(3, [0, 1], [1, 2], free, bound, 0.,
                                 SupportTransportParams(rate, .4, 0.))
        chain.advance(free, bound, np.ones(2), np.zeros(2), np.ones(2, bool), .1, trophic())
        assert chain.cargo.sum() == 0 and bound[0] == 0 and free[0] == 0
        for _ in range(499):
            chain.advance(free, bound, np.ones(2), np.zeros(2), np.ones(2, bool), .1, trophic())
        results.append((free.copy(), bound.copy(), chain.cargo.copy()))
    assert results[0][0][0] == 0 and results[0][1][0] == 0
    assert results[0][2].sum() == 0
    assert results[1][0][0] > .8


def test_frozen_reactions_converge_to_independent_matrix_exponential():
    # Variables [F_0,F_1,T_01,H_0,H_1]. No external drive or maintenance.
    matrix = np.zeros((5, 5))
    a, nu, rho, dn, dt, dh = .7, .4, .6, .1, .2, .15
    matrix[0, 0] = matrix[1, 1] = -dn
    matrix[1, 1] -= a
    matrix[2, 1] = a
    matrix[2, 2] = -(nu + dt)
    matrix[3, 2] = nu
    matrix[3, 3] = matrix[4, 4] = -(rho + dh)
    matrix[0, 3] = matrix[1, 4] = rho
    start = np.array([.2, 1., .3, .1, .05])
    exact = expm(matrix * 2.) @ start
    errors = []
    for h in (.1, .05, .025):
        free, bound = start[:2].copy(), start[2:3].copy()
        chain = SupportTransport(2, [0], [1], free, bound, 0.,
                                 SupportTransportParams(nu, rho, dh))
        chain.cargo[:] = start[3:]
        for _ in range(round(2. / h)):
            chain.advance(free, bound, np.ones(1), np.zeros(1), np.ones(1, bool),
                          h, trophic(tau_N=1/dn, tau_T=1/dt))
        errors.append(np.linalg.norm(np.r_[free, bound, chain.cargo] - exact))
    assert 1.8 < errors[0] / errors[1] < 2.2
    assert 1.8 < errors[1] / errors[2] < 2.2


def test_edge_permutation_has_no_priority():
    pre, post = np.array([0, 1, 0]), np.array([2, 2, 1])
    order = np.array([2, 0, 1])
    results = []
    for idx in (np.arange(3), order):
        free, bound = np.array([.2, .3, 1.]), np.array([.2, .5, .1])[idx].copy()
        chain = SupportTransport(3, pre[idx], post[idx], free, bound, [.1, .2, .3])
        for _ in range(50):
            chain.advance(free, bound, np.array([1., 2., 3.])[idx], np.ones(3),
                          np.ones(3, bool), .2, trophic())
        results.append((free, bound[np.argsort(idx)], chain.cargo))
    for left, right in zip(*results):
        np.testing.assert_allclose(left, right, atol=2e-15, rtol=1e-14)


def make_medium():
    m = Medium([CellClass()], np.zeros(4, int), [0, 1, 2, 0], [1, 2, 3, 2],
               [.01, .02, .03, .01], SynapseParams(eta=1e-5), dt=.25, seed=37)
    m.trophic = trophic(q0=.01, tau_N=200., tau_T=20000.)
    m.trophic_edges[:] = True
    m.z[:] = [1., 2., 3., 4.]
    return m


def test_integrated_checkpoint_resume_all_pools_ledgers_rng_and_material(tmp_path):
    m = make_medium()
    m.enable_material(initial_free=1.)
    m.enable_support_transport(SupportTransportParams(.01, .02, .001), source=[0, 0, 0, .04])
    for _ in range(80):
        m.step(forced=m.rng.random(m.N) < .02)
    path = tmp_path / 'transport.npz'
    save_medium(path, m, {'candidate': 'conserved support, not capability'})
    fork, provenance = load_medium(path)
    assert provenance['candidate'] == 'conserved support, not capability'
    for _ in range(120):
        for obj in (m, fork):
            obj.step(forced=obj.rng.random(obj.N) < .02)
    for name in m._state_arrays + m._physiology_arrays + ('g',):
        np.testing.assert_array_equal(getattr(m, name), getattr(fork, name))
    for name in m.support_transport._fields:
        np.testing.assert_array_equal(getattr(m.support_transport, name), getattr(fork.support_transport, name))
    for name in m.material._fields:
        np.testing.assert_array_equal(getattr(m.material, name), getattr(fork.material, name))
    assert m.rng.bit_generator.state == fork.rng.bit_generator.state
    np.testing.assert_allclose(m.support_transport.balance(m.Nf, m.T), 0., atol=2e-12)
    np.testing.assert_allclose(m.material.balance(m.g), 0., atol=2e-12)


def test_legacy_schema_without_transport_key_preserves_historical_path():
    m = make_medium()
    state = m.state_dict()
    del state['support_transport']
    fork = Medium.from_state_dict(state)
    assert fork.support_transport is None
    # Independent historical E9 calculation before the next plastic update.
    m.plastic = fork.plastic = False
    dt, p = m.dt, m.trophic
    free = m.Nf * np.exp(-dt/p.tau_N) + dt*p.q0
    up = p.kappa * np.maximum(m.z, 0) * free[m.post]
    demand = np.bincount(m.post, up*dt, m.N)
    up *= np.minimum(1., free / np.maximum(demand, 1e-30))[m.post]
    expected_free = free - np.bincount(m.post, up*dt, m.N)
    expected_bound = np.maximum(m.T + dt*(up-m.T/p.tau_T-p.phi*m.g), 0.)
    expected_g = m.g * np.exp(-dt*p.lambda0/(1+expected_bound/p.T_star))
    m.step()
    fork.step()
    for name, expected in (('Nf', expected_free), ('T', expected_bound), ('g', expected_g)):
        np.testing.assert_array_equal(getattr(m, name), expected)
        np.testing.assert_array_equal(getattr(fork, name), expected)


def test_enabling_keeps_existing_support_and_rejects_ledger_reset():
    m = make_medium()
    m.Nf[:] = 1.
    m.T[:] = .5
    m.enable_support_transport()
    np.testing.assert_array_equal(m.Nf, np.ones(4))
    np.testing.assert_array_equal(m.T, np.full(4, .5))
    np.testing.assert_allclose(m.support_transport.balance(m.Nf, m.T), 0.)
    with pytest.raises(ValueError, match='already enabled'):
        m.enable_support_transport()


def test_invalid_configuration_and_archive_fail_explicitly():
    with pytest.raises(ValueError):
        SupportTransportParams(transport_rate=-1.)
    with pytest.raises(ValueError):
        SupportTransport(1, [0], [0], np.zeros(1), np.zeros(1), -1.)
    m = make_medium()
    m.trophic.rule = 'residual'
    with pytest.raises(ValueError, match='constitutive'):
        m.enable_support_transport()
    m.trophic.rule = 'constitutive'
    m.enable_support_transport()
    state = m.state_dict()
    state['support_transport']['arrays']['cargo'][0] = 1.
    with pytest.raises(ValueError, match='balance'):
        Medium.from_state_dict(state)


def test_disabled_capture_still_transports_previously_captured_support():
    free, bound = np.zeros(2), np.ones(1)
    chain = SupportTransport(2, [0], [1], free, bound, 0., SupportTransportParams(1., 1., 0.))
    for _ in range(20):
        chain.advance(free, bound, np.ones(1), np.ones(1), np.zeros(1, bool),
                      .5, trophic(phi=1.))
    assert free[0] > .99 and chain.spent.sum() == 0.
    np.testing.assert_allclose(chain.balance(free, bound), 0., atol=1e-14)


def test_restore_accepts_floating_roundoff_in_long_cumulative_ledger():
    free, bound = np.array([.2]), np.zeros(0)
    chain = SupportTransport(1, [], [], free, bound, .001)
    for _ in range(9601):
        chain.advance(free, bound, np.zeros(0), np.zeros(0), np.zeros(0, bool),
                      .25, trophic(tau_N=200.))
    restored = SupportTransport.from_state_dict(chain.state_dict(), free, bound)
    np.testing.assert_array_equal(restored.imported, chain.imported)
    assert np.abs(restored.balance(free, bound)).max() < 1e-9
