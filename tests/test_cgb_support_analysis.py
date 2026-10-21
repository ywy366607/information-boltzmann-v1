"""Analytic reaction-network and passive-label contracts, no learning task."""
from dataclasses import replace

import numpy as np
import pytest

from information_boltzmann.cgb.medium import TrophicParams
from information_boltzmann.cgb.support_analysis import (
    choose_support_boundary, frozen_support_flow, reverse_distances)
from information_boltzmann.cgb.support_transport import SupportTransport, SupportTransportParams


def test_steady_elimination_satisfies_original_three_pool_equations():
    pre, post = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 2])
    carrier, q = np.array([1., .5, 2., 3.]), np.array([0., .3, .2])
    p = SupportTransportParams(.4, .6, .15)
    result = frozen_support_flow(pre, post, carrier, q, kappa=.7,
                                free_clearance=.1, terminal_clearance=.2, params=p)
    f, t, h = (result[k] for k in ('free', 'bound', 'cargo'))
    a = .7 * carrier
    free_rhs = q + .6*h - .1*f - np.bincount(post, weights=a*f[post], minlength=3)
    bound_rhs = a*f[post] - .6*t
    cargo_rhs = np.bincount(pre, weights=.4*t, minlength=3) - .75*h
    for rhs in (free_rhs, bound_rhs, cargo_rhs):
        np.testing.assert_allclose(rhs, 0., atol=1e-15)
    assert result['total_clearance'] == pytest.approx(q.sum(), abs=1e-15)
    assert result['max_one_hop_fraction'] < result['residence_survival'] < 1
    assert min(f.min(), t.min(), h.min()) >= 0


def test_two_hop_flow_requires_both_actual_carriers_and_transport():
    pre, post = np.array([0, 1]), np.array([1, 2])
    q = np.array([0., 0., 1.])
    p = SupportTransportParams(.5, .4, .1)
    args = dict(kappa=1., free_clearance=.1, terminal_clearance=.2)
    yes = frozen_support_flow(pre, post, np.ones(2), q, params=p, **args)
    assert yes['two_hop_return'][0] > 0
    for carrier, par in ((np.array([0., 1.]), p), (np.ones(2), replace(p, transport_rate=0.))):
        result = frozen_support_flow(pre, post, carrier, q, params=par, **args)
        assert result['two_hop_return'].sum() == 0


def test_repeated_returns_are_bounded_even_on_closed_graph_with_clearance():
    result = frozen_support_flow(np.array([0, 1]), np.array([1, 0]),
                                 np.array([10., 10.]), np.array([1., 0.]),
                                 kappa=2., free_clearance=.01, terminal_clearance=.02,
                                 params=SupportTransportParams(.4, .6, .03))
    assert result['flow_residual'] < 1e-13
    assert result['total_clearance'] == pytest.approx(1., abs=1e-13)


def test_topological_boundary_selection_is_deterministic_and_distinguishes_capture():
    pre, post = np.array([0, 1, 2]), np.array([1, 2, 3])
    np.testing.assert_array_equal(reverse_distances(4, pre, post, 3), [3, 2, 1, 0])
    assert choose_support_boundary(4, pre, post, np.ones(3, bool)) == 3
    # Breaking a capture link does not remove its anatomical edge.
    d = reverse_distances(4, pre, post, 3, [True, False, True])
    np.testing.assert_array_equal(d, [-1, -1, 1, 0])


def test_passive_source_label_separates_initial_inventory_without_feedback():
    pre, post = [0, 1], [1, 2]
    free, bound = np.array([.2, .3, 1.]), np.array([.4, .5])
    old_f, old_t = free.copy(), bound.copy()
    fresh_f, fresh_t = np.zeros(3), np.zeros(2)
    p = SupportTransportParams(.4, .6, .1)
    actual = SupportTransport(3, pre, post, free, bound, [0, 0, .1], p)
    old = SupportTransport(3, pre, post, old_f, old_t, 0., p)
    fresh = SupportTransport(3, pre, post, fresh_f, fresh_t, [0, 0, .1], p)
    trophic = TrophicParams(kappa=.2, tau_N=5., tau_T=7., phi=0.)
    for step in range(100):
        carrier = np.array([.5+step/100, 2.])
        for state, f, t in ((actual, free, bound), (old, old_f, old_t), (fresh, fresh_f, fresh_t)):
            state.advance(f, t, carrier, np.ones(2), np.ones(2, bool), .1, trophic)
    np.testing.assert_allclose(free, old_f + fresh_f, atol=1e-14)
    np.testing.assert_allclose(bound, old_t + fresh_t, atol=1e-14)
    np.testing.assert_allclose(actual.cargo, old.cargo + fresh.cargo, atol=1e-14)
    assert fresh_f[0] > 0


@pytest.mark.parametrize('free_clearance,terminal_clearance', [(0., .2), (-1., .2), (.1, -1.)])
def test_invalid_steady_assumptions_rejected(free_clearance, terminal_clearance):
    with pytest.raises(ValueError):
        frozen_support_flow(np.array([0]), np.array([0]), np.ones(1), np.ones(1),
                            kappa=1., free_clearance=free_clearance,
                            terminal_clearance=terminal_clearance,
                            params=SupportTransportParams())
