"""Finite observation-sector invariants, not a brain capability benchmark."""
import numpy as np
from dataclasses import replace
from scipy.integrate import quad_vec
from scipy.linalg import expm

from information_boltzmann.cgb.mobile_exchange import MobileExchangeParams
from scripts.ib.generative.r7_structure_constraints import (
    STATES, physical_state, structure_generator, sampled_entry,
    observation_probabilities, structure_score, count_closure)
from scripts.ib.generative.r7_structure_calibration import shift_structure_energy
from scripts.ib.generative.r7_structure_joint_constraints import formation_energy_for_occupancy


def test_structure_generator_is_native_reversible_and_preserves_composition():
    params = MobileExchangeParams()
    q, rho, receipts = structure_generator(params, 3)
    np.testing.assert_allclose(q.sum(axis=1), 0., atol=1e-17)
    np.testing.assert_allclose(rho @ q, 0., atol=1e-16)
    for row in receipts:
        model = physical_state(params, 3, STATES[row['source']])
        target = physical_state(params, 3, STATES[row['target']])
        events = [e for e in model.events() if e.kind == row['kind'] and e.sign == row['sign']]
        assert len(events) == 1
        event = events[0]
        assert model.event_rate(event) == row['rate']
        model.apply_event(event)
        for name in ('coordinates', 'pre', 'post', 'free', 'bound', 'trace',
                     'material', 'charged', 'depleted', 'charge'):
            np.testing.assert_array_equal(getattr(model, name), getattr(target, name))
        for value in model.balances().values():
            np.testing.assert_allclose(value, 0., atol=1e-13)


def test_trace_loss_does_not_kill_installed_paid_structure():
    q, _, receipts = structure_generator(MobileExchangeParams(), 2)
    # r=1,C=0 -> r=0,C=0 keeps an installed bridge. Only idle can disassemble.
    assert q[3, 1] > 0 and q[3, 0] == 0 and q[2, 0] == 0 and q[4, 0] == 0
    assert q[1, 0] > 0
    assert any(r['source'] == 1 and r['target'] == 3 for r in receipts)


def test_first_visible_entry_integral_and_censor_mass_match_independent_integral():
    q, _, _ = structure_generator(MobileExchangeParams(), 4)
    t = q[1:, 1:]
    for clock in (0.5, 31., 2500.):
        entry, visible, _ = sampled_entry(t, clock)
        integral, _ = quad_vec(lambda u: expm(t*clock*u)[0], 0., 1., epsabs=1e-12)
        np.testing.assert_allclose(entry*visible, integral, atol=1e-12, rtol=1e-10)
        survival, events, _, _ = observation_probabilities(t, clock)
        np.testing.assert_allclose(events[:59].sum()+survival[59], 1., atol=1e-12)
        assert np.all(events >= 0) and np.all(np.diff(survival) <= 1e-14)
        score = structure_score(t, clock, np.array([0., 59.]), np.array([1., 0.]))
        assert np.isfinite(score['mean_nll'])


def test_stationary_birth_visibility_and_counts_obey_same_renewal_balance():
    # The actual archived scalar material has these constitutive values.
    params = MobileExchangeParams(assembly_rate=.05, trace_energy=2.,
                                  capture_coupling=2., bound_energy=-2.)
    q, rho, _ = structure_generator(params, 2)
    # Keep this independent discrete resolvent well conditioned; the much
    # stiffer default material is checked against quadrature above instead.
    clock = 71000.
    t = q[1:, 1:]
    entry, visible, _ = sampled_entry(t, clock)
    # Expected number of visible sampled frames of one newly visible identity.
    step = expm(t*clock)
    frames = float(entry @ np.linalg.solve(np.eye(4)-step, np.ones(4)))
    visible_rate = clock*rho[0]*q[0, 1]*visible
    np.testing.assert_allclose(visible_rate*frames, 1-rho[0], rtol=1e-10)
    closure = count_closure(q, rho, clock, visible, np.array([1, 2, 3]))
    np.testing.assert_allclose(closure['predicted_visible_arrivals_per_minute']*frames,
                               closure['predicted_mean_count'], rtol=1e-10)


def test_fitted_formation_energy_changes_both_native_directions_and_weights():
    params = MobileExchangeParams(assembly_rate=.05, trace_energy=2.,
                                  capture_coupling=2., bound_energy=-2.)
    q, rho, receipts = structure_generator(params, 3)
    for energy in (-10., 2., 7., 18., 30.):
        cached, weights = shift_structure_energy(q, rho, receipts, energy-params.structure_energy)
        actual, actual_weights, _ = structure_generator(replace(params, structure_energy=energy), 3)
        np.testing.assert_allclose(cached, actual, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(weights, actual_weights, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(weights @ cached, 0., atol=1e-15)


def test_count_energy_inversion_reproduces_native_partition_not_extra_rate():
    params = MobileExchangeParams(assembly_rate=.05, trace_energy=2.,
                                  capture_coupling=2., bound_energy=-2.)
    for support in (1, 4, 16):
        for probability in (.1, .5, .9):
            energy = formation_energy_for_occupancy(params, support, probability)
            q, rho, _ = structure_generator(replace(params, structure_energy=energy), support)
            np.testing.assert_allclose(1-rho[0], probability, atol=1e-12)
            np.testing.assert_allclose(rho @ q, 0., atol=1e-15)
