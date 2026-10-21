"""Finite trace storage and reciprocal capture, numerical/physical only."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import expit

from information_boltzmann.cgb.driven_exchange import DrivenExchange
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.trace_exchange import TraceExchange, TraceExchangeParams


def panel(params=None, trace=.2):
    return TraceExchange(np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3),
        [1., .8, .9], [.2, .3, .4], [.2, .3, .4], [.8, .7, .6], [.2, .3, .4],
        np.full(3, 2.), np.full(3, .5), [2., .5, -.2], [1., 2., 1.],
        [.1, .2, .1], params, trace=trace, port_nodes=[0, 2],
        world_charge=[.4, -.3], world_capacitance=[2., 3.])


def test_trace_affinities_use_the_same_joint_free_energy():
    m = panel()
    f = m.fluxes()
    pre, post = m.pre[0], m.post[0]
    er = m.trace_local_energy()[0]
    expected = m.params.charged_energy-er+m.params.charge_per_event*(m.voltage[pre]-m.voltage[post])
    expected += np.log(m.charged[pre]*(m.trace_capacity[0]-m.trace[0])/(m.depleted[pre]*m.trace[0]))
    assert abs(np.log(f['trace_write'][0][0]/f['trace_write'][1][0])-expected) < 1e-14
    old, h = m.free_energy(), 1e-7
    m.trace[0] += h
    m.charged[pre] -= h
    m.depleted[pre] += h
    m.charge[pre] -= m.params.charge_per_event*h
    m.charge[post] += m.params.charge_per_event*h
    assert abs((m.free_energy()-old)/h+expected) < 2e-6


@pytest.mark.parametrize('reaction', ['binding', 'transport'])
def test_reciprocal_capture_interaction_changes_both_support_reaction_affinities(reaction):
    m = panel()
    plus, minus = m.fluxes()[reaction]
    affinity = np.log(plus[0]/minus[0])
    old, h = m.free_energy(), 1e-7
    if reaction == 'binding':
        m.free[m.post[0]] -= h
        m.bound[0] += h
    else:
        m.bound[0] -= h
        m.carried[m.pre[0]] += h
        m.charged[m.pre[0]] -= h
        m.depleted[m.pre[0]] += h
    assert abs((m.free_energy()-old)/h+affinity) < 2e-6


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_closed_trace_capture_structure_world_obey_all_balances_and_g(dt):
    m = panel()
    for _ in range(12):
        update = m.step(dt)
        assert update['free_energy_drop'] >= -1e-10
        assert np.all(m.trace >= 0.) and np.all(m.trace <= m.trace_capacity)
        for value in m.balances().values():
            assert np.max(np.abs(value), initial=0.) < 1e-10


def test_trace_writing_is_a_fuel_paid_charge_transfer_not_an_activity_ema():
    p = TraceExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=0., active_rate=0., recovery_rate=0., write_rate=.8)
    m = panel(p, trace=0.)
    before_p, before_q = m.charged.sum(), m.charge.copy()
    m._write_trace(.2)
    written = m.trace_written.sum()
    assert written > 0.
    assert abs(before_p-m.charged.sum()-written) < 1e-14
    assert np.max(np.abs(m.charge-before_q)) > 0.
    assert m.work_in == 0.


def test_thermal_trace_relaxes_to_binding_dependent_finite_equilibrium():
    p = TraceExchangeParams(write_rate=0., recovery_rate=.3)
    m = panel(p, trace=0.)
    expected = m.trace_capacity*expit(-m.trace_local_energy())
    m._recover_trace(100.)
    np.testing.assert_allclose(m.trace, expected, atol=1e-13)
    assert expected[2] > expected[0]


def test_capture_and_channel_connections_have_independent_causal_controls():
    a = panel(TraceExchangeParams(capture_coupling=0., gate_low=1., gate_high=1.), trace=.1)
    b = panel(TraceExchangeParams(capture_coupling=0., gate_low=1., gate_high=1.), trace=.8)
    np.testing.assert_array_equal(a.fluxes()['binding'][1], b.fluxes()['binding'][1])
    np.testing.assert_array_equal(a.conductance, b.conductance)
    connected_a, connected_b = panel(trace=.1), panel(trace=.8)
    assert np.all(connected_b.fluxes()['binding'][1] < connected_a.fluxes()['binding'][1])
    assert np.all(connected_b.conductance > connected_a.conductance)


def test_trace_capture_protects_structure_in_a_matched_turnover_process():
    p = TraceExchangeParams(binding_rate=.1,transport_rate=0.,release_rate=0.,
        assembly_rate=.1,conversion_rate=0.,active_rate=0.,write_rate=0.,recovery_rate=0.,
        gate_low=1.,gate_high=1.)
    def make(r):
        return TraceExchange(np.array([0]),np.array([1]),[1.],[.01,.01],[.5],[0.,0.],
            [0.,0.],[1.,1.],[2.],[0.,0.],[0.,0.],params=p,trace=r)
    low,high = make(.1),make(.8)
    for _ in range(20):
        low.step(1.)
        high.step(1.)
    assert high.bound[0] > low.bound[0]
    assert high.capacity[0] > low.capacity[0]
    assert high.work_in == low.work_in == 0.


def test_switching_off_new_routes_preserves_parent_bitwise():
    p = TraceExchangeParams(write_rate=0., recovery_rate=0., capture_coupling=0.,
                            gate_low=1., gate_high=1.)
    a = panel(p, trace=0.)
    s = a.state_dict()
    s['kind'] = 'driven_exchange'
    for name in ('write_rate', 'recovery_rate', 'trace_energy', 'capture_coupling', 'gate_low', 'gate_high'):
        s['params'].pop(name)
    b = DrivenExchange.from_state_dict(s)
    for _ in range(4):
        a.step(.4, world_voltage=[1., -1.])
        b.step(.4, world_voltage=[1., -1.])
    for name in DrivenExchange._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in DrivenExchange._ledgers:
        assert getattr(a, name) == getattr(b, name)


def test_trace_full_restore_preserves_future_channels_support_and_structure(tmp_path):
    a = panel()
    for _ in range(5):
        a.step(.3, world_voltage=[2., -1.])
    save_exchange(tmp_path/'trace.npz', a, {'scope': 'trace physical audit'})
    b, meta = load_exchange(tmp_path/'trace.npz')
    assert meta['scope'] == 'trace physical audit'
    for _ in range(7):
        a.step(.3, world_voltage=[-1., 2.])
        b.step(.3, world_voltage=[-1., 2.])
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in a._ledgers:
        assert getattr(a, name) == getattr(b, name)


def test_trace_write_step_refines_toward_independent_ode():
    p = TraceExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=0., active_rate=0., recovery_rate=0., write_rate=.8,
        charged_energy=1., trace_energy=.5, capture_coupling=1.)
    def make():
        return TraceExchange(np.array([0]), np.array([1]), [1.], [1., 1.], [.2], [0., 0.],
            [.8, .5], [.2, .5], [2.], [0., 0.], [.1, -.1], unit_conductance=0., params=p, trace=.2)
    def rhs(_, y):
        r, fuel, waste, qa, qb = y
        a = 1.-(.5-.2)+qa-qb
        j = .8*(fuel*(1-r)*expit(a)-waste*r*expit(-a))
        return [j, -j, j, -j, j]
    exact = solve_ivp(rhs, [0., 2.], [.2, .8, .2, .1, -.1], rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for dt in (.2, .1, .05):
        m = make()
        for _ in range(round(2/dt)):
            m.step(dt)
        errors.append(np.linalg.norm(np.r_[m.trace[0], m.charged[0], m.depleted[0], m.charge]-exact))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2
