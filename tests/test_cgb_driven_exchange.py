"""Local physical and boundary contracts, not a language capability test."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import expit

from information_boltzmann.cgb.driven_exchange import DrivenExchange, DrivenExchangeParams
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.flow_exchange import FlowExchange


def isolated(charge=(0., .1), fuel=(.9, .5), active_rate=.8, **ports):
    p = DrivenExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=0., charged_energy=1., active_rate=active_rate)
    return DrivenExchange(np.array([0]), np.array([1]), [1.], [0., 0.], [0.],
        [0., 0.], fuel, 1.-np.array(fuel), [2.], [0., 0.], charge,
        unit_conductance=0., params=p, **ports)


def panel():
    return DrivenExchange(np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3),
        [1., .8, .9], [.2, .3, .4], [.2, .3, .4], [.8, .7, .6], [.2, .3, .4],
        np.full(3, 2.), np.full(3, .5), [2., .5, -.2], [1., 2., 1.],
        [.1, .2, .1], port_nodes=[0, 2], world_charge=[.4, -.3],
        world_capacitance=[2., 3.], port_conductance=[.1, .2], source_conductance=.2)


def test_active_affinity_includes_both_electrical_terminals_and_paid_fuel():
    m = isolated()
    plus, minus = m.fluxes()['active']
    a = 1.+np.log(.9/.1)+m.voltage[0]-m.voltage[1]
    assert abs(np.log(plus[0]/minus[0])-a) < 1e-14
    before, e = m.free_energy(), 1e-7
    m.charged[0] -= e
    m.depleted[0] += e
    m.charge[0] -= e
    m.charge[1] += e
    assert abs((m.free_energy()-before)/e+a) < 1e-6


def test_uphill_transfer_consumes_finite_drive_and_reverse_recharges_it():
    a = isolated()
    before = a.field_energy()
    a.step(.2)
    assert a.field_energy() > before
    assert a.charged[0] < .9 and a.active_net[0] > 0.
    b = isolated(charge=(0., 5.), fuel=(.1, .5))
    b.step(.2)
    assert b.charged[0] > .1 and b.active_net[0] < 0.


def test_no_fuel_composition_cannot_create_active_transfer():
    m = isolated(fuel=(0., 0.))
    m.depleted[:] = 0.
    # Declare a new physical initial composition, not a living-state reset.
    m.initial_drive[:] = 0.
    m.initial_energy, m.initial_free_energy = m.energy(), m.free_energy()
    before = m.charge.copy()
    m.step(100.)
    np.testing.assert_array_equal(before, m.charge)
    np.testing.assert_array_equal(m.active_net, [0.])


def test_shared_donor_cannot_be_spent_separately_by_each_edge():
    p = DrivenExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=0., active_rate=10.)
    m = DrivenExchange(np.array([0, 0, 0]), np.array([1, 2, 3]), np.ones(3),
        np.zeros(4), np.zeros(3), np.zeros(4), [.2, 0., 0., 0.],
        [.8, 1., 1., 1.], np.full(3, 2.), np.zeros(4), np.zeros(4),
        unit_conductance=0., params=p)
    m.step(100.)
    assert m.active_net.sum() <= .2+1e-14
    assert np.all(m.charged >= 0.) and np.all(m.depleted >= 0.)
    for value in m.balances().values():
        assert np.max(np.abs(value), initial=0.) < 1e-11


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_closed_active_graph_and_world_conserve_and_decrease_free_energy(dt):
    m = panel()
    for _ in range(15):
        update = m.step(dt)
        assert update['free_energy_drop'] >= -1e-10
        for value in m.balances().values():
            assert np.max(np.abs(value), initial=0.) < 1e-10
    assert m.work_in == 0.


def test_world_port_has_finite_storage_and_reacts_back_without_a_source():
    m = isolated(charge=(1., 0.), active_rate=0., port_nodes=[0],
                 world_charge=[0.], world_capacitance=[2.], port_conductance=[1.])
    old_brain, old_world = m.charge.copy(), m.world_charge.copy()
    m.step(1.)
    sent = (1.-np.exp(-1.5))/1.5
    assert abs(m.charge[0]-(1.-sent)) < 1e-14
    assert abs(m.world_charge[0]-sent) < 1e-14
    assert not np.array_equal(old_brain, m.charge)
    assert not np.array_equal(old_world, m.world_charge)
    assert m.port_heat > 0. and m.work_in == 0.


def test_source_uses_finite_conductance_and_accounts_work_and_heat():
    m = isolated(charge=(0., 0.), active_rate=0., port_nodes=[0],
                 world_charge=[0.], world_capacitance=[1.],
                 port_conductance=[0.], source_conductance=[1.])
    m.step(.5, world_voltage=[2.])
    q = 2.*(1.-np.exp(-.5))
    assert abs(m.world_charge[0]-q) < 1e-14
    assert abs(m.source_work-2*q) < 1e-14
    assert abs(m.source_heat-(2*q-.5*q*q)) < 1e-14
    assert abs(m.energy()+m.heat_out-m.initial_energy-m.work_in) < 1e-13


def test_zero_active_and_no_world_ports_reproduce_passive_parent_bitwise():
    a = isolated(active_rate=0.)
    s = a.state_dict()
    s['kind'] = 'flow_exchange'
    s['params'].pop('active_rate')
    b = FlowExchange.from_state_dict(s)
    for _ in range(10):
        a.step(.3, np.array([.1, -.1]))
        b.step(.3, np.array([.1, -.1]))
    for name in FlowExchange._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in FlowExchange._ledgers:
        assert getattr(a, name) == getattr(b, name)


def test_single_active_event_matches_continuous_ode_under_refinement():
    def rhs(_, y):
        qa, qb, p, d = y
        a = 1.+qa-qb
        j = .8*(p*expit(a)-d*expit(-a))
        return [-j, j, -j, j]
    exact = solve_ivp(rhs, [0., 2.], [0., .1, .9, .1], rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for dt in (.2, .1, .05):
        m = isolated()
        for _ in range(round(2./dt)):
            m.step(dt)
        errors.append(np.linalg.norm(np.r_[m.charge, m.charged[0], m.depleted[0]]-exact))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_active_coloring_order_error_vanishes_with_smaller_steps():
    # Two interacting edges in opposite execution orders implement one ODE.
    def build(reverse):
        p = DrivenExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
            assembly_rate=0., conversion_rate=0., charged_energy=1., active_rate=.8)
        idx = [1, 0] if reverse else [0, 1]
        return DrivenExchange(np.array([0, 1])[idx], np.array([1, 2])[idx], np.ones(2),
            np.zeros(3), np.zeros(2), np.zeros(3), np.full(3, .8), np.full(3, .2),
            np.full(2, 2.), np.zeros(3), np.zeros(3), unit_conductance=0., params=p)
    errors = []
    for dt in (.2, .1, .05):
        a, b = build(False), build(True)
        for _ in range(round(2./dt)):
            a.step(dt)
            b.step(dt)
        errors.append(np.linalg.norm(a.charge-b.charge))
    assert errors[1] < .6*errors[0] and errors[2] < .6*errors[1]


def test_driven_world_checkpoint_preserves_full_future_and_rejects_tampering(tmp_path):
    a = panel()
    for _ in range(4):
        a.step(.3, world_voltage=[2., -1.])
    path = tmp_path/'driven.npz'
    save_exchange(path, a, {'scope': 'physical only'})
    b, meta = load_exchange(path)
    assert meta['scope'] == 'physical only'
    assert isinstance(b, DrivenExchange)
    for _ in range(6):
        a.step(.3, world_voltage=[-1., 2.])
        b.step(.3, world_voltage=[-1., 2.])
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in a._ledgers:
        assert getattr(a, name) == getattr(b, name)
    state = a.state_dict()
    state['arrays']['world_charge'][0] += 1.
    with pytest.raises(ValueError, match='accounting'):
        DrivenExchange.from_state_dict(state)


def test_invalid_world_input_is_rejected_before_advancing_any_state():
    m = panel()
    before = m.charge.copy()
    with pytest.raises(ValueError):
        m.step(1., world_voltage=[np.nan, 0.])
    np.testing.assert_array_equal(before, m.charge)
    with pytest.raises(ValueError):
        DrivenExchangeParams(active_rate=-1.)
