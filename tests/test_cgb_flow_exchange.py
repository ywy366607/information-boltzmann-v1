"""Physical/numerical interface checks, not a learning capability study."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import expit

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.flow_exchange import FlowExchange, FlowExchangeParams


def panel(params=None):
    return FlowExchange(np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3),
        [1., .8, .9], [.2, .3, .4], [.2, .3, .4], [.8, .7, .6], [.2, .3, .4],
        np.full(3, 2.), np.full(3, .5), [2., .5, -.2], [1., 2., 1.],
        [.1, .2, .1], params)


def isolated(charge=2., drive=.2, rate=.8):
    params = FlowExchangeParams(binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=rate, charged_energy=1.)
    return FlowExchange(np.array([], int), np.array([], int), [], [0.], [],
        [0.], [drive], [1.-drive], [], [0.], [charge], params=params)


def test_conversion_affinity_matches_joint_field_and_pool_free_energy():
    m = panel()
    plus, minus = m.fluxes()['conversion']
    affinity = (np.log(m.depleted/m.charged)
                +m.params.charge_per_event*m.voltage-m.params.charged_energy)
    np.testing.assert_allclose(np.log(plus/minus), affinity, atol=2e-15)
    # Independent finite difference includes field work and ideal mixing.
    g0, epsilon = m.free_energy(), 1e-7
    m.charged[0] += epsilon
    m.depleted[0] -= epsilon
    m.charge[0] -= m.params.charge_per_event*epsilon
    assert abs((m.free_energy()-g0)/epsilon+affinity[0]) < 1e-6
    assert np.all((plus-minus)*affinity >= 0.)


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_closed_coupled_process_conserves_compositions_charge_energy_and_decreases_g(dt):
    m = panel()
    for _ in range(20):
        result = m.step(dt)
        assert result['free_energy_drop'] >= -1e-10
        for value in m.balances().values():
            assert np.max(np.abs(value), initial=0.) < 1e-9
        assert np.all(m.charged >= 0.) and np.all(m.depleted >= 0.)
    assert m.work_in == 0.
    assert m.conduction_numerical_heat >= 0.


def test_boundary_current_pays_exact_self_energy_and_can_extract_work():
    m = isolated(charge=0., rate=0.)
    m.step(1., np.array([2.]))
    assert m.field_energy() == m.work_in == 2.
    assert m.heat_out == 0.
    result = m.step(1., np.array([-1.]))
    assert result['work_in'] == -1.5
    assert m.field_energy() == m.work_in == .5
    for v in m.balances().values():
        assert np.max(np.abs(v), initial=0.) < 1e-14


def test_transducer_returns_energy_to_fast_field_as_well_as_charging_drive():
    forward = isolated(charge=5., drive=.1)
    reverse = isolated(charge=0., drive=.9)
    forward.step(.5)
    reverse.step(.5)
    assert forward.charged[0] > .1 and forward.charge[0] < 5.
    assert reverse.charged[0] < .9 and reverse.charge[0] > 0.
    assert reverse.returned_charge[0] < 0.


def test_zero_transduction_blocks_boundary_work_from_recharging_pool():
    m = isolated(charge=0., rate=0.)
    p = m.charged.copy()
    for _ in range(10):
        m.step(.2, np.ones(1))
    np.testing.assert_array_equal(m.charged, p)
    np.testing.assert_array_equal(m.returned_charge, np.zeros(1))
    assert m.work_in > 0.


def test_parallel_installed_structure_changes_subsequent_identical_fast_response():
    a, b = panel(), panel()
    b.capacity *= 1.2
    # Reinitialize bookkeeping after declaring a different physical initial
    # structure; do not compare different fast starting states.
    state = b.state_dict()
    state['arrays']['initial_material'] = b.material+b._sum(b.post, b.capacity)
    state['initial_energy'], state['initial_free_energy'] = b.energy(), b.free_energy()
    b = FlowExchange.from_state_dict(state)
    np.testing.assert_array_equal(a.charge, b.charge)
    a._conduction(1.)
    b._conduction(1.)
    assert np.max(np.abs(a.charge-b.charge)) > .01


def test_conduction_matches_independent_linear_solution_and_heat_split():
    m = panel()
    old_q, old_energy = m.charge.copy(), m.field_energy()
    incidence = np.zeros((m.n, len(m.pre)))
    for e, (pre, post) in enumerate(zip(m.pre, m.post)):
        incidence[pre, e] += 1.
        incidence[post, e] -= 1.
    lap = (incidence*m.conductance)@incidence.T
    expected = np.linalg.solve(np.diag(m.capacitance)+.7*lap, old_q)
    m._conduction(.7)
    np.testing.assert_allclose(m.voltage, expected, atol=1e-13)
    assert abs(old_energy-m.field_energy()-m.conduction_heat-m.conduction_numerical_heat) < 1e-12


def test_conversion_step_refinement_matches_independent_continuous_ode():
    def rhs(_, y):
        q, p, d = y
        j = .8*(d*expit(q-1.)-p*expit(1.-q))
        return [-j, j, -j]
    exact = solve_ivp(rhs, [0., 2.], [2., .2, .8], rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for dt in (.2, .1, .05):
        m = isolated()
        for _ in range(round(2./dt)):
            m.step(dt)
        errors.append(np.linalg.norm(np.array([m.charge[0], m.charged[0], m.depleted[0]])-exact))
    assert 1.8 < errors[0]/errors[1] < 2.2
    assert 1.8 < errors[1]/errors[2] < 2.2


def test_complete_flow_restore_continues_bitwise_with_work_and_structure(tmp_path):
    a = panel()
    current = np.array([.1, -.02, 0.])
    for _ in range(10):
        a.step(.3, current)
    path = tmp_path/'flow.npz'
    save_exchange(path, a, {'claim': 'physical interface only'})
    b, provenance = load_exchange(path)
    assert isinstance(b, FlowExchange)
    assert provenance['claim'] == 'physical interface only'
    for _ in range(15):
        a.step(.3, current)
        b.step(.3, current)
    for name in a._arrays:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in a._ledgers:
        assert getattr(a, name) == getattr(b, name)
    state = a.state_dict()
    state['work_in'] += 1.
    with pytest.raises(ValueError, match='accounting'):
        FlowExchange.from_state_dict(state)


def test_edge_permutation_preserves_coupled_process():
    a = panel()
    s = a.state_dict()
    idx = [2, 0, 1]
    for name in ('pre', 'post', 'capacity'):
        s[name] = s[name][idx]
    for name in ('bound', 'positions', 'unit_conductance', 'binding_net'):
        s['arrays'][name] = s['arrays'][name][idx]
    b = FlowExchange.from_state_dict(s)
    for _ in range(5):
        a.step(.3)
        b.step(.3)
    np.testing.assert_allclose(a.charge, b.charge, atol=1e-13)
    np.testing.assert_allclose(a.capacity, b.capacity[np.argsort(idx)], atol=1e-13)


def test_invalid_constitutive_inputs_rejected_before_current_changes_state():
    with pytest.raises(ValueError):
        FlowExchangeParams(conversion_rate=-1.)
    with pytest.raises(ValueError):
        FlowExchangeParams(charge_per_event=0.)
    m = panel()
    q = m.charge.copy()
    with pytest.raises(ValueError):
        m.step(1., np.full(3, np.nan))
    np.testing.assert_array_equal(m.charge, q)
    with pytest.raises(ValueError):
        m.step(1., np.full(3, 1e6))
    np.testing.assert_array_equal(m.charge, q)
