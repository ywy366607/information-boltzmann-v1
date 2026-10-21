"""Finite-position structure/energy contracts; no capability task."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.structural_exchange import StructuralExchange, StructuralExchangeParams


def panel():
    return StructuralExchange(np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3),
        [1., .8, .9], [.2, .3, .4], [.2, .3, .4], [.8, .7, .6], [.2, .3, .4],
        np.full(3, 2.), np.full(3, .5))


def test_assembly_affinity_is_the_same_material_and_drive_free_energy():
    m = panel()
    p = m.params
    plus, minus = m.fluxes()['assembly']
    expected = (np.log(m.material[m.post])+np.log(m.positions-m.capacity)
                +p.charged_energy+np.log(m.charged[m.post])
                -p.structure_energy-np.log(m.capacity-m.bound)-np.log(m.depleted[m.post]))
    np.testing.assert_allclose(np.log(plus/minus), expected, atol=1e-14)
    assert m.entropy_rate() >= 0.


@pytest.mark.parametrize('dt', [.01, 1., 100.])
def test_growth_turnover_share_material_and_drive_with_nonincreasing_free_energy(dt):
    m = panel()
    initial_structure = m.capacity.copy()
    for _ in range(30):
        result = m.step(dt)
        assert result['free_energy_drop'] >= -1e-11
        assert np.all(m.bound <= m.capacity) and np.all(m.capacity <= m.positions)
        for value in m.balances().values():
            assert np.max(np.abs(value), initial=0.) < 3e-11
    assert not np.array_equal(m.capacity, initial_structure)


def test_bound_positions_protect_by_transition_availability_not_a_turnover_formula():
    p = StructuralExchangeParams(binding_rate=0., transport_rate=0., release_rate=0., assembly_rate=1.)
    m = StructuralExchange(np.array([0]), np.array([0]), [1.], [0.], [1.], [0.],
                           [0.], [1.], [2.], [0.], p)
    m.step(100.)
    assert m.capacity[0] == 1.
    assert m.material[0] == 0.
    # A separate physically unbound state has a disassembly path.
    unbound = StructuralExchange(np.array([0]), np.array([0]), [1.], [1.], [0.], [0.],
                                 [0.], [1.], [2.], [0.], p)
    unbound.step(100.)
    assert unbound.capacity[0] < 1.
    assert unbound.material[0] > 0.


@pytest.mark.parametrize('material,drive', [(0., .5), (.5, 0.)])
def test_empty_structure_cannot_grow_without_both_required_local_inputs(material, drive):
    p = StructuralExchangeParams(binding_rate=0., transport_rate=0., release_rate=0., assembly_rate=1.)
    m = StructuralExchange(np.array([0]), np.array([0]), [0.], [0.], [0.], [0.],
                           [drive], [1.-drive], [2.], [material], p)
    m.step(100.)
    assert m.capacity[0] == 0.


def test_edge_order_preserves_structure_and_material_allocation():
    a = panel()
    idx = np.array([2, 0, 1])
    b = StructuralExchange(a.pre[idx], a.post[idx], a.capacity[idx], a.free,
        a.bound[idx], a.carried, a.charged, a.depleted, a.positions[idx], a.material, a.params)
    for _ in range(30):
        a.step(.5)
        b.step(.5)
    np.testing.assert_allclose(a.capacity, b.capacity[np.argsort(idx)], atol=1e-14)
    for name in ('free','carried','charged','depleted','material'):
        np.testing.assert_allclose(getattr(a, name),getattr(b, name),atol=1e-14)


def test_assembly_time_refinement_matches_independent_continuous_process():
    p = StructuralExchangeParams(binding_rate=0.,transport_rate=0.,release_rate=0.,
                                 assembly_rate=.7,structure_energy=.5,charged_energy=1.)
    start = np.array([1., .5, .8, .2])  # n,B,P,D; bound C=.2 fixed
    def rhs(_, y):
        n,b,drive,waste = y
        net=.7*(b*(2.-n)*drive-np.exp(-.5)*(n-.2)*waste)
        return [net,-net,-net,net]
    exact=solve_ivp(rhs,[0,2.],start,rtol=1e-11,atol=1e-13).y[:,-1]
    errors=[]
    for dt in (.1,.05,.025):
        m=StructuralExchange(np.array([0]),np.array([0]),[1.],[1.],[.2],[.1],
                             [.8],[.2],[2.],[.5],p)
        for _ in range(round(2/dt)):m.step(dt)
        errors.append(np.linalg.norm(np.array([m.capacity[0],m.material[0],m.charged[0],m.depleted[0]])-exact))
    assert 1.8<errors[0]/errors[1]<2.2
    assert 1.8<errors[1]/errors[2]<2.2


def test_structure_and_exchange_archive_restore_all_future_state(tmp_path):
    m=panel()
    for _ in range(20):m.step(.5)
    path=tmp_path/'structural.npz'
    save_exchange(path,m,{'scope':'structure reaction interface'})
    fork,meta=load_exchange(path)
    assert isinstance(fork,StructuralExchange)
    assert meta=={'scope':'structure reaction interface'}
    for _ in range(20):m.step(.5);fork.step(.5)
    for name in m._arrays+('capacity',):
        np.testing.assert_array_equal(getattr(m,name),getattr(fork,name))
    assert m.energy()==fork.energy() and m.free_energy()==fork.free_energy()


def test_invalid_structure_capacity_and_tampered_material_ledger_fail():
    m=panel()
    state=m.state_dict()
    state['arrays']['material'][0]+=1.
    with pytest.raises(ValueError,match='accounting'):
        StructuralExchange.from_state_dict(state)
    with pytest.raises(ValueError):
        StructuralExchangeParams(assembly_rate=-1.)
