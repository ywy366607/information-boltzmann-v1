"""Conservation, numerical integration and persistent-state contracts only."""
import copy
import io
import pickle

import numpy as np
import pytest

from information_boltzmann.cgb.medium import CellClass, Medium, SynapseParams, TrophicParams
from information_boltzmann.cgb.structural_budget import MaterialParams, StructuralBudget


def assert_balanced(budget, g):
    np.testing.assert_allclose(budget.balance(g), 0, atol=2e-12, rtol=0)
    assert np.all(budget.free >= 0)


def test_growth_pays_local_stock_and_has_no_edge_order_priority():
    g = np.zeros(3)
    b = StructuralBudget(2, [0, 0, 1], g, MaterialParams(cost=2), [1, 0])
    funded = b.transfer(g, np.array([1., 3., 100.]))
    np.testing.assert_allclose(funded, [.125, .375, 0])
    assert_balanced(b, funded)


@pytest.mark.parametrize('recycle', [0, .3, 1])
def test_dismantling_returns_only_recoverable_material(recycle):
    g = np.array([1., 2.])
    b = StructuralBudget(1, [0, 0], g, MaterialParams(cost=2, recycle=recycle))
    g2 = b.transfer(g, np.zeros(2))
    np.testing.assert_allclose(b.free, [6*recycle])
    np.testing.assert_allclose(b.discarded, [6*(1-recycle)])
    assert_balanced(b, g2)


def test_supply_clearance_maintenance_and_starvation_have_a_complete_ledger():
    g = np.array([2., 1.])
    b = StructuralBudget(2, [0, 1], g,
                         MaterialParams(supply=.1, clearance=.2, maintenance=.5), [.3, .1])
    for dt in [.01, 1, 7, .03]:
        b.advance(g, dt)
        assert_balanced(b, g)
    assert b.unpaid.sum() > 0


def test_closed_system_cannot_accumulate_structure_by_recycling():
    rng = np.random.default_rng(21)
    g = np.zeros(13)
    b = StructuralBudget(3, np.arange(13) % 3, g, MaterialParams(cost=1.7, recycle=.8), .2)
    for _ in range(200):
        b.advance(g, .25)
        g = b.transfer(g, rng.uniform(0, 2, len(g)))
        assert_balanced(b, g)
        assert 1.7*g.sum() <= .6 + 1e-12


def make_medium(material=True):
    m = Medium([CellClass(), CellClass(E_out=-.3, inhibitory_plastic=True)],
               np.array([0, 0, 1]), np.array([0, 1, 2]), np.array([1, 2, 0]),
               np.array([.01, .02, .03]),
               SynapseParams(eta=.02, gmax=.1, lambda0=.0001), dt=.25, seed=5)
    m.inh_eta = .0001
    m.trophic = TrophicParams(rule='constitutive', capture='tag', lambda0=.0002)
    m.trophic_edges[:] = True
    if material:
        m.enable_material(MaterialParams(supply=.0001, maintenance=.0001, recycle=.7), .01)
    return m


@pytest.mark.parametrize('material', [False, True])
def test_complete_restore_is_bitwise_and_does_not_alias_parent(material):
    m = make_medium(material)
    for _ in range(70):
        m.step(I_ext=m.rng.uniform(0, .4, m.N), forced=m.rng.random(m.N) < .05)
    original = m.state_dict()
    fork = Medium.from_state_dict(pickle.load(io.BytesIO(pickle.dumps(original))))
    before = copy.deepcopy(original)
    fork.u[0] += 1
    np.testing.assert_array_equal(m.u, before['arrays']['u'])
    fork.u[0] = m.u[0]
    for _ in range(100):
        for obj in (m, fork):
            obj.step(I_ext=obj.rng.uniform(0, .4, obj.N), forced=obj.rng.random(obj.N) < .05)
        for name in Medium._state_arrays + ('g',):
            np.testing.assert_array_equal(getattr(m, name), getattr(fork, name))
        if material:
            for name in m.material._fields:
                np.testing.assert_array_equal(getattr(m.material, name), getattr(fork.material, name))
            assert_balanced(m.material, m.g)


def test_large_positive_tags_cannot_overshoot_finite_slots_or_stock():
    m = Medium([CellClass()], np.zeros(2, int), np.array([0]), np.array([1]),
               np.array([.001]), SynapseParams(eta=10, m0=1, gmax=.04))
    m.enable_material(MaterialParams(), .01)
    m.z[:] = 1e9
    m.step()
    assert 0 <= m.g[0] <= .04
    assert m.g[0] <= .011 + 1e-12
    assert_balanced(m.material, m.g)


def test_trophic_turnover_returns_material_even_when_learning_is_off():
    m = make_medium()
    m.plastic = False
    m.material.params = MaterialParams(recycle=.7)
    m.trophic = TrophicParams(rule='constitutive', q0=0, lambda0=.1)
    old_g, old_free = m.g.copy(), m.material.free.copy()
    m.step()
    assert (m.g < old_g).all()
    np.testing.assert_allclose(m.material.free-old_free,
                             .7 * m.material.total(old_g-m.g))
    assert_balanced(m.material, m.g)


def test_cannot_reinitialize_material_of_a_continuing_individual():
    with pytest.raises(ValueError, match='already enabled'):
        make_medium().enable_material()


def test_medium_constructor_owns_structure_and_connectivity():
    g, pre = np.array([.1]), np.array([0])
    m = Medium([CellClass()], np.zeros(2, int), pre, np.array([1]), g)
    m.g[:] = 0
    m.pre[:] = 1
    np.testing.assert_array_equal(g, [.1])
    np.testing.assert_array_equal(pre, [0])


def test_atomic_archive_restores_every_array_and_material_ledger(tmp_path):
    from information_boltzmann.cgb.checkpoint import load_medium, save_medium
    m = make_medium()
    for _ in range(20):
        m.step(forced=m.rng.random(m.N) < .1)
    path = tmp_path/'continuous.npz'
    save_medium(path, m, {'source': 'numerical_contract'})
    other, provenance = load_medium(path)
    assert provenance == {'source': 'numerical_contract'}
    for name in Medium._state_arrays + ('g',):
        np.testing.assert_array_equal(getattr(m, name), getattr(other, name))
    assert_balanced(other.material, other.g)
    np.testing.assert_array_equal(m.rng.random(10), other.rng.random(10))
    m.step()
    save_medium(path, m, {'source': 'second_complete_state'})
    restored, provenance = load_medium(path)
    assert restored.t == m.t
    assert provenance['source'] == 'second_complete_state'
    assert not path.with_name(path.name+'.tmp').exists()


def test_restore_preserves_changed_physiology_and_fast_edge_parameters():
    m = make_medium()
    m.theta[:] *= 1.3
    m.U_edge[:] *= .7
    m.C[:] *= 2
    m.E_edge[:] += .2
    other = Medium.from_state_dict(m.state_dict())
    for name in Medium._physiology_arrays:
        np.testing.assert_array_equal(getattr(m, name), getattr(other, name))
    for obj in (m, other):
        obj.step(forced=np.array([True, False, True]))
    for name in Medium._state_arrays:
        np.testing.assert_array_equal(getattr(m, name), getattr(other, name))


def test_unfunded_positive_tag_cannot_create_structure_legacy_control_can():
    def create():
        obj = Medium([CellClass()], np.zeros(2, int), np.array([0]), np.array([1]),
                     np.zeros(1), SynapseParams(eta=.1, m0=.1, gmax=.1))
        obj.z[:] = 1
        return obj
    legacy, funded = create(), create()
    funded.enable_material(MaterialParams(), 0)
    legacy.step()
    funded.step()
    assert legacy.g[0] > 0
    assert funded.g[0] == 0
    assert_balanced(funded.material, funded.g)
