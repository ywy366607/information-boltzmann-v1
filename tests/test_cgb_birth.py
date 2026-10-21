import numpy as np
import pytest

from information_boltzmann.cgb.birth import BirthParams, seed_from_contacts
from information_boltzmann.cgb.medium import CellClass, Medium, SynapseParams
from information_boltzmann.cgb.structural_budget import MaterialParams
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator


def generator():
    return TypeStrengthGenerator(dict(code=np.array([[1., 0.], [0., 1.]]),
        affinity=np.array([[0., 2.], [-2., 0.]]), out_bias=np.zeros(2),
        in_bias=np.zeros(2), bias=np.array(0.), geometry_weight=np.zeros(1),
        feature_mean=np.zeros(1), feature_scale=np.ones(1),
        types=np.array(['a', 'b']), symmetric=np.array(False)))


def birth(p, **kwargs):
    return seed_from_contacts(generator(), [0, 1], [CellClass()], [0, 0],
                              [0, 1], [1, 0], np.zeros((2, 1)), p,
                              params=BirthParams(1, seed=9),
                              syn=SynapseParams(gmax=.04), **kwargs)


def test_birth_separates_occurrence_and_directed_strength():
    m, report = birth([1, 1])
    assert m.g[0] > m.g[1]
    assert report['sampled_edges'] == 2
    assert report['unverified']
    none, _ = birth([0, 0])
    assert none.E == 0
    none.step()


def test_birth_material_is_paid_and_restore_does_not_rebirth():
    m, report = birth([1, 1], material=MaterialParams(), initial_free=.001)
    assert report['funded_structure'] < report['desired_structure']
    assert m.g.sum() <= .002 + 1e-12
    np.testing.assert_allclose(m.material.balance(m.g), 0, atol=1e-15)
    restored = Medium.from_state_dict(m.state_dict())
    np.testing.assert_array_equal(m.g, restored.g)
    np.testing.assert_array_equal(m.rng.random(8), restored.rng.random(8))


def test_seed_replays_occurrence_and_initial_structure():
    a, ra = birth([.5, .5])
    b, rb = birth([.5, .5])
    np.testing.assert_array_equal(a.pre, b.pre)
    np.testing.assert_array_equal(a.g, b.g)
    assert ra == rb


@pytest.mark.parametrize('p', [[-.1, 1], [np.nan, 0], [1, 2]])
def test_invalid_occurrence_is_rejected(p):
    with pytest.raises(ValueError, match='Occurrence'):
        birth(p)
