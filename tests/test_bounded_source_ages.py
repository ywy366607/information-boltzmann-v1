"""Age bookkeeping and shared compression identities; not capability tests."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ib_local.bounded_source_ages import BoundedSourceAges
from scripts.ib_local.bounded_causal_tag_reference import Code
from scripts.ib_local.bounded_causal_tag_blocked import ArraySource, insert_source_inplace


def test_projection_drop_and_transport_match_explicit_cohort_bookkeeping():
    layout = SimpleNamespace(size=5, fields=['x'], slices={'x': slice(0,5)})
    code = Code.empty(2, 3, layout, dtype=np.float32)
    ages = BoundedSourceAges(2, 3, window=2)
    rng = np.random.default_rng(17)
    expected = [np.zeros((2,3)) for _ in range(3)]
    for window in range(3):
        ages.begin(2*window)
        for tick in range(2):
            # All cohorts share the existing transported basis.
            code.U *= .9
            owner = rng.normal(size=3)
            receipt = insert_source_inplace(code, ArraySource(rng.normal(size=5)), owner, layout, 3)
            for cohort, value in enumerate(expected):
                for j, a in enumerate(receipt['coefficients']):
                    if j == receipt['drop']:
                        value[j] = owner if cohort == window else 0
                    elif cohort == window:
                        value[j] += a*owner
            ages.insert(owner, receipt)
        ages.finish(2*(window+1))
        np.testing.assert_allclose(ages.older, sum(expected[:window]), atol=1e-14)
        np.testing.assert_allclose(ages.previous, expected[window], atol=1e-14)
        np.testing.assert_allclose(sum(expected)+ages.common_residual(code.V), code.V, atol=1e-14)
    assert ages.nbytes == 3*2*3*8
    c = rng.normal(size=2)
    np.testing.assert_allclose(ages.older_credit(c), c @ sum(expected[:2]))


def test_inherited_and_rounding_residual_are_not_called_strict_old():
    ages = BoundedSourceAges(2, 3)
    inherited = np.ones((2,3), np.float32)
    np.testing.assert_array_equal(ages.older_credit(np.ones(2)), np.zeros(3))
    np.testing.assert_array_equal(ages.common_residual(inherited), inherited)
    ages.begin(500)
    ages.insert(np.ones(3), dict(coefficients=np.array([1.,0.]), drop=None))
    ages.finish(628)
    assert not ages.older.any()  # Latest actual source is627: only1 event old.
    ages.begin(628)
    ages.finish(756)
    assert ages.older[0].sum() == 3  # Source627 now129 events old.


def test_age_state_roundtrip_and_invalid_clock_do_not_fabricate_history():
    ages = BoundedSourceAges(2,3)
    ages.begin(500)
    ages.insert(np.ones(3), dict(coefficients=np.ones(2), drop=0))
    stored = ages.state_dict()
    restored = BoundedSourceAges(2,3)
    restored.load_state_dict(stored)
    np.testing.assert_array_equal(restored.current, ages.current)
    assert restored.active
    with pytest.raises(ValueError):
        restored.older_credit(np.ones(2))
    restored.finish(628)
    with pytest.raises(ValueError, match='contiguous'):
        restored.begin(629)
    corrupted = copy.deepcopy(stored)
    corrupted['older'][0,0] = np.nan
    with pytest.raises(ValueError):
        restored.load_state_dict(corrupted)
