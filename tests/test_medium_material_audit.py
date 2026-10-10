"""Numerical invariants of read-only spatial statistics; no capability tasks."""
import numpy as np

from scripts.ib.diagnose_medium_material import components, neighbor_correlation


def test_periodic_components_join_across_faces():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = mask[3, 0, 0] = mask[2, 2, 2] = True
    assert components(mask) == [2, 1]
    assert components(np.zeros_like(mask)) == []


def test_neighbor_correlation_is_translation_and_scale_invariant():
    field = np.random.default_rng(42).normal(size=(4, 4, 4, 3))
    first = neighbor_correlation(field, (4, 4, 4))
    second = neighbor_correlation(np.roll(field, 1, 0) * 2 + 5, (4, 4, 4))
    np.testing.assert_allclose(first, second, atol=1e-14)
    assert neighbor_correlation(np.ones_like(field), (4, 4, 4)) is None
