"""Numerical min-plus bounds, not a neural capacity experiment."""
import numpy as np
import pytest

from scripts.ib.audit_fly_port_arrival import earliest_arrival


def test_arrival_preserves_direction_delay_and_parallel_edges():
    edges = [(np.array([0, 0, 1, 1, 2, 4]), np.array([1, 1, 2, 3, 3, 0]),
              np.array([3, 1, 1, 3, 1, 1]))]
    result = earliest_arrival(6, np.array([0]), edges, 3)
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 4, 4])
    # Multiple hops fit the same horizon; the higher-delay parallel edge
    # cannot replace the faster one as a summed adjacency would do.
    limited = earliest_arrival(6, np.array([0]), edges, 1)
    np.testing.assert_array_equal(limited, [0, 1, 2, 2, 2, 2])


def test_arrival_requires_positive_chemical_delay():
    with pytest.raises(ValueError):
        earliest_arrival(2, np.array([0]), [(np.array([0]), np.array([1]), np.array([0]))], 1)
